import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.fakes import FakeEmbed, FakeLLM
from latentself.backends.embed import TEXT, VL
from latentself.calibration import calibrate, cascade_bounds, choose_model
from latentself.db import connect
from latentself.embed import index
from latentself.evaluation import evaluation_db, evaluate, prepare
from latentself.ingest.gemini import ingest
from latentself.search import rank_conversations, read_policy, VectorIndex


class CalibrationTest(unittest.TestCase):
    def test_model_cutoff_and_calibrated_border(self):
        self.assertEqual(choose_model({TEXT:0.97,VL:0.94}),TEXT)
        self.assertEqual(choose_model({TEXT:0.969,VL:0.94}),VL)
        low,high = cascade_bounds([(0.3,True),(0.8,False)])
        self.assertLess(low,0.3)
        self.assertGreater(high,0.8)

    def test_pool_to_report_with_two_offline_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = [{"title":"Prompted "+term,"time":f"2024-02-0{i}T12:00:00Z",
                     "details":[{"url":"https://gemini.google.com/app/"+cid}]}
                    for i,(term,cid) in enumerate([("fixture","aa"),("unrelated","bb")],1)]
            (root/"MyActivity.json").write_text(json.dumps(data))
            con = connect(root/"test.db")
            self.addCleanup(con.close)
            ingest(con,root)
            backends = []
            for name in [VL,TEXT]:
                backend = FakeEmbed()
                backend.model = name
                index(con,backend,emit=lambda _:None)
                backends.append(backend)
            other = con.execute("SELECT id FROM turn WHERE prompt='unrelated'").fetchone()[0]
            con.execute("""INSERT INTO attachment(id,turn_id,pos,stored_name,path,match,mime,origin)
                VALUES ('visual',?,0,'fixture.png','fixture.png','exact','image/png','user')""",(other,))
            con.execute("""INSERT INTO embedding SELECT 'attachment','visual',e.model,e.dim,e.input_hash,e.vec
                FROM embedding e JOIN turn t ON t.id=e.item_id
                WHERE e.item_kind='turn' AND e.model=? AND t.prompt='fixture'""",(VL,))
            def reply(messages):
                if "candidates" not in messages[-1]["content"]:
                    return {"terms":[f"alias{i}" for i in range(8)]}
                batch = json.loads(messages[-1]["content"])["candidates"]
                return {"results":[{"conv_id":r["conv_id"],"relevant":"fixture" in r["excerpt"],"reason":"fixture"} for r in batch]}
            llm = FakeLLM(reply)
            class Reranker:
                model = "fixture-reranker"

                def score(self,query,docs):
                    return [1.0 if "fixture" in doc else 0.0 for doc in docs]
            rows = [{"q":"fixture"}]
            ev = evaluation_db(root/"labels.db")
            self.addCleanup(ev.close)
            with patch.dict(rows[0],photo=True):
                con.execute("SAVEPOINT without_photo")
                con.execute("DELETE FROM embedding WHERE item_kind='attachment'")
                with self.assertRaisesRegex(ValueError,"1 more visual"):
                    list(prepare(con,ev,rows,backends,llm))
                con.execute("ROLLBACK TO without_photo")
                con.execute("RELEASE without_photo")
            con.execute("SAVEPOINT stale_text")
            con.execute("UPDATE embedding SET input_hash='outdated' WHERE item_kind='turn' AND model=?",(TEXT,))
            with self.assertRaisesRegex(ValueError,"changed between model runs"):
                list(prepare(con,ev,rows,backends,llm))
            con.execute("ROLLBACK TO stale_text")
            con.execute("RELEASE stale_text")
            # Make the unrelated fixture a single-route candidate in this tiny corpus.
            original_recall = VectorIndex.recall
            def one_disagreement(idx,*args):
                scores,omitted = original_recall(idx,*args)
                if idx.backend.model==TEXT and other in scores:
                    del scores[other]
                    omitted += 1
                return scores,omitted
            with patch.object(VectorIndex,"recall",one_disagreement):
                summary = list(prepare(con,ev,rows,backends,llm))[0]
            self.assertEqual((summary["candidates"],summary["consensus_labels"],summary["pending_owner_labels"]),(2,1,1))
            self.assertEqual((summary["strict_candidates"],summary["strict_unneeded"]),(1,1))
            with self.assertRaisesRegex(ValueError,"owner review"):
                calibrate(con,ev,rows,backends,llm,Reranker())
            ev.execute("INSERT INTO label VALUES ('fixture','bb',0,NULL)")
            calls_before = [list(b.calls) for b in backends]
            scorer = Reranker()
            for winner in (VL,TEXT):
                with patch("latentself.calibration.choose_model",return_value=winner), \
                        patch("latentself.calibration.rank_conversations",wraps=rank_conversations) as ranked:
                    prepared = calibrate(con,ev,rows,backends,llm,scorer,score_only=True)
                    self.assertGreater(ranked.call_args.args[1][other],0.99)
            self.assertEqual(prepared["candidates"],2)
            self.assertEqual(read_policy(con),{})
            def offline(*args):
                self.fail("the reranker should be unloaded during LLM calibration")
            scorer.score = offline
            result = calibrate(con,ev,rows,backends,llm,scorer)
            replay = result["strict"]["cascade"]["queries"][0]["replayed_rerank_seconds"]
            self.assertAlmostEqual(replay,prepared["rerank_serving_seconds"])
            self.assertEqual([b.calls for b in backends],calls_before)
            self.assertEqual(read_policy(con),result["selected"])
            self.assertEqual(result["selected"]["text_model"],VL)
            policy = {**result["selected"],"judge":"llm"}
            report = list(evaluate(con,ev,rows,indexes=[VectorIndex(con,backends[0],("turn",))],policy=policy,llm=llm))
            self.assertEqual([r["accepted"] for r in report],[True,True])
            photo = list(prepare(con,ev,[{"q":"photo fixture","photo":True}],backends,llm))[0]
            self.assertEqual((photo["strict_candidates"],photo["pending_owner_labels"],photo["strict_unneeded"]),(0,0,1))
            def disagree(messages):
                value = reply(messages)
                for item in value.get("results",[]):
                    item["relevant"] = False
                return value
            revised = FakeLLM(disagree)
            revised.model = "another-fixture-judge"
            # Both retrievers still vote for the same fixture despite a changed strict verdict.
            revised_pool = list(prepare(con,ev,rows,backends,revised))[0]
            self.assertEqual((revised_pool["consensus_labels"],revised_pool["pending_owner_labels"]),(1,0))
            self.assertEqual(dict(ev.execute("SELECT cid,relevant FROM label WHERE q='fixture'")),{"aa":1,"bb":0})
            # After recall changes, only the keyword route remains for the former consensus.
            with patch.object(VectorIndex,"recall",return_value=({},2)):
                revised_pool = list(prepare(con,ev,rows,backends,revised))[0]
            self.assertEqual((revised_pool["consensus_labels"],revised_pool["pending_owner_labels"]),(0,1))
            self.assertEqual(dict(ev.execute("SELECT cid,relevant FROM label WHERE q='fixture'")),{"bb":0})
