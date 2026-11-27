import json
from pathlib import Path
import tempfile
import unittest

from tests.fakes import FakeEmbed, FakeLLM
from latentself.backends.embed import TEXT, VL
from latentself.calibration import calibrate, cascade_bounds, choose_model
from latentself.db import connect
from latentself.embed import index
from latentself.evaluation import evaluation_db, evaluate, prepare
from latentself.ingest.gemini import ingest
from latentself.search import read_policy, VectorIndex


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
            def reply(messages):
                if "candidates" not in messages[-1]["content"]:
                    return {"terms":[f"alias{i}" for i in range(8)]}
                batch = json.loads(messages[-1]["content"])["candidates"]
                return {"results":[{"conv_id":r["conv_id"],"relevant":r["conv_id"]=="aa","reason":"fixture"} for r in batch]}
            llm = FakeLLM(reply)
            class Reranker:
                def score(self,query,docs):
                    return [1.0 if "fixture" in doc else 0.0 for doc in docs]
            rows = [{"q":"fixture"}]
            ev = evaluation_db(root/"labels.db")
            self.addCleanup(ev.close)
            self.assertEqual(list(prepare(con,ev,rows,backends,llm))[0]["candidates"],2)
            with self.assertRaisesRegex(ValueError,"owner review"):
                calibrate(con,ev,rows,backends,llm,Reranker())
            ev.execute("INSERT INTO label VALUES ('fixture','bb',0,NULL)")
            result = calibrate(con,ev,rows,backends,llm,Reranker())
            self.assertEqual(read_policy(con),result["selected"])
            self.assertEqual(result["selected"]["text_model"],VL)
            policy = {**result["selected"],"judge":"llm"}
            report = list(evaluate(con,ev,rows,indexes=[VectorIndex(con,backends[0],("turn",))],policy=policy,llm=llm))
            self.assertEqual([r["accepted"] for r in report],[True,True])
