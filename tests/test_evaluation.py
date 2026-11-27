from pathlib import Path
import tempfile
import unittest

from latentself.db import connect
from latentself.evaluation import evaluation_db,extend_pool,label_batch,metrics,prepare_photo_review,read_queries
from tests.fakes import FakeLLM
from unittest.mock import patch


class EvaluationTest(unittest.TestCase):
    def test_early_photo_labels_survive_full_pool_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            con = connect(Path(folder)/"main.db")
            ev = evaluation_db(Path(folder)/"labels.db")
            self.addCleanup(con.close)
            self.addCleanup(ev.close)
            con.execute("INSERT INTO turn VALUES ('t','fixture','prompt','2020',?,'snap')",("striped kite"*50,))
            con.execute("INSERT INTO conversation(id,source,n_turns) VALUES ('c','fixture',1)")
            con.execute("INSERT INTO turn_conv VALUES ('t','c',0)")
            con.execute("INSERT INTO fts_turn VALUES ('t','striped kite','','')")
            con.execute("""INSERT INTO attachment(id,turn_id,pos,path,stored_name,match,mime,origin)
                VALUES ('a','t',0,'kite.jpg','kite.jpg','exact','image/jpeg','user')""")
            rows = [{"q":"kite","photo":True}]
            llm = FakeLLM({"terms":[f"alias{i}" for i in range(8)]})
            for _ in range(2):
                summary = list(prepare_photo_review(con,ev,rows,llm))[0]
                self.assertEqual(summary["candidates"],1)
                self.assertGreater(summary["judge_prompt_chars_omitted"],0)
            self.assertEqual(ev.execute("SELECT count(*) FROM pool").fetchone()[0],0)
            responses = iter(["v 1","y","1"])
            output = []
            def answer(_):
                self.assertIn("1  kite.jpg",output)
                return next(responses)
            with patch("latentself.evaluation.subprocess.run") as preview:
                preview.return_value.returncode = 0
                self.assertEqual(label_batch(con,ev,rows,ask=answer,emit=output.append),1)
                preview.assert_called_once_with(["/usr/bin/qlmanage","-p",str(Path("kite.jpg").absolute())],capture_output=True)
            ev.execute("""INSERT INTO pool VALUES ('kite','c','["keyword","strict"]','updated evidence')""")
            self.assertEqual(label_batch(con,ev,rows,ask=lambda _:self.fail("duplicate review")),0)
            self.assertEqual(tuple(ev.execute("SELECT relevant,photo FROM label").fetchone()),(1,"kite.jpg"))

    @patch("latentself.search.configured_indexes")
    @patch("latentself.evaluation.search")
    def test_full_index_candidates_extend_once(self, search, configured):
        from types import SimpleNamespace
        configured.return_value = [SimpleNamespace(ids=["turn"])]
        search.return_value = {"notices":[],"results":[{"conv_id":"a","first_prompt":"fixture","snippets":[]}]}
        with tempfile.TemporaryDirectory() as folder:
            con = connect(Path(folder)/"main.db")
            ev = evaluation_db(Path(folder)/"labels.db")
            self.addCleanup(con.close)
            self.addCleanup(ev.close)
            for expected in (1,0):
                self.assertEqual(list(extend_pool(con,ev,[{"q":"fixture"}])),[(1,expected)])

    def test_metrics_preserve_unknowns_and_require_actual_photo(self):
        rows = [{"conv_id":str(i),"images":[f"{i}.png"]} for i in range(6)]
        gold = {"1":(True,"5.png"),"0":(False,None),"miss":(True,None)}
        result = metrics(rows,gold,True)
        self.assertEqual(result["recall"],0.5)
        self.assertEqual(result["precision"],1/6)
        self.assertFalse(result["photo_top5"])
        self.assertEqual(result["unjudged"],["2","3","4","5"])
        self.assertEqual(result["missed"],["miss"])
        self.assertTrue(metrics(rows[1:],gold,True)["photo_top5"])

    def test_yaml_and_resumable_owner_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "queries.yaml"
            path.write_text('queries:\n  - {q: "synthetic: label", photo: false}\n')
            rows = read_queries(path)
            con = connect(root/"main.db")
            ev = evaluation_db(root/"labels.sqlite3")
            self.addCleanup(con.close)
            self.addCleanup(ev.close)
            ev.executemany("INSERT INTO pool VALUES ('synthetic: label',?,?,'fixture')",
                           [('a','["keyword","strict"]'),('b','["long synthetic model name"]')])
            for choice,expected in [('y',1),('q',0),('n',1)]:
                self.assertEqual(label_batch(con,ev,rows,limit=1,ask=lambda _:choice,emit=lambda _:None),expected)
            self.assertEqual(dict(ev.execute("SELECT cid,relevant FROM label")),{'b':1,'a':0})
