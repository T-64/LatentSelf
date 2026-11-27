from pathlib import Path
import tempfile
import unittest

from latentself.db import connect
from latentself.evaluation import evaluation_db,label_batch,metrics,read_queries


class EvaluationTest(unittest.TestCase):
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
            ev.executemany("INSERT INTO pool VALUES ('synthetic: label',?,'[]','fixture')",[('a',),('b',)])
            for choice,expected in [('y',1),('q',0),('n',1)]:
                self.assertEqual(label_batch(con,ev,rows,limit=1,ask=lambda _:choice,emit=lambda _:None),expected)
            self.assertEqual(dict(ev.execute("SELECT cid,relevant FROM label")),{'a':1,'b':0})
