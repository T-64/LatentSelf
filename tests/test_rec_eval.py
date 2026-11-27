from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from latentself.catalog import dumps
from latentself.db import connect
from latentself.rec_eval import scores,split
from latentself.rec_output import valid_sentence
from latentself.recommend import feedback_factors,ranked_items


class RecommendationEvaluationTest(unittest.TestCase):
    def test_first_hit_after_rank_fifty_contributes_only_to_reciprocal_rank(self):
        films = [dict(id=str(i),components=(70-i,0,0,0),wanted=False,creators=[],vec=np.array([1.]))
                 for i in range(70)]
        order = (r["id"] for r in ranked_items(films))
        value = scores(order,["60"])
        self.assertEqual(value,dict(recall20=0,recall50=0,ndcg20=0,mrr=1/61))
        self.assertEqual(next(order),"61")

    def test_equal_timestamps_stay_on_one_side_and_metrics(self):
        cutoff,train,held = split([(str(i),f"2024-01-{min(i+1,9):02d}") for i in range(10)])
        self.assertEqual(held,["8","9"])
        self.assertTrue(all(ts<cutoff for _,ts in train))
        result = scores(["x","8","9"],held)
        self.assertEqual(result["recall20"],1)
        self.assertEqual(result["mrr"],.5)
        self.assertLess(result["ndcg20"],1)
        self.assertTrue(valid_sentence("你提到“读书”。",["周末读书。"]))
        self.assertFalse(valid_sentence("你提到“骑马”。",["周末读书。"]))
        self.assertFalse(valid_sentence("“缺少右引号",["缺少右引号"]))

    def test_feedback_caps_and_cutoff(self):
        with TemporaryDirectory() as tmp:
            con = connect(Path(tmp)/"db")
            self.addCleanup(con.close)
            for i in range(12):
                iid,tid = f"film{i}",str(i)
                date = f"2024-01-{i+1:02d}"
                con.execute("INSERT INTO turn VALUES (?,'s','feedback',?,'','s',NULL)",(tid,date))
                con.execute("INSERT INTO signal VALUES (?,'rec_feedback',?,NULL)",(tid,dumps(dict(item=iid,action="like" if i%2 else "dislike"))))
                con.execute("INSERT INTO sae_act VALUES ('r',?,?,2,1,1)",("item:"+iid,i%2))
            self.assertEqual(feedback_factors(con,"r"),{0:.5,1:1.5})
            self.assertEqual(feedback_factors(con,"r","2024-01-02"),{0:.8})
