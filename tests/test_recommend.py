from collections import Counter
import unittest
import numpy as np

from latentself.recommend import prior,rank


class RankingTest(unittest.TestCase):
    def test_components_diversity_and_caps(self):
        pool = [dict(id=str(i),components=[1-i/11,.4,.5,.6],vec=np.array([i%2,1-i%2],dtype=float),
                     creators=["shared" if i<5 else str(i)],wanted=i%2==0) for i in range(10)]
        results = rank(pool,10)
        self.assertAlmostEqual(results[0]["score"],.4+.16+.05+.06)
        self.assertLessEqual(sum(r["wanted"] for r in results),4)
        self.assertLessEqual(max(Counter(c for r in results for c in r["creators"]).values()),2)
        self.assertEqual(prior([dict(src="tspdt-top1000",rank=1)]),1)
        self.assertEqual(prior([dict(src="tspdt-recommendations")]),.3)
