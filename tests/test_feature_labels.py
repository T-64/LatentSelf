import unittest
from unittest.mock import Mock

from latentself.interp.label import evaluate,snippet,split_examples


class LabelTest(unittest.TestCase):
    def test_no_holdout_reuse_balanced_scoring_and_exact_span(self):
        for n in (5,30,50):
            best,random,neg,holdout = split_examples(range(n),range(n,n+30),42)
            self.assertFalse(set(best+random+neg)&{u for u,_ in holdout})
            self.assertFalse(set(best)&set(random))
            self.assertTrue(holdout)
        llm = Mock()
        llm.chat.return_value = dict(active=[True,True,True,True])
        self.assertEqual(evaluate(llm,{},[("a",True),("b",True),("c",True),("d",False)]),.5)
        self.assertEqual(snippet("abc",[(0,1),(1,2),(2,3)],1,True),"a«b»c")
