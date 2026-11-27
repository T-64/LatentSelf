import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock,patch

from latentself.interp.label import evaluate,measure,snippet,split_examples


class LabelTest(unittest.TestCase):
    def test_labeler_and_reader_can_run_at_different_times(self):
        from latentself.db import connect
        from latentself.interp.experiment import register
        with TemporaryDirectory() as tmp:
            db = connect(Path(tmp)/"db")
            self.addCleanup(db.close)
            run = register(db,"9b","bf16",8,"card","turn_self")
            llm = Mock(model="fixture")
            bundle = dict(training=dict(strong=["x"],random_active=[],inactive=["y"]),held=[("a",True),("b",False)])
            with patch("latentself.interp.label.prepare_examples",return_value=bundle) as reader:
                self.assertEqual(measure(db,None,llm,run,[7],stage="prepare")["prepared"],1)
                stats = measure(db,None,llm,run,[7,8,9],stage="prepare",limit=1)
                self.assertEqual((stats["skipped"],stats["prepared"],stats["limited"]),(1,1,1))
                llm.chat.assert_not_called()
                llm.chat.side_effect = [dict(label="标签",note="说明"),dict(active=[True,False])]
                self.assertEqual(measure(db,None,llm,run,[7],stage="score")["scores"],[1.0])
                self.assertEqual(measure(db,None,llm,run,[7],stage="score")["skipped"],1)
                self.assertEqual(reader.call_count,2)

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
