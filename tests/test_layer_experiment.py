import unittest

from latentself.interp.experiment import MODELS,select


class SelectionTest(unittest.TestCase):
    def test_reader_upgrade_requires_all_thresholds(self):
        metrics = {}
        for name,precision,accuracy,error,hours in [("9b","bf16",.7,.5,5),("27b","bf16-offload",.78,.4,25),
                                                    ("27b","int8",.78,.48,8)]:
            metrics[name+"/"+precision] = [dict(model=name,precision=precision,layer=layer,sample=2000,
                features_scored=50,detect_score=accuracy,fvu=error,full_hours=hours) for layer in MODELS[name][3]]
        self.assertEqual(select(metrics)["model"],"9b")
        metrics["27b/int8"][1]["fvu"] = .42
        self.assertEqual(select(metrics)["layer"],24)
        metrics["27b/int8"][1]["features_scored"] = 49
        with self.assertRaises(ValueError):
            select(metrics)
