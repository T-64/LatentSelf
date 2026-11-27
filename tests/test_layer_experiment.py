import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from latentself.db import connect
from latentself.interp.experiment import MODELS,experiment,select


class SelectionTest(unittest.TestCase):
    def test_formula_pilot_reuses_the_same_sample_and_does_not_publish_selection_metrics(self):
        with TemporaryDirectory() as root:
            db = connect(Path(root)/"fixture")
            self.addCleanup(db.close)
            db.execute("INSERT INTO turn(id,source,kind,ts,prompt,snapshot) VALUES ('t','test','prompt','2024','sample','s')")
            db.execute("INSERT INTO turn(id,source,kind,ts,prompt,snapshot) VALUES ('u','test','prompt','2024','second sample','s')")
            cli = Mock()
            reader,repo,_,layers = MODELS["9b"]
            def encode(texts,chosen,*,formula,experiment):
                loss = 1 if formula=="card" else 2
                return [{str(chosen[0]):dict(reader=reader,sae_repo=repo,precision="bf16",formula=formula,sse=loss,
                         variance=2,fvu=loss/2,n_tokens=2,n_chunks=1,features=[[1,1.,1]])} for _ in texts]
            cli.encode.side_effect = encode
            for _ in range(2):
                metrics = experiment(db,cli,None,"9b","bf16",stage="pilot",emit=lambda _:None)
                self.assertEqual([m["layer"] for m in metrics],layers)
                self.assertTrue(all(m["formula"]=="card" and m["pilot_units"]==2 for m in metrics))
            self.assertEqual(cli.encode.call_count,2*len(layers))
            db.execute("DELETE FROM meta WHERE k=(SELECT min(k) FROM meta WHERE k LIKE 'sae_exp:%')")
            experiment(db,cli,None,"9b","bf16",stage="pilot",emit=lambda _:None)
            self.assertEqual(cli.encode.call_count,2*len(layers)+1)
            self.assertEqual(len(cli.encode.call_args.args[0]),1)
            self.assertFalse(db.execute("SELECT 1 FROM meta WHERE k LIKE 'sae_metric:%'").fetchone())

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
