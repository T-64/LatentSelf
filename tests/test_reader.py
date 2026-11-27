import importlib.util
import unittest
from unittest.mock import patch

from latentself.interp.reader import Reader


@unittest.skipUnless(importlib.util.find_spec("torch"),"server-side PyTorch")
class TorchReaderTest(unittest.TestCase):
    def test_post_layer_hook_and_reconstructed_logits(self):
        import torch
        from types import SimpleNamespace
        torch.manual_seed(1)

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(16,8)
                self.layers = torch.nn.ModuleList([torch.nn.Linear(8,8)])
            def forward(self,input_ids,use_cache=False):
                return self.layers[0](self.emb(input_ids))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()
                self.lm_head = torch.nn.Linear(8,16)
            def forward(self,ids,use_cache=False):
                return SimpleNamespace(logits=self.lm_head(self.model(ids)))

        reader = Reader.__new__(Reader)
        reader.torch,reader.model,reader.device = torch,Model().eval(),torch.device("cpu")
        reader.reader,reader.sae_repo,reader.precision,reader.formula = "fixture","fixture","test","card"
        reader.layers = reader.model.model.layers
        reader.tokenizer = lambda text,**_:dict(input_ids=[1,2,3,4],offset_mapping=[(i,i+1) for i in range(4)])
        reader.loaded = {0:{k:torch.randn(*shape) for k,shape in
                            dict(W_enc=(64,8),W_dec=(8,64),b_enc=(64,),b_dec=(8,)).items()}}
        for formula in ("card","centered"):
            result = reader.encode_layer("demo",0,formula=formula,experiment=True,feature=1)
            self.assertEqual(result["ce_tokens"],3)
            self.assertIsNotNone(result["delta_ce"])
            self.assertEqual(len(result["offsets"]),4)
            self.assertEqual(result["token_acts"][0],0)
            self.assertTrue(all(0<idx<4 for _,_,idx in result["features"]))
            self.assertGreater(result["fvu"],0)
            self.assertEqual(len(reader.layers[0]._forward_hooks),0)
            # This fixture has no cross-token attention. Rechunking cannot change its loss.
            with patch("latentself.interp.reader.chunks", return_value=[(0,3),(2,4)]):
                split = reader.encode_layer("demo",0,formula=formula,experiment=True)
            self.assertEqual(split["ce_tokens"],result["ce_tokens"])
            self.assertAlmostEqual(split["delta_ce"],result["delta_ce"],places=4)
        self.assertEqual(len(reader.neighbors(0,3)),10)
