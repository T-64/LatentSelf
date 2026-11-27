from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from latentself.db import connect
from latentself.interp.encode import standardize,store_unit
from latentself.interp.sae_io import chunks,Pool,topk_sae


class SparseTest(unittest.TestCase):
    def test_reference_sae_and_overlap_max(self):
        rng = np.random.default_rng(7)
        weights = {k:rng.normal(size=s) for k,s in
                   dict(W_enc=(256,64),W_dec=(64,256),b_enc=256,b_dec=64).items()}
        x = rng.normal(size=(9,64))
        for formula in ("card","centered"):
            ix,acts,recon = topk_sae(x,weights,8,formula)
            self.assertEqual(recon.shape,x.shape)
            self.assertEqual(ix.shape,(9,8))
            self.assertTrue(np.all(np.diff(acts,axis=1)<=0))
        self.assertEqual(list(chunks(9,5,2)),[(0,5),(3,8),(6,9)])
        pooled = Pool()
        pooled.add(0,[[1],[1],[1]],[[99],[2],[4]])
        pooled.add(2,[[1],[2]],[[4],[7]])
        self.assertEqual(pooled.strongest(),[(2,7,3),(1,4,2)])

    def test_zeros_contribute_to_length_bucket_moments(self):
        with TemporaryDirectory() as tmp:
            db = connect(Path(tmp)/"fixture")
            self.addCleanup(db.close)
            for i in range(20):
                store_unit(db,"fixture",str(i),"x",dict(n_tokens=i+2,n_chunks=1,fvu=.25,
                           features=[(3,2,1)] if i%2==0 else []))
            self.assertEqual(standardize(db,"fixture"),dict(buckets=10,units=20))
            self.assertEqual({row[0] for row in db.execute("SELECT act_z FROM sae_act")},{1.0})
            with self.assertRaises(ValueError):
                store_unit(db,"fixture","0","y",dict(n_tokens=2,n_chunks=1,fvu=0,features=[(3,4,2)]))
            self.assertEqual(db.execute("SELECT count(*) FROM sae_act").fetchone()[0],10)
