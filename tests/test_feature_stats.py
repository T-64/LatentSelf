import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from latentself.analysis.features import adjust,compare,difference,enrichment,npmi,profile
from latentself.db import connect
from latentself.interp.encode import store_unit


class FeatureStatisticsTest(unittest.TestCase):
    def test_smoothed_odds_and_multiple_tests(self):
        odds,z,p = difference(5,10,0,10)
        self.assertAlmostEqual(odds,math.log(21))
        self.assertAlmostEqual(z,2.581988897)
        self.assertLess(p,.01)
        adjusted = adjust(dict(a=.01,b=.04,c=.2))
        for key,expected in dict(a=.03,b=.06,c=.2).items():
            self.assertAlmostEqual(adjusted[key],expected)

    def test_retraces_sources_and_excludes_later_units(self):
        with TemporaryDirectory() as tmp:
            db = connect(Path(tmp)/"db")
            self.addCleanup(db.close)
            for i in range(10):
                tid = str(i)
                db.execute("INSERT INTO turn(id,source,kind,ts,prompt,snapshot,is_paste) VALUES (?,'s','prompt',?,'p','s',0)",
                           (tid,f"2024-01-{i+1:02d}"))
                data = dict(n_tokens=2,n_chunks=1,fvu=0,features=[(1,2,1),(2,2,1)] if i<5 else [])
                store_unit(db,"r",tid,"h",data)
                store_unit(db,"b",tid,"h",dict(data,features=[]))
            self.assertEqual(enrichment(db,"r","b")["rankable"],2)
            self.assertEqual(npmi(db,"r",lambda a,b:0,3)[0]["npmi"],1)
            self.assertFalse(npmi(db,"r",lambda a,b:1,3))
            left = set("01234")
            self.assertEqual(set(compare(db,"r",left,set("56789"))["features"][0]["samples"]),set("012"))
            db.execute("UPDATE sae_act SET act_z=1 WHERE run_id='r'")
            self.assertEqual(set(profile(db,"r",before="2024-01-03")[0]["evidence"]),set("01"))
