from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from latentself.db import connect
from latentself.units import is_paste,review_paste,units


class AnalysisUnitsTest(unittest.TestCase):
    def test_paste_rules(self):
        self.assertFalse(is_paste("我想了解这位作者"))
        self.assertTrue(is_paste("来源：一本书"))
        self.assertTrue(is_paste("English text "*80))
        self.assertTrue(is_paste("中文"*800+"\n# 标题\n- 条目"))
        self.assertFalse(is_paste("中文"*800))

    def test_half_hour_boundary_and_review_threshold(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            con = connect(root/"db")
            self.addCleanup(con.close)
            times = ["00:00","00:30","01:01"]+["02:00"]*17
            con.executemany("INSERT INTO turn(id,ts,prompt,source,kind,snapshot) VALUES (?,?,?,'fixture','prompt','s')",
                            [(str(i),"2024-05-08T"+ts+":00Z","study") for i,ts in enumerate(times)])
            self.assertEqual(units(con),dict(turns=20,episodes=3))
            self.assertEqual(units(con,True),dict(turns=20,episodes=3))
            answers = iter(["n"]*2+["y"]*18)
            report = review_paste(con,root,100,answer=lambda _:next(answers))
            self.assertTrue(report["passed"])
            self.assertTrue(Path(report["path"]).exists())
