from pathlib import Path
import tempfile
import unittest

from latentself.db import connect
from latentself.evidence import SourceTurn
from latentself.titles import sample_turns,titles
from tests.fakes import FakeLLM


class TitlesTest(unittest.TestCase):
    def test_long_dialogue_keeps_both_ends_and_counts_sampling(self):
        turns = [SourceTurn(str(i),"2020","",str(i)*900) for i in range(12)]
        text,loss = sample_turns(turns)
        self.assertLessEqual(len(text),6000)
        self.assertGreater(loss,0)
        self.assertTrue(text.startswith("0"*900))
        self.assertTrue(text.endswith("11"*900))
        self.assertIn("5"*100,text)

    def test_incremental_summary_includes_only_user_material(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = connect(Path(tmp)/"test.db")
            self.addCleanup(db.close)
            db.execute("INSERT INTO conversation(id,source) VALUES ('c','fixture')")
            db.execute("INSERT INTO turn VALUES ('t','fixture','prompt','2020','reading','s')")
            db.execute("INSERT INTO turn_conv VALUES ('t','c',0)")
            db.execute("INSERT INTO answer VALUES ('t',0,'','assistant-only')")
            db.execute("""INSERT INTO attachment(id,turn_id,pos,stored_name,match,caption,origin)
                        VALUES ('a','t',0,'x','exact','generated-only','generated')""")
            llm = FakeLLM({"title":"阅读","summary":"用户在阅读。","kinds":["学习"]})
            self.assertEqual(titles(db,llm,limit=0)["limited"],1)
            self.assertEqual(titles(db,llm,emit=lambda _:None)["success"],1)
            self.assertEqual(llm.calls[0][0][-1]["content"],"reading")
            self.assertEqual(titles(db,llm)["skipped"],1)
            db.execute("UPDATE attachment SET origin='user',caption='owner image'")
            self.assertEqual(titles(db,llm,emit=lambda _:None)["success"],1)
            self.assertIn("owner image",llm.calls[-1][0][-1]["content"])
