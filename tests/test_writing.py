import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from latentself.db import connect
from latentself.read_api import Reader
from latentself.writing import write,writer


class WritingTest(unittest.TestCase):
    def test_consent_each_time(self):
        for answer in ("","N","no"):
            with self.assertRaises(ValueError):
                writer("claude",lambda _:answer)

    def test_outline_is_editable_and_unverified_sentences_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"fixture"
            db = connect(path)
            self.addCleanup(db.close)
            db.execute("INSERT INTO turn VALUES ('t','fixture','prompt','2024-03-02','I built a bridge.','s')")
            db.execute("INSERT INTO conversation(id,source,title) VALUES ('c','fixture','Bridge')")
            db.execute("INSERT INTO turn_conv VALUES ('t','c',0)")
            db.commit()
            reader = Reader(path)
            item = dict(conv_id="c",snippets=[dict(turn_id="t")])
            with patch.object(reader,"search",return_value=dict(results=[item],omitted=0)),patch("latentself.writing.ask") as ask:
                ask.return_value = "## Original outline"
                report = write(db,reader,object(),tmp,"bridge")
                self.assertTrue(Path(report["timeline"]).is_file())
                self.assertNotIn("draft",report)
                outline = Path(report["outline"])
                outline.write_text("## Revised\nKeep this detail.")
                write(db,reader,object(),tmp,"bridge")
                ask.assert_called_once()
                ask.side_effect = [dict(sentences=[dict(text="Built a bridge. Won an award.",sources=[1])]),
                                   dict(verdict="supported"),dict(verdict="insufficient")]
                report = write(db,reader,object(),tmp,"bridge",True)
                self.assertIn("Keep this detail",ask.call_args_list[1].args[2]["chapter"])
                self.assertEqual(report["unverified"],1)
                draft = Path(report["draft"]).read_text()
                self.assertIn("[待核实] Won an award.",draft)
                self.assertIn("https://gemini.google.com/app/c",draft)
                self.assertIn("[t:t]",draft)
                self.assertEqual(outline.read_text(),"## Revised\nKeep this detail.")
