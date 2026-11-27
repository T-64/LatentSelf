import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from latentself.db import connect
from latentself.evidence import SourceTurn
from latentself.facts import extract,fact_batches,valid_date
from tests.fakes import FakeLLM


class FactExtractTest(unittest.TestCase):
    def test_complete_long_input_and_calendar_date_validation(self):
        text = "abcd"*5000
        batches = list(fact_batches([SourceTurn("t","2024-01-01",text,text)]))
        self.assertEqual(len(batches),3)
        parts = [b[0][1] for b in batches]
        self.assertEqual(parts[0]+"".join(s[119:] for s in parts[1:]),text)
        with self.assertRaises(ValueError):
            valid_date("2021-02-29")

    def test_rejects_non_owner_quotes_and_resumes_extraction(self):
        with TemporaryDirectory() as tmp:
            db = connect(Path(tmp)/"db")
            self.addCleanup(db.close)
            db.execute("INSERT INTO turn (id,source,kind,ts,prompt,snapshot) VALUES ('t','fixture','prompt','2024-02-03','在看 书','s')")
            db.execute("INSERT INTO conversation(id,source) VALUES ('c','fixture')")
            db.execute("INSERT INTO turn_conv VALUES ('t','c',0)")
            item = dict(text="用户在看书。",kind="event",confidence=3,evidence=[dict(turn_id="1",quote="看书")])
            llm = FakeLLM({"facts":[item,{**item,"evidence":[dict(turn_id="1",quote="去跑步")]}]})
            result = extract(db,llm,emit=lambda _:None)
            self.assertEqual((result["candidates"],result["invalid_quotes"],result["invalid_candidates"]),(1,1,1))
            stored = json.loads(db.execute("SELECT candidates FROM fact_source").fetchone()[0])[0]
            self.assertEqual((stored["date_from"],stored["evidence"]),("2024-02-03",{"t":"看 书"}))
            self.assertEqual(extract(db,llm)["skipped"],1)
