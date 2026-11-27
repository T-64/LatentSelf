import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from latentself.db import connect
from latentself.ingest.aistudio import ingest


class StudioIngest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = connect(self.root / "test.db")
        self.addCleanup(self.db.close)
        (self.root / "objects").mkdir()
        data = b'{"chunkedPrompt":{}}'
        digest = hashlib.sha256(data).hexdigest()
        (self.root / "objects" / digest).write_bytes(data)
        self.row = dict(source="aistudio", drive_id="synthetic-id", title="Demo",
                        url="https://aistudio.google.com/app/prompts/synthetic-id", source_sha256=digest,
                        drive_created="2025-01-01", drive_modified="2025-01-02", messages=[
                            dict(index=0, role="user", time=None, text="sample phrase", attachments=[]),
                            dict(index=2, role="model", time=None, text="sample answer", attachments=[])])
        (self.root / "media.json").write_text("{}")

    def write(self, rows):
        (self.root / "conversations.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))

    def test_same_title_distinct_sources_and_repeat(self):
        self.write([self.row, dict(self.row,drive_id="another-id")])
        self.assertEqual(ingest(self.db,self.root)["imported"],2)
        changes=self.db.total_changes
        self.assertEqual(ingest(self.db,self.root),{"skipped":2})
        self.assertEqual(changes,self.db.total_changes)
        self.assertEqual(self.db.execute("SELECT count(*) FROM fts_turn WHERE fts_turn MATCH 'sample'").fetchone()[0],2)
        self.assertEqual(self.db.execute("SELECT count(*) FROM turn WHERE ts=''").fetchone()[0],2)

    def test_read_api_keeps_original_message_order_and_link(self):
        from latentself.read_api import Reader
        self.row["messages"].append(dict(index=3, role="user", time=None, text="second", attachments=[]))
        self.write([self.row])
        ingest(self.db,self.root)
        rows=Reader(self.root / "test.db").show_conversation("aistudio:synthetic-id")["results"]
        self.assertEqual([r["prompt"] for r in rows],["sample phrase","second"])
        self.assertEqual({r["url"] for r in rows},{self.row["url"]})

    def test_undated_prompts_stay_searchable_outside_calendar_buckets(self):
        from latentself.units import units
        from latentself.analysis.stats import overview
        self.write([self.row])
        ingest(self.db,self.root)
        counts=units(self.db)
        self.assertEqual((counts["turns"],counts["episodes"],counts["undated"]),(1,0,1))
        self.assertEqual(overview(self.db)["histograms"]["date_quality"]["unknown"],1)

    def test_invalid_record_rolls_back_batch(self):
        self.write([self.row,dict(self.row,drive_id="../invalid")])
        with self.assertRaises(ValueError):
            ingest(self.db,self.root)
        self.assertEqual(self.db.execute("SELECT count(*) FROM conversation").fetchone()[0],0)
