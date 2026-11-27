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



    def test_export_reuses_bytes_by_hash_and_preserves_colliding_names(self):
        from unittest.mock import patch
        from scripts.export_aistudio import export
        from base64 import b64encode
        raw=json.dumps({"chunkedPrompt":{"chunks":[
            {"role":"user","text":"question","inlineImage":{"data":b64encode(b"image").decode(),"mimeType":"image/png"}},
            {"role":"model","isThought":True,"text":"private intermediate"},
            {"role":"model","parts":[{"thought":True,"text":"internal"},{"text":"visible"}]}],
            "pendingInputs":[{"text":"unsent"}]}}).encode()
        h=hashlib.sha256(raw).hexdigest()
        local=self.root/"cached"
        local.mkdir()
        (local/"existing").write_bytes(raw)
        records=[dict(id=k,name="Same title",size=str(len(raw)),sha256Checksum=h,
                      createdTime="2025-01-01",modifiedTime="2025-01-02",
                      webViewLink="https://aistudio.google.com/app/prompts/"+k) for k in ("one","two")]
        def remote(args,**kwargs):
            if args[0]=="lsjson":
                return b"[]"
            self.assertEqual(args[:2],["backend","query"])
            return json.dumps(records).encode()
        output=self.root/"new"
        with patch("scripts.export_aistudio.drive",side_effect=remote):
            counts=export("remote:","Google AI Studio",output,[local])
        self.assertEqual(counts["conversations"],2)
        self.assertEqual(counts["thought_chunks_excluded"],2)
        self.assertEqual(counts["unsent_nonempty_drafts_excluded"],2)
        stats=ingest(self.db,output)
        self.assertEqual(stats["imported"],2)
        answers=self.db.execute("SELECT DISTINCT text FROM answer").fetchall()
        self.assertEqual([r[0] for r in answers],["visible"])

    def test_invalid_record_rolls_back_batch(self):
        self.write([self.row,dict(self.row,drive_id="../invalid")])
        with self.assertRaises(ValueError):
            ingest(self.db,self.root)
        self.assertEqual(self.db.execute("SELECT count(*) FROM conversation").fetchone()[0],0)
