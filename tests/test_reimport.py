import json
import tempfile
import unittest
from pathlib import Path

from latentself.db import connect
from latentself.ingest.gemini import ingest


class ReimportTest(unittest.TestCase):
    def test_keeps_enrichment_and_discards_replaced_or_removed_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            con = connect(root / "test.db")
            self.addCleanup(con.close)
            entry = {"time": "2024-01-01T00:00:00Z", "title": "Live Prompt fixture",
                     "details": [{"url": "https://gemini.google.com/app/abcd"}],
                     "attachedFiles": ["frames.zip", "old.png", "gone.png"]}
            source = root / "MyActivity.json"
            source.write_text(json.dumps([entry]))
            ingest(con, root)
            with con:
                con.execute("UPDATE conversation SET title='retained'")
                con.execute("UPDATE attachment SET caption='saved', ocr='text', tags='[]'")
                for kind, table in [('turn', 'turn'), ('turn_self', 'turn'), ('attachment', 'attachment')]:
                    con.execute(f"INSERT INTO embedding SELECT ?,id,'fixture',1,'hash',X'003C' FROM {table}", (kind,))
            entry["attachedFiles"] = ["frames.zip", "watermarked_img_fixture.png"]
            source.write_text(json.dumps([entry]))
            ingest(con, root)
            self.assertEqual([tuple(r) for r in con.execute(
                "SELECT origin,caption,ocr,tags FROM attachment ORDER BY pos")],
                [('live_frame', 'saved', 'text', '[]'), ('generated', None, None, None)])
            self.assertEqual(dict(con.execute("SELECT item_kind,count(*) FROM embedding GROUP BY item_kind")),
                             {'turn': 1, 'turn_self': 1, 'attachment': 1})
            self.assertEqual(con.execute("SELECT title FROM conversation").fetchone()[0], 'retained')
            self.assertIn('saved', con.execute("SELECT extra FROM fts_turn").fetchone()[0])
