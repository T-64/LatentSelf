"""Synthetic Takeout covering every edge case found in the real export (no personal data)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from latentself.db import connect
from latentself.ingest import gemini
from latentself import cli


def url(cid):
    return {"name": f"https://gemini.google.com/app/{cid}", "url": f"https://gemini.google.com/app/{cid}"}


BASE = {"header": "Gemini Apps", "products": ["Gemini Apps"], "activityControls": ["Gemini Apps Activity"]}

ENTRIES = [
    # plain prompt with answer, two drafts
    {**BASE, "title": "Prompted 我的牙齿有点疼怎么办", "time": "2026-01-01T10:00:00.000Z", "details": [url("aaaa")],
     "safeHtmlItem": [{"html": "<p>建议去看<strong>牙医</strong></p><ul><li>先漱口</li></ul>"}, {"html": "<p>第二个草稿</p>"}]},
    # turn shared by two branches
    {**BASE, "title": "Prompted 补牙之后能吃饭吗", "time": "2026-01-01T10:05:00.000Z", "details": [url("aaaa"), url("bbbb")],
     "safeHtmlItem": [{"html": "<p>两小时后</p>"}]},
    {**BASE, "title": "Branched 那智齿呢", "time": "2026-01-01T10:10:00.000Z", "details": [url("bbbb")],
     "safeHtmlItem": [{"html": "<p>看情况</p>"}]},
    # prompt with attachments: exact, missing extension, (n) duplicate, missing
    {**BASE, "title": "Prompted 树上这个标记是什么", "time": "2026-01-02T09:00:00.000Z", "details": [url("cccc")],
     "subtitles": [{"name": "Attached 4 files."}, {"name": "-  IMG_1.heic", "url": "IMG_1-abc.heic"}],
     "attachedFiles": ["IMG_1-abc.heic", "notes-def.md", "photo-xyz.jpg", "gone-000.png"], "imageFile": "IMG_1-abc.heic",
     "safeHtmlItem": [{"html": "<p>林业标记</p>"}]},
    # same stored name uploaded again -> photo-xyz(2).jpg on disk
    {**BASE, "title": "Prompted 再看一张", "time": "2026-01-02T09:01:00.000Z", "details": [url("cccc")],
     "attachedFiles": ["photo-xyz.jpg"], "safeHtmlItem": [{"html": "<p>同上</p>"}]},
    # canvas without URL, 5 minutes after a dialogue turn
    {**BASE, "title": "Created Gemini Canvas titled 树木标记图鉴", "time": "2026-01-02T09:06:00.000Z",
     "subtitles": [{"name": "```html\n<html>canvas body</html>"}]},
    # feedback without URL
    {**BASE, "title": "Gave feedback: Good response", "time": "2026-01-02T09:07:00.000Z"},
    # housekeeping
    {**BASE, "title": "Cleared previous feedback", "time": "2026-01-03T00:00:00.000Z"},
    # prompt with no answer
    {**BASE, "title": "Prompted 上一条没说吗", "time": "2026-01-04T00:00:00.000Z", "details": [url("dddd")]},
]


class IngestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.snap = root / "raw" / "gemini" / "2026-01-05"
        (self.snap / "attachments").mkdir(parents=True)
        (self.snap / "MyActivity.json").write_text(json.dumps(ENTRIES, ensure_ascii=False))
        for n in ["IMG_1-abc.heic", "notes-def", "photo-xyz(1).jpg", "photo-xyz(2).jpg"]:
            (self.snap / "attachments" / n).write_bytes(b"x")
        self.db_path = root / "ls.db"
        self.con = connect(self.db_path)
        self.stats = gemini.ingest(self.con, self.snap)

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def q(self, sql, *p):
        return self.con.execute(sql, p).fetchall()

    def test_every_entry_kept_and_classified(self):
        self.assertEqual(self.stats["turns"], len(ENTRIES))
        kinds = dict(self.q("SELECT kind, COUNT(*) FROM turn GROUP BY kind"))
        self.assertEqual(kinds, {"prompt": 5, "branch": 1, "canvas": 1, "feedback": 1, "cleared": 1})

    def test_branch_shared_turn_belongs_to_both(self):
        convs = [r[0] for r in self.q("SELECT conv_id FROM turn_conv tc JOIN turn t ON t.id=tc.turn_id WHERE t.prompt LIKE '补牙%' ORDER BY 1")]
        self.assertEqual(convs, ["aaaa", "bbbb"])
        n = dict(self.q("SELECT id, n_turns FROM conversation"))
        self.assertEqual(n["aaaa"], 2)
        self.assertEqual(n["bbbb"], 2)

    def test_drafts_and_text(self):
        rows = self.q("SELECT draft_idx, text FROM answer a JOIN turn t ON t.id=a.turn_id WHERE t.prompt LIKE '我的牙齿%' ORDER BY 1")
        self.assertEqual(len(rows), 2)
        self.assertIn("牙医", rows[0][1])
        self.assertIn("- 先漱口", rows[0][1])

    def test_attachment_resolution(self):
        m = dict(self.q("SELECT stored_name || '#' || pos, match FROM attachment"))
        self.assertEqual(m["IMG_1-abc.heic#0"], "exact")
        self.assertEqual(m["notes-def.md#1"], "no_ext")
        self.assertEqual(m["photo-xyz.jpg#2"], "dup_suffix")
        self.assertEqual(m["gone-000.png#3"], "missing")
        self.assertEqual(m["photo-xyz.jpg#0"], "dup_suffix")
        paths = [r[0] for r in self.q("SELECT path FROM attachment WHERE stored_name='photo-xyz.jpg' ORDER BY path")]
        self.assertEqual([os.path.basename(p) for p in paths], ["photo-xyz(1).jpg", "photo-xyz(2).jpg"])
        self.assertEqual(self.q("SELECT orig_name FROM attachment WHERE stored_name='IMG_1-abc.heic'")[0][0], "IMG_1.heic")
        # imageFile duplicates attachedFiles[0] and must not create a second row
        self.assertEqual(self.q("SELECT COUNT(*) FROM attachment WHERE stored_name='IMG_1-abc.heic'")[0][0], 1)

    def test_recovers_unique_export_identifiers_with_different_extensions(self):
        folder = self.snap/"attachments"
        stem = "fixture-0123456789abcdef"
        actual = stem+".png"
        (folder/actual).touch()
        (folder/"record.pdf").touch()
        (folder/"generic.jpg").touch()
        resolver = gemini.AttachmentResolver(folder)
        self.assertEqual(resolver.resolve(stem+".jpg"),(folder/actual,"image_extension"))
        self.assertEqual(resolver.resolve("record.PDF"),(folder/"record.pdf","extension_case"))
        self.assertEqual(resolver.resolve("generic.png"),(None,"missing"))
        self.assertEqual(gemini.AttachmentResolver(folder,[actual]).resolve(stem+".jpg"),(None,"missing"))
        (folder/(stem+".webp")).touch()
        self.assertEqual(gemini.AttachmentResolver(folder).resolve(stem+".jpg"),(None,"missing"))

    def test_canvas_and_signal_linked_by_time(self):
        self.assertEqual([tuple(r) for r in self.q("SELECT conv_id, inferred FROM turn_conv tc JOIN turn t ON t.id=tc.turn_id WHERE t.kind='canvas'")], [("cccc", 1)])
        self.assertIn("canvas body", self.q("SELECT content FROM canvas")[0][0])
        target = self.q("SELECT t2.prompt FROM signal s JOIN turn t2 ON t2.id=s.target_id")[0][0]
        self.assertEqual(target, "再看一张")

    def test_reingest_is_idempotent(self):
        gemini.ingest(self.con, self.snap)
        self.assertEqual(self.q("SELECT COUNT(*) FROM turn")[0][0], len(ENTRIES))
        self.assertEqual(self.q("SELECT COUNT(*) FROM attachment")[0][0], 5)
        self.assertEqual(self.q("SELECT COUNT(*) FROM turn_conv")[0][0], 8)

    def test_search_short_and_long_queries(self):
        self.assertEqual(len(cli._matching_turns(self.con, "牙齿", False)), 1)          # 2 chars -> instr fallback
        self.assertEqual(len(cli._matching_turns(self.con, "牙医", False)), 0)          # only in the answer
        self.assertEqual(len(cli._matching_turns(self.con, "牙医", True)), 1)
        self.assertEqual(len(cli._matching_turns(self.con, "树上这个", False)), 1)      # trigram path
        self.assertEqual(len(cli._matching_turns(self.con, "IMG_1", False)), 1)        # attachment original name
        self.assertEqual(len(cli._matching_turns(self.con, "canvas body", False)), 1)  # canvas content


if __name__ == "__main__":
    unittest.main()
