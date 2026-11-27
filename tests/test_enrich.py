from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from tests.fakes import FakeLLM
from latentself.backends import InferenceUnavailable
from latentself.db import connect
from latentself.enrich import enrich


class EnrichmentTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.con = connect(self.root / "test.db")
        self.addCleanup(self.con.close)
        self.con.execute("INSERT INTO turn VALUES ('t','gemini','prompt','2020',?,'snapshot')", ('x'*330,))
        self.llm = FakeLLM(dict(caption="fixture", ocr="label", tags=["sign"], content_kind="photo"))

    def attachment(self, name, mime, origin="user"):
        path = self.root / name
        self.con.execute("""INSERT INTO attachment (id,turn_id,pos,stored_name,path,match,mime,origin)
                         VALUES (?,'t',0,?,?,'exact',?,?)""", (name,name,str(path),mime,origin))
        self.con.commit()
        return path

    def test_docs_truncation_and_fts_survive_resume(self):
        path = self.attachment("sample.docx", "application/zip")
        with ZipFile(path, 'w') as z:
            z.writestr('word/document.xml', '<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                       + '<w:r><w:t>' + 'x'*20004 + '</w:t></w:r></w:p>')
        result = enrich(self.con,self.llm,emit=lambda _:None)
        self.assertEqual((result["success"],result["omitted_chars"]), (1,4))
        self.assertEqual(self.con.execute("SELECT length(ocr) FROM attachment").fetchone()[0], 20000)
        self.assertIn("xxxx",self.con.execute("SELECT extra FROM fts_turn").fetchone()[0])
        self.assertEqual(enrich(self.con,self.llm)["skipped"],1)

    @patch('latentself.enrich.images.subprocess.run')
    def test_heic_live_frame_selection_and_limit(self, run):
        read = []
        def convert(args, **kwargs):
            if args[1]=="-g":
                from types import SimpleNamespace
                width = 3000 if args[-1].endswith(".heic") else 640
                return SimpleNamespace(stdout=f"  pixelWidth: {width}\n  pixelHeight: 480\n")
            source = Path(args[-3])
            self.assertEqual("-Z" in args,source.suffix==".heic")
            read.append(source.read_bytes())
            Path(args[-1]).write_bytes(b'jpeg fixture')
        run.side_effect = convert
        self.attachment("a.heic","image/heic","generated").write_bytes(b'heic')
        with ZipFile(self.attachment("b.zip","application/zip"),'w') as z:
            for i in (10,2,9,1,3):
                z.writestr(f'frame_{i}.png', str(i))
        result = enrich(self.con,self.llm,limit=1,emit=lambda _:None)
        self.assertEqual((result["success"], result["limited"], result["omitted_chars"]), (1,1,30))
        result = enrich(self.con,self.llm,emit=lambda _:None)
        self.assertEqual((result["skipped"],result["omitted_frames"]), (1,2))
        self.assertEqual(read,[b'heic',b'1',b'3',b'10'])
        self.assertEqual(self.con.execute("SELECT origin FROM attachment WHERE id='b.zip'").fetchone()[0], 'live_frame')
        self.assertTrue(self.con.execute("SELECT caption FROM attachment WHERE id='a.heic'").fetchone()[0].startswith('[Gemini 生成]'))

    @patch("latentself.enrich.images.prepare",return_value=(["fixture"],0))
    def test_network_outage_does_not_mark_photos_as_broken(self, prepare):
        self.attachment("photo.jpg","image/jpeg")
        def offline(messages):
            raise InferenceUnavailable("offline")
        result = enrich(self.con,FakeLLM(offline),emit=lambda _:None)
        self.assertEqual((result["failed"],result["interrupted"]),(0,"offline"))
        self.assertIsNone(self.con.execute("SELECT enrich_error FROM attachment").fetchone()[0])
        self.assertEqual(enrich(self.con,self.llm,emit=lambda _:None)["success"],1)

    @patch('latentself.enrich.images.prepare', side_effect=ValueError("fixture error"))
    def test_failure_retries_once_and_requires_retry_flag(self, prepare):
        self.attachment('bad.png', 'image/png')
        self.assertEqual(len(enrich(self.con,self.llm)["failures"]),1)
        self.assertEqual(prepare.call_count,2)
        self.assertEqual(enrich(self.con,self.llm)["skipped"],1)
        self.assertEqual(enrich(self.con,self.llm,retry_errors=True)["failed"],1)
        self.assertEqual(prepare.call_count,4)
