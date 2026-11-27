from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from latentself.db import connect
from latentself.exports import export_markdown


class ExportTest(unittest.TestCase):
    def test_generated_and_manually_edited_exports(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"report.md"
            db = connect(Path(tmp)/"fixture")
            self.addCleanup(db.close)
            self.assertEqual(export_markdown(db,path,"v1"),path)
            self.assertEqual(export_markdown(db,path,"v2"),path)
            self.assertEqual(path.read_text(),"v2\n")
            path.write_text("owner edit")
            alternative = export_markdown(db,path,"v3")
            self.assertNotEqual(alternative,path)
            self.assertEqual(export_markdown(db,path,"v3"),alternative)
            self.assertEqual((path.read_text(),alternative.read_text()),("owner edit","v3\n"))
