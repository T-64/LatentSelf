import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from latentself.catalog import aliases,import_catalog,normalized,set_state
from latentself.db import connect


class CatalogTest(unittest.TestCase):
    def test_match_by_external_id_then_canonical_key_and_preserve_feedback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = dict(canonicalKey="film:sample:2001",title="The Sample",year=2001)
            (root/"tspdt-2026.json").write_text(json.dumps(dict(top1000=[dict(record,rank=1)],
                recommendations=[dict(record,imdb="https://www.imdb.com/title/tt0000001/")])))
            (root/"movies-final-2026.json").write_text(json.dumps(dict(records=[dict(record,sourceKey="queue")])))
            con = connect(root/"db")
            self.addCleanup(con.close)
            self.assertEqual(import_catalog(con,root,{"queue":"wanted"})["imported"],1)
            iid = con.execute("SELECT id FROM item").fetchone()[0]
            self.assertEqual(iid,"film:tt0000001")
            with con:
                set_state(con,iid,"watched",1,"feedback")
            import_catalog(con,root,{"queue":"wanted"})
            self.assertEqual(con.execute("SELECT status FROM item_state").fetchone()[0],"watched")
            aliases(con)
            self.assertIn("sample",[r[0] for r in con.execute("SELECT alias FROM item_alias")])
            self.assertEqual(normalized("An Example!"),"example")
