import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from latentself.catalog import aliases,import_catalog,normalized,set_state
from latentself.db import connect


class CatalogTest(unittest.TestCase):
    def test_late_identifier_merges_existing_entries_and_keeps_feedback(self):
        from latentself.catalog import current_id
        from latentself.tspdt_ids import signature
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = dict(canonicalKey="film:older:2001",title="Older Title",creator="Director",year=2001,rank=1)
            b = dict(canonicalKey="film:alias:2001",title="Alternate",creator="Director",year=2001,imdb="tt0000001")
            (root/"tspdt-2026.json").write_text(json.dumps(dict(top1000=[a],recommendations=[b])))
            (root/"movies-final-2026.json").write_text('{"records":[]}')
            con = connect(root/"db")
            self.addCleanup(con.close)
            import_catalog(con,root)
            old = a["canonicalKey"]
            with con:
                set_state(con,old,"watched",1,"feedback")
            index = {"1":dict(signature=signature(a["title"],a["creator"],a["year"]),imdb=b["imdb"])}
            import_catalog(con,root,public_index=index)
            import_catalog(con,root)
            self.assertEqual(con.execute("SELECT count(*) FROM item").fetchone()[0],1)
            self.assertEqual(current_id(con,old),"film:tt0000001")
            self.assertEqual(tuple(con.execute("SELECT item_id,rating FROM item_state").fetchone()),("film:tt0000001",1))

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
