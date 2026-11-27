from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from latentself.db import connect


class ReviewMigrationTest(unittest.TestCase):
    def test_upgrade_from_v2_preserves_vectors_and_reopens_with_facts(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder)/"fixture.db"
            with patch("latentself.units.create_tables"):
                con = connect(db)
            for table in ("fact_evidence","fact_source","fact","digest","conv_summary"):
                con.execute(f"DROP TABLE {table}")
            con.execute("PRAGMA user_version=2")
            con.execute("INSERT INTO embedding VALUES ('turn','t','fixture',1,'hash',X'003C')")
            con.commit()
            con.close()
            for attempt in range(2):
                con = connect(db)
                try:
                    self.assertEqual(con.execute("PRAGMA user_version").fetchone()[0],4)
                    self.assertEqual(con.execute("SELECT vec FROM embedding").fetchone()[0],b"\x00\x3c")
                    if attempt==0:
                        con.execute("""INSERT INTO fact(id,text,kind,confidence,status)
                                       VALUES ('f','synthetic','event',2,'active')""")
                        con.commit()
                    self.assertEqual(con.execute("SELECT verified FROM fact").fetchone()[0],0)
                finally:
                    con.close()
