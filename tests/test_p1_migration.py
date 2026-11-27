import sqlite3
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from latentself.db import SCHEMA, connect


class MigrationTest(unittest.TestCase):
    def test_recognizes_frame_archives_on_ordinary_turns_in_existing_v2_databases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db = root/"db"
            con = connect(db)
            con.execute("DELETE FROM meta WHERE k='frame_archive_origins_v1'")
            con.execute("INSERT INTO turn (id,source,kind,ts,prompt,snapshot) VALUES ('t','gemini','prompt','2020','fixture','snap')")
            for name,member in [("camera.zip","nested/frame_0.png"),("document.zip","notes.txt")]:
                with ZipFile(root/name,"w") as archive:
                    archive.writestr(member,b"fixture")
                con.execute("""INSERT INTO attachment(id,turn_id,pos,stored_name,path,match,origin,caption)
                    VALUES (?,'t',0,?,?,'exact','user','preserved')""",(name,name,str(root/name)))
            con.commit()
            con.close()
            for _ in range(2):
                con = connect(db)
                self.assertEqual([tuple(r) for r in con.execute("SELECT origin,caption FROM attachment ORDER BY id")],
                                 [("live_frame","preserved"),("user","preserved")])
                con.close()

    def test_upgrade_preserves_legacy_data_and_reopens(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.db"
            old = sqlite3.connect(path)
            old.executescript(SCHEMA)
            old.execute("INSERT INTO turn (id,source,kind,ts,prompt,snapshot) VALUES ('t', 'gemini', 'live', '2020', '', 'snapshot')")
            old.executemany("""
                INSERT INTO attachment (id, turn_id, pos, stored_name, match, caption)
                VALUES (?, 't', 0, ?, 'exact', 'existing caption')
            """, [('a', 'watermarked_img_fixture.png'), ('b', 'frames.zip'), ('c', 'photo.png')])
            old.commit()
            old.close()
            for attempt in range(2):
                con = connect(path)
                try:
                    self.assertEqual(con.execute("PRAGMA user_version").fetchone()[0], 5)
                    self.assertEqual(dict(con.execute("SELECT id, origin FROM attachment")),
                                     {'a': 'generated', 'b': 'live_frame', 'c': 'user'})
                    self.assertEqual(con.execute("SELECT DISTINCT caption FROM attachment").fetchall()[0][0],
                                     'existing caption')
                    if attempt == 0:
                        con.execute("INSERT INTO embedding VALUES ('turn','t','fixture',1,'hash',X'003C')")
                        con.execute("INSERT INTO search_cache VALUES ('key','expand','[]','2020')")
                        con.commit()
                    self.assertEqual(con.execute("SELECT count(*) FROM embedding").fetchone()[0], 1)
                    self.assertEqual(con.execute("SELECT count(*) FROM search_cache").fetchone()[0], 1)
                finally:
                    con.close()
