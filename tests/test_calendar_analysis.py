import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from latentself.analysis.report import reports
from latentself.analysis.stats import overview,split_groups
from latentself.db import connect


class CalendarTest(unittest.TestCase):
    def test_historical_timezone_changes_group_and_local_month(self):
        with TemporaryDirectory() as tmp,patch.dict(os.environ,{"LATENTSELF_TZ_PERIODS":
            '[{"from":"2024-01-01","zone":"Europe/Berlin"},{"from":"2024-02-01","zone":"America/New_York"}]'}):
            root = Path(tmp)
            db = connect(root/"db")
            self.addCleanup(db.close)
            db.executemany("INSERT INTO turn(id,ts,prompt,source,kind,snapshot) VALUES (?,?,'test','s','prompt','s')",
                           [("a","2024-01-01T23:30:00Z"),("b","2024-02-01T03:00:00Z")])
            db.commit()
            self.assertEqual(overview(db)["histograms"]["month"],{"2024-01":2})
            self.assertEqual(split_groups(db,"night"),({"a"},{"b"}))
            self.assertEqual(len(reports(db,root,"overview")["files"]),1)
