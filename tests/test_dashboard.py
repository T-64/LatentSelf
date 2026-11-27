from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch

from latentself.db import connect
from latentself.interp.dashboard import build,highlight
from latentself.interp.experiment import register


def fixture(root):
    con = connect(root/"db")
    run = register(con,"9b","bf16",8,"card","turn_self")
    for f,name in ((1,"读书"),(2,"旅行")):
        con.execute("""INSERT INTO feature VALUES (?,?,.5,.2,1,2,?,'文字来源',.6,'fixture')""",(run,f,name))
        for i in range(5):
            uid = f"t{f}-{i}"
            con.execute("INSERT INTO turn VALUES (?,'fixture','prompt','2024-01-03','阅读 <sample>','s',0)",(uid,))
            con.execute("INSERT INTO turn_conv VALUES (?,'fixture',0)",(uid,))
            con.execute("INSERT INTO sae_act VALUES (?,?,?,3,1,1)",(run,uid,f))
    con.commit()
    def post(url,*args,**kwargs):
        if url.endswith("/neighbors"):
            return dict(results=[[2,.5]])
        return dict(results=[{"8":dict(offsets=[(0,1),(1,2)],token_acts=[0,3])}])
    client = Mock(url="http://127.0.0.1")
    with patch("latentself.interp.dashboard.post",side_effect=post):
        result = build(con,client,run,root)
    con.close()
    return result


class DashboardTest(unittest.TestCase):
    def test_html_links_and_token_peak(self):
        with TemporaryDirectory() as tmp:
            result = fixture(Path(tmp))
            self.assertEqual(result["pages"],2)
            self.assertIn('href="1.html"',Path(result["index"]).read_text())
            detail = (Path(tmp)/"features/1.html").read_text()
            self.assertIn('class="peak"',detail)
            self.assertIn('&lt;sample&gt;',detail)
            self.assertIn('https://gemini.google.com/app/fixture',detail)
            with self.assertRaises(ValueError):
                highlight("ab",[(0,1),(1,2)],[0,3],0)
