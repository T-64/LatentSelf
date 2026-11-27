from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
import numpy as np

from latentself.catalog import dumps
from latentself.db import connect
from latentself.film_state import feedback
from latentself.interp.encode import store_unit
from latentself.interp.experiment import register
from latentself.rec_eval import BACKTEST_VERSION
from latentself.rec_output import recommend
from latentself.recommend import representations,taste


class RecommendationFlowTest(unittest.TestCase):
    def test_ten_film_fixture_retraces_reasons_and_feedback_changes_list(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            con = connect(root/"db")
            self.addCleanup(con.close)
            run = register(con,"9b","bf16",8,"card","turn_self")
            base = register(con,"9b","bf16",8,"card","baseline")
            con.execute("INSERT INTO topic(id,name,resolution,last_ts) VALUES ('culture','电影',1,'2024-01-20')")
            con.execute("INSERT INTO topic_member VALUES ('culture','c',1)")
            con.execute("INSERT INTO feature VALUES (?,7,.5,.0,2,3,'人文故事','fixture',.8,'fake')",(run,))
            vec = np.array([1,0],dtype="<f2")
            for i in range(20):
                uid = f"t{i:02d}"
                con.execute("INSERT INTO turn VALUES (?,'fixture','prompt',?,'阅读电影评论','s',0)",(uid,f"2024-01-{i+1:02d}"))
                con.execute("INSERT INTO turn_conv VALUES (?,'c',0)",(uid,))
                con.execute("INSERT INTO embedding VALUES ('turn_self',?,'embed',2,'x',?)",(uid,vec.tobytes()))
                sample = dict(n_tokens=2+i,n_chunks=1,fvu=.1,features=[(7,2,1)] if i%2 else [])
                store_unit(con,run,uid,"h",sample)
                store_unit(con,base,"public"+uid,"h",dict(sample,features=[]))
            for i in range(10):
                con.execute("INSERT INTO item(id,domain,title,creators,sources) VALUES (?,'film',?,?,?)",
                            (str(i),f"影片{i}",dumps([f"导演{i}"]),dumps([dict(src="tspdt-top1000",rank=i+1)])))
            con.commit()
            client,embed,llm = Mock(),Mock(model="embed"),Mock(model="local")
            embed.embed_texts.side_effect = lambda texts,kind:[vec for _ in texts]
            client.encode.return_value = [{"8":dict(reader="Qwen/Qwen3.5-9B-Base",sae_repo="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",
                precision="bf16",formula="card",features=[(7,3,1)],n_tokens=4)}]
            self.assertEqual(representations(con,embed,client,run)["success"],10)
            self.assertEqual(representations(con,embed,client,run)["skipped"],10)
            self.assertIn(7,taste(con,run,"embed")["weights"])
            with self.assertRaisesRegex(ValueError,"backtest"):
                recommend(con,llm,run,"embed",root,3)
            report = dict(passed=True,model="embed",profile_run=run,weights=[.4,.4,.1,.1])
            with con:
                con.execute("INSERT INTO meta VALUES ('rec_backtest',?)",(dumps(report),))
            with self.assertRaisesRegex(ValueError,"backtest"):
                recommend(con,llm,run,"embed",root,3)
            # Exercise publication separately from the measured quality gate.
            with con:
                con.execute("UPDATE meta SET v=? WHERE k='rec_backtest'",(dumps(dict(report,version=BACKTEST_VERSION)),))
            llm.chat.return_value = "你曾写过“阅读电影评论”。"
            first = recommend(con,llm,run,"embed",root,3)
            self.assertEqual(first["returned"],3)
            self.assertIn("https://gemini.google.com/app/c",Path(first["path"]).read_text())
            iid = first["items"][0]["item_id"]
            feedback(con,iid,"dismiss")
            second = recommend(con,llm,run,"embed",root,3)
            self.assertNotIn(iid,[r["item_id"] for r in second["items"]])
