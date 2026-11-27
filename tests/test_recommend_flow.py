from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
import numpy as np

from latentself.catalog import dumps,set_state
from latentself.db import connect
from latentself.film_state import feedback,mention_table
from latentself.interp.encode import store_unit
from latentself.interp.experiment import register
from latentself.rec_eval import BACKTEST_VERSION,backtest
from latentself.rec_output import recommend
from latentself.recommend import representations,taste


class RecommendationFlowTest(unittest.TestCase):
    def test_measured_backtest_gates_sourced_recommendations_and_feedback(self):
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
            mention_table(con)
            for iid,tid,date in [("0","early","2024-02-01"),("9","late","2024-02-02")]:
                con.execute("INSERT INTO turn VALUES (?,'fixture','prompt',?,'已看这部合成影片','s',0)",(tid,date))
                con.execute("INSERT INTO item_mention VALUES (?,?,'watched',1,'fixture',NULL)",(iid,tid))
                set_state(con,iid,"watched",1,"chat_mention",[tid],date)
            measured = backtest(con,run,"embed")
            self.assertEqual((measured["training"],measured["held"],measured["candidates"]),(1,1,9))
            self.assertTrue(all(b["recall50"]==1 for b in measured["baselines"].values()))
            self.assertFalse(measured["passed"])
            with self.assertRaisesRegex(ValueError,"backtest"):
                recommend(con,llm,run,"embed",root,3)
            report = dict(passed=True,model="embed",profile_run=run,weights=[.4,.4,.1,.1])
            with con:
                con.execute("UPDATE meta SET v=? WHERE k='rec_backtest'",(dumps(report),))
            with self.assertRaisesRegex(ValueError,"backtest"):
                recommend(con,llm,run,"embed",root,3)
            # Add fifty higher-prior, equally dense films with no matching SAE feature.
            # The held-out film now falls beyond both baselines' recall cutoff.
            with con:
                con.execute("UPDATE item SET sources=? WHERE id='9'",(dumps([dict(src="tspdt-top1000",rank=999)]),))
                con.executemany("INSERT INTO item(id,domain,title,sources) VALUES (?,'film',?,?)",
                                [(f"00-extra-{i:02d}",f"合成对照{i}",dumps([dict(src="tspdt-top1000",rank=i+1)]))
                                 for i in range(50)])
            client.encode.return_value[0]["8"]["features"] = []
            self.assertEqual(representations(con,embed,client,run)["success"],50)
            measured = backtest(con,run,"embed")
            self.assertEqual(measured["version"],BACKTEST_VERSION)
            self.assertTrue(all(b["recall50"]==0 for b in measured["baselines"].values()))
            self.assertEqual(measured["metrics"]["recall50"],1)
            self.assertTrue(measured["passed"])
            llm.chat.return_value = "你曾写过“阅读电影评论”。"
            first = recommend(con,llm,run,"embed",root,3)
            self.assertEqual(first["returned"],3)
            self.assertIn("https://gemini.google.com/app/c",Path(first["path"]).read_text())
            iid = first["items"][0]["item_id"]
            feedback(con,iid,"dismiss")
            second = recommend(con,llm,run,"embed",root,3)
            self.assertNotIn(iid,[r["item_id"] for r in second["items"]])
