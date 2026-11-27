import unittest

from latentself.interp.baseline import sample


class BaselineTest(unittest.TestCase):
    def test_only_chinese_first_user_prompt_and_seeded_sample(self):
        rows = [dict(conversation_hash=str(i),conversation=[dict(role="user",language="Chinese",content=f"测试{i}"),
                    dict(role="assistant",language="Chinese",content="unused")]) for i in range(30)]
        rows.append(dict(conversation_hash="other",conversation=[dict(role="user",language="English",content="Hi"),
                         dict(role="user",language="Chinese",content="无关")]))
        rows.append(dict(conversation_hash="paste",conversation=[dict(role="user",language="Chinese",content="作者：样例")]))
        a,total = sample(rows,6)
        self.assertEqual(total,30)
        self.assertEqual(sample(rows,6)[0],a)
        self.assertNotEqual(sample(rows,6,7)[0],a)
        self.assertTrue(all(text.startswith("测试") for _,text in a))
