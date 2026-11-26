# LatentSelf v2

本地个人数据底座：把 AI 对话记录（目前支持 Gemini Takeout）导入成统一模型，在此之上做查询、画像与推荐、AI 写作，以及可视化。设计见 [docs/DESIGN.md](docs/DESIGN.md)。

数据不进仓库：原始快照放在 `$LATENTSELF_DATA/raw/gemini/<YYYY-MM-DD>/`（默认 `~/data`），数据库放在 `$LATENTSELF_DATA/processed/latentself/ls.db`。

```bash
uv tool install --editable .     # 安装 lself 命令
lself ingest                     # 导入最新的快照（约 15 秒）
lself stats
lself grep 咖啡                   # 关键词搜索，按对话分组
lself grep 路牌 --images          # 只看带附件的轮次，显示文件路径
lself show <conv_id>             # 按顺序显示整段对话
lself open <attachment_id>       # 打开附件
```

运行测试：`python3 -m unittest discover -s tests`。测试用的是合成数据，不含真实数据。
