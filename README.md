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

## P1 附件与语义搜索

Mac 端需要 Python 3.12+ 和 NumPy。图片处理使用 macOS 自带的 `sips`，PDF 需要 Poppler 的 `pdftotext`。DOCX 在本地解析；Live 使用已有转写，并从 ZIP 选择首、中、尾帧。每个附件单独提交，重新运行可以续接。

```bash
lself enrich --docs --limit 5
lself enrich --images --limit 5
lself enrich --retry-errors
lself embed --limit 50
lself search '检索主题' --terms '同义词,译名' --sort score
lself search '检索主题' --strict --images --since 2025-01 --until 2025-12 --json
```

默认检索提问及附件、Canvas 内容，`--answers` 同时检索回答。结果中的 `counts` 记录候选截断，`omitted_snippets` 记录每个对话省略的摘录数。模型离线时输出会说明降级原因。JSON 中的 `mode=unjudged` 表示候选仍待相关性判断。

### GPU 服务

先运行 `scripts/server/status.sh` 记录容器、原有 Qwen 服务和显存，再按需要启停。模型文件下载到 GPU 主机的 Hugging Face 缓存。服务角色分别为 `embed`（视觉向量）、`text`（对照模型）、`rerank` 和 `llm`。独立模型服务的端口仅发布到服务器回环地址。既有 Qwen 栈也需要满足相同的绑定要求。

```bash
scripts/server/fetch-models.sh             # 使用镜像时加 mirror
scripts/server/up.sh embed
ssh -N -L 127.0.0.1:8000:127.0.0.1:8000 -L 127.0.0.1:8001:127.0.0.1:8001 \
       -L 127.0.0.1:8002:127.0.0.1:8002 -L 127.0.0.1:8003:127.0.0.1:8003 server
scripts/server/down.sh embed
```

长任务可用 `scripts/run-logged.sh enrich --images` 等方式在 Mac 后台运行。它会打印 PID 和私有日志路径。默认推理端点可用 `LATENTSELF_LLM_URL`、`LATENTSELF_EMBED_URL`、`LATENTSELF_TEXT_EMBED_URL`、`LATENTSELF_RERANK_URL` 覆盖。任务结束后按开始前记录的状态恢复服务。

如果已有 Qwen 栈发布到所有网卡，可运行 `scripts/server/up.sh llm-private`。它使用已安装的镜像和只读模型目录，在服务器回环端口 8010 提供独立的单槽实例。将 SSH 隧道的本地 18000 转发到远端 8010，并设置 `LATENTSELF_LLM_URL=http://127.0.0.1:18000/v1`。结束后运行 `scripts/server/down.sh llm-private`。

向量服务默认按单模型运行分配显存，`LATENTSELF_GPU_MEMORY_FRACTION` 可指定 vLLM 使用的显存比例。实测多服务共用显卡时，可以用 `LATENTSELF_KV_CACHE_BYTES` 指定每个新建向量或重排实例的 KV 缓存字节数。比较模型时可依次加载，每种模型的评测查询向量会保存在 Mac 的私有评测库中。

### 模型比较与验收

私有查询文件默认为 `$LATENTSELF_DATA/processed/latentself/eval/queries.yaml`，顶层 `queries` 列表的每项使用 `q`，照片项加 `photo: true`。标签保存在同目录的 `labels.sqlite3`。以下步骤先构建小范围比较语料（扩展关键词命中和照片轮次），模型选择之后再建立全量索引。

```bash
lself eval --index-pool Qwen/Qwen3-VL-Embedding-8B
lself eval --index-pool Qwen/Qwen3-Embedding-8B
lself eval --prepare
lself eval --label        # 每次最多 30 项，照片查询可标记具体文件
lself eval --score-reranker  # 单卡可先只加载 reranker，保存实测分数和耗时
lself eval --calibrate    # 输出匿名汇总，选定策略存入数据库
lself embed
lself eval --extend-pool  # 将全库检索新增的候选加入同一份标注清单
lself eval --label
lself eval
scripts/smoke_p1.sh
```

两个候选模型建索引期间也可以开始核对照片。运行 `lself eval --prepare-photo-review` 后，`lself eval --label` 会显示照片查询已经命中的关键词候选。它们一定进入最终候选并集，标注会沿用。完整模型比较仍由 `lself eval --prepare` 生成。

校准比较两种模型的文本召回率，再比较逐个判断和分级判断的质量与耗时。分开加载 reranker 和 LLM 时，分级判断的耗时包含同一批候选先前实测的 reranker 请求时间，报告也单列这部分秒数。最终评测使用全量索引，逐查询列出漏报、误报、待标注项和照片前五命中。其明细仅保存在本地。
