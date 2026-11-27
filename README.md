# LatentSelf v2

本地个人数据底座：把 AI 对话记录（支持 Gemini Takeout 和 Google AI Studio）导入成统一模型，在此之上做查询、画像与推荐、AI 写作，以及可视化。设计见 [docs/DESIGN.md](docs/DESIGN.md)。

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

## Google AI Studio 对话

用已授权的 rclone Drive 远端导出本人保存的 AI Studio 对话。把示例中的远端名换成自己的配置，并选用新的本机目录。

```bash
snapshot="$HOME/data/processed/latentself/exports/aistudio-$(date +%Y%m%d-%H%M%S)"
python3 scripts/export_aistudio.py --remote remote: --output "$snapshot"
lself ingest --source aistudio --snapshot "$snapshot"
```

导出按 Drive 文件 ID 区分重名对话，用 SHA-256 校验原件。可用 `--reuse DIR` 从已有的本机文件复用相同内容。语料保留消息顺序、说话人、原始链接和附件引用。未发送草稿与模型思考块仅保留在原件里。缺少事件时间的问答仍可全文检索，按日期的统计跳过这些问答。重复导入跳过相同版本，已导入对话的修订会计入 `revised_pending`，供对照两个快照核查。

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

如果已有 Qwen 栈发布到所有网卡，可运行 `scripts/server/up.sh llm-private`。它使用已安装的镜像和只读模型目录，在服务器回环端口 8010 提供独立实例。默认只开一个推理槽。批量评测时，可以在新建实例和运行评测的同一环境中设置 `LATENTSELF_JUDGE_SLOTS=2`，并行处理两批候选，每槽保留 24,576 token 的上下文。将 SSH 隧道的本地 18000 转发到远端 8010，并设置 `LATENTSELF_LLM_URL=http://127.0.0.1:18000/v1`。结束后运行 `scripts/server/down.sh llm-private`。

向量服务默认按单模型运行分配显存，`LATENTSELF_GPU_MEMORY_FRACTION` 可指定 vLLM 使用的显存比例。实测多服务共用显卡时，可以用 `LATENTSELF_KV_CACHE_BYTES` 指定每个新建向量或重排实例的 KV 缓存字节数。比较模型时可依次加载，每种模型的评测查询向量会保存在 Mac 的私有评测库中。

建完多模态索引后，日常文字搜图和文字检索可以使用 `scripts/server/up.sh query`。该实例使用同一份 VL 权重，只加载文字查询所需部分，默认分配 1.5 GiB KV 缓存。已用 9 条查询核对过它与完整实例的输出，并实测可与 `llm-private` 共用 48 GiB 显卡。它与 `embed` 共用端口 8001。切换前用 `scripts/server/down.sh` 停掉当前角色，补算图片向量时启动 `embed`。

### 模型比较与验收

私有查询和标注存放于 `$LATENTSELF_DATA/processed/latentself/eval/`。默认 fast 只用原词和缓存中的扩展词。`--expand` 允许冷查询主动请求扩展，`--terms` 使用指定词；strict 会并行请求扩展和查询向量。

```bash
lself eval --index-pool Qwen/Qwen3-VL-Embedding-8B --pool-source aistudio
lself eval --index-pool Qwen/Qwen3-Embedding-8B --pool-source aistudio
lself eval --prepare                   # 在对齐的候选索引上冻结配置输出、候选池和人工样本
lself eval --checklist review.html     # 独立的只读清单，须使用新文件名
lself eval --status
lself eval --label                     # y 是、n 否、s 跳过、q 暂停，每次默认 30 条
lself eval --photos                    # 本机照片页可选择目标文件
```

两个来源的候选按 S1/S2/S3 分别抽 4/6/6 条。名额按层内来源比例分配，有 AI Studio 候选时为它至少保留一个名额。已有本人标签计入样本。指定目标照片与相关性标注分别保存，也可用 `lself eval --target-photo 查询序号 "/完整/本机路径"` 登记。抽样完成后等待本人标注。

标注完成后，`lself eval --score-reranker` 可以单独保存重排评分，`--calibrate` 按文本查询合并后的加权指标选型。`lself eval` 输出六个子层的统计、按查询及来源的结果、照片前五、样本误报漏报和 1,000 次分层 bootstrap 区间。新索引或配置改变输出时，使用新的 `--review-db` 和 `--carry-review 原标注库` 重新准备。明细只留在本机。

### 标题、摘要与可追溯写作

这些命令使用 P2 数据库格式。请先完成检索模型校准，再在自己的数据上运行。

```bash
lself titles --limit 50
lself digest --level month --period 2024-03
lself digest
lself facts extract
lself facts merge --vectors-only
lself facts merge
lself facts verify
lself facts export
lself facts review
lself facts review --label
lself write "一段经历"
lself write "一段经历" --continue
```

首次写作会显示时间线和大纲的本机路径。编辑大纲，保留每章的 `##` 标题，再执行 `--continue`。草稿按句核对用户原文，证据不足处带有 `[待核实]`。默认经已有隧道使用自己的模型。选用 `--llm claude` 时，每次运行都需在终端同意把写作材料发送给 Anthropic。

### Claude Code 的只读 MCP 接口

在本机安装带官方 MCP SDK 的 LatentSelf 后，可在需要查询的项目的 `.mcp.json` 使用以下配置。MCP 通过标准输入输出启动，每次查询使用只读 SQLite 连接。将本机档案的查询结果交给云端客户端之前，应先确认该客户端的数据去向。

```json
{"mcpServers":{"latentself":{"command":"lself","args":["mcp"]}}}
```

也可以复用本机已有 NumPy 和官方 Python MCP SDK 的解释器。将 `command` 改为该 Python 的绝对路径，`args` 设为 `["-B","-m","latentself.mcp_server"]`，并在同一服务的 `env.PYTHONPATH` 填写 LatentSelf 仓库的绝对路径即可。已验证 SDK 1.29 和 2.3 的 stdio 接口。

可查询对话、日期时间线、摘要、核查过的事实和附件元数据。结果超过上限时会带有 `truncated` 和省略计数。

### 可解释特征与话题

服务器端运行 `bash scripts/server/fetch-readers.sh` 缓存固定版本的 9B、27B 读者和实验层。可从 Mac 执行 `ssh server 'bash -s' < scripts/server/fetch-readers.sh`。读者监听服务器回环端口 8003；同端口的文本向量服务须先停止。把 `LATENTSELF_READER_URL` 指向 SSH 转发的本机地址。

```bash
lself units --sample-paste 100
lself units --sample-paste 100 --label
scripts/server/up.sh reader9
lself interp experiment --model 9b --precision bf16
scripts/server/down.sh reader9
# 另两组分别运行 reader27-offload / bf16-offload 和 reader27-int8 / int8
lself interp experiment --select
lself interp encode
lself interp baseline
lself interp label
lself topics build
lself topics name
lself topics report
lself analyze all
lself features build-dashboard
lself search --feature 123 --json
```

选型要求三个运行条件各有五层、同一组 2,000 个单元、每层 50 个带独立留出集的检测分数。`--count` 可用于小样本连通性检查。代码将这类运行排除在正式选型之外。实验指标和校验清单写在本机。公开基准采用 AllenAI 的 WildChat-1M 清理版（ODC-BY）。在服务器用 `python -m latentself.interp.baseline ~/.cache/latentself/wildchat-zh-unique.jsonl` 生成固定种子的 15,000 个不同对话样本后，再启动读者。服务器可复用此样本，返回给 Mac 的是编号、哈希与编码结果。

历史时区用 `LATENTSELF_TZ_PERIODS` 指定，例如 `[{"from":"2024-01-01","zone":"Europe/Berlin"}]`。未覆盖时段按 UTC 统计。`LATENTSELF_LIFE_DATE` 可设置人生节点对照日期。本机看板在输出目录的 `features/index.html`，可按名称筛选、查看原文高亮、打开出处。

在共用一张显卡时，可以先为每个实验运行条件执行 `lself interp experiment --model … --precision … --stage prepare`。停掉读者、启动本地 LLM 后，使用同样的参数和 `--stage score`。前一步只需要读者，后一步从 Mac 的私有检查点读取样本并打分。全量特征命名 `lself interp label` 也支持这两个阶段。

单独验证公式时可运行 `lself interp experiment --model 9b --precision bf16 --stage pilot`。它使用同一批 2,000 个单元中的前 100 个，逐层比较两种公式并保存检查点。`--stage prepare` 遇到相同输入会复用这些计算。

话题聚类默认在服务器运行 Leiden。运行 `bash scripts/server/setup-analysis.sh` 可在服务器建立隔离的 Python 环境。Mac 只通过 SSH 发送本次计算的整数节点编号、邻接边及权重，返回四层社区编号。已在本机安装可选依赖时也可使用 `lself topics build --compute local`。

### 电影推荐

设置 `LATENTSELF_CATALOG_SRC` 指向包含两个目录导出文件的文件夹。导入会按 IMDb ID 和 canonicalKey 合并。若私人目录只有精选清单、没有明确的观看状态，导入会报告数量并保留未知状态。可通过 `LATENTSELF_OWNER_LIST_STATUS` 将你明确指定的来源键映射到 `watched` 或 `wanted`。

```bash
lself catalog import
lself catalog enrich --limit 50
lself catalog aliases
lself rec detect-watched
lself rec detect-watched --review
lself rec detect-watched --label
lself rec encode
lself rec backtest
lself rec --n 20 --explain
lself rec --json
lself rec feedback <影片ID> like
```

元数据补全从环境变量 `TMDB_API_KEY` 读取访问凭据，按公开影片 ID 请求，并在本机缓存。缺少 ID 的影片会单列。检测和编码可使用 `--limit` 分批运行。影片理由来自选定特征和用户原话，结果在输出目录的 `rec/`。可用 `LATENTSELF_CULTURE_TOPICS` 配置文化话题名的关键词列表。

TSPDT 的原始排名工作簿提供公开 IMDb 链接。在服务器用带 `xlrd` 的 Python 执行 `python -m latentself.tspdt_ids StartingList.xls ~/.cache/latentself/tspdt-2026-ids.json`，即可提取 2026 年前一千名的校验值和 ID。Mac 运行 `lself catalog import --public-ids-server`，按排名、片名、导演、年份匹配。后续导入会沿用已核实的对应关系。工作簿来源见 [TSPDT 2026 榜单与下载](https://theyshootpictures.com/gf1000.htm)。

回测按对话证据时间排序，最后五分之一用于评估。时间相同的影片留在同一侧。画像、长度归一化、观看状态和显式反馈均按切分点重算。两个基线分别只按榜单先验和向量相似度排序。正式推荐的 Recall@50 须超过两者，MRR 则追踪到第一次命中。使用旧版基线的库需重跑 `lself rec backtest`。网格选权和指标共用这份留出集，结果仍需结合后续反馈判断。
