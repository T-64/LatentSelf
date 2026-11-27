# P1 Spec：语义搜索与附件理解

> 状态：已确认，可以开工（2026-11-27）。所有决定见 §10。
> 前置：P0 已完成（`docs/DESIGN.md`、`latentself/ingest/gemini.py`、`ls.db`）。实现者开工前先读 DESIGN.md §1–§4.1 和 PIPELINE.md §0–§2（L2–L4a），再读本文件。

## 1. Intent（owner 想要的）

> 「我搜一个主题，就把我所有关于这个主题的对话重新弹出来。」
> 「我之前给它发过一张图片，我想找到那张图片。」

拆开来说：

- **I1 全量召回**：要的是「所有相关的对话」，不是前 10 条。宁可让我多看几条，也不要漏掉。
- **I2 按意思搜，不按字面搜**：换一种说法（同义词、外语、只描述症状）也要能找到；只是字面碰巧包含查询词、其实在聊别的事的，不算。
- **I3 图片能用文字找到**：照片本身没有文字，所以要先让模型看图，生成描述。
- **I4 数据在本地**：原始数据和索引都在 Mac 上；计算可以借用 owner 自己的 4090 服务器，但服务器上不留副本。
- **I5 是后续的地基**：P1 产出的向量和图片描述，P3 的 SAE 特征、P4 的推荐都会直接复用。所以向量要存得规整，并且可以增量更新。

## 2. 范围

**要做**

1. 附件理解：图片（含 HEIC）、Gemini Live 的摄像头帧、PDF/DOCX 文本 → 写进 `attachment.caption / ocr`。
2. 文本 embedding：每个对话轮次生成一个向量，可以增量更新。
3. 混合检索：查询扩展 → 关键词召回 ∪ 向量召回 → 聚合到对话 → （strict 模式下）模型逐个判断。
4. CLI：`lself enrich`、`lself embed`、`lself search`。
5. 评测集和评测脚本，用来证明 I1、I2、I3 做到了。

**不做（留到之后的阶段）**

- 对话标题、事实层、摘要、MCP（P2）；SAE、话题、画像（P3）；推荐（P4）；Web UI 和可视化（P5）。
- 视频（mp4，20 个）、zip 里的非帧文件、代码文件的语义理解。
- AI Studio、Claude 导出的导入。

## 3. 已核实的事实（实现时不用再查）

| 事实 | 数值 / 结论 |
|---|---|
| 对话类轮次（prompt / branch / link_import / live） | 15,392 |
| 提问长度 | p50 32 字，p90 152，p99 约 5,000，最长约 19.5 万（是粘贴进来的文档） |
| 回答长度（draft 0，纯文本） | p50 1,081，p90 2,015 |
| 已找到的附件文件 | 3,115 个不同文件：jpg 2,209、png 559、heic 172、wav 96、pdf 74、zip 24、mp4 20、docx 13，其余很少 |
| Gemini **生成**的图片 | 文件名是 `watermarked_img_*`（59 个），对应 `subtitles` 里的 `N generated image.`。**它们不是 owner 拍的照片** |
| `Audio included.` 的 96 条 | 是 Gemini Live 轮次：一段约 1–2 秒的 wav，加一个 `*.zip`，里面是 `frame_0.png…frame_N.png` 摄像头帧。提问文字（title）**已经是语音转写结果**，所以 wav 不需要再转写 |
| 服务器 | RTX 4090 D 48GB、62GB 内存。Qwen3.8-27B（原生视觉，llama.cpp docker，`:8000`，OpenAI 兼容，`-np 1` 单槽，默认开 thinking）用 `qwen-start` / `qwen-off` 启停。Qwen3-Embedding-8B 的权重在 `~/.cache/huggingface`，但**目前没有可用的推理服务**（服务器上没有 vllm 环境） |
| 网络 | Mac 和服务器不在同一地区，跨境延迟较高；通过 `~/.ssh/config` 里已配置好的主机别名 `server`（公网）或 `server-ts`（Tailscale）连接。建议走 SSH 隧道访问服务器端口 |
| Mac | M5，16GB，**磁盘空间紧张**。只负责调度、存 `ls.db` 和做轻量预处理（`sips` 转 HEIC、`ffmpeg`）。**任何模型都不在 Mac 上下载或运行** |
| 服务器的使用时段 | 因为时差，服务器大部分时间空闲，模型服务可以常驻；磁盘剩余约 356GB，新模型直接下载到服务器上的 HF 缓存 |
| `attachment.mime` | 按文件名猜的。`watermarked_img_*` 需要另外标成 `origin='generated'` |

## 4. 数据模型变更

在 `db.py` 里加一个**可重复执行**的迁移（P0 的库没有设置 `user_version`，即为 0；P1 迁移后设为 2，后续每个阶段各加 1）：

```sql
ALTER TABLE attachment ADD COLUMN origin TEXT;          -- user | generated | live_frame
ALTER TABLE attachment ADD COLUMN enrich_model TEXT;    -- e.g. qwen3.8-27b@q6
ALTER TABLE attachment ADD COLUMN enrich_at TEXT;
ALTER TABLE attachment ADD COLUMN enrich_error TEXT;
ALTER TABLE attachment ADD COLUMN tags TEXT;            -- JSON array
ALTER TABLE attachment ADD COLUMN content_kind TEXT;    -- photo | screenshot | document | chart | meme | other

CREATE TABLE IF NOT EXISTS embedding (
    item_kind  TEXT NOT NULL,          -- turn | turn_self | attachment | item（P4）
    item_id    TEXT NOT NULL,          -- turn.id or attachment.id
    model      TEXT NOT NULL,          -- e.g. Qwen3-VL-Embedding-8B
    dim        INTEGER NOT NULL,
    input_hash TEXT NOT NULL,          -- sha1 of the exact text (or image bytes + text); changes → re-embed
    vec        BLOB NOT NULL,          -- float16, L2-normalized
    PRIMARY KEY (item_kind, item_id, model)
);
-- embedding has no FK to turn/attachment; a cleanup step deletes orphans after ingest.

CREATE TABLE IF NOT EXISTS search_cache (
    key        TEXT PRIMARY KEY,       -- sha1(kind|model|query[|conv_id])
    kind       TEXT NOT NULL,          -- expand | judge
    value      TEXT NOT NULL,          -- JSON
    created_at TEXT NOT NULL
);
```

- 重新导入（`lself ingest`）**不能覆盖** enrich 和 embedding 的结果。现在的 `ingest()` 会先删再插 attachment，需要改成保留已有的 caption / ocr / tags 等列（按 `attachment.id` 合并）。这一点**必须有测试**。
- turn 改用 `INSERT … ON CONFLICT(id) DO UPDATE`，避免 `INSERT OR REPLACE` 的级联删除把已有结果一起删掉。
- 一个 Live 轮次的 zip 展开成多张帧，但**不要**为每张帧建一行 attachment。只抽 3 张有代表性的帧（首、中、尾）一起交给模型，描述写回 zip 那一行，并设 `origin='live_frame'`。

## 5. 模块与接口

### 5.0 模型（武器库，2026-11 调研）

所有模型都在服务器上运行，Mac 只通过 SSH 隧道调用 HTTP 接口。

| 角色 | 候选 | 要点 |
|---|---|---|
| **Embedding（主候选）** | [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B)（2026-01，Apache-2.0） | 文字、图片、截图、视频在**同一个向量空间**；4096 维，支持 MRL（64–4096）；vLLM `runner="pooling"`；MMEB-V2 77.9，MMTEB 67.88 |
| Embedding（纯文本对照） | [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)（2025-06，服务器上已下载） | 中文证据最充分：C-MTEB 73.84，MMTEB 70.58；只支持文字；支持 TEI 和 vLLM。语料里中文提问占 93.5%，所以对照组选它，不选主打多语言的模型 |
| Reranker | [Qwen/Qwen3-VL-Reranker-8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B)（2026-01，Apache-2.0） | 交叉编码器，查询和文档都可以是图文混合；用于 strict 模式的第一道判断 |
| LLM / 视觉 | Qwen3.8-27B，Unsloth UD-Q6_K_XL GGUF（服务器上已有，llama.cpp :8000） | 图片描述、查询扩展、最终判断；调用时关闭 thinking |

**选型流程（实现的第一步，必须先于全量计算）：**

1. 在服务器上下载 Qwen3-VL-Embedding-8B 和 Qwen3-VL-Reranker-8B（Qwen3-Embedding-8B 已在服务器上）。
2. 用评测集（§7）的候选池，对两个 embedding 模型各算一遍向量，比较 fast 模式的召回率。评测集只覆盖评测查询召回到的轮次，不需要先全量计算。
3. 决策规则：
   - 默认选 **Qwen3-VL-Embedding-8B**：一个空间同时装下文字和图片，P4 的推荐依赖这一点。
   - 只有当 Qwen3-Embedding-8B 在文本查询上的召回率比 VL **高出 ≥ 3 个百分点**时，才改用双向量方案：文本轮次用 Qwen3-Embedding-8B，图片用 VL，检索时两路融合。
4. Reranker：在 strict 模式里比较「reranker → LLM 只判断边界样本」和「LLM 判断全部候选」两种做法的精确率和耗时，选更好的那种。
5. 把比较结果（各模型在每个查询上的召回率、精确率、耗时）写进 `docs/P1-MODEL-EVAL.md`。只放汇总数字，不放查询词和对话内容。


```
latentself/
├── backends/
│   ├── llm.py        # chat(messages, images=None, json_schema=None) -> dict|str
│   └── embed.py      # embed(texts, kind="doc"|"query") -> np.ndarray[float32]
├── enrich/
│   ├── images.py     # 图片和 Live 帧 → caption / ocr / tags / content_kind
│   └── docs.py       # PDF / DOCX → ocr 列（纯文本抽取，不调模型）
├── embed.py          # 拼接文本、增量计算、读写 embedding 表
├── search.py         # expand → recall → aggregate → judge → results
└── cli.py            # 新增 enrich / embed / search
```

### 5.1 backends

- 通过环境变量配置：`LATENTSELF_LLM_URL`（默认 `http://127.0.0.1:8000/v1`，经 SSH 隧道）、`LATENTSELF_EMBED_URL`、`LATENTSELF_LLM=local|claude`（默认 `local`）。
- `llm.chat` 调用本地 Qwen 时必须**关闭 thinking**：传 `chat_template_kwargs: {"enable_thinking": false}`，`temperature 0.2`；需要结构化输出时用 `response_format` 的 JSON schema。
- 必须有 `FakeLLM` 和 `FakeEmbed`（确定性、离线），所有单元测试只用它们。
- `embed.py` 后端：接口统一为 `embed_texts(texts, kind)` 和 `embed_images(items)`（图片可以附带文字）。每个模型的查询前缀和指令格式按各自模型卡实现（Qwen3-Embedding 用 `Instruct: …\nQuery: …`，Qwen3-VL-Embedding 用 system prompt），文档一律不加前缀。输出做 L2 归一化，用 float16 存储，维度取模型原生维度（Q7）。
- **查询时也必须连服务器**，因为查询向量要用同一个模型来算。服务器不可用时，`lself search` 自动降级为只用关键词，并在输出第一行明确说明。

### 5.2 enrich/images.py

- 输入：所有 `mime LIKE 'image/%'`、找到了文件、`caption IS NULL` 的 attachment，加上 Live 的 zip。
- 预处理（在 Mac 上做）：HEIC 用 `sips` 转 JPEG，长边缩到 1536px，以 base64 放进请求。**不往服务器拷贝任何文件**。
- 一次请求处理一张图（Live 一次处理 3 帧）。上下文里附上该轮的提问原文（截断到 300 字），让描述贴合当时在问什么。
- 输出 JSON：
  ```json
  {"caption": "中文 2–4 句：主体、场景、动作，以及能判断出的地点或物品类型",
   "ocr": "图中可读文字原样照录，没有则为空字符串",
   "tags": ["名词短语", "..."],
   "content_kind": "photo|screenshot|document|chart|meme|other"}
  ```
- `origin='generated'` 的图同样生成描述，但在 caption 前加上 `[Gemini 生成]`。
- 如果选定的是多模态 embedding 模型，每张图还要额外算一个**图片向量**，写进 `embedding(item_kind='attachment')`。输入是图片加上当轮提问。描述（caption）仍然要生成：关键词检索、结果展示和 LLM 判断都需要文字。
- 断点续跑：每张图处理完立即 commit；失败写 `enrich_error` 并跳过，最多重试 1 次；最后汇总成功、失败、跳过的数量。
- 预计：约 2,900 张 × 8–10 秒 ≈ 6–8 小时（单槽、关闭 thinking 时）。（Q4 已确认）

### 5.3 enrich/docs.py

- PDF 用 `pdftotext`（poppler）或 `pypdf`；DOCX 用 `python-docx` 或直接解析 zip 里的 XML。只抽前 20,000 字写进 `ocr`，不调用模型。（Q6 已确认）

### 5.4 embed.py：每轮嵌入的文本

```
{提问原文，截断到 2,000 字}
[回答] {draft 0 纯文本的前 500 字}
[附件] {每个附件的 caption；ocr 的前 300 字}
[Canvas] {标题 + 正文前 500 字}          # 仅 canvas 轮次
```

（Q2 已确认）

- 只嵌入对话类轮次和 canvas 轮次。
- 另外再算一份 `turn.self` 向量（只含提问和附件描述，不含回答），供 P3 画像使用，定义见 PIPELINE.md §L3。存储时把 `item_kind` 设为 `turn_self`。
- `text_hash` 不变就跳过；附件补上 caption 后 hash 会变，自动重算。
- 批量大小 32–64；断点续跑；最后打印新增、跳过、失败的数量。
- 预计 15.7k 条，在 4090 上用 8B 模型，几分钟到十几分钟。

### 5.5 search.py

```
search(query, mode="fast"|"strict", images_only=False, since=None, until=None) -> list[ConvResult]
```

1. **查询扩展**（**仅 strict 模式**，结果缓存）：调用 LLM，返回 8–15 个相关词，要覆盖中文同义词、上下位词、德语和英语说法，例如「咖啡」→ 手冲、意式、咖啡豆、拿铁、磨豆机、Kaffee、espresso…。CLI 的 `--terms "a,b"` 可以直接指定扩展词，跳过模型。
   - **fast 模式不调用任何 LLM**：只用原词，加上 `search_cache` 里已经缓存的扩展词（如果有的话，不额外花时间）。`--expand` 可以强制在 fast 模式里也做扩展。
   - strict 模式里，扩展请求和查询向量请求**并行**发出。
2. **关键词召回**：原词和每个可用的扩展词分别做 FTS（≥3 字走 trigram MATCH，<3 字走 instr），默认搜 `prompt` 和 `extra`（附件、Canvas），`--answers` 时也搜回答。
3. **向量召回**：查询向量对所有轮次向量和图片向量算余弦相似度（图片命中会映射回它所在的轮次），取 `cos ≥ τ` 与 top-300 的并集。τ 在评测集上标定后写进配置，不要硬编码猜一个值。
4. **聚合到对话**：每个对话的分数 = `max(轮次分数) + 0.05·log(1+命中轮数)`，关键词命中和向量命中各自归一化后取较大值。候选上限 200 个对话；**超出时要打印被截掉了多少**，不能悄悄截断。
5. **判断**（仅 strict 模式，结果缓存）：先用 reranker 给每个候选对话打分；分数明显高或明显低的直接定论，中间区间再交给 LLM（两个阈值都在评测集上标定）。如果选型阶段证明「LLM 判断全部候选」更好，就省去 reranker。LLM 判断的做法：每次请求批量判断 8–10 个候选对话。给模型的材料是：首轮提问 + 命中的轮次（每轮提问 200 字，加附件 caption），每个对话不超过 1,500 字。输出 `[{conv_id, relevant: bool, reason: "一句话"}]`。判断标准写在 prompt 里：**这个对话是不是在讨论查询主题本身**，只是字面出现查询词不算。
6. **输出**：按时间排序（`--sort score` 可选）。每个对话显示：日期范围、轮数、命中轮数、首轮提问、最多 3 条命中原句（标出是关键词命中还是语义命中）、附件路径（`--images` 时显示）、Gemini 链接；strict 模式下再加判断理由。`--json` 输出结构化结果，供后续 MCP 使用。

性能目标（2026-11-28 修订）：
- fast 模式：首次查询（查询向量没有缓存，需要一次远程 embedding 请求）端到端中位数 ≤ 3 秒，同时报告 p90。查询向量已缓存时 ≤ 1 秒。
- strict 模式：< 3 分钟（约 200 个候选）。
- 如果单次远程 embedding 请求的中位数就超过 2 秒，先分别测量网络往返时间和服务端处理时间，在报告里说明瓶颈在哪，再考虑优化（例如服务端预热、减少 padding、换协议）。不达标时先报告，不阻塞验收。

> 修订原因：初版把 LLM 查询扩展放进了 fast 模式，实测首次查询中位数 5.3 秒，其中扩展请求就占了 5.2 秒。扩展的价值主要在 strict 模式的全量召回里，fast 模式要的是快。

### 5.6 CLI

```
lself enrich [--images] [--docs] [--limit N] [--retry-errors]
lself embed  [--limit N]
lself search <query> [--strict] [--images] [--answers] [--terms a,b] [--since YYYY-MM] [--until YYYY-MM] [--sort time|score] [--json]
lself eval   [--queries tests/eval/queries.yaml]
```

## 6. 服务器运行规范

- 所有调用都通过 SSH 隧道访问：`ssh -N -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 -L 8002:127.0.0.1:8002 server`。不要把端口暴露到公网。
- 模型服务：在服务器上用 docker 运行 embedding 服务（`127.0.0.1:8001`）和 reranker 服务（`127.0.0.1:8002`），具体用 vLLM、TEI 还是 sentence-transformers，由实现者按模型卡决定。挂载服务器上的 HF 缓存，新模型也下载到那里。
- 常驻：服务器大部分时间空闲，**查询时用的 embedding 服务应该常驻**。实现者需要实测显存：embedding 服务（8B，约 17GB）、reranker（约 17GB）、Qwen3.8-27B（权重约 24GB，加上 KV cache）三者能否同时运行。放不下时的优先级是：embedding 常驻 > Qwen 按需启动 > reranker 按需启动。必要时可以调小 Qwen 的上下文长度 `-c`，但要记录原值，以便恢复。
- 在仓库里提供 `scripts/server/{up,down,status}.sh`，所有启停都通过它们执行，不在服务器上手工操作。
- llama.cpp 用现有的 `qwen-start` / `qwen-off`。这个栈 owner 平时也在用（Hermes、Claude Code 代理）；批量任务会独占它，事先记录它原来是否在运行，结束后恢复（Q5）。
- 服务器上**不落盘任何原始数据**：图片以 base64 放在请求体里发送，文本也只在请求里。运行过程中产生的临时文件（如果有）放在 `/tmp/latentself-*`，结束时删除。
- 长任务在 Mac 端放到后台运行，并输出进度日志：`~/data/processed/latentself/logs/<task>-<ts>.log`。

## 7. 评测（验收的依据）

`tests/eval/queries.yaml`（只放查询词和标注，**不放对话内容**），放在仓库外：`$LATENTSELF_DATA/processed/latentself/eval/`。⚠ 因为对话 ID 加上查询词也会透露个人信息，所以评测文件不进 git。

- 查询集（Q8）：由 owner 提供约 10 条，覆盖健康、学习、生活、兴趣几类主题，其中至少 2 条是「找某张照片」。查询词本身也放在仓库外的评测文件里。
- 标注流程（2026-11-28 修订为分层抽样）：
  1. **候选池**：对每个查询，在**索引里的全部来源**（Gemini + AI Studio，2026-11-28 决定）上取「宽松关键词召回（含扩展词）∪ 每个候选 embedding 模型的向量 top-300 ∪ 每种待评测配置的输出」的并集。任何待评测配置的输出如果不在池里，必须先补进池，再抽样。
  2. **分层**：每个查询的池分成三层：
     - S1：两路及以上命中；
     - S2：只有一路命中，且 strict 判断为相关；
     - S3：只有一路命中，且 strict 判断为不相关。

     strict 判断**只用来分层，不作为标签**：用被测系统自己的判断当答案，评测就成了循环论证。

     每一层再按来源（`gemini` / `aistudio`）拆开，共 6 个子层。
  3. **抽样**：每个查询从 S1、S2、S3 中分别抽 4、6、6 条（某层不够就全取），固定随机种子。每层的名额按两个来源在该层的池大小比例分配；只要该层里有 AI Studio 的候选，就至少给 AI Studio 1 条名额。9 个查询合计不超过 180 条，owner 大约需要 40 分钟。照片查询另外由 owner 在审阅页里指定目标文件。
  4. **标注**：用 CLI 交互式地按 y/n/s（s = 跳过，不计入）。已有的人工标签保留并计入样本。实现 agent 生成清单后**停下来等 owner 标完**。
  5. **估计**：每条已标注样本的权重 w = 该层池大小 ÷ 该层已标注数（Horvitz–Thompson 估计）。
     - 召回率 = Σ w·[相关且被检出] ÷ Σ w·[相关]
     - 精确率 = Σ w·[相关且被检出] ÷ Σ w·[被检出]

     同时用分层 bootstrap（1,000 次）给出 95% 置信区间。权重按 6 个子层分别计算。
- 指标与门槛（按全部文本查询合并后的**加权点估计**判定；按单个查询的数字只报告，不作门槛，因为每个查询的样本太少）：
  - strict 模式：召回率 ≥ 95%，精确率 ≥ 90%；
  - fast 模式：召回率 ≥ 95%，精确率不设门槛，但要报告；
  - 照片类查询：目标照片出现在前 5 个结果里；
  - 置信区间下限低于门槛时，在报告里明确写出，不阻塞验收。
  - 除了合并后的数字，还要按来源分别报告召回率和精确率。按来源的数字样本更少，只报告，不作门槛。
- 模型选型（§5.0 第 3 步）使用同一批加权估计：纯文本模型的召回率点估计比 VL 高 ≥ 3 个百分点时，才改用双向量方案。

> 修订原因：初版规定「只有一路命中的候选全部交给人工」，实测 9 个查询的池里有 2,492 条这样的候选，人工不可能标完。改成分层抽样后，人工量固定在约 150 条，并且能给出带置信区间的无偏估计。

- `lself eval` 输出：合并后的加权召回率和精确率（带置信区间）、每个查询的数字、各层的池大小和标注数、样本中漏检和误报的对话 ID（只输出到终端和仓库外的报告，不进 git）。`lself eval --status` 显示标注进度。

## 8. 测试要求

- 单元测试全部离线，只用 FakeLLM 和 FakeEmbed：
  - 迁移可以重复执行；
  - 重新导入后 caption 和 embedding 都还在；
  - `text_hash` 变化时触发重算；
  - 召回、聚合、截断时会打印被截掉的数量；
  - 判断结果的缓存命中；
  - HEIC 预处理（可以用 mock 代替 `sips`）。
- 继续沿用合成 Takeout，不使用真实数据。
- 有一个手动执行的冒烟脚本 `scripts/smoke_p1.sh`，在真实库上跑：`lself search <评测集第 1 条> --strict`、`lself search <评测集中的照片查询> --images`。

## 9. 完成的定义

1. `python3 -m unittest` 全部通过；
2. 真实库上 enrich 全部完成，失败率 < 2%，失败清单打印出来；
3. 真实库上 embed 全部完成；
4. `lself eval` 达到 §7 的门槛，并把报告贴进 PR 描述；
5. README 更新命令说明；DESIGN.md 中 P1 一行标记为完成；
6. 服务器上没有残留的容器或临时文件，显存恢复原状。

## 10. 决定（2026-11-27 已确认）

| # | 问题 | 决定 |
|---|---|---|
| Q1 | 结果单位 | 以**对话**为单位返回，命中的轮次在对话里高亮 |
| Q2 | 每轮向量的文本 | 提问 + 回答前 500 字 + 附件描述（见 §5.4）。P3 画像时另算一份只含提问的向量 |
| Q3 | 搜索模式 | 默认 fast，不调用 LLM；`--strict` 才做查询扩展和逐个判断（2026-11-28 修订） |
| Q4 | 图片描述耗时 | 可以一次性跑 6–8 小时，夜里跑，支持断点续跑 |
| Q5 | 服务器 | 实现 agent 可以自己通过 ssh 启停 Qwen 和 embedding 容器；结束后恢复原状（如果原来在跑 Qwen，就重新启动它） |
| Q6 | PDF / DOCX | 抽取文本进索引 |
| Q7 | 向量维度 | 用模型的原生维度（两个候选都是 4096），float16 存储 |
| Q8 | 评测集 | owner 已确认查询列表，放在 `$LATENTSELF_DATA/processed/latentself/eval/queries.yaml`（不进 git）。人工标注按 §7 的分层抽样进行（约 150 条，2026-11-28 修订）：实现 agent 生成 y/n 清单，然后**停下来等 owner 标完** |
| Q10 | 模型在哪跑 | 全部在服务器上跑。Mac 不下载、不运行任何模型（磁盘紧张）。查询时调用服务器上常驻的 embedding 服务 |
| Q11 | 模型选型 | 按 §5.0 的流程，在评测集上比较 Qwen3-VL-Embedding-8B 和 Qwen3-Embedding-8B 后再定，默认选 VL。harrier 已移除：语料中文占 93.5%，主打多语言没有优势 |
| Q12 | 评测范围 | 包含 AI Studio：候选池和抽样覆盖索引里的全部来源，按来源分子层抽样，按来源分别报告（2026-11-28） |
| Q9 | 云端模型 | 判断和查询扩展只用本地模型；如果评测不达标，先报告，不要自行切换到 Claude |
