# LatentSelf v2 设计

> 2026-11-26 起草。旧版 `~/code/LatentSelf-private` 保留为参考，不再迭代。

## 0. 一句话

把我所有 AI 对话（先做 Gemini，之后加入 AI Studio 和 Claude）变成一个**本地**的个人数据底座，在上面支撑四件事：查询、画像与推荐、AI 写作复盘、可视化。

这四件事共用一个核心：**同一个 embedding 空间里的潜在特征（latent features）**。查询是在这个空间里找邻居，画像是在这个空间里找我反复出现的方向，推荐是把电影或书投进同一个空间、看它们点亮了哪些我的方向，可视化就是把这个空间画出来。

## 1. 原则

1. **数据不出本地。** 原始数据、索引、向量、模型都放在 Mac 或自己的 GPU 服务器上。批量计算（embedding、图片描述、逐条判断）一律用本地模型。只有在我明确选择时，单次高质量生成（例如写书）才调用云端模型。
2. **代码和数据分离。** 仓库里只有代码；数据放在 `$LATENTSELF_DATA`（默认 `~/data`），产出放在 `$LATENTSELF_OUT`（默认 `~/t/_LatentSelf`）。
3. **一切可溯源。** 每条事实、推荐、段落都带上它来自哪个对话、哪一轮（`conv:<id>#<turn>`），并且能跳回原文和 Gemini 链接。
4. **增量。** 新的 Takeout 快照放进 `raw/gemini/<日期>/`，只处理新增或变化的轮次。
5. **先能用，再好看。** CLI 和 MCP 优先，Web UI 和可视化放到最后。

## 2. 数据层

### 2.1 原始快照（只读）

```
~/data/raw/gemini/<YYYY-MM-DD>/MyActivity.json
~/data/raw/gemini/<YYYY-MM-DD>/attachments/        # 上传的照片、语音、PDF、生成的图片
~/data/raw/aistudio/ ...  ~/data/raw/claude/ ...   # 之后接入
```

### 2.2 Gemini 字段 → 统一模型

| Takeout 字段 | 去向 |
|---|---|
| `title`（前缀 + 原话） | `turn.kind`（prompt / branch / link_import / live / canvas / feedback …）+ `turn.prompt` |
| `time` | `turn.ts`（UTC） |
| `details[].url` → `app/<id>` | `turn_conv`（多对多；分支共享的轮次同时属于多个对话） |
| `safeHtmlItem[]` | `answer`（每个草稿一行，`draft_idx`）；同时存 HTML 和纯文本 |
| `subtitles` | 附件原名、「N generated image」「Audio included」；Canvas 的全文 |
| `attachedFiles` / `imageFile` | `attachment`（原名、存储名、路径、类型、是否找到） |
| `Gave feedback` / `Selected preferred draft` | `signal`（显式偏好），按时间挂到最近的轮次 |
| 没有链接的 Canvas | `canvas`，按时间挂到最近的对话 |

没有用的字段（`header` / `products` / `activityControls`）丢弃。

### 2.3 存储

单个 SQLite 文件 `~/data/processed/latentself/ls.db`：

- `conversation(id, source, first_ts, last_ts, n_turns, title*)`：`title` 由模型生成并缓存
- `turn(id, source, kind, ts, prompt, snapshot)`：`id` 由来源和内容哈希得到，跨快照保持稳定
- `turn_conv(turn_id, conv_id)`
- `answer(turn_id, draft_idx, html, text)`
- `attachment(id, turn_id, orig_name, stored_name, path, mime, caption*, ocr*, transcript*)`
- `canvas(id, ts, title, content, conv_id*)`
- `signal(id, ts, kind, value, turn_id*)`
- `fts_turn`：FTS5 trigram 全文索引，覆盖提问、回答、附件描述，中文可直接子串匹配
- `vec(item_kind, item_id, model, dim, blob)`：向量；规模约 2 万条，用 numpy 暴力计算余弦相似度即可，不需要向量数据库
- `fact(id, ts, text, evidence[])`、`digest(period, text, evidence[])`：AI 整理层，带证据
- `feature(id, model, …)`、`feature_act(feature_id, item_id, act)`：LatentSelf 特征
- `item(id, domain, meta, …)`：推荐候选（电影、书、音乐）

（标 * 的列由后续阶段填充。）

## 3. 计算层

| 任务 | 模型 | 在哪跑 |
|---|---|---|
| 文本和图片 embedding | Qwen3-VL-Embedding-8B（主候选，文字和图片同一个空间），在评测集上与 Qwen3-Embedding-8B 比较后定（见 P1-SPEC §5.0） | 4090 |
| 可解释特征 | 读者模型 Qwen3.5-9B-Base + 官方 Qwen-Scope SAE（W64K），见 PIPELINE.md L4b | 4090 |
| 重排 | Qwen3-VL-Reranker-8B | 4090 |
| 图片描述 + OCR | Qwen3.8-27B（服务器上已有，原生视觉，llama.cpp :8000） | 4090 |
| 语音提问转写 | faster-whisper-large-v3（服务器上已有） | 4090 |
| 检索相关性判断、对话标题、事实抽取、每期摘要 | Qwen3.8-27B（本地，默认）；可选 Claude | 4090 / 云端 |
| 写书、长篇复盘 | Claude（需要单次明确同意）或本地模型 | 云端 / 4090 |

图片的处理策略是**先描述，再嵌入**：每张图生成一段中文描述，加上 OCR 文字，写进 `attachment.caption`，再一起建全文索引和向量。这样「我发过一张路牌的照片」就能被文字查询找到。如果选用多模态 embedding（Qwen3-VL-Embedding），每张图还会直接算一个和文字同空间的图片向量，这样以图搜图、以文搜图都可以做。

服务器只负责算，算完的结果回写到 Mac 上的 `ls.db`。服务器上不长期保存原始数据副本，每次处理完就删掉。

## 4. 四个功能

### 4.1 查询（第一阶段）

```
lself search "咖啡"            # 返回所有相关对话（不是前 k 条）
lself search "路牌" --images   # 只看带图片的轮次，直接给出文件路径
lself show <conv_id>          # 按顺序显示整段对话（提问、回答、附件）
lself open <attachment_id>    # 在 Finder 或预览中打开
```

检索流程（目标是「全部找出来」）：

1. **召回**：FTS5 关键词结果 ∪ 向量近邻结果（门槛放宽，并做查询扩展：让模型把「咖啡」扩展成「手冲、意式、咖啡豆、拿铁、Kaffee…」）。
2. **聚合**：把命中的轮次合并到对话，用对话内最高分加上命中轮数来排序。
3. **判断**：本地模型逐个对话判断「是否在讨论这个主题」，给出 是/否 和一句理由。结果按 `(query, conv)` 缓存。
4. **输出**：按时间排列，每个对话显示日期、轮数、标题、命中的原句、附件和 Gemini 链接。

### 4.2 画像与推荐（第三、四阶段）——LatentSelf 核心算法

> **2026-11-27 更新**：潜在特征改用「读者 LLM + 官方 Qwen-Scope SAE」，不再在 embedding 上自己训练 SAE；话题、画像、推荐的流程都以 [PIPELINE.md](PIPELINE.md) 为准。下面保留原始思路，作为背景说明。

#### 潜在特征：稀疏字典学习（SAE）

旧版用 UMAP + HDBSCAN，把每个点硬分到一个簇里。问题在于：一段对话往往同时属于好几个方向（比如「在国外看医生」同时属于健康、制度、花钱），而且簇只能是一个「语义标签」。

新版在 embedding 空间里学习一组**稀疏方向**：

- 输入：每一轮的 embedding $x \in \mathbb{R}^d$（约 1.5 万轮，加上附件描述和 Canvas）。
- 训练一个 TopK 稀疏自编码器：$z = \mathrm{TopK}(W_e(x - b))$，$\hat{x} = W_d z + b$。字典大小约 512–2048，每轮只激活 k≈8–16 个特征。
- 每个特征是一个方向，它的含义**不靠起名**，而是由「最能激活它的那些轮次」和「它在时间上的出现模式」来呈现。很多特征无法用一句话概括（比如「周末、出门、拍照问路」），但它们确实存在，而且可以指向别的东西。这正是最初的设想。
- 每个特征算三组统计：
  - 激活强度随时间的变化：周期性、持续还是突发；
  - 和其他特征的共现关系；
  - 最强激活样本，供人和模型阅读理解。
- 模型可以给特征加一个**可选的**暂定名，但界面始终以样本为准。

#### 话题：kNN 图 + Leiden 社区（用来回答「哪些是同一个爱好」）

- 先以**对话**为单位（取对话内各轮 embedding 的均值，再按长度加权），构建互为 k 近邻的图（k≈15，余弦相似度）。
- 用 Leiden 算法做社区发现，扫描多个分辨率，得到层级话题树：「运动 → 跑步 → 装备」。
- 为什么不用 K-means：K-means 要预先指定 K，默认簇是球形、大小相近，而且每个点都必须归某个簇。我的数据里话题大小差异极大（一个长期话题可能有几百轮，一次随口一问只有 1 轮），形状也不规则。Leiden 只依赖局部邻接关系，不用预设数量，也能保留孤立的小话题。
- 判断「是不是同一个爱好反复出现」：对每个话题生成活跃时间序列，按天或按周统计。再把「间隔 >14 天后再次出现」定义为一次复发，统计复发次数和跨度。

两层的关系：**话题**是硬的、可命名的，方便导航；**特征**是软的、可叠加的，方便画像和推荐。

#### 画像

画像 = 一组特征的权重向量。权重由激活总量、时间衰减（半衰期约 6 个月，可调）、显式信号（点赞/点踩、看过/喜欢）共同决定，并给出证据轮次。另外由模型从每期摘要中抽取**事实层**（`fact`，带证据），比如「某天开始学吉他」。

#### 推荐（先做电影）

- 候选库：公开影单（如 TSPDT，含导演和 IMDb ID）、TMDB 元数据、自己的观影记录表（看过或想看）。每部电影把「简介 + 类型 + 导演 + 年代 + 关键词」用同一个 embedding 模型编码，再用同一个 SAE 编码成特征激活。
- 打分：`score(item) = Σ_f profile[f] · act_item[f]`，再加上和最近若干轮的相似度（短期兴趣）、多样性约束（MMR），并排除已经看过的（从对话和观影记录中识别）。
- 解释：每条推荐给出「它点亮了你的哪些特征」，以及这些特征最强的 2–3 段原话。
- 反馈闭环：`lself rec feedback <item> like|dislike|watched` 写入 `signal`，下次打分时生效。
- 书和音乐使用同一套框架，只换候选库。

### 4.3 AI 写作与复盘（第二阶段搭基础设施，之后随时可用）

- **MCP 服务**：`latentself-mcp` 暴露 `search` / `show_conversation` / `timeline` / `facts` / `digests` / `features` / `attachment`。Claude Code 或本地 agent 可以直接调用，写「某个人生阶段」这样的小书时，靠检索取材，而不是把全部数据塞进上下文。
- **多级摘要**：月度 → 季度 → 全年，以 `digest` 表缓存，每条带证据。新快照只需重算受影响的月份。
- **写作流程**（`lself write "<主题>"`）：
  1. 检索相关对话和事实；
  2. 生成时间线；
  3. 生成大纲，给我确认；
  4. 逐章写作，每段都带引用；
  5. 由一个核查 agent 回查原文，核对引用是否属实。

### 4.4 可视化（最后再定）

候选方向，等前三项做完再选：

- **特征时间河流图**：x 轴是时间，每条带子是一个特征的激活强度，用来看兴趣的迁移和复发。
- **日历热力图**：选中一个话题或特征，看它落在一年里的哪些天、一天中的哪些小时。
- **话题层级图**：话题树的 icicle 或 sunburst 图，点击下钻到对话。
- **二维地图**：对话级的 UMAP 或 PaCMAP 投影，保留局部结构，颜色表示话题、亮度表示时间；可以叠加我的推荐物品，看它们落在哪里。
- 关于高维的说明：二维投影只用来**导航**，不用来下结论。所有「相似」或「重复」的判断都在原始高维空间（或特征空间）里计算，图上只负责展示。

## 5. 仓库结构

```
latentself/
├── latentself/
│   ├── config.py          # 路径、模型端点
│   ├── db.py              # schema + 迁移
│   ├── ingest/gemini.py   # Takeout → 统一模型（之后加 aistudio.py、claude.py）
│   ├── enrich/            # 图片描述、语音转写、对话标题（调用服务器）
│   ├── embed.py           # 增量 embedding
│   ├── search.py          # 召回 + 聚合 + 判断
│   ├── latent/sae.py      # 稀疏自编码器
│   ├── latent/topics.py   # kNN 图 + Leiden
│   ├── profile.py
│   ├── rec/               # 候选库 + 打分 + 解释
│   ├── write/             # 摘要 + 写作流程
│   ├── mcp_server.py
│   └── cli.py             # `lself ...`
├── tests/                 # 用小型合成 Takeout 做测试，不放真实数据
└── docs/DESIGN.md
```

依赖：Mac 端用 Python 3.12，核心只用标准库加 numpy。**所有模型（embedding、reranker、LLM），以及 SAE 训练，都在服务器的 4090 上运行**，Mac 不下载、不运行模型（2026-11-27 决定，原因是 Mac 磁盘紧张，而服务器因为时差大多空闲）。Leiden 用 `leidenalg` + `igraph`。

## 6. 阶段与验收

> 2026-11-27 起，各阶段的详细 spec 见 [P1](P1-SPEC.md) · [P2](P2-SPEC.md) · [P3](P3-SPEC.md) · [P4](P4-SPEC.md)，整体流程见 [PIPELINE.md](PIPELINE.md)。下表保留为概览。

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | 仓库骨架、导入 Gemini（全部字段）、SQLite 和 FTS、`lself show` | 真实导出的记录全部入库；每条的去向都能统计；分支共享的轮次归属正确 |
| P1 | embedding、图片描述和 OCR、语音转写、混合检索与判断、`lself search` | 搜一个主题能给出完整清单，没有只是字面碰巧相同的误报；能用文字找到指定照片 |
| P2 | 对话标题、月度摘要和事实层、MCP 服务 | Claude Code 通过 MCP 写出带引用的某段经历时间线 |
| P3 | SAE 特征、Leiden 话题、画像 | 特征浏览器（CLI 或 HTML）：每个特征显示最强样本和时间曲线 |
| P4 | 电影推荐和反馈 | 给出 20 部推荐，每部都附带证据；排除已看过的 |
| P5 | 可视化 | 待定 |
| — | 接入 AI Studio 和 Claude 导出 | 同一套 schema，只增加 ingest |

## 7. 已定（2026-11-26）

- 新仓库：`~/code/latentself`。
- 批量模型（检索判断、事实抽取、摘要）：默认用本地 Qwen3.8-27B；后端可插拔（`LATENTSELF_LLM=local|claude`）。
- 长篇写作：默认用本地模型，要用 Claude 时每次单独确认。
