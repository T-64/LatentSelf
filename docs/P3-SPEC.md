# P3 Spec：可解释空间、话题与分析

> 状态：已确认，可以开工（2026-11-27）。前置：P1 完成（`embedding` 表里有 `turn` 和 `turn_self` 两种向量）。P2 不是前置。
> 先读 PIPELINE.md 全文（本阶段对应 L3 的 episode 和 is_paste、L4b、L5a、L5b、L6），再读本文件。通用约定见 P1-SPEC §5.1、§6、§8。

## 1. Intent

- **I1 说不清，但存在**：找出「我」在高维空间里反复出现的方向。这些方向不一定能用一个词概括，但可以统计、追踪、对比，还能指向别的东西（推荐）。
- **I2 Anthropic 式的可解释性**：每个特征都能看到它在哪些原话、哪几个字上被点亮，什么时候出现，和哪些特征一起出现。
- **I3 和一般人比较**：不只看「我常聊什么」，还要看「我和一般人有什么不同」。
- **I4 普通统计**：什么时候用、怎么用、哪些爱好是长期的、哪些会反复出现。
- **I5 结论可复现、可回溯**：每个数字都能追到具体单元；所有实验都有报告。

## 2. 范围

**要做**：episode 和 is_paste；读者模型和 SAE 的选型实验；全量 SAE 编码；dense 话题和复发统计；特征自动起名；基准频率与特有特征；L6 的全部分析报告；按特征搜索。
**不做**：推荐（P4）、交互式可视化（P5）。本阶段的报告是静态 Markdown 或 HTML，可以附带 matplotlib 生成的 PNG。

## 3. 数据模型（迁移 `user_version` 4）

```sql
ALTER TABLE turn ADD COLUMN is_paste INTEGER;            -- 1 = 主要是粘贴进来的别人的文字
CREATE TABLE IF NOT EXISTS episode (
    id TEXT PRIMARY KEY, start_ts TEXT, end_ts TEXT, n_turns INTEGER
);
CREATE TABLE IF NOT EXISTS episode_turn (episode_id TEXT, turn_id TEXT, PRIMARY KEY (episode_id, turn_id));

-- 可解释空间
CREATE TABLE IF NOT EXISTS sae_run (
    id TEXT PRIMARY KEY,                 -- 例如 q35-9b-base_L16_w64k_k50
    reader TEXT, layer INTEGER, sae_repo TEXT, width INTEGER, topk INTEGER,
    precision TEXT,                      -- bf16 | bf16-offload | int8
    unit_kind TEXT,                      -- turn_self | turn_answer | baseline
    created_at TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS sae_act (
    run_id TEXT, unit_id TEXT, feature INTEGER,
    act REAL,                            -- 原始 max 池化激活
    act_z REAL,                          -- 按长度桶标准化的值
    argmax_tok INTEGER,                  -- 取到最大值的 token 位置（第 0 个 token 已排除）
    PRIMARY KEY (run_id, unit_id, feature)
);
CREATE INDEX IF NOT EXISTS sae_act_feature ON sae_act(run_id, feature);
CREATE TABLE IF NOT EXISTS sae_unit (                    -- 每个单元的元数据
    run_id TEXT, unit_id TEXT, n_tokens INTEGER, n_chunks INTEGER, fvu REAL,
    PRIMARY KEY (run_id, unit_id)
);
CREATE TABLE IF NOT EXISTS feature (
    run_id TEXT, feature INTEGER,
    freq_self REAL,                      -- 在我的单元中激活的比例
    freq_base REAL,                      -- 在基准语料中激活的比例
    enrich_logodds REAL, enrich_z REAL,  -- 特有程度
    label TEXT, label_note TEXT,
    detect_score REAL,                   -- 自动起名的检测打分（0–1）
    label_model TEXT,
    PRIMARY KEY (run_id, feature)
);
CREATE TABLE IF NOT EXISTS baseline_unit (               -- 基准语料（只存在服务器上；这里只记 id、长度和来源，不存原文）
    id TEXT PRIMARY KEY, source TEXT, n_tokens INTEGER
);

-- 话题
CREATE TABLE IF NOT EXISTS topic (
    id TEXT PRIMARY KEY,                 -- r<resolution>_<community>
    resolution REAL, parent_id TEXT, name TEXT, size INTEGER,
    first_ts TEXT, last_ts TEXT, n_recur INTEGER, active_days INTEGER
);
CREATE TABLE IF NOT EXISTS topic_member (topic_id TEXT, conv_id TEXT, weight REAL, PRIMARY KEY (topic_id, conv_id));
```

`sae_act` 每个单元只存非零的前 128 个特征。1.5 万个单元 × 128 ≈ 200 万行，每个 run 约 100MB。只保留选定的主 run 和基准 run，实验用的 run 跑完就删。

## 4. 模块

```
latentself/
├── units.py              # episode、is_paste、单元文本（turn.self / turn.answer）
├── interp/
│   ├── reader.py         # 服务器端：加载读者模型 + SAE，批量编码，返回稀疏激活
│   ├── sae_io.py         # 读取 Qwen-Scope 的 layer{n}.sae.pt；encode(resid) -> topk
│   ├── experiment.py     # 选模型、选层实验 → docs/P3-LAYER-EVAL.md
│   ├── encode.py         # Mac 端：调度全量编码，写入 sae_act
│   ├── baseline.py       # 基准语料的采样和编码
│   ├── label.py          # 自动起名 + 检测打分
│   └── dashboard.py      # 生成特征看板（HTML）
├── topics.py             # kNN 图 + Leiden，多分辨率，复发统计
└── analysis/
    ├── stats.py          # 基础统计
    ├── diff.py           # 两组单元的特征差异
    ├── corr.py           # NPMI 相关性
    ├── profile.py        # 画像
    └── report.py         # 汇总成报告
```

### 4.1 服务器端的读者服务
- `interp/reader.py` 以 HTTP 服务形式运行在服务器上（`127.0.0.1:8003`，经 SSH 隧道访问）。接口：`POST /encode {texts:[...], layers:[...]}` → 每段文本、每一层返回 `{n_tokens, features:[[id, act, argmax_tok], ...top128], fvu}`。
- 文本在请求里传递，**服务器上不落盘**；基准语料是例外，它本来就是公开数据。
- 实现细节：
  - 用 transformers 加载 bf16 模型；在 `model.model.layers[n]` 的输出上挂 hook，取残差流（先对照 Qwen-Scope 模型卡里的示例代码确认 hook 的位置）。
  - **SAE 编码前先核对模型卡的公式**：卡上的示例没有 ReLU，也没有减 `b_dec`。实现时要按论文核对 TopK SAE 的标准形式，并在 100 个样本上比较两种写法的 FVU，选重构更好的那种，并在报告里记录。
  - 排除第 0 个 token；输入长度超过 1,024 token 时切块（相邻块重叠 64 token），各块取 max 后再取 max。
  - 用 Qwen 的 tokenizer 统计 token 数。
- Qwen3.5 是比较新的架构：要确认 transformers 的版本支持它，并在报告里记录版本号。

### 4.2 单元（`units.py`）
- `turn.self` 的文本：提问原文，加上每个附件的 `[图片] caption`。不含回答。
- **is_paste 判定**（规则在样本上核对后再定）：满足任意一条就算粘贴：
  - 长度 > 1,500 字，且包含多段标题、列表、代码块或引用格式；
  - 出现明显的转载标记（例如「作者」「来源」「赞同」「原文链接」之类）；
  - 英文占比 > 80%，且长度 > 800 字。

  实现时抽 100 条已判定的样本，让 owner 抽查 20 条。抽查准确率 ≥ 90% 才算通过，否则调整规则后重抽。
- **episode**：所有对话类轮次按时间排序，相邻两轮间隔超过 30 分钟就切一段。

### 4.3 选型实验（`interp/experiment.py`）
按 PIPELINE.md L4b 第 1–2 步执行（9B 对比 27B，27B 试 offload 和 int8 两种跑法），样本是固定随机种子抽取的 2,000 个 `turn.self`。

每个（模型 × 跑法 × 层）记录：
- FVU；
- 把重构结果代回模型后，下一个 token 的交叉熵增加了多少（ΔCE）；
- 死特征比例（在 2,000 个样本里从未激活的特征占比）；
- 平均 L0；
- 检测分数：每层随机取 50 个活跃特征，用 §4.6 的方法起名并打分；
- 耗时。

决策规则见 PIPELINE.md L4b。报告 `docs/P3-LAYER-EVAL.md` 只放汇总表，不放样本原文和特征样例。

### 4.4 全量编码（`interp/encode.py`）
- 用选定的 run，编码所有 `turn.self`（约 1.5 万条）。可选：另开一个 run，编码 `turn.answer`。
- 断点续跑：已有 `(run_id, unit_id)` 的就跳过；文本变了（按 hash 判断）就重算。
- 编码完成后计算 `act_z`：按 `n_tokens` 分成 10 个等频桶，在每个桶内对每个特征做 z-score。

### 4.5 基准语料（`interp/baseline.py`）
- 来源：WildChat-1M 或 LMSYS-Chat-1M 中**语言为中文的用户第一轮提问**，用固定随机种子抽 15,000 条。实现者要确认数据集的许可和 gated 状态：如果需要在 HF 上点「同意条款」，**停下来请 owner 操作**，不要代为同意。
- 预处理和 `turn.self` 保持一致：同样的长度截断，同样的 is_paste 规则（粘贴类样本同样剔除或降权）。
- 编码用同一个 run 配置，`unit_kind='baseline'`。原文只放在服务器上；`ls.db` 里只记 id 和长度。
- 特有程度的计算：对每个特征，
  - `freq_self` = 我的非粘贴单元中 `act > 0` 的比例；
  - `freq_base` 同理，基于基准语料；
  - `enrich_logodds` = 加 0.5 平滑后的 log-odds 差；
  - `enrich_z` 用两比例 z 检验得到。

  频率太低的特征（在我的数据里激活少于 5 次）只记录，不参与排名。

### 4.6 自动起名（`interp/label.py`）
- 对象：在我的数据里激活 ≥ 5 次的特征。
- 每个特征的材料：
  - 10 个最强激活片段（以 argmax token 为中心，前后各 40 字，被点亮的字用 `«»` 标出）；
  - 10 个随机激活片段；
  - 10 个不激活的片段（随机抽取）。
- 本地 LLM（Qwen3.8-27B，关闭 thinking）输出 `{label: "≤12 字", note: "一句话说明"}`。
- **检测打分**：另取 10 个激活片段和 10 个不激活片段，打乱顺序后连同 label 一起交给 LLM，让它判断每个片段是否激活。`detect_score` = 平衡准确率。
- 不设硬阈值来删除特征。界面上，`detect_score < 0.65` 的特征标为「难以命名」，并按特有程度单独列一个榜单。
- 只用我的数据做起名材料。论文的建议是：在和目标分布相似的数据上重新起名，效果更好。

### 4.7 话题（`topics.py`）
- 输入：对话级向量。由各轮的 `turn_self` 向量按「提问长度的平方根」加权求均值；粘贴轮次的权重乘以 0.3。
- 建图：kNN（k=15，余弦相似度），只保留互为近邻的边，边权为余弦相似度。
- 聚类：Leiden（`leidenalg`，RBConfiguration），分辨率扫描 `[0.2, 0.5, 1.0, 2.0]`，得到四层。用 Jaccard 重叠找出每个社区在上一层的父社区（重叠最大者），生成层级树。
- 话题名：每个话题取离中心最近的 8 个对话的标题（来自 P2，没有的话就用首轮提问），交给 LLM 生成一个 ≤8 字的名字。
- 复发统计：对每个话题，取成员对话的所有活跃日期，间隔超过 14 天算一次复发。记录 `n_recur`、`active_days`、`first_ts`、`last_ts`。
- **特征聚类**（对照用）：把对话级 SAE 向量（各轮取 max）二值化，计算 Jaccard 矩阵，做谱聚类（簇数取 dense 分辨率 1.0 那一层的话题数）。输出两种聚类的对照表：调整兰德指数（ARI），以及每个 SAE 簇里占比最高的 dense 话题。

### 4.8 分析（`analysis/`）
每个分析都是一个函数，接收「单元集合或时间范围」作为参数，返回数据结构；`report.py` 负责把它渲染成 Markdown 和 PNG。

| 函数 | 输出 |
|---|---|
| `stats.overview()` | 每月、每星期几、每小时的提问量（按当时所在地的时区计算；时区随时间变化的区间表写在配置里 `LATENTSELF_TZ_PERIODS`）；提问长度分布；语言占比；附件类型；对话轮数分布；episode 时长分布 |
| `topics.timeline(level)` | 话题 × 月的热力表；复发次数排名；长期话题和一阵子的话题 |
| `diff.compare(A, B)` | 两组单元的特征频率差：log-odds、z 值、Benjamini–Hochberg 校正后的 q 值，取前 30 个，每个附上 3 条样例 |
| `corr.npmi(min_count=10)` | 特征两两之间的 NPMI，取前 50 对，过滤掉 label 的 embedding 相似度 > 0.8 的组合（太显然的） |
| `profile.build(half_life_days=180)` | 特征权重 = Σ act_z × 时间衰减 × (1 − 0.7·is_paste)，再乘以 max(0, enrich_z)；输出前 50 个，每个附上证据轮次 |
| `profile.unnamed()` | 「难以命名，但我特有」的特征榜单 |

预置的对比组（`lself analyze diff` 的预设）：
- 按时间：某个人生节点之前和之后，切分日期写在配置里，不硬编码；
- 深夜（00:00–05:00，按当时所在地时区）和白天；
- 带图片的轮次和纯文字轮次；
- 长对话（≥10 轮）和短对话。

### 4.9 特征看板（`interp/dashboard.py`）
- 一个静态 HTML 站点：`$LATENTSELF_OUT/features/index.html`，加上每个特征一页。
- 索引页：三个榜单（特有程度、频率、难以命名），可以按 label 搜索。
- 特征页包括：
  - label、note、detect_score；
  - freq_self 和 freq_base；
  - 激活直方图；
  - 按周的时间序列；
  - 前 20 个激活样本：逐 token 高亮。高亮需要现场重算 token 级激活，所以生成看板时要调用服务器的 `/encode?per_token=1`；
  - 共现最多的 10 个特征；
  - decoder 方向余弦最近的 10 个特征；
  - 指向 Gemini 原对话的链接。
- 页面样式遵循 `~/code/AGENTS.md` 里的前端审美要求。这只是本地文件，不发布。

### 4.10 按特征搜索
`lself search --feature <id>`：列出激活了该特征的全部轮次，按 act 排序，输出格式和 P1 一致。

## 5. CLI

```
lself units      [--recompute-paste] [--sample-paste 100]
lself interp     experiment | encode [--answers] | baseline | label [--limit N]
lself topics     build | name | report
lself analyze    overview | timeline | diff <preset|A..B> | corr | profile | unnamed | all
lself features   build-dashboard
lself search     --feature <id>
```

## 6. 测试
- 离线：用随机初始化的小 SAE（维度 64，特征数 256，k=8）和一个 FakeReader（输出固定的残差向量）。
- 覆盖：TopK 编码和 max 池化（排除第 0 个 token）；切块与重叠后 max 的合并；`act_z` 分桶；log-odds 和 z 检验的数值；BH 校正；NPMI；Leiden 建图（小型合成图，社区结构已知）；复发统计；is_paste 规则；episode 切分。

## 7. 验收
1. `docs/P3-LAYER-EVAL.md` 完成，选出读者模型和主层，选择理由符合决策规则。
2. 全量编码完成：覆盖所有 `turn.self`，失败率 < 1%。
3. 基准语料编码完成，特有程度榜单生成。
4. 自动起名：所有活跃特征都有 label 和 detect_score。报告里给出 detect_score 的分布。
5. 话题四层，加上复发统计。dense 和 SAE 两种聚类的对照表生成。
6. `lself analyze all` 在 `$LATENTSELF_OUT/analysis/<日期>/` 下生成全部报告。
7. 特征看板可以在浏览器里打开；随机抽 5 个特征页，确认高亮的位置和 argmax 一致。
8. owner 走读：实现 agent 列出特有程度前 30 名和「难以命名」前 20 名的看板链接，然后**停下来等 owner 反馈**。根据反馈调整起名的 prompt 或排名公式，最多迭代 2 轮。
9. 服务器上的基准原文保留，其他临时文件清理掉；显存恢复原状。不往 git 提交任何个人数据、特征样例或分析报告。
