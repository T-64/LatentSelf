# P4 Spec：个人推荐（先做电影）

> 状态：已确认，可以开工（2026-11-27）。前置：P1（dense 向量）、P3（SAE run、画像、特征起名）。
> 先读 PIPELINE.md §L7「推荐」，再读本文件。通用约定见 P1-SPEC §5.1、§6、§8。

## 1. Intent

- **I1 「我接下来该看什么电影」**：从我的对话里推断口味，给出一份短而准的片单。
- **I2 有理由**：每部推荐都说清「它点亮了我的哪些方向」，并给出我说过的原话作为证据。
- **I3 会学习**：我说「看过」「喜欢」「不喜欢」之后，下一次推荐会跟着变。
- **I4 框架通用**：同一套流程以后换个候选库，就能推荐书和音乐。

## 2. 范围

**要做**：候选库导入和补全、看过的识别、物品表示（dense + SAE）、打分、多样性约束、理由生成、反馈、回测、CLI。
**不做**：书和音乐的候选库（只要接口能复用）；Web UI；把推荐结果推送到 Curator（以后可以对接）。

## 3. 数据模型（迁移 `user_version` 5）

```sql
CREATE TABLE IF NOT EXISTS item (
    id TEXT PRIMARY KEY,                 -- film:<imdb_id> 或 film:<canonicalKey>
    domain TEXT NOT NULL,                -- film | book | music
    title TEXT, title_zh TEXT, original_title TEXT,
    year INTEGER, creators TEXT,         -- JSON 数组（导演）
    countries TEXT, genres TEXT, keywords TEXT,   -- JSON
    overview TEXT, overview_zh TEXT,
    runtime INTEGER, imdb_id TEXT, tmdb_id INTEGER,
    sources TEXT,                        -- JSON：[{"src":"tspdt-top1000","rank":12}, ...]
    meta_hash TEXT
);
CREATE TABLE IF NOT EXISTS item_alias (item_id TEXT, alias TEXT, PRIMARY KEY (item_id, alias));  -- 中英文名、常见简称
CREATE TABLE IF NOT EXISTS item_state (
    item_id TEXT PRIMARY KEY,
    status TEXT,                         -- watched | wanted | dismissed | unknown
    rating INTEGER,                      -- -1 / 0 / 1（不喜欢 / 一般 / 喜欢）
    source TEXT,                         -- owner_list | chat_mention | feedback
    evidence TEXT,                       -- JSON：[turn_id, ...]
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS rec_run (
    id TEXT PRIMARY KEY, created_at TEXT, params TEXT, profile_run TEXT
);
CREATE TABLE IF NOT EXISTS rec_item (
    run_id TEXT, item_id TEXT, rank INTEGER, score REAL,
    score_dense REAL, score_feat REAL, score_recent REAL,
    reasons TEXT,                        -- JSON：[{feature, label, contribution, evidence:[turn_id,...]}]
    PRIMARY KEY (run_id, item_id)
);
```

物品的向量复用 `embedding` 表（`item_kind='item'`）；物品的 SAE 激活复用 `sae_act` 表（`unit_id='item:<id>'`，run 与画像用的主 run 相同）。

## 4. 流程

### 4.1 候选库
- 来源（路径用环境变量 `LATENTSELF_CATALOG_SRC` 指定，不写死）：
  - Curator 仓库导出的 `data/imports/tspdt-2026.json`：`top1000` 共 1,000 部，`recommendations` 共 3,088 部，字段有 title、creator、year、country、minutes、imdb、canonicalKey；
  - owner 的私人片单 `movies-final-2026.json`：既用作候选，也用来识别「看过 / 想看」，字段要在实现时先读取核对。
- 去重：优先按 `imdb_id`，其次按 `canonicalKey`。
- 补全：用 TMDB API 按 imdb_id 查询，取简介（中文和英文）、类型、关键词、片长、原名、中文名。
  - API key 从环境变量 `TMDB_API_KEY` 读取；**没有就停下来请 owner 提供**，不要去别的项目里找。
  - 发给 TMDB 的只有公开的影片 ID，不含任何个人数据。
  - 有本地缓存，同一部片不重复请求；遵守速率限制。
- 别名：中文名、英文名、原名、去掉标点和冠词的形式，写进 `item_alias`。

### 4.2 看过的识别
1. **owner 的片单**：按片单的字段映射成 watched / wanted，`source='owner_list'`。
2. **对话里的提及**：
   1. 用别名表在 `turn.prompt` 里做子串匹配。别名少于 3 个字的不参与匹配，避免误报。
   2. 匹配到的轮次交给本地 LLM 判断，输出 `{status: watched|wanted|mentioned_only, sentiment: -1|0|1}`。
   3. 只有 watched 和 wanted 才写进 `item_state`，`source='chat_mention'`，证据记下 turn_id。
3. 冲突处理：`feedback` > `owner_list` > `chat_mention`，以优先级高的为准。

### 4.3 物品表示
- 物品文本模板（中文优先；没有中文时用英文）：
  ```
  《{title_zh or title}》（{original_title}，{year}，{countries}）导演：{creators}
  类型：{genres}；关键词：{keywords}
  简介：{overview_zh or overview}
  ```
- 用 P1 选定的 embedding 模型（文档模式，不加查询前缀）算 dense 向量。
- 用 P3 选定的 SAE run 编码，得到物品的特征激活。

### 4.4 打分
对每个状态为 unknown 或 wanted 的物品：

```
score = α · score_dense + β · score_feat + γ · score_recent + δ · prior
score_dense  = cos(item_vec, profile_center)，其中 profile_center 是我所有 turn_self 向量按画像时间衰减加权的均值（只用和文化、娱乐相关话题的轮次；这个话题子集由 P3 话题的名字和成员决定，做成配置项）
score_feat   = Σ_f profile[f] · act_z_item[f] / ‖profile‖  （只算画像前 200 个特征）
score_recent = 与最近 30 天的 turn_self 向量的最大余弦相似度
prior        = 来源先验：TSPDT 排名换算成 0–1；只在推荐列表里出现过的给 0.3
```

- **多样性约束（MMR）**：λ = 0.7，用物品 dense 向量之间的余弦相似度。同一位导演最多出现 2 部。
- **默认输出**：20 部，其中最多 4 部是 wanted（已经在我的想看清单里）。
- **权重**：初始值 α = 0.4，β = 0.4，γ = 0.1，δ = 0.1，最终由 §4.6 的回测来调。

### 4.5 理由
- 对每部推荐，取对 `score_feat` 贡献最大的 3 个特征。每个特征给出 label（「难以命名」的特征显示为「一个难以命名的方向」，并附上它的看板链接），再加上该特征在我数据里最强的 2 条原话（截取 60 字）。
- 本地 LLM 把以上材料写成一句中文推荐语（≤ 50 字）。prompt 里**禁止编造原话里没有的事实**，代码层面再校验推荐语里引号内的内容都来自材料。
- 输出：`$LATENTSELF_OUT/rec/<run_id>.md`，同时用 `--json` 输出结构化结果。

### 4.6 回测
- 正样本：`item_state` 里 `rating = 1` 或 status = watched，并且有对话证据的影片。
- 留出法：按证据时间排序，留出最后 20% 作为测试集。用证据时间之前的数据重算画像和 profile_center，再把这些影片放回候选池（视为 unknown），看它们的排名。
- 指标：Recall@20、Recall@50、nDCG@20、MRR。在留出集上用网格搜索调 α、β、γ、δ。报告写进 `docs/P4-BACKTEST.md`，只放汇总数字。
- 基线：①只用 `prior`（TSPDT 排名）；②只用 `score_dense`。新方法必须在 Recall@50 上超过两个基线，否则先报告，不要上线。

### 4.7 反馈
`lself rec feedback <item> like|dislike|watched|wanted|dismiss`：写入 `item_state`（`source='feedback'`），同时写入 P0 的 `signal` 表，`kind='rec_feedback'`。下一次 `lself rec` 时生效：dismissed 永远排除；对于 like 或 dislike 的物品，它点亮最多的前 10 个特征在画像里分别 ×1.2 或 ×0.8，上限 ±50%。

## 5. CLI

```
lself catalog import | enrich | aliases
lself rec detect-watched
lself rec [--n 20] [--explain] [--json]
lself rec feedback <item_id|标题> <like|dislike|watched|wanted|dismiss>
lself rec backtest
```

## 6. 测试
- 离线：用一个合成的小片库（10 部）、FakeEmbed、FakeLLM 和一份合成画像。
- 覆盖：去重；别名匹配（少于 3 个字的别名不参与）；状态优先级；打分公式；MMR 和同导演上限；理由的引号校验；回测的切分没有用到未来数据；反馈对画像权重的调整及上限。

## 7. 验收
1. 候选库导入完成，其中 ≥ 95% 的影片有简介。
2. 看过识别：随机抽 30 条 chat_mention 结果由 owner 核对，准确率 ≥ 90%（实现 agent 生成清单后**停下来等 owner 标完**）。
3. 回测报告完成，新方法在 Recall@50 上超过两个基线。
4. `lself rec --explain` 输出 20 部推荐，每部都有理由和可回溯的原话。owner 试用一轮反馈后再出一次推荐，确认结果有变化。
5. 不往 git 提交任何片单、状态、推荐结果或回测明细。
