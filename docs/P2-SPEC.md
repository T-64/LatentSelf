# P2 Spec：标题、摘要、事实层与 MCP

> 状态：已确认，可以开工（2026-11-27）。前置：P1 完成（`search.py`、`backends/`、`embedding` 表可用）。
> 先读 DESIGN.md §1、§4.3，PIPELINE.md §L7，再读本文件。通用约定见 P1-SPEC §5.1（backends）、§6（服务器）、§8（测试）。

## 1. Intent

- **I1 复盘和写作要有材料**：「某段经历从开始到现在」这样的小书，AI 要能按时间找到发生过什么，而且每一句都能回到原话。
- **I2 我的 Fact**：问「我什么时候开始学某样东西」「我现在住哪」，能直接回答，并给出处。
- **I3 让 agent 能用**：Claude Code 或本地 agent 通过 MCP 调用检索、时间线、事实，不用把全部数据塞进上下文。

## 2. 范围

**要做**：对话标题、对话摘要、月 / 季 / 年三级摘要、事实层（抽取、合并、核查）、MCP 服务、`lself write` 的最小流程。
**不做**：SAE 和话题（P3）、推荐（P4）、Web UI。

## 3. 数据模型（迁移 `user_version` 3）

```sql
-- conversation.title 已存在（P0）
CREATE TABLE IF NOT EXISTS conv_summary (
    conv_id    TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,          -- 1–3 句，第三人称「用户……」
    kinds      TEXT,                   -- JSON：["健康","学习",...]，从固定词表中选
    model      TEXT NOT NULL,
    input_hash TEXT NOT NULL,          -- 对话内容变了就重算
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS digest (
    period     TEXT PRIMARY KEY,       -- 2026-03 | 2026-Q1 | 2026
    level      TEXT NOT NULL,          -- month | quarter | year
    body_md    TEXT NOT NULL,          -- Markdown，每条要点末尾带 [c:<conv_id>] 或 [t:<turn_id>]
    model      TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact (
    id         TEXT PRIMARY KEY,       -- sha1(normalized text)[:16]
    text       TEXT NOT NULL,          -- 一句话，第三人称
    kind       TEXT NOT NULL,          -- event | state | preference | plan | possession | relation | identity
    date_from  TEXT,                   -- YYYY-MM-DD 或 YYYY-MM；event 填发生日
    date_to    TEXT,                   -- state 的结束日（被新事实取代时填写）
    confidence INTEGER NOT NULL,       -- 3 原话明确 | 2 单一来源明确 | 1 推断
    status     TEXT NOT NULL,          -- active | superseded | rejected
    superseded_by TEXT,
    verified   INTEGER NOT NULL DEFAULT 0,  -- 核查通过 = 1
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_evidence (
    fact_id  TEXT NOT NULL REFERENCES fact(id) ON DELETE CASCADE,
    turn_id  TEXT NOT NULL,
    quote    TEXT NOT NULL,            -- 原文片段（≤120 字），必须是 turn.prompt 的子串
    PRIMARY KEY (fact_id, turn_id)
);
```

**固定的话题词表**（`kinds`，用于粗分，P3 的话题会更细）：健康、身体与运动、饮食、学习、语言、工作与求职、钱与消费、居住与生活、出行旅行、人际与情感、文化娱乐、技术与工具、社会与思想、其他。

## 4. 流程

### 4.1 对话标题和摘要
- 输入：对话的全部 `turn.self`（提问和附件描述），按时间排列。总长超过 6,000 字时，取首尾各 2,000 字，加上中间均匀抽样的若干轮。
- 本地 LLM 一次请求输出 JSON：`{title: "≤15 字", summary: "1–3 句", kinds: [...]}`。
- 只基于用户说的话来写摘要；Gemini 的回答只在必要时作为上下文，不写成用户的观点。
- 增量：按 `input_hash`；新快照里对话有新轮次时重算。
- 预计：约 2,600 个对话，每个约 3–5 秒，总共 2–4 小时。

### 4.2 三级摘要
- **月**：把该月所有对话摘要和标题（加上每个对话的轮数、日期）交给 LLM，按固定的小节输出：
  ```
  ## 概况（2–3 句）
  ## 事件（带日期，按时间排列）
  ## 进行中的事（学习、项目、计划）
  ## 健康与身体
  ## 兴趣与消费
  ## 想法与情绪
  ```
  每条要点必须带至少一个 `[c:<conv_id>]`。超过上下文长度时，先按周分组摘要，再合并。
- **季、年**：由下一级的摘要生成，引用直接沿用下一级的引用。
- **校验**（代码，不用 LLM）：每个 `[c:…]` 必须存在于该时间段内；去掉不合法的引用，并记录数量。
- 导出：`$LATENTSELF_OUT/digests/<level>/<period>.md`。

### 4.3 事实层
1. **抽取**：以对话为单位，输入对话的 `turn.self`（带 turn_id），LLM 输出候选事实：`[{text, kind, date_from?, date_to?, confidence, evidence:[{turn_id, quote}]}]`。规则写在 prompt 里：
   - 只抽**关于用户本人**的事实，不抽常识，也不抽 Gemini 说的话；
   - `quote` 必须逐字摘自 evidence 里那一轮的提问；
   - 日期：优先取原话里的日期，其次取该轮的时间戳；
   - 推断出来的事实，confidence 设为 1。
2. **代码校验**：`quote` 必须是 `turn.prompt` 的子串（允许空白差异），不满足就丢弃，并记录数量。
3. **合并**：对新事实的 text 算 embedding（复用 P1 的 embedding 后端），和已有事实比较余弦相似度：
   - ≥ 0.92 且 kind 相同：视为同一事实，合并 evidence，confidence 取最大值；
   - 0.80–0.92：交给 LLM 判断是「同一事实」「新状态取代旧状态」还是「不同事实」。如果是取代（例如换了住处），旧事实设为 `status='superseded'`，填上 `date_to` 和 `superseded_by`。
4. **核查**：对每条 active 事实，把它和它的 evidence 原文交给 LLM，问「原文是否支持这条事实」。支持则 `verified=1`；不支持则 `status='rejected'`，保留记录，不删除。
5. 导出：`$LATENTSELF_OUT/facts.md`，按 kind 分节，每条形如 `事实（日期，★★★）[t:…]`。

### 4.4 MCP 服务 `latentself-mcp`
- 运行方式：stdio，Python 官方 MCP SDK。**只读**，只在本机运行，不监听网络端口。
- 工具（输入和输出都是 JSON，结果都带 `turn_id` 或 `conv_id` 以及 Gemini 链接）：

| 工具 | 输入 | 输出 |
|---|---|---|
| `search` | `query, mode=fast|strict, since?, until?, images_only?, limit?` | P1 的 `--json` 结果 |
| `show_conversation` | `conv_id, full=false` | 按顺序排列的轮次：提问、回答（可截断）、附件描述和路径 |
| `timeline` | `since, until, kinds?` | 对话标题、摘要、日期，按时间排列 |
| `digest` | `period` | 摘要的 Markdown |
| `facts` | `query?, kind?, active_only=true` | 事实和证据 |
| `attachment` | `attachment_id` | 元数据、描述、本地路径（不返回文件内容） |
| `stats` | `metric`（P3 之后可用） | 统计结果 |

- 在 README 里给出 Claude Code 的 `.mcp.json` 示例，指向 `lself mcp`。
- 每个工具的返回值都有长度上限（默认 20,000 字符），超出时截断，并在返回值里标明 `truncated: true` 和被截掉的条数。

### 4.5 `lself write <主题>`（最小流程）
1. 用 `search` 和 `facts` 收集材料，再用 `timeline` 生成该主题的时间线，写到 `$LATENTSELF_OUT/write/<slug>/timeline.md`。
2. 生成大纲 `outline.md`，然后**停下来**，等 owner 编辑或确认。
3. `lself write <主题> --continue`：逐章写作，每段末尾都带引用；写完由一个核查步骤逐条验证引用（同 4.3 第 4 步），不支持的句子标为 `[待核实]`。
4. 默认用本地 LLM。`--llm claude` 必须显式传入，而且每次运行前要在终端里确认（y/N），提示「材料会发给 Anthropic」。

## 5. CLI

```
lself titles   [--limit N]
lself digest   [--level month|quarter|year] [--period 2026-03]
lself facts    extract|merge|verify|export
lself mcp                      # 启动 MCP stdio 服务
lself write <主题> [--continue] [--llm local|claude]
```

## 6. 测试
- 离线，只用 FakeLLM 和 FakeEmbed。
- quote 子串校验；非法引用被剔除；事实合并的三种情况（同一事实、取代、不同事实）；`superseded` 链条；摘要的增量重算；MCP 各工具的输入输出 schema 和截断。

## 7. 验收
1. 所有对话都有标题和摘要，所有月份都有摘要；季度和年度摘要齐全。
2. 摘要里的引用 100% 合法（代码校验）。
3. 事实：随机抽 50 条 active 且 verified 的事实，由 owner 判断，正确率 ≥ 90%（实现 agent 生成核对清单后**停下来等 owner 标完**）。
4. 在 Claude Code 里通过 MCP 完成一次「某段经历的时间线」任务，输出里的每一条都带可点击的出处。
5. 不往 git 提交任何摘要、事实或评测结果。
