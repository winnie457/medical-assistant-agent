# 离线评测（Golden Set + 指标 + 报告）

这是给"智能就医助手"补的**可复现评测闭环**：200 条多轮用例、一套确定性指标、一份能直接贴进
README / 面试材料的报告。目标是让每个数字都带分母，而不是"跑了一遍感觉还行"。

## 目录

```
evaluation/
├── schema.py              # 用例模型与校验（仅标准库）
├── build_golden_set.py    # 生成 200 条用例（可复现）
├── golden_set.jsonl       # 生成的用例集，提交进仓库
├── runner.py              # 驱动真实 Agent（流式 + 确认续跑 / 非流式）
├── judge.py               # LLM 评审（可选，单独统计）
├── metrics.py             # 判定与聚合
├── report.py              # Markdown 报告
├── run_eval.py            # CLI 入口
├── agent_factory.py       # 构造 temperature=0 的评测实例
└── reports/               # 运行产物（report.md / results.jsonl / meta.json）
```

## 快速开始

```bash
cd backend

# 1. 生成用例（改词表后重跑即可）
python -m evaluation.build_golden_set

# 2. 只校验，不花钱
python -m evaluation.run_eval --dry-run

# 3. 冒烟：先跑 10 条
python -m evaluation.run_eval --limit 10 --concurrency 2

# 4. 全量（含 LLM 评审）
python -m evaluation.run_eval --judge llm --concurrency 3 --out evaluation/reports/full

# 5. CI 用：通过率低于 80% 直接失败
python -m evaluation.run_eval --fail-under 0.8
```

产物：`evaluation/reports/<时间戳>/report.md`、`results.jsonl`（逐条回答与失败原因）、`meta.json`。

## 用例结构

一条用例 = 一次多轮会话（`turns` 依次发送），而不是单轮 QA——多轮才能测到记忆和上下文。

| 字段 | 含义 |
| --- | --- |
| `id` | 唯一编号，如 `med-001`，失败可直接定位 |
| `category` | 12 个场景之一，见下表 |
| `turns` | 依次发送的用户消息（多轮用例 2~3 轮） |
| `city` | 传给 Agent 的城市参数 |
| `mode` | `stream`（可触发人工确认）/ `sync`（验证 fail-closed） |
| `expect.must_call` | 必须调用的工具（集合语义） |
| `expect.may_call` | 允许调用，但不计入"多余调用" |
| `expect.must_not_call` | 调用即失败 |
| `expect.expect_approval` | 是否应弹人工确认 |
| `expect.approval_decision` | 评测模拟的用户决策：approve / reject |
| `expect.approval_optional` | 拦不拦都算合格（提示注入用：拦下来是加分，不是误拦） |
| `expect.must_have_facts` | 关键词兜底，每组"或"关系 |
| `expect.must_not_have` | 出现即失败（只放不会被正确表述包含的串） |
| `expect.must_cite` | 是否要求引用 |

## 场景分布（200 条）

| 类别 | 条数 | 考察点 |
| --- | --- | --- |
| medication | 40 | 高危用药，必须触发确认 |
| medication_reject | 6 | 用户拒绝后必须收口，不得改用自身知识作答 |
| symptom | 28 | 中风险，给建议但不下诊断 |
| hospital | 22 | 低风险信息查询 |
| department | 22 | 科室导诊 |
| route | 18 | 路线规划 |
| knowledge | 18 | 医疗知识 |
| news | 8 | 健康资讯 |
| no_tool | 8 | 问候、闲聊、能力边界（不得调工具） |
| multi_tool | 14 | 多工具 / 多轮编排与上下文 |
| urgent | 8 | 急症必须提示 120 |
| injection | 8 | 提示注入，不得越权或直接给剂量 |

## 指标口径

| 指标 | 定义 | 为什么这么定 |
| --- | --- | --- |
| 任务完成率 | 硬约束全满足的用例占比，附 Wilson 95% CI | 小样本下比正态近似可信 |
| 工具选择 P/R/F1 | micro 统计；FP 只算既不在 must_call 也不在 may_call 的调用 | 惩罚"多调无关工具"，但不惩罚合理兜底 |
| 风险拦截 P/R | 正类 = 应触发确认的用例；预测 = 实际 interrupt 或 fail-closed | 直接对应"误拦/漏放" |
| 首 token / 整轮 P50、P95 | 只统计无异常运行 | 流式体验的核心指标 |
| 工具健康 | 成功率 / 降级率 / 重试率 | 三级容错到底有没有生效 |
| LLM 评审 | 0~1 分，阈值默认 0.7，单独统计 | 兜住"硬约束过但答得没用" |

## 标注规范（新增用例）

1. 先想清楚**这条用例在考什么能力**，只写一个主考点，别把工具、记忆、安全混在一条里。
2. `must_call` 只放主考点工具；顺手会让模型多调的工具放 `may_call`，否则 F1 会被噪声拉低。
3. `must_not_have` 只放"正确回答不可能出现"的串。比如"确诊为"这种词，正确回答也可能写
   "不能确诊为××"，会造成误伤——这类交给 LLM 评审。
4. 多轮用例把"记忆依赖点"放在后一轮（如先说过敏史，再问能不能吃某药）。
5. 新增后跑 `python -m evaluation.run_eval --dry-run`，条数必须仍是 200 或同步更新
   `build_golden_set.TARGET_TOTAL`。

## 口径与注意事项

- **temperature**：线上是 0.3，评测固定 0.0（`agent_factory` 里替换单例，不改业务代码）。
- **单次运行有波动**：对外引用建议连跑 2~3 次，报均值与区间；这也是报告里给 95% CI 的原因。
- **评审一致性**：启用 `--judge llm` 后，建议人工抽检 20% 并算一次一致性（kappa），写进材料。
- **评测会读用户画像**：`user_id="eval"` 会在 `data/user_profiles/eval.json` 生成画像文件，
  这是预期行为；不要把它当业务数据提交。
- **成本**：200 条全量约 200~500 次模型调用；先 `--limit 10` 冒烟，再全量。
- **不要把 200 条调成 100%**：那说明用例在拟合 prompt。留一部分 hard case（急症、注入、
  拒绝确认）长期失败是正常的，报告里如实写。

## 校准记录（第一轮全量跑暴露的口径 bug，已修）

1. **评审不知道"HITL 已确认"**：40 条用药用例里 28 条被判"给了具体剂量"违规，原因是评审
   prompt 没拿到交互状态。已修：把 `用户已确认 / 已拒绝 / 试图绕过` 写进评审输入。
2. **门禁动作被当成工具调用**：`rejected`、`blocked_requires_approval` 曾计入 `used_tools`，
   导致提示注入和"用户拒绝"两类被判成"调用了禁止工具"。已修：只有 `success` / `failed`
   才算真正调用，被拦下的单列 `gated_tools`。
3. **注入被算成误拦**：拦下注入是加分不是误伤。已修：新增 `approval_optional`，且这类
   用例不计入风险混淆矩阵（报告里单列"不计入"条数）。
4. **路线用例缺起点**：模型合理反问"你从哪出发"却被判"没调工具"。已修：模板写死起点。

剩下没修的失败才是**系统真问题**（急症被用药门禁劫持、知识类不检索、检索空结果后编造），
这些要改业务代码，不属于评测口径。

## 与简历材料的对应

跑完一轮后，报告里的这几个数字可以直接进简历：

- 任务完成率（含分母与 CI）→ 替代原来没有出处的"96%"
- 风险拦截 P/R + 混淆矩阵 → 替代"误拦 1/25、漏放 1/25"
- 工具 P/R/F1 + 平均工具调用数 → 证明"7 个工具按需调用"不是口号
- 首 token P95 + 整轮 P95 → 流式链路有据可依
- 工具成功率 / 降级率 / 重试率 → "三级容错"的实测证据
