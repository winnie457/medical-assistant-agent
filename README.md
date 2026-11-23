# 智能就医助手 · Medical Assistant Agent

> 基于 LangGraph 的 ReAct 医疗工具调用 Agent：**三层护栏 · 有状态 · 可离线评测**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)
![Vue](https://img.shields.io/badge/Vue-3.5-42B883?logo=vue.js&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

面向普通用户的智能就医助手：用户描述症状或用药疑问，Agent 自主决定调用哪个医疗工具，并在高风险环节强制人工确认。

这个项目想解决三个具体问题：

1. **通用大模型会给出没有依据的医疗建议** → 出口做确定性校验，工具没返回的信息不允许写进答案；
2. **就医是多步骤任务**（症状 → 科室 → 医院 → 路线）→ 用 ReAct 状态图做工具编排；
3. **用药建议风险高**，不能让模型自主给出 → 三级风险分级 + 人工确认，且急症优先级高于用药门禁。

---

## 核心特性

| 能力 | 做法 | 效果 |
| --- | --- | --- |
| **工具编排** | LangGraph 手写 ReAct 状态图（`precheck → agent ⇄ approval → tools`），7 个医疗工具按需调用，单轮最多 10 次迭代 | 工具选择 micro **F1 0.906** |
| **风险拦截** | 按医疗后果分低/中/高三级；用药执行前 `interrupt` 挂起确认，拒绝即收口；非流式入口 fail-closed | 192 条可判定用例**零误拦、零漏放** |
| **急症优先** | 误服、卒中、抽搐、大出血、头孢配酒等规则先于用药判定，命中直接给 120 并跳过确认 | 8/8 急症用例走急救话术 |
| **幻觉治理** | 出口校验三类无依据断言（排名/号源、工具失败后的距离时长、缺医嘱的剂量），命中追加来源声明 | 200 条中命中 36 条并全部补正 |
| **容错** | Never-Throw 执行器：参数纠偏 → 指数退避重试 → 降级兜底 → 结构化错误回传模型 | 工具失败不中断会话 |
| **状态与恢复** | 会话、运行记录与 LangGraph checkpoint 落 SQLite，重启后按 `run_id` 续跑中断的确认 | 跨进程恢复 **5/5** |
| **可复现评测** | 200 条多轮 Golden Set、12 类场景、确定性指标 + Wilson 95% CI + LLM 评审 | 任务完成率 **90.5%** |

## 效果数据

评测集：200 条多轮用例、12 类场景；被测模型 DeepSeek-chat，temperature=0。评测方法与指标定义见 [`backend/evaluation/README.md`](backend/evaluation/README.md)；每次运行的完整报告（逐条回答、失败原因、混淆矩阵）输出到 `backend/evaluation/reports/<run>/`，该目录不入库。

| 指标 | 结果 | 口径 |
| --- | --- | --- |
| 任务完成率 | **90.5%**（181/200，95% CI 85.6%~93.8%） | 硬约束全部满足；三次运行 89.5% / 90.5% / 91.0% |
| 工具选择 | P 0.896 / R 0.915 / **F1 0.906** | micro 统计；被门禁拦下的不算调用 |
| 风险拦截 | TP 48 / FP 0 / FN 0 | 48 条应确认 + 144 条不应确认 |
| 输出护栏命中 | 36 / 200 | 命中即追加来源声明 |
| 首 token 延迟 | P50 634ms / P95 6.7s | 200 条评测集统计 |
| 并发（50 路） | 100% 成功，0 错误 | 单机进程内压测，吞吐 6.12 req/s |

复现：

```bash
cd backend
python -m evaluation.run_eval --judge llm --out evaluation/reports/round8   # 200 条全量评测
python -m evaluation.recovery_bench --cases 5                               # 跨进程中断恢复
python -m evaluation.concurrency_bench --concurrency 50                     # 并发压测
```

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (Vue3 + Vite)                    │
│              聊天界面 / 工具调用面板 / 风险确认弹窗            │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP (SSE 流式 / REST 非流式)
┌───────────────────────▼─────────────────────────────────────┐
│                     FastAPI 接入层                            │
│  /chat  /chat/stream  /approve  /reject  /runs/{run_id}      │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                MedicalAgent (LangGraph 状态图)                │
│                                                             │
│   ┌──────────┐   ┌───────┐   ┌──────────┐   ┌───────┐       │
│   │ precheck │──▶│ agent │⇄▶│ approval │──▶│ tools │──▶ END │
│   │急症+用药  │   │ (LLM) │   │(风险门禁) │   │(执行) │       │
│   └──────────┘   └───────┘   └──────────┘   └───────┘       │
│                                                             │
│   三层护栏：入口（急症 + 用药意图）/ 中段（三级风险 + 确认）    │
│             出口（事实性断言校验）                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼─────────────────┬──────────────────┐
        ▼               ▼                 ▼                  ▼
   7 个医疗工具      LLM(DeepSeek)    长期记忆(JSON)      会话/运行/中断态
  症状/医院/科室/    OpenAI 兼容        MemoryManager       SQLite
  知识/路线/用药/资讯                 + 冲突消解      (app.db + checkpoints.sqlite)
```

一次请求的完整链路：

1. 取会话历史（SQLite）与用户健康画像，登记 `run_id`；
2. `precheck`：**先判急症**（命中直接 120 并结束），再判用药意图（流式挂起确认 / 非流式 fail-closed）；
3. `agent`：状态校验 → 拼 Prompt → 流式调模型 → 输出 `thought / tool_call / answer_delta`；
4. `approval`：高风险工具挂起，用户确认后 `Command(resume)` 续跑；
5. `tools`：Never-Throw 执行器（纠偏 → 重试 → 降级）；
6. 循环直到无工具调用或达到迭代上限；
7. 出口护栏校验 → 推 `answer` + `done`（带 `guard_issues`）→ 落库、异步抽取记忆。

---

## 快速开始

### 1. 后端

```bash
cd backend

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
copy .env.example .env          # 填入模型与地图 API Key
python run.py                   # http://localhost:8000
```

依赖里已包含 `langgraph-checkpoint-sqlite` 与 `aiosqlite`。缺少这两个包时，Agent 会自动降级为进程内存储（重启后中断态丢失），启动日志会出现 `[checkpointer] SQLite 不可用 ...` 告警。

### 2. 前端

```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

`.env` 里配置 `VITE_API_BASE_URL`（默认指向 `http://localhost:8000`）。

### 3. 环境变量

| 变量 | 说明 | 必填 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 模型 Key（默认对接 DeepSeek 的 OpenAI 兼容端点） | ✅ |
| `OPENAI_BASE_URL` | 例如 `https://api.deepseek.com/v1` | ✅ |
| `OPENAI_MODEL` | 例如 `deepseek-chat` | ✅ |
| `AMAP_API_KEY` | 高德 Web 服务 Key，用于医院搜索与路线规划 | 建议 |
| `TAVILY_API_KEY` | 资讯检索（缺失时回落到搜索降级） | 可选 |
| `AGENT_MAX_ITERATIONS` | 单轮最大迭代次数，默认 10 | 可选 |
| `CORS_ORIGINS` | 允许的前端来源，逗号分隔 | 可选 |
| `APP_DB_PATH` | 会话库路径，默认 `backend/data/app.db` | 可选 |

---

## 项目结构

```
backend/
├── app/
│   ├── agents/
│   │   ├── medical_agent.py      # 状态图、precheck/approval/tools 节点、流式协议
│   │   ├── emergency.py          # 急症规则与急救话术
│   │   ├── output_guard.py       # 出口护栏：事实性断言校验
│   │   ├── checkpointer.py       # SQLite checkpointer（缺依赖时降级）
│   │   └── state_middleware.py   # 状态校验与自动修复
│   ├── tools/                    # 7 个医疗工具 + Never-Throw 执行器 + 风险分级
│   ├── memory/                   # 长期健康画像：提取、冲突消解
│   ├── services/                 # 会话（SQLite）、运行记录、LLM、记忆后台任务
│   ├── storage/db.py             # SQLite 连接与建表
│   └── api/                      # FastAPI 入口与路由
├── evaluation/                   # 200 条 Golden Set、指标、报告、基准脚本
├── tests/                        # 不依赖网络的回归测试
└── requirements.txt
frontend/
└── src/                          # Vue3 + TypeScript 聊天界面
```

---

## 设计要点

### 1. ReAct 状态图，而不是现成 Agent

用 `StateGraph` 显式建模「预检 → 决策 ⇄ 确认 → 执行」，而不是调 `create_react_agent`。原因是需要在标准 ReAct 循环里插入两个非标准节点：

- `precheck`：入口确定性判断（急症 / 用药），堵住"模型不调工具、直接凭自身知识作答"的绕过路径；
- `approval`：工具级风险门禁，模型真要调高风险工具时再拦一次。

流式与非流式共用同一张图，靠 `state["interactive"]` 区分是否有确认通道。

### 2. 三层护栏

| 层 | 位置 | 拦什么 |
| --- | --- | --- |
| 入口 | `precheck` | 急症（直接 120）、用药意图（挂起确认 / fail-closed） |
| 中段 | `approval` | 模型选择的高风险工具，`interrupt` 挂起等确认 |
| 出口 | `output_guard` | 工具没返回却写出的事实性断言，追加来源声明 |

急症判定放在用药门禁**之前**：像"误服了半瓶安眠药"这种问题本身带"药"字，如果先走用药门禁，用户只会看到"请确认是否查看用药信息"，而正确动作是立即拨打 120。

### 3. 状态与恢复

- `data/app.db`：会话、消息、运行记录（`running → interrupted → completed/failed`）；
- `data/checkpoints.sqlite`：LangGraph checkpoint，按 `thread_id = run_id` 存中断状态；
- `GET /api/chat/runs/{run_id}` 可跨进程查询运行状态与待确认内容。

LangGraph 1.x 的同步与异步 checkpointer 是两套接口，而流式链路本来就是异步，因此统一使用 `AsyncSqliteSaver`，`chat()` 也改为 async 走 `ainvoke`，保证两条链路共用同一份状态。

实测：进程 A 制造挂起后退出，进程 B 按 `run_id` 续跑，**5/5 成功**。

### 4. 记忆分层

短期记忆是会话消息窗口；长期记忆是用户健康画像（JSON 持久化），每轮对话结束后异步抽取健康信息，按「否定 > 置信度 > 时间」的规则做冲突消解，避免"我青霉素不过敏了"这类新信息被旧记忆覆盖，或反过来污染后续建议。

### 5. 评测体系

- **用例集**：200 条多轮用例、12 类场景（用药确认、拒绝收口、症状、医院、导诊、路线、知识、资讯、无需工具、多工具、急症、提示注入）；
- **三层期望**：硬约束（工具、确认行为、禁止表述）→ 弱约束（关键词兜底）→ 软约束（LLM 评审分）；
- **指标**：任务完成率（Wilson 95% CI）、工具 P/R/F1、风险混淆矩阵、首 token 与整轮 P50/P95、护栏命中率。

详见 [`backend/evaluation/README.md`](backend/evaluation/README.md)。

---

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 非流式对话（无确认通道，高风险 fail-closed） |
| POST | `/api/chat/stream` | 流式对话（SSE，可触发人工确认） |
| POST | `/api/chat/approve` | 确认高风险操作并续跑 |
| POST | `/api/chat/reject` | 拒绝对高风险操作，直接收口 |
| GET | `/api/chat/runs/{run_id}` | 查询运行状态与待确认内容 |
| GET | `/api/chat/conversations` | 会话列表 |
| GET | `/api/chat/profile` | 用户健康画像 |
| DELETE | `/api/chat/profile/memories/{memory_id}` | 删除单条记忆 |
| GET | `/health` | 健康检查 |

SSE 事件类型：`thought / tool_call / tool_result / answer_delta / answer_reset / approval_required / answer / done`。

---

## 已知局限

- **知识类问题检索不稳定**：18 条知识用例中有 9 条模型直接凭参数记忆作答，没调用知识库；下一步把"该不该调工具"从模型手里收回到路由层。
- **工具数据边界**：医院搜索依赖高德 POI，没有医院等级/医保定点字段；路线工具失败时会降级，答案只能标注"未获取实时数据"。
- **SQLite 单写者**：仅适合单实例部署，多实例需要换 Postgres/Redis。
- **未做鉴权与限流**：公网部署前必须补，否则 API Key 有被刷的风险。
- **checkpoint 无 TTL**：长期运行会持续增长，需要定期清理。

---

## 免责声明

本项目仅用于学习、研究与技术演示，**不构成任何医疗建议**，也不能替代医生的诊断与治疗。如出现身体不适，请及时就医；紧急情况请立即拨打 120。

## License

MIT。项目骨架参考 hello-agents 系列开源示例（MIT）二次开发，特此致谢；主要改造包括 ReAct 状态图重构、三层护栏、SQLite 状态持久化与 200 条评测体系。
