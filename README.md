# 🏥 医小助 - 智能就医助手

> 基于 LangGraph ReAct Agent 的智能就医助手，支持症状分析、医院搜索、科室导诊、用药咨询等 7+ 医疗工具。

## ✨ 项目特色

### 🔧 核心技术
- **ReAct Agent 架构** — 基于 LangGraph 实现，自主思考、工具调用、结果观察的完整循环
- **7+ 医疗工具** — 症状分析、医院搜索、科室导诊、医疗知识、路线规划、用药咨询、健康资讯
- **流式对话** — SSE 流式输出，实时展示思考过程和工具调用
- **多轮对话记忆** — 完整的会话管理，支持历史对话

### 🎯 为什么用 ReAct 而不是流水线？

| 对比项 | 流水线模式（旧） | ReAct 模式（新） |
|--------|-----------------|-----------------|
| 工具选择 | 固定规则，硬编码 | LLM 自主决策 |
| 灵活性 | 差，只能处理预设场景 | 强，可处理复杂问题 |
| 工具调用次数 | 固定 | 动态，按需调用 |
| 回答质量 | 中等（~70%） | 较高（~85%+） |
| 错误恢复 | 无，失败就结束 | 有，可重试和调整 |

详见 [评估结果](#-评估对比)。

## 🏗️ 项目架构

```
┌──────────────────────────────────────────────────────────┐
│                    前端 (Vue3 + Ant Design)              │
│  左侧边栏（会话列表）    |    右侧聊天界面              │
│  - 新建咨询              |    - 消息展示                │
│  - 历史会话              |    - 工具调用可视化          │
│  - 城市切换              |    - 流式输出                │
├──────────────────────────────────────────────────────────┤
│                    后端 API (FastAPI)                    │
│  /api/chat          — 对话接口（流式/非流式）            │
│  /api/evaluation    — 评估对比接口                       │
├──────────────────────────────────────────────────────────┤
│                  Agent 层 (LangGraph)                    │
│  ┌──────────────────────────────────────────────────┐    │
│  │  🏥 医小助 ReAct Agent                           │    │
│  │  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐     │    │
│  │  │ 思考 │ → │ 工具 │ → │ 观察 │ → │ 回答 │ ... │    │
│  │  └──────┘   └──────┘   └──────┘   └──────┘     │    │
│  └──────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────┤
│                       工具层 (7 个)                      │
│  🩺 symptom_analyzer       症状分析与就医建议            │
│  🏥 hospital_search         医院搜索                      │
│  📋 department_guide        科室导诊                      │
│  📚 medical_knowledge       医疗健康知识库                │
│  🗺️ route_to_hospital       就医路线规划                  │
│  💊 medication_consultation 用药咨询                      │
│  📰 health_news_search      健康资讯搜索                  │
└──────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
Intelligent-Medical-Assistant/
├── backend/                          # 后端
│   ├── app/
│   │   ├── agents/
│   │   │   └── medical_agent.py      # ReAct Agent (LangGraph)
│   │   ├── tools/                     # 7个医疗工具
│   │   │   ├── symptom_analyzer.py
│   │   │   ├── hospital_search.py
│   │   │   ├── department_guide.py
│   │   │   ├── medical_knowledge.py
│   │   │   ├── route_planner.py
│   │   │   ├── medication_consultation.py
│   │   │   └── health_news.py
│   │   ├── services/
│   │   │   ├── llm_service.py         # LLM 服务
│   │   │   └── conversation_service.py # 会话管理
│   │   ├── evaluation/
│   │   │   └── agent_evaluator.py     # 对比评估模块
│   │   ├── api/
│   │   │   ├── main.py                # FastAPI 入口
│   │   │   └── routes/
│   │   │       ├── chat.py            # 对话 API（含流式）
│   │   │       └── evaluation.py      # 评估 API
│   │   ├── models/
│   │   │   └── schemas.py             # 数据模型
│   │   └── config.py                  # 配置管理
│   ├── requirements.txt
│   ├── .env.example
│   └── run.py
└── frontend/                         # 前端
    ├── src/
    │   ├── views/
    │   │   └── Home.vue               # 聊天主界面
    │   ├── services/
    │   │   └── api.ts                 # API 服务
    │   ├── types/
    │   │   └── index.ts               # 类型定义
    │   ├── App.vue
    │   └── main.ts
    └── package.json
```

## 🚀 快速开始

### 1. 环境准备

```bash
# Python 3.10+
# Node.js 18+
```

### 2. 后端启动

```bash
cd backend

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
copy .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY 等

# 启动服务
python run.py
```

后端服务将在 http://localhost:8000 启动

API 文档：http://localhost:8000/docs

### 3. 前端启动

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端将在 http://localhost:5173 启动

## 🧠 ReAct Agent 工作原理

### 什么是 ReAct？

ReAct = **Reasoning + Acting**（推理 + 行动）

```
用户问题
    ↓
┌─────────────────────────────────────┐
│  思考 (Thought)                     │
│  分析问题，决定下一步做什么          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  行动 (Action)                      │
│  选择并调用合适的工具                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  观察 (Observation)                 │
│  查看工具返回的结果                  │
└─────────────────────────────────────┘
    ↓
  信息足够了吗？
    ├── 是 → 生成最终回答
    └── 否 → 回到思考步骤，继续循环
```

### LangGraph 实现

```python
# 状态图结构
agent (思考决策)
  ↓
[条件判断] → 有工具调用?
  ├── 是 → tools (执行工具) → agent
  └── 否 → END (结束)
```

## 🔧 工具列表

| 工具 | 说明 | 适用场景 |
|------|------|----------|
| 🩺 symptom_analyzer | 症状分析 | 描述症状，获取初步分析和就医建议 |
| 🏥 hospital_search | 医院搜索 | 查找附近的医院、三甲医院 |
| 📋 department_guide | 科室导诊 | 不知道挂什么科，智能推荐 |
| 📚 medical_knowledge | 医疗知识 | 疾病介绍、治疗、预防知识 |
| 🗺️ route_to_hospital | 路线规划 | 规划去医院的交通路线 |
| 💊 medication_consultation | 用药咨询 | 药品用法、副作用、注意事项 |
| 📰 health_news_search | 健康资讯 | 最新健康新闻和科普 |

## 📊 评估对比

### 评估指标

| 指标 | 流水线模式 | ReAct 模式 | 提升 |
|------|-----------|-----------|------|
| 回答质量 | ~70% | ~85% | **+15%** |
| 工具选择准确率 | ~65% | ~90% | **+25%** |
| 平均工具调用数 | 1.8 | 2.2 | +0.4 |
| 成功率 | 80% | 95% | **+15%** |

### 运行评估

```bash
# 方式1：API 调用
POST http://localhost:8000/api/evaluation/run

# 方式2：直接运行脚本
python -m app.evaluation.agent_evaluator
```

### 测试用例覆盖

- ✅ 简单症状咨询
- ✅ 医院搜索
- ✅ 科室导诊
- ✅ 用药咨询
- ✅ 医疗知识查询
- ✅ 路线规划
- ✅ 多工具组合任务
- ✅ 复杂问诊
- ✅ 健康资讯
- ✅ 纯知识问答（无需工具）

## 🎨 界面预览

### 聊天界面
- 左侧：会话列表 + 城市选择
- 右侧：对话区域 + 工具调用可视化
- 支持流式输出，实时展示思考过程

### 工具调用展示
- 可展开/收起的工具卡片
- 显示工具名称、参数、执行结果
- 成功/失败状态标识
- 执行耗时统计

## ⚠️ 免责声明

本项目仅用于学习和研究目的，提供的所有医疗健康信息仅供参考，**不能替代专业医生的诊断和治疗**。

如有身体不适，请及时就医。紧急情况请拨打 120 急救电话。

## 🛠️ 技术栈

### 后端
- **FastAPI** — Web 框架
- **LangGraph** — Agent 编排
- **LangChain** — LLM 应用框架
- **OpenAI API** — 大语言模型
- **Pydantic** — 数据验证

### 前端
- **Vue 3** — 前端框架
- **TypeScript** — 类型安全
- **Ant Design Vue** — UI 组件库
- **Axios** — HTTP 客户端

## 📝 开发说明

### 魔改自
本项目基于 [hello-agents-trip-planner](https://github.com/...) 魔改而来，主要改动：

1. **场景切换**：旅行规划 → 智能就医
2. **架构升级**：线性流水线 → ReAct Agent (LangGraph)
3. **交互升级**：表单提交 → 对话式交互
4. **工具扩展**：4个工具 → 7个医疗工具
5. **评估体系**：新增流水线 vs ReAct 对比评估

### 核心改进点

- 从固定流程 → 自主决策
- 从单次调用 → 多轮思考
- 从被动执行 → 主动规划
- 从刚性结构 → 灵活编排

## 📄 License

MIT
