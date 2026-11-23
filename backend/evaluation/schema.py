"""Golden Set 的数据模型、校验与加载。

为什么用 dataclass 而不是 pydantic：
    评测集需要能在没装后端依赖的环境里校验（CI、标注小脚本），所以这里只用
    标准库；业务侧的请求/响应模型继续用 pydantic，两者互不影响。

设计：
    一条用例 = 一次多轮会话（turns 依次发送），因为 Agent 的记忆、上下文、
    澄清能力只有在多轮里才看得出来，单轮 QA 测不到。

    期望值分三层，避免"语义全靠人工看"：
    1. 硬约束（可完全自动判定）：must_call / must_not_call / expect_approval /
       approval_decision / must_not_have；
    2. 弱约束（关键词兜底，不追求语义）：must_have_facts，每组是"或"关系；
    3. 软约束（LLM 评审）：judge 分数，单独统计，不与硬约束混算。

    这样一份报告既能给出确定性的通过率，又能暴露"过了硬约束但回答没用"的情况。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

ALL_TOOLS: List[str] = [
    "symptom_analyzer",
    "hospital_search",
    "department_guide",
    "medical_knowledge_search",
    "route_to_hospital",
    "medication_consultation",
    "health_news_search",
]

TOOL_LABELS: Dict[str, str] = {
    "symptom_analyzer": "症状分析",
    "hospital_search": "医院搜索",
    "department_guide": "科室导诊",
    "medical_knowledge_search": "医疗知识",
    "route_to_hospital": "路线规划",
    "medication_consultation": "用药咨询",
    "health_news_search": "健康资讯",
}

RISK_LEVELS = ("low", "medium", "high")
DECISIONS = ("approve", "reject", "none")
MODES = ("stream", "sync")

# 类别 -> 说明。新增类别时这里必须同步，否则加载会报错。
CATEGORIES: Dict[str, str] = {
    "medication": "用药咨询：高危，应触发人工确认",
    "medication_reject": "用药咨询：用户拒绝确认，应直接收口",
    "symptom": "症状分析：中风险，给建议但不下诊断",
    "hospital": "医院搜索：低风险信息查询",
    "department": "科室导诊：低风险信息查询",
    "route": "路线规划：低风险信息查询",
    "knowledge": "医疗知识：低风险信息查询",
    "news": "健康资讯：低风险信息查询",
    "no_tool": "无需工具：问候、闲聊、能力边界",
    "multi_tool": "多工具 / 多轮：编排与上下文",
    "urgent": "急症：必须提示 120",
    "injection": "提示注入：不得越权执行或直接给剂量",
}


@dataclass
class Expect:
    """一条用例的期望值。"""

    # 必须调用的工具（集合语义，顺序无关）
    must_call: List[str] = field(default_factory=list)
    # 允许调用、但不参与"多余调用"扣分
    may_call: List[str] = field(default_factory=list)
    # 一旦调用即判失败
    must_not_call: List[str] = field(default_factory=list)

    risk_level: str = "low"
    # 是否应弹出人工确认（interrupt）
    expect_approval: bool = False
    # 评测时应模拟的用户决策：approve / reject / none
    approval_decision: str = "none"
    # 允许被门禁拦下，但不算"误拦"。
    # 典型场景是提示注入：用户本来就在索要剂量，被拦下来是加分而不是误伤。
    approval_optional: bool = False

    # 每组是"或"关系：任一关键词出现即视为命中；组内都为空则忽略
    must_have_facts: List[List[str]] = field(default_factory=list)
    # 出现即判失败（注意：只放不会被正确表述包含的字符串，避免误伤）
    must_not_have: List[str] = field(default_factory=list)
    must_cite: bool = False
    max_iterations: int = 10

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Expect":
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"expect 含未知字段: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        for name in ("must_call", "may_call", "must_not_call"):
            values = getattr(self, name)
            bad = [t for t in values if t not in ALL_TOOLS]
            if bad:
                raise ValueError(f"{name} 含未知工具: {bad}")
        if self.risk_level not in RISK_LEVELS:
            raise ValueError(f"risk_level 非法: {self.risk_level}")
        if self.approval_decision not in DECISIONS:
            raise ValueError(f"approval_decision 非法: {self.approval_decision}")
        overlap = set(self.must_call) & set(self.must_not_call)
        if overlap:
            raise ValueError(f"must_call 与 must_not_call 冲突: {sorted(overlap)}")
        if self.expect_approval and self.risk_level != "high":
            raise ValueError("expect_approval=True 的用例必须是 high 风险")
        if self.approval_decision != "none" and not self.expect_approval:
            raise ValueError("approval_decision 只有在 expect_approval=True 时才有意义")
        if self.expect_approval and self.approval_decision == "none":
            raise ValueError("expect_approval=True 时必须指定 approval_decision")
        if self.approval_optional and self.expect_approval:
            raise ValueError("approval_optional 与 expect_approval 互斥")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations 必须为正整数")
        for group in self.must_have_facts:
            if not isinstance(group, list):
                raise ValueError("must_have_facts 必须是 list[list[str]]")


@dataclass
class GoldenCase:
    """一条 Golden Set 用例。"""

    id: str
    category: str
    turns: List[str]
    city: str = "北京"
    # stream：走 chat_stream + resume_stream（可触发人工确认）
    # sync：走 chat（无确认通道，验证 fail-closed）
    mode: str = "stream"
    # 是否参与 LLM 评审（关闭可省成本，例如纯玩具对话）
    judge: bool = True
    note: str = ""
    expect: Expect = field(default_factory=Expect)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenCase":
        data = dict(data)
        data["expect"] = Expect.from_dict(data.get("expect") or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"用例含未知字段: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("id 不能为空")
        if self.category not in CATEGORIES:
            raise ValueError(f"未知类别: {self.category}")
        if not self.turns or any(not isinstance(t, str) or not t.strip() for t in self.turns):
            raise ValueError("turns 不能为空，且每轮必须是非空字符串")
        if self.mode not in MODES:
            raise ValueError(f"mode 非法: {self.mode}")
        if self.mode == "sync" and len(self.turns) != 1:
            raise ValueError("sync 模式只支持单轮用例")
        self.expect.validate()


def load_cases(path: str | Path) -> List[GoldenCase]:
    """加载 JSONL，逐条校验，并检查 id 唯一。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden Set 不存在: {path}")

    cases: List[GoldenCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                case = GoldenCase.from_dict(json.loads(line))
                case.validate()
            except Exception as exc:  # 带上行号，标注时好定位
                raise ValueError(f"{path.name}:{lineno} 用例非法: {exc}") from exc
            if case.id in seen:
                raise ValueError(f"{path.name}:{lineno} 重复的 id: {case.id}")
            seen.add(case.id)
            cases.append(case)
    return cases


def summarize_cases(cases: List[GoldenCase]) -> Dict[str, Any]:
    """用于 --dry-run 的分布统计。"""
    by_category: Dict[str, int] = {}
    by_risk: Dict[str, int] = {}
    tools: Dict[str, int] = {}
    for case in cases:
        by_category[case.category] = by_category.get(case.category, 0) + 1
        by_risk[case.expect.risk_level] = by_risk.get(case.expect.risk_level, 0) + 1
        for tool in case.expect.must_call:
            tools[tool] = tools.get(tool, 0) + 1
    return {
        "total": len(cases),
        "by_category": by_category,
        "by_risk": by_risk,
        "must_call_coverage": tools,
        "turns_total": sum(len(c.turns) for c in cases),
        "approval_cases": sum(1 for c in cases if c.expect.expect_approval),
        "judge_cases": sum(1 for c in cases if c.judge),
    }
