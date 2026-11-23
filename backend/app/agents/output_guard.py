"""输出后置校验（第三层护栏）—— 防止"工具没给的数据，回答里补出来"。

为什么需要这一层：
    前两层（入口意图识别、工具级 interrupt）管的是"要不要执行"，管不住
    "执行完之后怎么写"。实测里最典型的三类越界：
      1) 路线工具明确返回"未获取到实时路线"，模型仍编排了地铁换乘和用时；
      2) 医院搜索只返回 POI（名称/地址/电话），回答里却出现"科室排名/号源紧张/专家"；
      3) 检索不到资讯时，用通用知识补成"最新消息"。
    这些内容看起来最可信，也最危险。所以在答案出口再做一次确定性检查：
    命中就往答案末尾追加限定说明，同时把命中的问题放进 SSE 的 done 事件，
    供评测统计"无依据断言发生率"。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

# 可以被核验的具体事实：手机号 / 座机 / 距离 / 时长
PRECISE_PATTERNS = (
    re.compile(r"1[3-9]\d{9}"),
    re.compile(r"0\d{2,3}-?\d{7,8}"),
    re.compile(r"\d+\s*(公里|千米|km)"),
    re.compile(r"\d+\s*(分钟|小时)"),
)

# 工具不会返回、只能靠猜的评价性表述
CLAIM_WORDS = ("排名", "全国知名", "权威", "号源", "余号", "专家号", "国家重点", "第一梯队", "口碑")

# 诚实标注数据边界的说法
CAVEAT_WORDS = (
    "未获取", "未能获取", "无法确认", "仅供参考", "以医院官方", "以实际", "为准",
    "估算", "未能实时", "未返回", "请核实", "以官方",
)

DISCLAIMER_WORDS = (
    "仅供参考", "不能替代", "及时就医", "请就医", "建议就医",
    "咨询医生", "咨询药师", "医生或药师", "遵医嘱",
)
DOCTOR_HINT = ("遵医嘱", "咨询医生", "咨询药师", "医生或药师", "药师", "医生")
DOSE_PATTERN = re.compile(r"(每次|一日|一天|每日)\s*[0-9]")

FAILED_STATUSES = {"failed"}


def _tool_names(events: Sequence[Dict[str, Any]]) -> set:
    return {event.get("tool_name") for event in events if event.get("tool_name")}


def inspect_answer(answer: str, tool_events: Sequence[Dict[str, Any]] | None) -> List[str]:
    """返回命中的问题列表；空列表表示通过。"""
    text = answer or ""
    events = list(tool_events or [])
    if not text:
        return []

    # 急症分支自带 120 指引，不再重复挂免责
    if any(event.get("status") == "emergency_shortcut" for event in events):
        return []

    issues: List[str] = []
    has_caveat = any(word in text for word in CAVEAT_WORDS)
    degraded = [
        event for event in events
        if event.get("status") in FAILED_STATUSES or event.get("used_fallback")
    ]

    if degraded and not has_caveat:
        if any(pattern.search(text) for pattern in PRECISE_PATTERNS):
            issues.append("工具失败/降级后仍给出具体地址或时长，未标注来源限制")
        if any(word in text for word in CLAIM_WORDS):
            issues.append("工具失败/降级后仍给出医院或科室评价")

    # 医院/科室评价只能来自知识检索；POI 搜索不提供这类信息
    if any(word in text for word in CLAIM_WORDS) and "medical_knowledge_search" not in _tool_names(events):
        if "医院/科室评价缺少检索依据" not in issues:
            issues.append("医院/科室评价缺少检索依据")

    if DOSE_PATTERN.search(text) and not any(word in text for word in DOCTOR_HINT):
        issues.append("给出具体剂量但未提示遵医嘱")

    if ("用药" in text or "药" in text) and not any(word in text for word in DISCLAIMER_WORDS):
        issues.append("用药相关回答缺少免责/就医提示")

    return issues


def enforce_final_answer(
    answer: str,
    tool_events: Sequence[Dict[str, Any]] | None,
) -> Tuple[str, List[str]]:
    """命中问题就在答案末尾追加限定说明；返回 (最终答案, 问题列表)。"""
    issues = inspect_answer(answer, tool_events)
    if not issues:
        return answer, []

    notes: List[str] = []
    if any(("未标注来源限制" in issue) or ("评价" in issue) or ("依据" in issue) for issue in issues):
        notes.append(
            "> 说明：以上部分信息未从检索结果中获得，医院等级、科室实力、号源与路线请以医院官方渠道或导航软件为准。"
        )
    if any(("剂量" in issue) or ("免责" in issue) for issue in issues):
        notes.append("> 用药请遵医嘱，具体剂量与禁忌请咨询医生或药师。")

    if not notes:
        return answer, issues
    return answer.rstrip() + "\n\n" + "\n".join(dict.fromkeys(notes)), issues
