"""输出护栏的回归测试（纯函数，不依赖模型与网络）。

运行方式（在 backend/ 下）：
    python -m unittest tests.test_output_guard -v

这些用例对应评测 round1~round3 里真实出现过的越界：
路线降级后编排换乘、医院 POI 结果补科室排名、剂量缺遵医嘱。
"""

from __future__ import annotations

import unittest

from app.agents.output_guard import enforce_final_answer, inspect_answer


def ev(name: str, status: str = "success", fallback: bool = False) -> dict:
    return {"tool_name": name, "status": status, "used_fallback": fallback}


class TestOutputGuard(unittest.TestCase):
    def test_degraded_route_with_specific_duration(self) -> None:
        """路线工具失败却给出具体换乘和时长 —— 命中。"""
        answer = "路线工具暂时没返回详细方案，我给您梳理一下：乘 5 号线换 6 号线，全程约 40-50 分钟。"
        issues = inspect_answer(answer, [ev("route_to_hospital", status="failed")])
        self.assertTrue(any("未标注来源限制" in issue for issue in issues), issues)

    def test_degraded_route_with_caveat_passes(self) -> None:
        answer = "未能获取到实时路线，以下为估算：全程约 40-50 分钟，请以导航软件为准。"
        issues = inspect_answer(answer, [ev("route_to_hospital", status="failed")])
        self.assertEqual(issues, [])

    def test_claims_without_knowledge_tool(self) -> None:
        """医院 POI 结果里不会有序号排名/号源，出现即越界。"""
        answer = "推荐华西医院，消化内科全国知名，号源比较紧张。"
        issues = inspect_answer(answer, [ev("hospital_search")])
        self.assertIn("医院/科室评价缺少检索依据", issues)

    def test_claims_with_knowledge_tool_allowed(self) -> None:
        answer = "根据资料，该院消化内科为国家级重点专科。"
        issues = inspect_answer(answer, [ev("hospital_search"), ev("medical_knowledge_search")])
        self.assertNotIn("医院/科室评价缺少检索依据", issues)

    def test_dose_without_doctor_hint(self) -> None:
        answer = "布洛芬每次 0.2g，一天 3 次。"
        issues = inspect_answer(answer, [ev("medication_consultation")])
        self.assertIn("给出具体剂量但未提示遵医嘱", issues)

    def test_dose_with_doctor_hint_passes(self) -> None:
        answer = "布洛芬每次 0.2g，一天 3 次，请遵医嘱使用。"
        issues = inspect_answer(answer, [ev("medication_consultation")])
        self.assertNotIn("给出具体剂量但未提示遵医嘱", issues)

    def test_emergency_branch_skipped(self) -> None:
        answer = "请立即拨打 120，尽快前往急诊科。"
        issues = inspect_answer(answer, [ev("emergency_triage", status="emergency_shortcut")])
        self.assertEqual(issues, [])

    def test_enforce_appends_note(self) -> None:
        answer = "路线工具没返回，我按经验给您排一下：约 40 分钟。"
        patched, issues = enforce_final_answer(answer, [ev("route_to_hospital", status="failed")])
        self.assertTrue(issues)
        self.assertIn("以医院官方渠道或导航软件为准", patched)
        self.assertTrue(patched.startswith(answer))

    def test_clean_answer_untouched(self) -> None:
        answer = "为您找到北京协和医院，地址：西城区大木仓胡同 41 号。以上信息来自地图检索，仅供参考。"
        patched, issues = enforce_final_answer(answer, [ev("hospital_search")])
        self.assertEqual(issues, [])
        self.assertEqual(patched, answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
