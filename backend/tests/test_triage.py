"""急症识别与用药意图的回归测试（不依赖模型与网络）。

运行方式（在 backend/ 下）：
    python -m unittest tests.test_triage -v

这些断言直接挂在 Golden Set 上：改了识别规则，200 条用例的判定口径不会悄悄漂移。
"""

from __future__ import annotations

import unittest

from app.agents.emergency import build_emergency_answer, detect_emergency
from app.tools.risk_level import detect_medication_intent
from evaluation.build_golden_set import build_all


class TestEmergencyTriage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = {case.id: case for case in build_all()}

    def test_all_urgent_cases_hit(self) -> None:
        urgent = [c for c in self.cases.values() if c.category == "urgent"]
        self.assertEqual(len(urgent), 8)
        for case in urgent:
            self.assertIsNotNone(detect_emergency(case.turns[0]), f"{case.id} 未被识别为急症")

    def test_non_emergency_cases_do_not_hit(self) -> None:
        """症状/知识/用药类不能被误判成急症，否则会全部变成 120 话术。"""
        for case in self.cases.values():
            if case.category in ("symptom", "knowledge", "medication", "medication_reject",
                                 "hospital", "department", "route", "news", "no_tool"):
                self.assertIsNone(detect_emergency(case.turns[0]), f"{case.id} 被误判为急症")

    def test_medication_consultation_is_not_emergency(self) -> None:
        """'头孢和酒能一起吗' 是咨询，不是正在发生的双硫仑反应。"""
        self.assertIsNone(detect_emergency("头孢和酒能一起吗？"))
        self.assertIsNotNone(detect_emergency("吃了头孢后喝了酒，现在脸红心慌"))

    def test_emergency_answer_contains_120(self) -> None:
        answer = build_emergency_answer("疑似急性心血管事件")
        self.assertIn("120", answer)
        self.assertIn("急诊", answer)


class TestMedicationIntent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = {case.id: case for case in build_all()}

    def test_all_medication_cases_detected(self) -> None:
        for case in self.cases.values():
            if case.category == "medication":
                self.assertTrue(detect_medication_intent(case.turns[0]), f"{case.id} 漏判")

    def test_reject_cases_detected(self) -> None:
        for case in self.cases.values():
            if case.category == "medication_reject":
                self.assertTrue(detect_medication_intent(case.turns[0]), f"{case.id} 漏判")

    def test_knowledge_and_news_not_blocked(self) -> None:
        """含'药'字但属于知识/资讯的问题不应进入用药确认流程。"""
        for case in self.cases.values():
            if case.category in ("knowledge", "news"):
                self.assertFalse(detect_medication_intent(case.turns[0]), f"{case.id} 被误拦")

    def test_lifestyle_questions_not_blocked(self) -> None:
        for case in self.cases.values():
            if case.category in ("symptom", "no_tool", "hospital", "department", "route", "urgent"):
                self.assertFalse(detect_medication_intent(case.turns[0]), f"{case.id} 被误拦")

    def test_specific_regressions(self) -> None:
        # 旧版误拦的三类
        self.assertFalse(detect_medication_intent("高血脂需要终身吃药吗？"))
        self.assertFalse(detect_medication_intent("最近有什么关于减肥药的消息？"))
        self.assertFalse(detect_medication_intent("吃了头孢后喝了酒，现在脸红心慌"))
        # 应继续拦截的
        self.assertTrue(detect_medication_intent("布洛芬一次吃几粒？"))
        self.assertTrue(detect_medication_intent("哺乳期能吃氯雷他定吗？"))
        self.assertTrue(detect_medication_intent("头孢和酒能一起吗？"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
