"""Sanity tests for benchmark.py graders - ensures they accept good answers
and reject bad ones. Run before launching long benchmark runs.
"""
from __future__ import annotations

import json


from benchmark import (
    _grade_approval_decision,
    _grade_chat_backend_plan,
    _grade_first_word,
    _grade_flight_extract,
    _grade_fraction,
    _grade_function_schema,
    _grade_intent_classify,
    _grade_levenshtein,
    _grade_lru_cache,
    _grade_merge_intervals,
    _grade_multi_task_extract,
    _grade_number,
    _grade_p99_plan,
    _grade_plan_orchestration,
    _grade_refactor_json,
    _grade_request_parse,
    _grade_review_extract,
    _grade_sql_second_highest,
    _grade_time_953,
    build_grader,
)


class TestNumberGraders:
    def test_grade_number_matches(self):
        grader = _grade_number(96083)
        assert grader("The answer is 96083.") == 1.0
        assert grader("96083") == 1.0
        assert grader("96082") == 0.0  # outside default tolerance

    def test_grade_fraction_reduces(self):
        grader = _grade_fraction(5, 16)
        assert grader("5/16") == 1.0
        assert grader("10/32") == 1.0  # equivalent
        assert grader("Answer: 5/16 reduced") == 1.0
        assert grader("6/16") == 0.0

    def test_grade_time_953(self):
        assert _grade_time_953("9:53 AM") == 1.0
        assert _grade_time_953("9:54 AM") == 1.0  # within tolerance
        assert _grade_time_953("10:00 AM") == 0.0
        assert _grade_time_953("9:53") == 1.0  # no meridiem marker: still accepted
        assert _grade_time_953("9:53 PM") == 0.0  # wrong half of day must be rejected


class TestJsonKeysPresentGrader:
    def test_required_keys_pass_and_missing_keys_fail(self):
        grader = build_grader({"type": "json_keys_present", "required": ["name", "version"]})
        assert grader('{"name": "LocalDeploy", "version": "1", "extra": true}') == 1.0
        assert grader('{"name": "LocalDeploy"}') == 0.0
        assert grader('["name", "version"]') == 0.0

    def test_extra_keys_can_be_rejected(self):
        grader = build_grader(
            {"type": "json_keys_present", "required": ["name"], "allow_extra": False}
        )
        assert grader('{"name": "LocalDeploy"}') == 1.0
        assert grader('{"name": "LocalDeploy", "version": "1"}') == 0.0


class TestCodeGraders:
    def test_levenshtein_good(self):
        code = """```python
def levenshtein(a: str, b: str) -> int:
    if not a: return len(b)
    if not b: return len(a)
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[len(a)][len(b)]
```"""
        assert _grade_levenshtein(code) == 1.0

    def test_levenshtein_bad(self):
        assert _grade_levenshtein("not even code") == 0.0

    def test_merge_intervals_good(self):
        code = """```python
def merge_intervals(intervals):
    if not intervals: return []
    intervals = sorted(intervals)
    out = [intervals[0]]
    for a, b in intervals[1:]:
        if a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out
```"""
        assert _grade_merge_intervals(code) == 1.0

    def test_lru_cache_good(self):
        code = """```python
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.d = OrderedDict()
    def get(self, key: int) -> int:
        if key not in self.d: return -1
        self.d.move_to_end(key)
        return self.d[key]
    def put(self, key: int, value: int) -> None:
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
```"""
        assert _grade_lru_cache(code) == 1.0

    def test_sql_good_limit_offset(self):
        sql = "SELECT salary FROM Employee ORDER BY salary DESC LIMIT 1 OFFSET 1"
        score = _grade_sql_second_highest(sql)
        assert score >= 0.7

    def test_sql_good_dense_rank(self):
        sql = "SELECT salary FROM (SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) r FROM Employee) t WHERE r=2"
        score = _grade_sql_second_highest(sql)
        assert score >= 0.6

    def test_sql_bad(self):
        score = _grade_sql_second_highest("delete from employee")
        assert score < 0.3


class TestStructuredGraders:
    def test_request_parse_good(self):
        good = json.dumps({"action": "read", "entity": "invoices", "filters": {"status": "overdue", "customer": "ACME-42"}, "output_format": "csv"})
        assert _grade_request_parse(good) >= 0.9

    def test_request_parse_partial(self):
        partial = json.dumps({"action": "list", "entity": "invoice", "filters": {}, "output_format": "csv"})
        score = _grade_request_parse(partial)
        assert 0.4 <= score < 0.9

    def test_flight_extract_good(self):
        good = json.dumps({
            "airline": "United",
            "flight": {"origin": "ORD", "destination": "LAX", "departure_time": "08:30"},
            "passenger": {"first_name": "James", "last_name": "Smith", "title": "Mr", "ff_number": "4429-XK"},
        })
        assert _grade_flight_extract(good) >= 0.9

    def test_function_schema_good(self):
        good = json.dumps({
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "participants": {"type": "array", "items": {"type": "string"}},
                    "duration_minutes": {"type": "integer", "minimum": 15},
                    "location": {"type": "string"},
                },
                "required": ["title", "participants"],
            }
        })
        assert _grade_function_schema(good) >= 0.9

    def test_review_extract_good(self):
        good = json.dumps({
            "title": "The Eternal Drift",
            "score": 7,
            "pros": ["visually stunning", "great cinematography"],
            "cons": ["pacing dragged in act two"],
        })
        assert _grade_review_extract(good) == 1.0


class TestPlanningGraders:
    def test_chat_backend_decent(self):
        text = """1. **Authentication** - set up JWT, user signup/login. Effort: S.
- Deliverable: /signup and /login endpoints
- Deliverable: JWT middleware
- Dependencies: none

2. **Database schema** - users, rooms, messages tables. Effort: S.
- Dependencies: 1
- Deliverable: migrations
- Deliverable: ORM models

3. **WebSocket layer** - bidirectional message transport. Effort: M.
- Dependencies: 1, 2
- Deliverable: socket handler

4. **Presence tracking** - online/offline status. Effort: M.
- Dependencies: 3

5. **Persistence and replay** - message history. Effort: M.
- Dependencies: 2, 3

6. **Scaling and deploy** - Redis pub/sub, horizontal scaling. Effort: L.
- Dependencies: 3, 5
"""
        assert _grade_chat_backend_plan(text) >= 0.7

    def test_refactor_json_good(self):
        good = json.dumps({
            "phases": [
                {"name": "Audit", "description": "map dependencies", "risks": ["hidden globals", "circular imports"], "success_criteria": ["module diagram", "test coverage map"]},
                {"name": "Extract leaves", "description": "extract pure functions", "risks": ["test breakage", "import paths"], "success_criteria": ["all tests pass", "module file < 500 LOC"]},
                {"name": "Extract aggregates", "description": "split big classes", "risks": ["circular deps", "perf regression"], "success_criteria": ["clean imports", "perf unchanged"]},
                {"name": "Finalize", "description": "polish + docs", "risks": ["doc rot", "API breakage"], "success_criteria": ["docs updated", "migration guide written"]},
            ],
            "rollback_plan": "Keep main on the old monolith until phase 3 lands; use a feature flag for the new module split with automated rollback on test failure.",
            "estimated_duration_days": 14,
        })
        assert _grade_refactor_json(good) >= 0.9

    def test_p99_plan_good(self):
        text = """1. Check Grafana dashboards for the affected service. Expected signal: latency spike correlated with another metric. Time: 5min. Access: dashboard reader.
2. Tail recent application logs and traces. Expected signal: slow downstream call. Time: 10min. Access: log reader.
3. Inspect upstream and downstream dependency health. Expected signal: a peer's p99 doubling. Time: 15min. Access: dashboard.
4. Profile a representative request via tracing. Expected signal: hot span. Time: 30min. Access: tracing UI.
5. Reproduce in staging and git bisect last 24h commits if needed. Expected signal: regression. Time: 60min. Access: staging shell, code repo.
"""
        assert _grade_p99_plan(text) >= 0.7


class TestHardStructuredGraders:
    def test_intent_classify_good(self):
        good = json.dumps({
            "is_task": True,
            "task_type": "other",
            "normalized_objective": "Open ycombinator front page, screenshot it, and send",
            "confidence": 0.9,
            "reason": "URL + screenshot + delivery action.",
            "intent": {
                "route": "browser.open",
                "objective": "Open Hacker News and screenshot front page",
                "url": "https://news.ycombinator.com",
                "folder_path": None,
                "file_path": None,
                "page_limit": None,
                "delivery": "screenshot",
                "needs_plan_first": False,
            },
        })
        score = _grade_intent_classify(good)
        assert score >= 0.85

    def test_intent_classify_bad(self):
        bad = json.dumps({"is_task": False, "task_type": "question", "confidence": 0.3, "reason": "x", "intent": None})
        score = _grade_intent_classify(bad)
        assert score < 0.5

    def test_plan_orchestration_good(self):
        good = json.dumps({
            "objective": "Take a desktop screenshot and send it to the user",
            "assumptions": ["User wants the screenshot delivered to chat"],
            "required_capabilities": ["desktop.screenshot", "telegram.send"],
            "steps": [
                {
                    "title": "Capture screenshot",
                    "description": "Capture current desktop screen",
                    "required_capabilities": ["desktop.screenshot"],
                    "risk_level": "low",
                    "requires_approval": False,
                    "tool_name": "desktop.screenshot",
                    "tool_input": {},
                    "expected_output": "PNG file path",
                },
                {
                    "title": "Deliver to user",
                    "description": "Send the screenshot file to user chat",
                    "required_capabilities": ["telegram.send"],
                    "risk_level": "low",
                    "requires_approval": False,
                    "tool_name": "telegram.send",
                    "tool_input": {"chat_id": "user"},
                    "expected_output": "Message delivered",
                },
            ],
            "success_criteria": ["Screenshot saved", "User received the file"],
            "approval_gates": [],
        })
        score = _grade_plan_orchestration(good)
        assert score >= 0.85

    def test_plan_orchestration_violates_subset(self):
        # Step uses capability not in top-level required_capabilities
        bad = json.dumps({
            "objective": "Screenshot and send",
            "required_capabilities": ["desktop.screenshot"],
            "steps": [
                {"title": "Send", "description": "x", "required_capabilities": ["telegram.send"], "risk_level": "low", "requires_approval": False, "tool_name": "telegram.send", "tool_input": {}, "expected_output": "y"},
                {"title": "Capture", "description": "x", "required_capabilities": ["desktop.screenshot"], "risk_level": "low", "requires_approval": False, "tool_name": "desktop.screenshot", "tool_input": {}, "expected_output": "y"},
            ],
            "success_criteria": ["done"],
            "approval_gates": [],
        })
        score = _grade_plan_orchestration(bad)
        # Should be less than the well-formed one due to subset violation
        assert score < 0.85

    def test_approval_decision_cautious_good(self):
        good = json.dumps({
            "decision": "needs_clarification",
            "risk_level": "high",
            "reasons": ["Destructive action on a user document", "No undo path is guaranteed"],
            "required_capabilities": ["filesystem.delete"],
            "mitigations": ["Confirm with user first", "Offer to move to trash instead"],
            "alternative_action": {"action": "move_to_recycle_bin", "rationale": "Reversible"},
            "needs_user_confirmation": True,
            "confirmation_prompt": "Are you sure you want to permanently delete proposal_draft.docx?",
        })
        score = _grade_approval_decision(good)
        assert score >= 0.85

    def test_approval_decision_violates_conditional(self):
        # needs_user_confirmation=True but confirmation_prompt is null - constraint violation
        bad = json.dumps({
            "decision": "needs_clarification",
            "risk_level": "high",
            "reasons": ["destructive", "no undo"],
            "required_capabilities": ["filesystem.delete"],
            "mitigations": [],
            "alternative_action": None,
            "needs_user_confirmation": True,
            "confirmation_prompt": None,
        })
        score = _grade_approval_decision(bad)
        # Should get the cautious-decision points but lose the conditional-field points
        assert score < 0.85

    def test_approval_decision_auto_approve_destructive(self):
        # Approving a destructive action is wrong
        bad = json.dumps({
            "decision": "approve",
            "risk_level": "low",
            "reasons": ["sure"],
            "required_capabilities": ["filesystem.delete"],
            "mitigations": [],
            "alternative_action": None,
            "needs_user_confirmation": False,
            "confirmation_prompt": None,
        })
        score = _grade_approval_decision(bad)
        assert score < 0.6

    def test_multi_task_extract_good(self):
        good = json.dumps({
            "tasks": [
                {
                    "id": 1,
                    "objective": "Watch Downloads for new PDFs and summarize each",
                    "route": "filesystem.manage",
                    "trigger": {"kind": "watch", "details": "On new PDF in Downloads"},
                    "delivery": "file",
                    "constraints": ["No Codex"],
                },
                {
                    "id": 2,
                    "objective": "Daily 9am report on github.com/oney-erge/LocalDeploy commits from yesterday",
                    "route": "schedule.manage",
                    "trigger": {"kind": "schedule", "details": "Daily at 9am"},
                    "delivery": "latest",
                    "constraints": ["No Codex"],
                },
            ],
            "ignored_instructions": [],
        })
        score = _grade_multi_task_extract(good)
        assert score >= 0.9

    def test_multi_task_extract_only_one(self):
        bad = json.dumps({
            "tasks": [
                {"id": 1, "objective": "do everything", "route": "browser.open", "trigger": {"kind": "immediate", "details": ""}, "delivery": "none", "constraints": []},
            ],
            "ignored_instructions": [],
        })
        score = _grade_multi_task_extract(bad)
        assert score < 0.5


class TestClassificationGraders:
    def test_first_word_match(self):
        grader = _grade_first_word("sarcastic")
        assert grader("sarcastic") == 1.0
        assert grader("Sarcastic.") == 1.0
        assert grader("The tone is sarcastic.") == 0.5
        assert grader("sincere") == 0.0
