"""Tests for the quality / trajectory-preservation bench metric code (minmax_bench/quality).

These metrics produce the repo's public findings — they deserve the same test
surface as the cost bench.
"""
import json
import os
from types import SimpleNamespace

from minmax_bench.quality import engine as eng
from minmax_bench.quality import report


def tool(name, **inp):
    return {"type": "tool_use", "name": name, "input": inp}


# ---------------------------------------------------------------- rework_count
def test_rework_true_redundant_reread_counts():
    acts = [tool("Read", file_path="/a.py"), tool("Read", file_path="/a.py")]
    assert report.rework_count(acts) == 1


def test_rework_post_edit_reread_is_verification_not_rework():
    acts = [tool("Read", file_path="/a.py"), tool("Edit", file_path="/a.py"),
            tool("Read", file_path="/a.py")]
    assert report.rework_count(acts) == 0


def test_rework_post_edit_shell_cat_is_verification_not_rework():
    acts = [tool("Read", file_path="/a.py"), tool("Edit", file_path="/a.py"),
            tool("Bash", command="cat /a.py")]
    assert report.rework_count(acts) == 0


def test_rework_post_edit_grep_rerun_is_verification_not_rework():
    acts = [tool("Bash", command="grep pat /a.py"), tool("Edit", file_path="/a.py"),
            tool("Bash", command="grep pat /a.py")]
    assert report.rework_count(acts) == 0


def test_rework_grep_rerun_without_edit_counts():
    acts = [tool("Bash", command="grep pat /a.py"), tool("Bash", command="grep pat /a.py")]
    assert report.rework_count(acts) == 1


def test_rework_partial_then_covered_read():
    acts = [tool("Read", file_path="/a.py", offset=1, limit=10),
            tool("Read", file_path="/a.py", offset=5, limit=3)]  # inside seen span
    assert report.rework_count(acts) == 1


# ---------------------------------------------------------------- score
def test_score_same_bash_command_agrees():
    a, b = tool("Bash", command="ls  -la"), tool("Bash", command="ls -la")
    exact, action, sim = eng.score(a, b)
    assert action and sim > 0.9


def test_score_different_tool_disagrees():
    exact, action, _ = eng.score(tool("Read", file_path="/a"), tool("Bash", command="x"))
    assert not exact and not action


def test_score_identical_inputs_sim_is_one():
    a = tool("Edit", file_path="/a.py", old_string="x" * 5000, new_string="y" * 5000)
    exact, action, sim = eng.score(a, dict(a))
    assert exact and action and sim == 1.0


def test_score_text_bailout_does_not_agree_with_substantive_answer():
    orig = {"type": "text", "text": "The fix is to rebind the socket with SO_REUSEADDR "
                                    "and retry the bind in a loop, see server.py:42."}
    bail = {"type": "text", "text": "I can't proceed."}
    _, action, _ = eng.score(orig, bail)
    assert not action


# ---------------------------------------------------------------- bands / overlap
def test_overlap_disjoint_bands_diverge():
    assert report.overlaps((5, 5.5, 6), (11, 12, 14)) is False
    assert report.overlaps((5, 5.5, 6), (6, 7, 9)) is True


# ---------------------------------------------------------------- pricing
def test_cost_usd_is_model_aware():
    usage = {"input_tokens": 1_000_000}
    assert eng.cost_usd(usage, "claude-haiku-4-5") == 1.0
    assert eng.cost_usd(usage, "claude-sonnet-4-6") == 3.0
    assert eng.cost_usd(usage, "claude-opus-4-8") == 5.0


# ---------------------------------------------------------------- build_request purity
def _args(**kw):
    base = dict(max_tokens=100, strip_thinking=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_build_request_does_not_mutate_source_messages():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1",
                                          "content": "out"}]}]
    tmpl = {"model": "claude-sonnet-4-6", "system": "s", "tools": []}
    before = json.dumps(msgs, sort_keys=True)
    for i in (1, 2, 3):  # sequential replays over growing prefixes, like the drivers do
        req = eng.build_request(tmpl, msgs[:i], _args(), "sid")
        assert req["messages"][-1]["content"][-1].get("cache_control")
    assert json.dumps(msgs, sort_keys=True) == before  # no cache_control leaked back


def test_build_request_keeps_mcp_stubs_only_when_asked():
    tmpl = {"model": "m", "system": "s",
            "tools": [{"name": "Bash"}, {"name": "mcp__x__y"}]}
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    default = eng.build_request(tmpl, msgs, _args(), "sid")
    assert [t["name"] for t in default["tools"]] == ["Bash"]
    kept = eng.build_request(tmpl, msgs, _args(keep_all_tools=True), "sid")
    assert [t["name"] for t in kept["tools"]] == ["Bash", "mcp__x__y"]


# ---------------------------------------------------------------- incremental join + fidelity
def test_load_incremental_joins_and_reports_fidelity_vs_control(tmp_path):
    d = tmp_path / "incremental"
    d.mkdir()

    def write(arm, recs):
        (d / f"kv-{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))

    def rec(step, agree, ctx, cost):
        return {"arm": "x", "step": step, "agree_action": agree, "cost_usd": cost,
                "usage": {"input_tokens": ctx}}

    write("control", [rec(0, True, 100, 1.0), rec(1, True, 100, 1.0), rec(2, True, 100, 1.0)])
    write("condense", [rec(0, True, 90, 0.9), rec(1, True, 50, 0.5), rec(2, False, 50, 0.5)])
    out = report._load_incremental(str(tmp_path), ["condense"])
    inc = out[("kv", "condense")]
    assert inc["steps"] == 2                      # step 0 (cold cache) excluded everywhere
    assert abs(inc["comp"] - 0.5) < 1e-9          # (50+50) vs (100+100)
    assert inc["fid"] == 0.5 and inc["fid_ctrl"] == 1.0


def test_incremental_faithfulness_docks_redundant_refetch(tmp_path):
    """A step that agrees with the original but redundantly re-fetches already-seen info is
    NOT faithful — faithfulness must fall below raw agreement (rework folded in at source)."""
    d = tmp_path / "incremental"
    d.mkdir()

    def write(arm, recs):
        (d / f"kv-{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))

    def rec(step, agree, redundant):
        return {"arm": "x", "step": step, "agree_action": agree, "redundant": redundant,
                "cost_usd": 1.0, "usage": {"input_tokens": 100}}

    write("control", [rec(0, True, False), rec(1, True, False), rec(2, True, False)])
    # step 2 agrees with the original but re-fetches -> agreement 2/2 but faithful only 1/2
    write("condense", [rec(0, True, False), rec(1, True, False), rec(2, True, True)])
    inc = report._load_incremental(str(tmp_path), ["condense"])[("kv", "condense")]
    assert inc["fid"] == 0.5 and inc["fid_ctrl"] == 1.0
    assert inc["redund"] == 1


def test_faithful_step_backward_compatible():
    # old artifacts have no 'redundant' key -> degrade gracefully to plain agreement
    assert report._faithful_step({"agree_action": True}) is True
    assert report._faithful_step({"agree_action": True, "redundant": True}) is False
    assert report._faithful_step({"agree_action": False}) is False


def test_report_shows_incremental_only_tasks(tmp_path):
    """A task with ONLY incremental data (no full run — e.g. a session-labelled run) still
    gets a row with a faithfulness number, and its scoring method is inferred."""
    d = tmp_path / "incremental"
    d.mkdir()

    def write(arm, recs):
        (d / f"sess-{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))

    def rec(step, agree, ctx):
        return {"step": step, "agree_action": agree, "redundant": False,
                "cost_usd": 1.0, "usage": {"input_tokens": ctx}}

    write("control", [rec(0, True, 100), rec(1, True, 100), rec(2, True, 100)])
    write("condense", [rec(0, True, 60), rec(1, True, 60), rec(2, False, 60)])  # comp 40%
    args = SimpleNamespace(arms="condense", tasks="kv-store-grpc", agent="claude-code",
                           ctx_gate=50_000, **{"from": str(tmp_path)})
    built = report.build(args)
    row = next((r for r in built["rows"] if r["task"] == "sess"), None)
    assert row is not None                                   # incremental-only task got a row
    a = row["arms"]["condense"]
    assert a["n"] == 0                                       # no full run
    assert a["incr"]["scoring"] == "structural"             # inferred: no LLM judge in records
    faith, _cost = report._faithful_cost(a, report._floor_for(row, ["condense"]))
    assert "%" in faith                                      # a real number, not '—'


def test_faithful_engagement_counts_ccr_retrieves():
    """Headroom's NET compression can be ~0 while it still engaged CCR (compressed a tool output
    to a marker, then the agent retrieved it back). The score is always shown, but only carries a
    green/red verdict when the arm actually engaged (compressed OR retrieved)."""
    floor = 0.5
    low_no_ccr = {"incr": {"fid": 0.4, "comp": 0.01, "retrieves": 0, "costd": 0.0}}
    low_with_ccr = {"incr": {"fid": 0.4, "comp": 0.01, "retrieves": 5, "costd": 0.0}}
    passthrough = report._faithful_cost(low_no_ccr, floor)[0]
    engaged = report._faithful_cost(low_with_ccr, floor)[0]
    assert "40%" in passthrough and "green" not in passthrough and "red" not in passthrough
    assert "40%" in engaged and ("green" in engaged or "red" in engaged)  # verdict colour
    assert report._engaged({"comp": 0.01, "retrieves": 3}) is True
    assert report._engaged({"comp": 0.01, "retrieves": 0}) is False
    assert report._engaged({"comp": 0.20, "retrieves": 0}) is True  # compression alone counts


def test_scoring_infers_llm_judge():
    """The scoring label distinguishes a structural match from an LLM goal/equivalence judge."""
    structural = {0: {"agree_action": True, "agree_semantic": True}}
    goal = {0: {"agree_action": True, "quality": "good"}}
    equiv = {0: {"agree_action": False, "agree_semantic": True}}  # upgraded a near-miss
    assert report._scoring(structural) == "structural"
    assert report._scoring(goal) == "LLM · goal"
    assert report._scoring(equiv) == "LLM · equivalence"


def test_check_arms_catches_unknown_arm_and_missing_keys():
    # headroom-kompress is a full-mode-only arm — not a valid teacher-forced replay arm
    problems = eng.check_arms(["control", "headroom-kompress", "condense"], {})
    text = "\n".join(problems)
    assert "headroom-kompress" in text
    assert "ANTHROPIC_API_KEY" in text and "CONDENSE_API_KEY" in text
    assert not eng.check_arms(["control"], {"ANTHROPIC_API_KEY": "k"})


# ---------------------------------------------------------------- offline demo end-to-end
def test_bundled_sample_still_reports(tmp_path):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runs", "quality-sample")
    if not os.path.isdir(root):
        return  # sample not present in this checkout
    args = SimpleNamespace(arms="condense", tasks="kv-store-grpc", agent="claude-code",
                           **{"from": root})
    d = report.build(args)
    a = d["rows"][0]["arms"]["condense"]
    assert a["n"] == 2 and d["rows"][0]["vanilla"]["n"] == 2
    assert a["length_ok"] is False  # the documented headline: condense diverges on kv-store


# ---------------------------------------------------------------- backtest plumbing
def test_loader_expands_comma_separated_paths(tmp_path):
    from minmax_bench.data.loaders import _expand
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("")
    b.write_text("")
    assert _expand(f"{a},{b}") == [str(a), str(b)]
    assert _expand(str(tmp_path / "*.jsonl")) == [str(a), str(b)]


def test_recorded_usage_reads_real_session():
    import glob as g

    from minmax_bench.counterfactual import recorded_usage
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runs", "quality-sample")
    sessions = g.glob(f"{root}/vanilla-*/**/agent/sessions/projects/-app/*.jsonl",
                      recursive=True)
    if not sessions:
        return
    usages = recorded_usage(__import__("pathlib").Path(sessions[0]))
    assert usages and all("output_tokens" in u or "input_tokens" in u for u in usages)


def test_score_bash_cwd_artifacts_are_same_action():
    # real pair from a replay: cd-prefix + absolute-path spelling, same decision
    a = tool("Bash", command="python server.py &\nsleep 2\nps aux | grep server.py")
    b = tool("Bash", command="cd /app && python server.py &\nsleep 2\nps aux | grep server.py")
    exact, action, _ = eng.score(a, b)
    assert action and not exact
    c = tool("Bash", command="python -m grpc_tools.protoc -I. --python_out=. kv.proto && ls *pb2*")
    d = tool("Bash", command="python -m grpc_tools.protoc -I/app --python_out=/app "
                             "/app/kv.proto && ls /app/")
    _, action, _ = eng.score(c, d)
    assert action


def test_score_bash_different_program_still_disagrees():
    a = tool("Bash", command="python server.py")
    b = tool("Bash", command="cat server.py")
    _, action, _ = eng.score(a, b)
    assert not action


def test_resolve_tasks_forms():
    assert eng.resolve_tasks(None) == eng.DEFAULT_TASKS[:5]
    assert eng.resolve_tasks("2") == eng.DEFAULT_TASKS[:2]
    assert eng.resolve_tasks("a,b") == ["a", "b"]
    pool = eng.dataset_tasks()
    assert pool[: len(eng.DEFAULT_TASKS)] == eng.DEFAULT_TASKS  # curated stay first


def test_resolve_tasks_random_is_seeded_and_bounded():
    a = eng.resolve_tasks("random:4", seed=7)
    b = eng.resolve_tasks("random:4", seed=7)
    assert a == b and len(a) == 4 and all(t in eng.dataset_tasks() for t in a)


def test_report_marks_sub_gate_tasks():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runs", "quality-sample")
    if not os.path.isdir(root):
        return
    args = SimpleNamespace(arms="condense", tasks="kv-store-grpc", agent="claude-code",
                           ctx_gate=50_000, **{"from": root})
    d = report.build(args)
    row = d["rows"][0]
    assert row["vanilla"]["peak_ctx"] > 0
    assert row["sub_gate"] is True  # kv-store peaks ~25-35k: compaction can't have fired


def test_solve_distinguishes_never_ran_from_crashed(tmp_path):
    """A cell requested but with no trial dir ever opened is aborted/out-of-scope, not a
    0/k failure — it must render as '—', while a cell whose trials opened but produced no
    reward is a genuine crash and must render as '⚠ lost'."""
    # never ran: only attempted.json, no trial subdirs
    never = tmp_path / "vanilla-taskA"
    never.mkdir()
    (never / "attempted.json").write_text(json.dumps({"k": 5, "arm": "vanilla"}))
    # crashed: attempted.json AND an opened trial dir, but no verifier/reward.txt
    crashed = tmp_path / "vanilla-taskB"
    (crashed / "2026-01-01__00-00-00" / "inst").mkdir(parents=True)
    (crashed / "attempted.json").write_text(json.dumps({"k": 5, "arm": "vanilla"}))

    idx = report.index_runs(str(tmp_path), "claude-code")
    s_never = report._cell_stats(idx.get("vanilla-taskA"))
    s_crash = report._cell_stats(idx.get("vanilla-taskB"))
    assert s_never["started"] == 0 and report._solve(s_never) == "—"
    assert s_crash["started"] >= 1 and "lost" in report._solve(s_crash)


def test_auth_mode_resolution(monkeypatch):
    import minmax_bench.quality.engine as e
    monkeypatch.setattr(e, "_CC_TOKEN_CACHE", ["unset"])
    monkeypatch.setattr(e, "cc_oauth_token", lambda: None)
    assert e.auth_mode({"ANTHROPIC_API_KEY": "k"}) == "api-key"
    assert e.auth_mode({}) is None
    monkeypatch.setattr(e, "cc_oauth_token", lambda: "tok")
    assert e.auth_mode({}) == "subscription"
    problems = e.check_arms(["control"], {})
    assert not problems  # subscription satisfies auth


def test_referenced_tool_names_includes_search_discovered_mcp():
    """Tool-search sessions reference MCP tools by name in results without ever
    calling them; those must still be stubbed or Anthropic 400s on the reference."""
    msgs = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "ToolSearch", "input": {"query": "resize"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "found: mcp__plugin_pw__browser_resize, mcp__plugin_pw__browser_click"}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "name": "mcp__plugin_pw__browser_click", "input": {}}]},
    ]
    names = eng.referenced_tool_names(msgs)
    assert "ToolSearch" in names                              # direct call
    assert "mcp__plugin_pw__browser_click" in names           # direct call
    assert "mcp__plugin_pw__browser_resize" in names          # search-only, never called
    built = {t["name"] for t in eng.build_tools(names, [])}
    assert "mcp__plugin_pw__browser_resize" in built          # now stubbed -> no 400


def test_peek_reports_peak_context(tmp_path):
    from minmax_bench.counterfactual import _peek
    sess = tmp_path / "s.jsonl"
    lines = [
        {"type": "user", "message": {"content": "hi"}},
        {"type": "assistant", "message": {"usage": {"input_tokens": 10,
                                                    "cache_read_input_tokens": 5000}}},
        {"type": "assistant", "message": {"usage": {"input_tokens": 20,
                                                    "cache_read_input_tokens": 40000}}},  # peak
        {"type": "assistant", "message": {"usage": {"input_tokens": 20,
                                                    "cache_read_input_tokens": 8000}}},  # after
    ]
    sess.write_text("\n".join(json.dumps(x) for x in lines))
    prompt, cwd, has_assistant, peak, capped = _peek(sess)
    assert has_assistant and peak == 40020 and not capped  # peak, not last, not first


def test_step_verdict_and_redundancy():
    from minmax_bench import counterfactual as cf
    good = {"step": 0, "orig": tool("Read", file_path="/a"), "replay": tool("Read", file_path="/a"),
            "agree_action": True}
    semi = {"step": 1, "orig": tool("Read", file_path="/a", offset=1),
            "replay": tool("Read", file_path="/a", offset=99), "agree_action": False, "sim": 0.6}
    bad = {"step": 2, "orig": tool("Write", file_path="/a"), "replay": tool("Bash", command="ls"),
           "agree_action": False, "sim": 0.0}
    assert cf._step_verdict(good) == "good"
    assert cf._step_verdict(semi) == "semi"
    assert cf._step_verdict(bad) == "bad"
    # redundancy: a Read of a file an earlier step already touched
    files, cmds = {"/a"}, set()
    assert cf._is_refetch(tool("Read", file_path="/a"), files, cmds)
    assert not cf._is_refetch(tool("Read", file_path="/b"), files, cmds)


def test_judge_equivalent_parses_json(monkeypatch):
    import minmax_bench.quality.engine as e
    monkeypatch.setattr(e, "call_api",
                        lambda *a, **k: ({"content": [{"text": '{"equivalent": true}'}]}, None))
    v, cost = e.judge_equivalent(tool("Bash", command="grep x f"),
                                 tool("Bash", command="rg x f"), {"ANTHROPIC_API_KEY": "k"})
    assert v is True
    monkeypatch.setattr(e, "call_api", lambda *a, **k: (None, "HTTP 500"))
    assert e.judge_equivalent({}, {}, {})[0] is None  # error -> None, not a crash


def test_patch_cwd_rewrites_the_templates_real_advertised_cwd():
    # the template advertises SOME capture project; a replay in another project must
    # not be told it's in the capture dir (the bug: it cd'd into the wrong repo)
    tmpl = {"system": [{"type": "text",
            "text": "<env>\nworking directory: /Users/x/dev/capture-proj\n"
                    "logs at -Users-x-dev-capture-proj/memory/\n</env>"}]}
    eng.patch_cwd(tmpl, "data/cc_request_template.json", "/Users/x/dev/real-session")
    s = json.dumps(tmpl["system"])
    assert "/Users/x/dev/capture-proj" not in s          # path form rewritten
    assert "-Users-x-dev-capture-proj" not in s          # CC slug form rewritten
    assert "working directory: /Users/x/dev/real-session" in s


def test_captured_reminders_and_ensure_reminders_carry_injected_context():
    # a captured request injects CLAUDE.md/env as <system-reminder> blocks in msg[0];
    # they must be carried into a recorded session that lacks them (non-mutating)
    captured = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "<system-reminder># claudeMd\nprefer ripgrep</system-reminder>"},
        {"type": "text", "text": "reply with just: OK"}]}]}
    rem = eng.captured_reminders(captured)
    assert len(rem) == 1 and "claudeMd" in rem[0]["text"]
    recorded = [{"role": "user", "content": [{"type": "text", "text": "fix server.py"}]}]
    before = json.dumps(recorded)
    merged = eng.ensure_reminders(recorded, rem)
    assert "claudeMd" in json.dumps(merged[0]["content"])       # reminder carried in
    assert "fix server.py" in json.dumps(merged[0]["content"])  # original prompt kept
    assert json.dumps(recorded) == before                        # source not mutated


def test_ensure_reminders_skips_when_already_present():
    rem = [{"type": "text", "text": "<system-reminder># claudeMd\nx</system-reminder>"}]
    already = [{"role": "user", "content": [
        {"type": "text", "text": "<system-reminder># claudeMd\ny</system-reminder>"},
        {"type": "text", "text": "hi"}]}]
    assert eng.ensure_reminders(already, rem) is already  # no duplication


def test_judge_action_quality_parses_verdict(monkeypatch):
    import minmax_bench.quality.engine as e
    monkeypatch.setattr(e, "call_api",
                        lambda *a, **k: ({"content": [{"text": '{"quality":"degraded"}'}]}, None))
    assert e.judge_action_quality("fix the bug", "",
                                  tool("Bash", command="ls"), {})[0] == "degraded"
    monkeypatch.setattr(e, "call_api",
                        lambda *a, **k: ({"content": [{"text": 'garbage'}]}, None))
    assert e.judge_action_quality("t", "", {}, {})[0] is None  # unparseable -> None, not crash


def test_step_verdict_prefers_goal_quality():
    from minmax_bench import counterfactual as cf
    # a structural disagreement that the goal judge rates 'good' -> shows good
    rec = {"step": 0, "orig": tool("Read", file_path="/a"), "replay": tool("Bash", command="rg x"),
           "agree_action": False, "sim": 0.0, "quality": "good"}
    assert cf._step_verdict(rec) == "good"
    rec["quality"] = "bad"
    assert cf._step_verdict(rec) == "bad"


def test_judge_text_match_compares_to_original(monkeypatch):
    import minmax_bench.quality.engine as e
    monkeypatch.setattr(e, "call_api",
                        lambda *a, **k: ({"content": [{"text": '{"quality":"good"}'}]}, None))
    orig = {"type": "text", "text": "ZDR keeps customer data out of retention via zdr_store.py"}
    rep = {"type": "text", "text": "Zero-data-retention is implemented in db/zdr_store.py so..."}
    assert e.judge_text_match(orig, rep, "explain ZDR", {})[0] == "good"


def test_recent_context_finds_the_live_user_question():
    import minmax_bench.quality.engine as e
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "explain ZDR"}]},
        {"role": "assistant", "content": [tool("Read", file_path="/zdr.py")]},
        {"role": "user", "content": [{"type": "tool_result", "content": "class ZDR: ..."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "here's how ZDR works"}]},
    ]
    ctx = e.recent_context(msgs, 3)  # decision point = the assistant answer after the read
    assert "explain ZDR" in ctx                       # the live question is surfaced
    assert "class ZDR" in ctx                          # and the latest observation (tool_result)


def test_common_step_aggregation_is_fair_when_an_arm_stops_early():
    from minmax_bench import counterfactual as cf
    # control hits budget at 2 steps; condense runs 4. Deltas must use the 2 common steps.
    def bs(n, ctx, cost):
        return {"by_step": {s: {"ctx": ctx, "cost": cost, "agree": True} for s in range(n)}}
    arms = {"control": bs(2, 100, 1.0), "condense": bs(4, 40, 0.4)}
    common = cf._common_steps(arms)
    assert common == {0, 1}                                  # only the shared steps
    c = cf._over(arms["condense"], common)
    assert c["n"] == 2 and c["cost"] == 0.8                  # 2 steps, not all 4
    ctrl = cf._over(arms["control"], common)
    # $ vs control over common steps = 1 - 0.8/2.0 = 60% (not 1 - 1.6/2.0 = 20% over own steps)
    assert abs((1 - c["cost"] / ctrl["cost"]) - 0.6) < 1e-9


def test_ccr_step_executes_retrieve_then_scores_the_real_action(monkeypatch):
    import minmax_bench.quality.engine as e
    calls = {"n": 0}
    def fake_call(arm, req, hdr, env):
        calls["n"] += 1
        if calls["n"] == 1:  # first response: the model asks to retrieve
            return {"content": [{"type": "tool_use", "id": "t1", "name": "headroom_retrieve",
                                 "input": {"hash": "abc"}}]}, None
        return {"content": [tool("Read", file_path="/real.py")]}, None  # then the real action
    monkeypatch.setattr(e, "call_api", fake_call)
    monkeypatch.setattr(e, "build_request", lambda *a, **k: {"messages": a[1]})

    class FakeMCP:
        ok = True
        def retrieve(self, args):
            return "full content for " + args["hash"]
    resp, err, nr, oh = e.ccr_step("headroom", {"model": "m"}, [], _args(), "sid", {}, {},
                                   FakeMCP())
    assert err is None and nr == 1                         # one retrieve executed
    assert e.extract_action(resp["content"])["name"] == "Read"  # scored the post-retrieval action


def test_ccr_step_no_mcp_falls_back_to_single_call(monkeypatch):
    import minmax_bench.quality.engine as e
    monkeypatch.setattr(e, "call_api",
                        lambda *a, **k: ({"content": [tool("Bash", command="ls")]}, None))
    monkeypatch.setattr(e, "build_request", lambda *a, **k: {})
    resp, err, nr, oh = e.ccr_step("headroom", {}, [], _args(), "s", {}, {}, None)
    assert nr == 0 and e.extract_action(resp["content"])["name"] == "Bash"
