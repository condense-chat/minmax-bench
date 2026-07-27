"""Tests for the quality / trajectory-preservation bench metric code (minmax_bench/quality).

These metrics produce the repo's public findings — they deserve the same test
surface as the cost bench.
"""
import io
import json
import os
import urllib.error
from types import SimpleNamespace

import pytest

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


def test_incremental_latency_vs_control(tmp_path):
    """Per-step wall-clock is aggregated per arm over the common steps and shown as a signed
    % vs control (+ = slower); artifacts without timing degrade to '—'."""
    d = tmp_path / "incremental"
    d.mkdir()

    def write(arm, recs):
        (d / f"kv-{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))

    def rec(step, lat):
        r = {"step": step, "agree_action": True, "cost_usd": 1.0, "usage": {"input_tokens": 100}}
        if lat is not None:
            r["latency_s"] = lat
        return r

    write("control", [rec(0, 1.0), rec(1, 1.0), rec(2, 1.0)])       # ~1s/step
    write("condense", [rec(0, 1.4), rec(1, 1.6), rec(2, 1.5)])      # steps 1,2 -> 1.55s
    write("headroom", [rec(0, None), rec(1, None), rec(2, None)])   # untimed -> —
    out = report._load_incremental(str(tmp_path), ["condense", "headroom"])
    con = out[("kv", "condense")]
    assert abs(con["latency_ctrl"] - 1.0) < 1e-9 and abs(con["latency"] - 1.55) < 1e-9
    assert "-55%" in report._latency_cell(con)         # 1 - 1.55/1.0 → slower = worse = negative
    assert "—" in report._latency_cell(out[("kv", "headroom")])     # untimed arm -> —


def test_faithful_step_backward_compatible():
    # old artifacts have no 'redundant' key -> degrade gracefully to plain agreement
    assert report._faithful_step({"agree_action": True}) is True
    assert report._faithful_step({"agree_action": True, "redundant": True}) is False
    assert report._faithful_step({"agree_action": False}) is False


def test_faithful_step_uses_goal_quality_when_present():
    """A goal-judged run rates each action on its own merit — so control (which takes valid
    steps) scores ~100%, not the low structural-agreement floor. The report must honour that:
    a 'good' step is faithful even if it structurally disagrees with the original."""
    # good toward the task but did NOT match the original action -> still faithful under goal
    assert report._faithful_step({"quality": "good", "agree_action": False}) is True
    assert report._faithful_step({"quality": "degraded", "agree_action": True}) is False
    assert report._faithful_step({"quality": "bad"}) is False
    # a good step that redundantly re-fetches is still docked
    assert report._faithful_step({"quality": "good", "redundant": True}) is False
    # llm:equiv upgrades a near-miss via agree_semantic when there's no goal quality
    assert report._faithful_step({"agree_action": False, "agree_semantic": True}) is True


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
    assert a["incr"]["scoring"] == "struct"                 # inferred: no LLM judge in records
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
    # normalised to control: fid 0.4 / floor 0.5 = 80% of control's faithfulness
    assert "80%" in passthrough and "green" not in passthrough and "red" not in passthrough
    assert "80%" in engaged and ("green" in engaged or "red" in engaged)  # verdict colour
    assert report._engaged({"comp": 0.01, "retrieves": 3}) is True
    assert report._engaged({"comp": 0.01, "retrieves": 0}) is False
    assert report._engaged({"comp": 0.20, "retrieves": 0}) is True  # compression alone counts


def test_scoring_infers_llm_judge():
    """The scoring label distinguishes a structural match from an LLM goal/equivalence judge."""
    structural = {0: {"agree_action": True, "agree_semantic": True}}
    goal = {0: {"agree_action": True, "quality": "good"}}
    equiv = {0: {"agree_action": False, "agree_semantic": True}}  # upgraded a near-miss
    assert report._scoring(structural) == "struct"
    assert report._scoring(goal) == "llm:goal"
    assert report._scoring(equiv) == "llm:equiv"


def test_check_arms_catches_unknown_arm_and_missing_keys(monkeypatch):
    # force "no subscription token" so the check is hermetic — cc_oauth_token reads the ambient
    # environment (os.environ / .env / keychain), which the dev machine may actually have
    from minmax_bench import auth as _auth
    _auth._CC_TOKEN_CACHE[0] = None
    # and force "no condense creds": condense_creds reads the real ~/.config/dense, which the
    # dev machine may have logged in — pin it to None so the missing-creds branch is exercised
    monkeypatch.setattr(eng, "condense_creds", lambda env: None)
    try:
        # headroom-kompress is a full-mode-only arm — not a valid teacher-forced replay arm
        problems = eng.check_arms(["control", "headroom-kompress", "condense"], {})
        text = "\n".join(problems)
        assert "headroom-kompress" in text
        assert "ANTHROPIC_API_KEY" in text  # anthropic auth missing
        assert "dense login" in text and "CONDENSE_API_KEY" in text  # condense creds missing
        assert not eng.check_arms(["control"], {"ANTHROPIC_API_KEY": "k"})
    finally:
        _auth._CC_TOKEN_CACHE[0] = "unset"


def _pin_dense_home(monkeypatch, home):
    """Point dense.load_profile at a temp home (its default home= is bound at def-time, so
    patch the function to inject home instead of patching DENSE_HOME)."""
    from minmax_bench import dense as dm
    orig = dm.load_profile
    monkeypatch.setattr(dm, "load_profile", lambda name=None: orig(name, home=home))


def test_condense_creds_prefers_dense_profile_over_key(monkeypatch, tmp_path):
    # a logged-in dense profile (prod: token + user files under ~/.config/dense)
    (tmp_path / "token").write_text("tok-from-dense\n")
    (tmp_path / "user").write_text("user-123\n")
    _pin_dense_home(monkeypatch, tmp_path)
    # even with CONDENSE_API_KEY set, the local dense profile wins and carries the user id
    creds = eng.condense_creds({"CONDENSE_API_KEY": "ak_headless"})
    assert creds["token"] == "tok-from-dense" and creds["user"] == "user-123"
    assert creds["url"].endswith("/anthropic")


def test_condense_creds_falls_back_to_key_when_no_profile(monkeypatch, tmp_path):
    _pin_dense_home(monkeypatch, tmp_path)  # empty dense home -> no profile
    creds = eng.condense_creds({"CONDENSE_API_KEY": "ak_headless", "CONDENSE_USER_ID": "u9"})
    assert creds["token"] == "ak_headless" and creds["user"] == "u9"
    assert eng.condense_creds({}) is None  # nothing configured at all


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


def _full_row(task="t", peak=36_000, sub_gate=True, vlens=(9, 9, 10), alens=(8, 8, 8)):
    """One built-report row with enough of a full-run cell for the verdict/table path."""
    def cell(lens):
        srt = sorted(lens)
        return {"_lens": list(lens), "length": (srt[0], srt[len(srt) // 2], srt[-1]),
                "_toks": [1_000_000] * len(lens), "_costs": [0.20] * len(lens),
                "peak_ctx": peak, "n": len(lens), "started": len(lens), "solve": len(lens),
                "attempted": len(lens), "lost": 0, "length_ok": True, "rework_ok": None,
                "milestone": None, "milestone_ok": None, "incr": None}
    v, a = cell(vlens), cell(alens)
    a["length_ok"] = True
    return {"task": task, "sub_gate": sub_gate, "vanilla": v, "arms": {}}


def test_compaction_gate_only_blanks_the_verdict_of_compaction_methods():
    """⊘ says "vanilla never grew big enough to compact, so nothing compacted". That is a real
    excuse for a HISTORY TRANSFORM and no excuse at all for a method that acts from step 1 —
    gating the latter turns a whole small-context run into a column of shrugs while its length,
    tokens and cost sit right there, measured."""
    r = _full_row()
    v, a = r["vanilla"], r["vanilla"]
    assert report._verdict(v, a, "condense", True)[0] == "⊘ too short"
    assert report._verdict(v, a, "headroom", True)[0] == "⊘ too short"
    # unclassified / non-compaction arms get the real verdict
    assert report._verdict(v, a, "vanilla-proxy", True)[0] != "⊘ too short"
    assert report._verdict(v, a, "a-brand-new-arm", True)[0] != "⊘ too short"
    # and above the gate nobody is excused
    assert report._verdict(v, a, "condense", False)[0] != "⊘ too short"
    assert report.gated("condense", True) and not report.gated("condense", False)


def test_summary_keeps_sub_gate_tasks_for_arms_the_gate_does_not_apply_to():
    """summarize() drops ⊘ tasks so a compaction claim isn't made about a task nothing
    compacted. For an arm that always acts, dropping them drops EVERY task — the arm reports no
    full-run quality at all, which reads as 'not measured' when it was measured fine."""
    def row(task):
        return {"task": task, "sub_gate": True,
                "vanilla": {"solve": 2, "attempted": 2, "n": 2},
                "arms": {"condense": {"solve": 1, "attempted": 2, "n": 1, "incr": None},
                         "vanilla-proxy": {"solve": 1, "attempted": 2, "n": 1, "incr": None}}}
    s = report.summarize({"rows": [row("a"), row("b")], "arms": ["condense", "vanilla-proxy"]})
    assert s["condense"]["full"] is None                      # gated: nothing comparable
    assert s["condense"]["full_short"] == ["a", "b"]
    assert s["vanilla-proxy"]["full"][0] == 0.5               # un-gated: a real pooled rate
    assert s["vanilla-proxy"]["full_short"] == []
    assert "too short" not in report._arm_note(s["vanilla-proxy"])


def test_full_table_deltas_read_as_savings_not_raw_change():
    """+ = better, the same convention the incremental table uses. An arm that got 18% cheaper
    printed −18% here, relying on the green to carry the meaning the sign was contradicting."""
    cheaper = report._cmp([10, 10], [8, 8])         # arm used less
    assert round(cheaper["saved"]) == 20 and cheaper["state"] == "good"
    costlier = report._cmp([10, 10], [13, 13])
    assert round(costlier["saved"]) == -30 and costlier["state"] == "bad"
    assert report._cmp([10, 10], [10, 10])["saved"] == 0.0    # never "-0%"
    cell = report._cmp_cell([10, 10], [8, 8], lambda x: f"{x:.0f}")
    assert "+20%" in cell and "10" in cell and "8" in cell


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
    monkeypatch.setattr(e, "cc_oauth_token", lambda: None)
    assert e.auth_mode({"ANTHROPIC_API_KEY": "k"}) == "api-key"
    assert e.auth_mode({}) is None
    monkeypatch.setattr(e, "cc_oauth_token", lambda: "tok")
    assert e.auth_mode({}) == "subscription"
    problems = e.check_arms(["control"], {})
    assert not problems  # subscription satisfies auth


def test_subscription_calls_carry_the_claude_code_identity(monkeypatch):
    """A system-less request on OAuth is not recognisable as Claude Code and gets throttled
    where the replay (which sends the captured system prompt) goes through — that is how a
    working subscription 429s on every judge call. The identity must be added for OAuth only,
    must never overwrite a real system prompt, and must not appear on api-key traffic."""
    sent = {}

    def fake_urlopen(req, *a, **kw):
        sent["body"] = json.loads(req.data)
        raise urllib.error.HTTPError(req.full_url, 400, "stop", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(eng.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(eng, "cc_oauth_token", lambda: "tok")
    monkeypatch.setattr(eng.time, "sleep", lambda *_: None)
    bare = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}

    eng.call_api("control", bare, {}, {})                       # subscription
    assert sent["body"]["system"] == [{"type": "text", "text": eng.CC_IDENTITY}]

    kept = {**bare, "system": [{"type": "text", "text": "captured CC prompt"}]}
    eng.call_api("control", kept, {}, {})                       # replay: left alone
    assert sent["body"]["system"] == kept["system"]

    eng.call_api("control", bare, {}, {"ANTHROPIC_API_KEY": "k"})  # api key: no spoof needed
    assert "system" not in sent["body"]
    assert "system" not in bare                                  # caller's dict never mutated


def test_milestone_judge_separates_a_failed_call_from_an_empty_answer(monkeypatch):
    """`judge returned no usable milestones` is a claim about the judge's OUTPUT. Printing it
    after a 429 sends the reader to the trajectory instead of the rate limit, so the API error
    must survive to the caller — and say the transient thing it is."""
    from minmax_bench.quality import generate as gen

    monkeypatch.setattr(gen.eng, "call_api", lambda *a, **k: (None, "HTTP 429: rate_limit"))
    obj, err = gen._ask("p", {})
    assert obj is None and "429" in err

    monkeypatch.setattr(gen.eng, "call_api",
                        lambda *a, **k: ({"content": [{"text": "no json here"}]}, None))
    obj, err = gen._ask("p", {})
    assert obj is None and err is None            # unparseable is NOT an API failure

    monkeypatch.setattr(gen.eng, "auth_mode", lambda _env: "subscription")
    why = gen._why("HTTP 429: rate_limit", {})
    assert "rate-limited" in why and "subscription" in why and "cached" in why
    assert "quota" in why                          # says what it is NOT, too


def test_judge_honours_auth_subscription_like_the_run_that_produced_it(monkeypatch):
    """`--mode judge` used to skip the key-drop that `full()` does, so re-judging a run made
    with --auth subscription silently went out over the API key instead. A command that
    reproduces on different credentials is useless for debugging an auth failure."""
    from minmax_bench.quality import generate as gen

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    env = {"ANTHROPIC_API_KEY": "k", "OTHER": "keep"}
    kept = gen._apply_auth(SimpleNamespace(auth="auto"), env)
    assert kept["ANTHROPIC_API_KEY"] == "k"                  # auto still prefers the key

    dropped = gen._apply_auth(SimpleNamespace(auth="subscription"), env)
    assert "ANTHROPIC_API_KEY" not in dropped and dropped["OTHER"] == "keep"
    assert "ANTHROPIC_API_KEY" not in os.environ             # harbor must not forward it either
    assert env["ANTHROPIC_API_KEY"] == "k"                   # caller's dict not mutated


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
    prompt, cwd, has_assistant, peak, total, turns, capped = _peek(sess)
    assert has_assistant and peak == 40020 and not capped  # peak, not last, not first
    assert total == 5010 + 40020 + 8020  # total = every turn's context summed
    assert turns == 3  # three usage-bearing assistant decision points


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


def test_redundant_refetch_downranks_goal_quality_one_notch():
    from minmax_bench import counterfactual as cf
    # a redundant re-fetch (compaction amnesia) drops the judge verdict one severity notch
    assert cf._penalize_redundant("good", True) == "degraded"
    assert cf._penalize_redundant("degraded", True) == "bad"
    assert cf._penalize_redundant("bad", True) == "bad"          # already floored
    # non-redundant steps and unknown/None verdicts pass through untouched
    assert cf._penalize_redundant("good", False) == "good"
    assert cf._penalize_redundant(None, True) is None


def test_quality_run_dir_auto_mints_under_configured_root(monkeypatch):
    from minmax_bench import config
    from minmax_bench.quality import paths
    # point the quality runs root somewhere custom (as QUALITY_RUNS_DIR / the wizard would)
    config.get_settings.cache_clear()
    monkeypatch.setenv("QUALITY_RUNS_DIR", "custom/runroot")
    try:
        assert paths.quality_runs_root() == "custom/runroot"
        d1 = paths.new_run_dir("incremental", "sess,name!")
        d2 = paths.new_run_dir("incremental", "sess,name!")
        assert d1.startswith("custom/runroot/incremental/")
        assert "sess-name-" in d1 and "!" not in d1  # slug sanitized
        assert d1 != d2  # a uid suffix guards against same-second collisions
        # discovery roots lead with the configured root and de-dup
        roots = paths.default_run_roots()
        assert roots[0] == "custom/runroot" and len(set(roots)) == len(roots)
    finally:
        config.get_settings.cache_clear()


def test_session_picker_paginates_and_shows_turns(tmp_path):
    from pathlib import Path

    from minmax_bench import counterfactual as cf
    sess = [cf.LocalSession(path=Path(f"/x/s{i}.jsonl"), project=f"/p/{i}", mtime=0.0,
                            size=1, prompt=f"p{i}", cwd="/p", peak_ctx=1000, total_ctx=2000,
                            turns=i) for i in range(cf._PER_PAGE + 3)]
    # a page holds _PER_PAGE rows; a cursor on the 2nd page renders that page's slice
    t0 = cf._session_table(sess, 0)
    t1 = cf._session_table(sess, cf._PER_PAGE)          # cursor into page 2
    assert t0.row_count == cf._PER_PAGE                 # full first page
    assert t1.row_count == 3                            # remainder on page 2
    # turns render as a lower bound when the peek read was capped (negative)
    assert cf._fmt_turns(80) == "80" and cf._fmt_turns(-50) == ">50"


def test_stop_reason_labels_distinguish_api_errors_from_budget():
    from minmax_bench import counterfactual as cf
    # a rate-limit / out-of-credits stop must NOT read as a budget cap
    assert cf._STOP_LABEL["budget"] == "budget cap"
    assert "credits" in cf._STOP_LABEL["out of credits"]
    assert "rate-limited" in cf._STOP_LABEL["rate-limited"]
    assert cf._STOP_LABEL["budget"] != cf._STOP_LABEL["rate-limited"]


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


def test_goal_mode_suppresses_the_same_action_verdict():
    from rich.console import Console

    from minmax_bench import counterfactual as cf

    def summary(judge):
        def arm(n, ctx, agree_frac, good):
            return {"steps_ok": n, "agree_action": int(n * agree_frac), "agree_exact": 0,
                    "quality": {"good": good, "degraded": 0, "bad": n - good},
                    "by_step": {i: {"ctx": ctx, "cost": 0.1, "agree": i < n * agree_frac,
                                    "quality": "good" if i < good else "bad"} for i in range(n)},
                    "avg_ctx_tokens": ctx, "cost_usd": 1.0, "errors": 0}
        return {"session": "/x/s.jsonl", "model": "opus", "steps": 50, "judge": judge,
                "arms": {"control": arm(50, 200, 0.5, 45), "condense": arm(50, 100, 0.25, 46)}}

    def render(judge):
        con = Console(width=100, record=True)
        cf.render_summary(summary(judge), con)
        return con.export_text()

    goal = render("goal")
    # goal is the chosen metric: the same-action "below the floor → trajectory loss" verdict
    # (low same-action reads as failure) must NOT appear; the goal verdict must
    assert "trajectory loss" not in goal and "same action vs original" not in goal
    assert "degrade" in goal                       # the goal-based verdict line is present
    assert "goal-quality" in goal                  # the bottom line names the chosen metric

    # structural is the metric here: the bottom line reports it as same-action FIDELITY (the
    # concise replacement for the old verbose "same action vs original → trajectory loss" prose)
    off = render("off")
    assert "same-action fidelity" in off and "faithful" in off
    assert "trajectory loss" not in off            # still no scary verbose verdict


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


# ---------------------------------------------------------------- shared ctx helpers
def test_ctx_tokens_sums_all_three_usage_tiers():
    u = {"input_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}
    assert eng.ctx_tokens(u) == 125
    assert eng.ctx_tokens({}) == 0


def test_peak_ctx_reads_a_session(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [
        {"type": "assistant", "requestId": "r1",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "a"}],
                     "usage": {"input_tokens": 10, "cache_read_input_tokens": 40}}},
        {"type": "assistant", "requestId": "r2",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "b"}],
                     "usage": {"input_tokens": 10, "cache_read_input_tokens": 990}}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert eng.peak_ctx(str(p)) == 1000


# ---------------------------------------------------------------- run discovery
def test_split_cell_handles_hyphenated_arms():
    assert report.split_cell("headroom-kompress-kv-store-grpc") == ("headroom-kompress",
                                                                    "kv-store-grpc")
    assert report.split_cell("vanilla-proxy-dna-assembly") == ("vanilla-proxy", "dna-assembly")
    assert report.split_cell("vanilla-dna-assembly") == ("vanilla", "dna-assembly")
    assert report.split_cell("mystery-thing") == (None, "mystery-thing")


def test_discover_runs_finds_full_and_incremental(tmp_path):
    full = tmp_path / "jobs" / "runA"
    (full / "vanilla-taskx" / "t1" / "inst" / "verifier").mkdir(parents=True)
    (full / "vanilla-taskx" / "t1" / "inst" / "verifier" / "reward.txt").write_text("1")
    (full / "run-manifest.json").write_text(json.dumps({"model": "claude-test-1"}))
    inc = tmp_path / "inc" / "runB"
    (inc / "incremental").mkdir(parents=True)
    (inc / "incremental" / "sess-control.jsonl").write_text("")
    (inc / "incremental" / "sess-condense.jsonl").write_text("")
    infos = {i["dir"]: i for i in report.discover_runs([str(tmp_path)])}
    a = infos[str(full)]
    assert a["modes"] == ["full"] and a["model"] == "claude-test-1"
    assert a["arms"] == ["vanilla"] and a["tasks"] == ["taskx"]
    b = infos[str(inc)]
    assert b["modes"] == ["incremental"]
    assert set(b["arms"]) == {"control", "condense"} and b["tasks"] == ["sess"]


# ---------------------------------------------------------------- rich-free analysis
def test_render_console_falls_back_to_plain_text_without_rich(tmp_path, capsys, monkeypatch):
    if not os.path.isdir("runs/quality-sample"):
        return  # sample not present in this checkout
    monkeypatch.setattr(report, "HAVE_RICH", False)
    args = SimpleNamespace(tasks="kv-store-grpc", arms="condense", agent="claude-code",
                           ctx_gate=50_000, **{"from": "runs/quality-sample"})
    report.render_console(report.build(args))
    out = capsys.readouterr().out
    assert "| task |" in out and "kv-store-grpc" in out  # the md fallback rendered


# ---------------------------------------------------------------- passthrough control arm
def test_passthrough_proxy_relays_status_headers_and_stream():
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from minmax_bench.quality.passthrough import PassthroughProxy

    class Up(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(n)
            assert self.headers.get("x-api-key") == "sk-test"
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: " + body + b"\n\n")

    up = HTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    try:
        with PassthroughProxy(port=0, upstream=f"http://127.0.0.1:{up.server_address[1]}") as p:
            req = urllib.request.Request(
                f"http://127.0.0.1:{p.port}/v1/messages", data=b'{"m":1}',
                headers={"x-api-key": "sk-test"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                assert r.status == 200
                assert r.headers["content-type"] == "text/event-stream"
                assert r.read() == b'data: {"m":1}\n\n'
    finally:
        up.shutdown()
        up.server_close()


def test_vanilla_proxy_arm_wiring_targets_the_local_forwarder():
    from minmax_bench.quality import generate as gen
    base, allow, agent, extra = gen._arm_wiring("vanilla-proxy", {})
    assert base == f"http://host.docker.internal:{gen.PTPORT}"
    assert allow == "host.docker.internal" and agent == "claude-code" and extra == []


def test_k_for_vanilla_defaults_to_k_plus_one():
    from minmax_bench.quality.generate import _k_for
    args = SimpleNamespace(k=4, k_vanilla=None)
    assert _k_for(args, "vanilla") == 5 and _k_for(args, "condense") == 4
    assert _k_for(SimpleNamespace(k=4, k_vanilla=2), "vanilla") == 2


def test_named_tasks_are_validated_with_a_did_you_mean(monkeypatch, capsys):
    """Regression: comma-separated task names were the ONE --tasks form returned verbatim, so
    a typo survived the wizard, the cost preview and Docker startup and only died inside
    harbor once per cell — after the run had already announced a spend ceiling."""
    monkeypatch.setattr(eng, "dataset_tasks",
                        lambda org="terminal-bench": ["dna-assembly", "protein-assembly",
                                                      "build-pmars"])
    assert eng.resolve_tasks("dna-assembly") == ["dna-assembly"]
    assert eng.resolve_tasks("dna-assembly,build-pmars") == ["dna-assembly", "build-pmars"]

    # a misspelling has a close neighbour -> refuse, and name it
    with pytest.raises(SystemExit) as e:
        eng.resolve_tasks("dna-assmebly")
    msg = str(e.value)
    assert "no such terminal-bench task" in msg and "dna-assembly" in msg

    # the local pool is a SUBSET of the dataset (harbor materializes on demand), so a name
    # with no close neighbour may be real-but-uncached — warn, never block
    assert eng.resolve_tasks("some-brand-new-task-xyz") == ["some-brand-new-task-xyz"]
    assert "not among the 3 terminal-bench tasks" in capsys.readouterr().err

    # nothing cached at all -> nothing to validate against, so don't invent an opinion
    monkeypatch.setattr(eng, "dataset_tasks", lambda org="terminal-bench": [])
    assert eng.resolve_tasks("anything") == ["anything"]
def test_saved_words_reads_as_plain_english():
    from minmax_bench.counterfactual import _saved_words
    assert "saved 40%" in _saved_words(0.40)      # positive delta = a saving
    assert "30% more" in _saved_words(-0.30)      # negative delta = the arm used MORE than control
    assert "~0" in _saved_words(0.001)            # negligible
    assert "—" in _saved_words(None)


def test_incremental_bottom_line_uses_chosen_metric_and_shows_savings():
    """The end-of-run bottom line must surface $ + context SAVED in plain words, and lead with
    the metric the user chose: goal-quality under --judge goal (not the structural `exact`),
    same-action fidelity otherwise."""
    import io

    from rich.console import Console

    from minmax_bench import counterfactual as cf

    def by(n, ctx, cost, agree, quality=None):
        step = {"ctx": ctx, "cost": cost, "agree": agree, "latency": 0.5}
        if quality:
            step["quality"] = quality
        return {i: dict(step) for i in range(n)}

    def summary(judged, cq, aq):
        return {"session": "/x/s.jsonl", "model": "m", "steps": 20, "judge": judged, "arms": {
            "control": {"steps_ok": 6, "agree_action": 6, "agree_exact": 6,
                        "by_step": by(6, 100_000, 1.0, True, cq)},
            "condense": {"steps_ok": 6, "agree_action": 4, "agree_exact": 3,
                         "by_step": by(6, 60_000, 0.6, False, aq)}}}

    # goal: bottom line leads with 'quality' (the chosen metric), never 'faithful'
    buf = io.StringIO()
    cf.render_summary(summary("goal", "good", "good"), Console(file=buf, width=140))
    tail = buf.getvalue().split("bottom line")[1]
    assert "quality" in tail and "faithful" not in tail
    assert "context saved 40%" in tail and "$ saved 40%" in tail

    # structural (judge off): bottom line leads with same-action fidelity
    buf2 = io.StringIO()
    cf.render_summary(summary("off", None, None), Console(file=buf2, width=140))
    tail2 = buf2.getvalue().split("bottom line")[1]
    assert "faithful" in tail2 and "context saved 40%" in tail2


# ---------------------------------------------------------------- overall summary (pooled)
def _incr_dir(tmp_path, task, arms):
    """Write incremental artifacts: arms = {arm: [(good, redundant, ctx), ...]} by step."""
    d = tmp_path / "incremental"
    d.mkdir(exist_ok=True)
    for arm, steps in arms.items():
        recs = [{"step": i, "quality": "good" if g else "bad", "redundant": r,
                 "agree_action": g, "cost_usd": 1.0, "usage": {"input_tokens": c}}
                for i, (g, r, c) in enumerate(steps)]
        (d / f"{task}-{arm}.jsonl").write_text("\n".join(json.dumps(x) for x in recs))


def _built(tmp_path, arms="condense"):
    args = SimpleNamespace(arms=arms, tasks="kv-store-grpc", agent="claude-code",
                           ctx_gate=50_000, **{"from": str(tmp_path)})
    return report.build(args)


def test_summary_pools_quality_and_redundancy_against_control(tmp_path):
    """The overall row is the paired pooled rate, not an average of per-task percentages."""
    ctrl = [(True, False, 100)] * 11 + [(False, False, 100)]      # 10/11 good after step 0
    arm = [(True, False, 40)] * 6 + [(False, True, 40)] * 6       # 5/11 good, 6/11 redundant
    _incr_dir(tmp_path, "sess", {"control": ctrl, "condense": arm})
    s = report.summarize(_built(tmp_path))["condense"]
    assert s["steps"] == 11                                       # step 0 excluded everywhere
    assert abs(s["good"][0] - 5 / 11) < 1e-9
    assert abs(s["good_ctrl"][0] - 10 / 11) < 1e-9
    assert abs(s["red"][0] - 6 / 11) < 1e-9
    assert abs(s["comp"] - 0.6) < 1e-9                            # 40 vs 100 per step
    assert s["good"][1] <= s["good"][0] <= s["good"][2]           # CI brackets the point value


def test_summary_shows_the_paired_delta_instead_of_a_verdict():
    """Each arm cell carries the paired difference and its CI — no cell is marked better or
    worse, because a difference the bootstrap can't separate from zero isn't one."""
    same = [1, 0] * 40
    assert report._boot_pair(same, same[:])[2] == (0.0, 0.0, 0.0)     # identical legs, no delta
    worse = report._boot_pair([0] * 80, same)[2]
    assert worse[2] < 0                                              # a real drop clears zero
    cell = report._cell((0.3, 0.2, 0.4), (-0.5, -0.62, -0.4))         # widest side sets the bar
    assert cell["txt"] == "30.0 ±10.0" and cell["sub"] == "Δ-50.0 ±12.0"


def test_summary_excludes_passthrough_sessions(tmp_path):
    """A session where the arm never compressed is dropped from the incremental columns —
    counting it would pull every arm toward control and read as 'trajectory preserved'."""
    ctrl = [(True, False, 100)] * 6
    _incr_dir(tmp_path, "fired", {"control": ctrl, "condense": [(False, False, 50)] * 6})
    _incr_dir(tmp_path, "flat", {"control": ctrl, "condense": [(True, False, 100)] * 6})
    s = report.summarize(_built(tmp_path))["condense"]
    assert s["sessions"] == ["fired"] and s["skipped"] == ["flat"]
    assert s["good"][0] == 0.0                     # only the engaged session counts


def test_summary_repeats_control_when_arms_cover_different_material(tmp_path):
    """Two arms measured on different sessions cannot share one baseline row: the table must
    repeat control per arm rather than compare an arm to material it never ran."""
    _incr_dir(tmp_path, "a", {"control": [(True, False, 100)] * 6,
                              "condense": [(True, False, 50)] * 6})
    _incr_dir(tmp_path, "b", {"control": [(False, False, 100)] * 6,
                              "headroom": [(False, False, 50)] * 6})
    sr = report.summary_rows(_built(tmp_path, arms="condense,headroom"))
    assert sr["shared"] is False
    assert [r["arm"] for r in sr["rows"]] == ["control", "condense", "control", "headroom"]


def test_summary_full_run_column_counts_lost_trials_as_failures(tmp_path):
    """Solve rate is over trials REQUESTED, so a crashed trial is a failure, not missing data —
    otherwise an arm whose bad trials died would outscore one whose bad trials finished."""
    def row(task, sub_gate):
        return {"task": task, "sub_gate": sub_gate,
                "vanilla": {"solve": 2, "attempted": 2, "n": 2},
                "arms": {"condense": {"solve": 1, "attempted": 2, "n": 1, "incr": None}}}
    s = report.summarize({"rows": [row("t", False)], "arms": ["condense"]})["condense"]
    assert s["full"][0] == 0.5 and s["full_ctrl"][0] == 1.0


def test_summary_full_run_column_drops_tasks_under_the_compaction_gate():
    """A task vanilla never grew big enough to compact cannot carry a compaction-quality
    claim — it is dropped from the pooled column and counted in the open, not averaged in."""
    def row(task, sub_gate):
        return {"task": task, "sub_gate": sub_gate,
                "vanilla": {"solve": 2, "attempted": 2, "n": 2},
                "arms": {"condense": {"solve": 1, "attempted": 2, "n": 1, "incr": None}}}
    s = report.summarize({"rows": [row("long", False), row("tiny", True)],
                          "arms": ["condense"]})["condense"]
    assert s["full_tasks"] == ["long"] and s["full_short"] == ["tiny"]
    assert "⊘1 too short" in report._arm_note(s)


# ---------------------------------------------------------------- wizard: parallel sessions
def _cli_defaults():
    """Every `quality run` parameter at its real default.

    Read off the signature rather than hand-listed: calling a typer command directly leaves
    OptionInfo sentinels in place of defaults, so a parameter added later would silently arrive
    as a sentinel and the test would pass while exercising nothing.
    """
    import inspect

    import typer

    from minmax_bench import cli
    return {name: (p.default.default if isinstance(p.default, typer.models.OptionInfo)
                   else p.default)
            for name, p in inspect.signature(cli.quality_run).parameters.items()}


def _argv_from_wizard(monkeypatch, **wizard_kw):
    """Run `quality run` down its WIZARD branch and return the argv it hands the driver.

    Every parameter is passed explicitly: typer defaults are OptionInfo objects when the
    command function is called directly, so an omitted one is a sentinel, not its value.
    """
    import sys

    from minmax_bench import cli
    from minmax_bench.interactive import QualityWizardResult
    from minmax_bench.quality import generate as gen

    seen = {}
    monkeypatch.setattr(gen, "main", lambda argv: seen.update(argv=argv))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "minmax_bench.interactive.run_quality_wizard",
        lambda _c: QualityWizardResult(mode="full", arms="condense", model="m", out="o",
                                       tasks="2", **wizard_kw))
    cli.quality_run(**_cli_defaults())
    return seen["argv"]


def test_wizard_parallel_sessions_default_is_one():
    """Sequential unless asked for: parallel containers contend for CPU/RAM/disk, which can
    shift the trajectories this bench measures."""
    from minmax_bench.interactive import QualityWizardResult
    assert QualityWizardResult(mode="full", arms="condense", model=None, out="o").concurrency == 1


def test_wizard_parallel_answer_beats_the_flag_default(monkeypatch):
    """The wizard's answer has to win over --concurrency's default. A knob that silently
    stays 1 is worse than no knob, because the run looks like it obeyed."""
    argv = _argv_from_wizard(monkeypatch, k=4, concurrency=3)
    assert argv[argv.index("--concurrency") + 1] == "3"

    argv = _argv_from_wizard(monkeypatch, k=4)                 # not asked -> sequential
    assert argv[argv.index("--concurrency") + 1] == "1"
# ------------------------------------------------------------------ caveman arm (full mode)
def test_caveman_arm_is_not_a_proxy():
    """caveman must reach api.anthropic.com directly. If it ever routes through a proxy it
    silently acquires the ~8-9k non-default-base-URL wiring cost that vanilla-proxy exists to
    isolate, and every token/cost number for the arm becomes incomparable to vanilla."""
    from minmax_bench.quality import generate as gen
    base, allow, agent, extra = gen._arm_wiring("caveman", {})
    assert base == "https://api.anthropic.com" and allow == "api.anthropic.com"
    assert agent == "harbor_agents.caveman_claude_code:CavemanClaudeCode"
    assert extra == ["--ae", "TMB_CAVEMAN_MODE=full"]
    # the mode is a knob, and it must reach the container
    _b, _a, _g, extra_ultra = gen._arm_wiring("caveman", {"TMB_CAVEMAN_MODE": "ultra"})
    assert extra_ultra == ["--ae", "TMB_CAVEMAN_MODE=ultra"]


def test_caveman_bad_mode_is_rejected_before_spending():
    """A bad level must fail at validate time, not after a container build on every cell."""
    from minmax_bench.quality import generate as gen
    args = SimpleNamespace(arms="caveman", dry_run=True)
    with pytest.raises(SystemExit) as e:
        gen._validate_full(args, {"ANTHROPIC_API_KEY": "k", "TMB_CAVEMAN_MODE": "wenyan-full"})
    # wenyan levels switch the OUTPUT LANGUAGE to classical Chinese — excluded on purpose
    assert "wenyan-full" in str(e.value) and "lite, full, ultra" in str(e.value)
    assert gen._validate_full(SimpleNamespace(arms="caveman", dry_run=True),
                              {"ANTHROPIC_API_KEY": "k"}) == ["vanilla", "caveman"]


def test_caveman_is_accepted_by_incremental_and_rides_control_transport():
    """caveman is a client-side policy, not an endpoint: check_arms must accept it (given
    control's anthropic auth) without demanding an ARMS entry, and its requests must go to
    control's transport."""
    assert eng.check_arms(["control", "caveman"], {"ANTHROPIC_API_KEY": "k"}) == []
    assert eng.caveman_api_arm("caveman") == "control"
    assert eng.caveman_api_arm("condense") == "condense"
    # an actually-unknown arm still fails, and the message now lists caveman as known
    prob = "\n".join(eng.check_arms(["control", "bogus"], {"ANTHROPIC_API_KEY": "k"}))
    assert "bogus" in prob and "caveman" in prob


def test_caveman_needs_controls_credentials_not_anthropics(monkeypatch):
    """Regression: credentials belong to the ENDPOINT, and caveman has none of its own.
    Under UPSTREAM_VIA=bedrock control IS the bedrock entry, so `--arms caveman` must ask for
    bedrock auth only — checking the literal arm name demanded an ANTHROPIC_API_KEY the run
    never uses and hard-aborted the preflight for a bedrock-only user."""
    monkeypatch.setitem(eng.ARMS, "control", {"base": "https://bedrock/anthropic",
                                              "bedrock": True})
    monkeypatch.setattr(eng, "auth_mode", lambda env: None)      # no anthropic key/oauth
    # bedrock creds resolve fine — isolate the ANTHROPIC demand, and keep the test off AWS
    monkeypatch.setattr("minmax_bench.bedrock.bearer_token", lambda region: "tok")
    ctrl_only = eng.check_arms(["control"], {})
    with_cav = eng.check_arms(["control", "caveman"], {})
    assert not any("no auth found" in p for p in with_cav), with_cav
    assert with_cav == ctrl_only  # caveman adds no credential demand of its own
    # a real anthropic-endpoint arm still does demand it
    assert any("no auth found" in p for p in eng.check_arms(["condense"], {}))


def test_caveman_ruleset_carries_marker_for_every_mode():
    """The frozen rulesets must each start with the activation marker — a stale/empty capture
    would silently inject nothing and make the arm a no-op."""
    for mode in eng.CAVEMAN_MODES:
        assert eng.caveman_ruleset(mode).startswith("CAVEMAN MODE ACTIVE")
    with pytest.raises(ValueError):
        eng.caveman_ruleset("wenyan-full")  # excluded (classical Chinese confounds metrics)


def test_caveman_inject_prepends_ruleset_into_first_user_turn():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "do the thing"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]
    out = eng.caveman_inject(msgs, "full")
    assert out[0]["content"][0]["text"].startswith("CAVEMAN MODE ACTIVE")
    assert out[0]["content"][1]["text"] == "do the thing"   # original block preserved after
    assert msgs[0]["content"][0]["text"] == "do the thing"  # source untouched (non-mutating)


def test_caveman_state_freezes_rails_and_drifts_only_prose():
    """On a matched action, CavemanState swaps in caveman's terse PROSE but keeps the recorded
    tool_use verbatim (frozen rails) — no id remap — and the recorded tool_result is unchanged
    with its ORIGINAL id, so the pair is valid by construction."""
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "assistant", "content": [   # decision point (recorded)
            {"type": "text", "text": "Let me carefully read the file to understand it."},
            {"type": "tool_use", "id": "rec_1", "name": "Read", "input": {"file_path": "a.py"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "rec_1", "content": "print(1)"}]},
    ]
    st = eng.CavemanState(msgs, [1], "full")
    # caveman's terse rendering of the SAME action (its own tool_use id is irrelevant now —
    # we keep the recorded rail, not caveman's action)
    rep = [{"type": "text", "text": "Read a.py."},
           {"type": "tool_use", "id": "cav_9", "name": "Read", "input": {"file_path": "a.py"}}]
    st.advance(1, rep, matched=True)
    assert st.native == 1 and st.reverted == 0
    asst = st.hist[-2]
    assert asst["content"][0]["text"] == "Read a.py."          # terse prose swapped in
    assert asst["content"][1]["id"] == "rec_1"                 # RECORDED tool_use kept (frozen)
    assert asst["content"][1]["input"] == {"file_path": "a.py"}
    tr = st.hist[-1]["content"][0]                             # recorded result, original id
    assert tr["tool_use_id"] == "rec_1" and tr["content"] == "print(1)"


def test_caveman_state_keeps_terse_prose_even_on_a_diverged_action():
    """caveman's terse prose is kept EVEN when its proposed action disagreed — the prose is a
    reflection on the (frozen) prior result, not a commitment to the next action, so it stays
    coherent with the frozen recorded action. The divergence is counted, not corrected."""
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "I'll read the file carefully to understand the bug."},
            {"type": "tool_use", "id": "rec_1", "name": "Read", "input": {"file_path": "a.py"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "rec_1", "content": "orig result"}]},
    ]
    st = eng.CavemanState(msgs, [1], "full")
    # caveman proposed a DIFFERENT action; its terse prose is kept anyway, the rail is frozen
    st.advance(1, [{"type": "text", "text": "Div by zero line 1."},
                   {"type": "tool_use", "id": "c", "name": "Bash", "input": {"command": "ls"}}],
               matched=False)
    assert st.native == 1 and st.reverted == 0 and st.diverged == 1
    asst = st.hist[-2]
    assert asst["content"][0]["text"] == "Div by zero line 1."   # caveman's terse prose kept
    assert asst["content"][1]["id"] == "rec_1"                   # rail frozen (recorded action)
    assert st.hist[-1]["content"][0]["content"] == "orig result"
    # only an API error (no caveman prose) keeps the recorded prose
    st_err = eng.CavemanState(msgs, [1], "full")
    st_err.advance(1, None, matched=False)
    assert st_err.reverted == 1 and st_err.native == 0
    assert st_err.hist[-2]["content"][0]["text"].startswith("I'll read the file")
    # a step the recording narrated with NO prose is neither native nor reverted — rail intact
    msgs2 = [{"role": "user", "content": [{"type": "text", "text": "t"}]},
             {"role": "assistant", "content": [
                 {"type": "tool_use", "id": "r", "name": "Read", "input": {"file_path": "a"}}]},
             {"role": "user",
              "content": [{"type": "tool_result", "tool_use_id": "r", "content": "x"}]}]
    st2 = eng.CavemanState(msgs2, [1], "full")
    st2.advance(1, [{"type": "text", "text": "Read a."},
                    {"type": "tool_use", "id": "z", "name": "Read", "input": {"file_path": "a"}}],
                matched=True)
    assert st2.native == 0 and st2.reverted == 0          # no prose existed to shrink
    assert st2.hist[-2]["content"][0]["id"] == "r"        # rail verbatim


def test_caveman_split_cell_roundtrips():
    assert report.split_cell("caveman-kv-store") == ("caveman", "kv-store")


def test_caveman_inactive_trials_are_surfaced_not_scored_as_preserved(tmp_path):
    """A trial whose transcript carries no activation marker ran WITHOUT the intervention.
    It is indistinguishable from vanilla, so it must be flagged — otherwise it lands as a
    clean ✓ 'trajectory preserved' while having measured nothing at all."""
    def trial(cell, name, marker):
        inst = tmp_path / cell / name / "inst"
        (inst / "verifier").mkdir(parents=True)
        (inst / "verifier" / "reward.txt").write_text("1")
        sess = inst / "agent" / "sessions" / "projects" / "-app"
        sess.mkdir(parents=True)
        line = {"type": "user", "timestamp": "2026-01-01T00:00:00Z",
                "message": {"role": "user",
                            "content": ("CAVEMAN MODE ACTIVE — level: full" if marker
                                        else "normal session start")}}
        (sess / "s.jsonl").write_text(json.dumps(line) + "\n")

    trial("caveman-taskA", "2026-01-01__00-00-00", marker=True)
    trial("caveman-taskA", "2026-01-01__00-00-01", marker=False)
    idx = report.index_runs(str(tmp_path), "claude-code")
    runs = idx["caveman-taskA"]["runs"]
    assert len(runs) == 2
    assert report._caveman_inactive(runs) == 1        # exactly the unmarked one


def test_caveman_inactive_warning_reaches_every_renderer(tmp_path):
    """md, html and console each build their own tables. A silent-inactivity warning that
    only lands in one of them is worse than none — html is the DEFAULT format, so a reader
    would see a clean ✓ with no hint the skill never loaded."""
    d = {"arms": ["caveman"], "model": "m", "has_milestone": False, "has_incr": False,
         "rows": [{"task": "t", "sub_gate": False,
                   "vanilla": dict(_STAT, _lens=[5, 5], length=(5, 5, 5)),
                   "arms": {"caveman": dict(_STAT, _lens=[5, 5], length=(5, 5, 5),
                                            inactive=2, length_ok=True, rework_ok=True,
                                            milestone=None, milestone_ok=None, incr=None)}}]}
    _head, rows = report.table(d)
    assert any("inactive" in c[0] for row in rows for c in row), "missing from markdown"
    assert "ran without the skill" in report._html_report_body(d), "missing from html"
    from rich.console import Console as _C
    con = _C(file=__import__("io").StringIO(), width=200, force_terminal=False)
    report._full_table(con, d, "m")
    assert "without the skill" in con.file.getvalue(), "missing from console"


_STAT = {"n": 2, "attempted": 2, "lost": 0, "started": 2, "solve": 2, "rework": (0, 0, 0),
         "peak_ctx": 100_000, "_costs": [], "_toks": [], "_lats": [], "_lens": [],
         "length": (0, 0, 0)}


def _tool_pairing_valid(msgs):
    """Every assistant tool_use is answered by a tool_result in the very next user turn, and
    there are no orphan results — the invariant Anthropic 400s on if broken."""
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        use_ids = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
        if not use_ids:
            continue
        nxt = msgs[i + 1]["content"] if i + 1 < len(msgs) else []
        res_ids = [b.get("tool_use_id") for b in nxt if b.get("type") == "tool_result"]
        if sorted(use_ids) != sorted(res_ids):
            return False
    return True


def test_caveman_accumulated_history_stays_api_valid_across_aligned_and_diverged():
    """Terse prose kept on an aligned step and again on a diverged one must still yield a
    message list whose tool_use/tool_result pairing is intact — trivially so, since the rails
    are frozen to the recording either way."""
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "assistant", "content": [                         # step 0 (ALIGNED)
            {"type": "text", "text": "I'll read the file to understand it."},
            {"type": "tool_use", "id": "rec_a", "name": "Read", "input": {"file_path": "a.py"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "rec_a", "content": "A"}]},
        {"role": "assistant", "content": [                         # step 1 (DIVERGED)
            {"type": "text", "text": "Now let me run the tests to confirm."},
            {"type": "tool_use", "id": "rec_b", "name": "Bash", "input": {"command": "pytest"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "rec_b", "content": "B"}]},
    ]
    st = eng.CavemanState(msgs, [1, 3], "full")
    st.advance(1, [{"type": "text", "text": "Read a.py."},
                   {"type": "tool_use", "id": "cav_a", "name": "Read",
                    "input": {"file_path": "a.py"}}], matched=True)     # aligned
    st.advance(3, [{"type": "text", "text": "Tests green."},
                   {"type": "tool_use", "id": "cav_x", "name": "Bash",
                    "input": {"command": "ls"}}], matched=False)        # diverged, prose still kept
    assert st.native == 2 and st.reverted == 0 and st.diverged == 1
    assert st.hist[-2]["content"][0]["text"] == "Tests green."          # terse prose on both
    # every tool_use in the history keeps its RECORDED id (no remap) and its recorded result
    assert {b["id"] for m in st.prefix() if m["role"] == "assistant"
            for b in m["content"] if b.get("type") == "tool_use"} == {"rec_a", "rec_b"}
    assert _tool_pairing_valid(st.prefix()), "caveman history has dangling/orphan tool blocks"
    # and the request built from that history carries the ruleset + is well-formed
    args = SimpleNamespace(max_tokens=16, strip_thinking=False, swechat=None,
                           keep_all_tools=True, drop_beta_config=True)
    req = eng.build_request({"model": "m", "system": [], "tools": []}, st.prefix(), args, "sid")
    assert "CAVEMAN MODE ACTIVE" in req["messages"][0]["content"][0]["text"]
    assert _tool_pairing_valid(req["messages"])


def _fake_session(path):
    """A minimal 2-decision-point Claude Code session jsonl: Read then a final text answer,
    with recorded usage above any sane ctx gate."""
    usage = {"input_tokens": 60_000, "output_tokens": 40, "cache_read_input_tokens": 0,
             "cache_creation_input_tokens": 0}
    def asst(rid, content):
        return {"type": "assistant", "requestId": rid,
                "message": {"role": "assistant", "model": "claude-sonnet-4-6",
                            "usage": usage, "content": content}}
    recs = [
        {"type": "user", "cwd": "/proj", "version": "2.1.0",
         "message": {"role": "user", "content": "fix the bug in a.py"}},
        asst("r1", [
            {"type": "text", "text": "I'll read the file to understand the bug first."},
            {"type": "tool_use", "id": "rec_a", "name": "Read", "input": {"file_path": "a.py"}}]),
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "rec_a", "content": "def f(): return 1/0"}]}},
        asst("r2", [{"type": "text", "text": "The bug is a division by zero on line 1."}]),
    ]
    path.write_text("\n".join(json.dumps(r) for r in recs))


def test_incremental_caveman_end_to_end_offline(tmp_path, monkeypatch):
    """Drive the real replay() loop for --arms caveman with a mocked API: proves the wiring
    (per-arm caveman history → build_request → advance) runs, writes a jsonl with the
    caveman_native flag, and reports the native/reverted split — no network, no spend."""

    from rich.console import Console

    from minmax_bench import counterfactual as cf

    sess = tmp_path / "s.jsonl"
    _fake_session(sess)

    # canned model reply per (arm, step): control mirrors the recording; caveman returns a
    # TERSE version of the SAME first action (aligned), then a divergent second action
    # (diverged — its terse prose is still kept, only the drift is counted).
    def fake_call_api(arm, req, headers, env):
        msgs = req["messages"]
        # step 1 (Read done already) vs step 0: infer by counting assistant turns in prefix
        n_asst = sum(1 for m in msgs if m["role"] == "assistant")
        if n_asst == 0:  # first decision point
            content = [{"type": "text", "text": "Read a.py."},
                       {"type": "tool_use", "id": "new_a", "name": "Read",
                        "input": {"file_path": "a.py"}}]
        else:            # second decision point — caveman diverges (runs a test)
            content = [{"type": "text", "text": "Run test."},
                       {"type": "tool_use", "id": "new_b", "name": "Bash",
                        "input": {"command": "pytest"}}]
        return {"content": content, "usage": {"input_tokens": 30_000, "output_tokens": 20,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}, None

    monkeypatch.setattr(cf.eng, "call_api", fake_call_api)
    monkeypatch.setattr(cf.eng, "auth_mode", lambda env: "api-key")
    # replay()'s api-key guard checks the env directly (not auth_mode); give it a dummy key so
    # the offline run — which mocks call_api and never actually spends — clears the guard.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-offline")

    out = tmp_path / "run"
    summary = cf.replay(sess, ["caveman"], budget_usd=5.0, limit=0, max_tokens=64,
                        out_dir=out, console=Console(quiet=True), assume_yes=True,
                        model=None, auth="api-key", task="t", judge="off", capture=False,
                        ctx_gate=0, caveman_mode="full")

    cav = summary["arms"]["caveman"]
    assert cav["steps_ok"] == 2
    # both steps had prose and caveman responded, so both are kept terse (native); the second
    # step's PROPOSED action diverged from the recording (the drift signal), but its prose is
    # kept anyway and the rail stays frozen
    assert cav["caveman_native"] == 2 and cav["caveman_reverted"] == 0
    assert cav["caveman_diverged"] == 1
    jsonl = (out / "incremental" / "t-caveman.jsonl").read_text().splitlines()
    lines = [json.loads(x) for x in jsonl]
    assert lines[0]["caveman_native"] is True and lines[1]["caveman_native"] is True
    assert lines[0]["replay"]["name"] == "Read"   # scored against the recording


def test_report_reads_caveman_like_condense_on_comp_and_stays_engaged(tmp_path):
    """caveman is read on comp (the accrued-context token measure) like every arm — no special
    'out' column. But because it applied its transform (native), a comp≈0 caveman run must still
    count as ENGAGED (fid a real verdict, not '⊘ passthrough') — its terse prose can move the
    next decision even when the net context change is ~0."""
    d = tmp_path / "incremental"
    d.mkdir()

    def write(arm, recs):
        (d / f"sess-{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in recs))

    def rec(step, ctx, agree=True, native=None):
        r = {"step": step, "agree_action": agree, "cost_usd": 1.0,
             "usage": {"input_tokens": ctx, "output_tokens": 50}}
        if native is not None:
            r["caveman_native"] = native
        return r

    # caveman barely moves the context (30k vs 30.1k) — comp ≈ 0, as on real coding sessions
    write("control", [rec(s, 30_100) for s in range(3)])
    write("caveman", [rec(s, 30_000, agree=(s == 0), native=True) for s in range(3)])
    inc = report._load_incremental(str(tmp_path), ["caveman"])[("sess", "caveman")]
    assert inc["native"] is True
    assert abs(inc["comp"]) < 0.02              # context essentially unchanged (read like condense)
    assert "outd" not in inc                    # the output-token special-casing is gone

    # comp cell is the plain context number — no 'out' tag
    assert "out" not in report._comp_cell(inc)
    # a caveman run that applied its transform stays engaged even at comp≈0
    assert report._engaged(inc) is True

    # md table: no 'out' column, and the comp≈0 caveman row is NOT a ⊘ passthrough (fid shown)
    dd = {"arms": ["caveman"], "model": "m", "has_milestone": False, "has_incr": True,
          "rows": [{"task": "sess", "sub_gate": False,
                    "vanilla": dict(_STAT, _lens=[5, 5], length=(5, 5, 5)),
                    "arms": {"caveman": dict(_STAT, _lens=[5, 5], length=(5, 5, 5),
                                             inactive=None, length_ok=True, rework_ok=True,
                                             milestone=None, milestone_ok=None, incr=inc)}}]}
    head, rows = report.table(dd)
    assert "out" not in head and "comp" in head
    flat = " ".join(c[0] for c in rows[0])
    assert "passthrough" not in flat

    # ...but a run whose EVERY step reverted to the recorded verbose prose applied nothing, and
    # must fall back to ⊘ passthrough. Regression: `native` read the KEY's presence, and
    # counterfactual writes caveman_native on every caveman step (True *or* False), so this run
    # was marked engaged and its fid shown as a verdict while measuring nothing.
    write("caveman", [rec(s, 30_000, agree=(s == 0), native=False) for s in range(3)])
    dead = report._load_incremental(str(tmp_path), ["caveman"])[("sess", "caveman")]
    assert dead["native"] is False
    assert report._engaged(dead) is False
    dd["rows"][0]["arms"]["caveman"]["incr"] = dead
    _h, rows2 = report.table(dd)
    assert "passthrough" in " ".join(c[0] for c in rows2[0])


def test_caveman_empty_prose_keeps_recorded_and_never_makes_an_empty_message():
    """Regression: a prose-only recorded turn + a caveman reply with no text must NOT collapse
    to {"content": []} (which 400s and, being append-only, poisons every later step). Keep the
    recorded prose instead (reverted)."""
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        {"role": "assistant", "content": [  # PROSE-ONLY decision point (no tool_use)
            {"type": "text", "text": "The bug is a division by zero on line 1."}]},
        {"role": "user", "content": [{"type": "text", "text": "thanks, now fix it"}]},
    ]
    st = eng.CavemanState(msgs, [1], "full")
    # caveman replied with a tool_use only — no text block (its no-narration case)
    st.advance(1, [{"type": "tool_use", "id": "c", "name": "Edit", "input": {}}], matched=False)
    assert st.native == 0 and st.reverted == 1
    asst = st.hist[-2]
    assert asst["content"], "assistant message must not be empty (would 400)"
    assert asst["content"][0]["text"].startswith("The bug is")   # recorded prose kept
    assert _tool_pairing_valid(st.prefix())
    # and an API error (rep_content None) on a prose-only turn is handled the same way
    st_err = eng.CavemanState(msgs, [1], "full")
    st_err.advance(1, None, matched=False)
    assert st_err.reverted == 1 and st_err.hist[-2]["content"]


def test_caveman_pin_agrees_across_all_three_sources():
    """The pin lives in three places that can't import each other: the harbor agent
    (CAVEMAN_SHA — what installs), generate.py (CAVEMAN_PIN + probe URL — the preflight), and
    data/caveman/PIN (which release the frozen rulesets came from). A bump in one without the
    others fails SILENTLY (the activation marker survives a version change), so assert they
    agree on both the tag and the sha."""
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # harbor agent — read as text (it imports `harbor`, absent from this env)
    agent_src = open(os.path.join(root, "harbor_agents", "caveman_claude_code.py")).read()
    agent_ref = re.search(r'CAVEMAN_REF\s*=\s*"([^"]+)"', agent_src).group(1)
    agent_sha = re.search(r'CAVEMAN_SHA\s*=\s*"([0-9a-f]{40})"', agent_src).group(1)
    # generate.py — importable (does not import harbor)
    from minmax_bench.quality import generate as gen
    probe_sha = re.search(r"[0-9a-f]{40}", gen.CAVEMAN_TARBALL_PROBE).group(0)
    # data/caveman/PIN — "vX.Y.Z (<40-hex>)"
    pin_txt = open(os.path.join(root, "data", "caveman", "PIN")).read().strip()
    pin_ref, pin_sha = re.match(r"(\S+)\s*\(([0-9a-f]{40})\)", pin_txt).groups()

    assert agent_ref == gen.CAVEMAN_PIN == pin_ref, "pin TAG disagrees across sources"
    assert agent_sha == probe_sha == pin_sha, "pin SHA disagrees across sources"


def _caveman_readout(native, reverted, diverged, stapled=None):
    """render_summary's caveman line for a run with these counts."""
    import io

    from rich.console import Console

    from minmax_bench import counterfactual as cf

    def by(n):
        return {i: {"ctx": 100_000, "cost": 1.0, "agree": True, "latency": 0.5} for i in range(n)}
    s = {"session": "/x/s.jsonl", "model": "m", "steps": 3, "judge": "off", "arms": {
        "control": {"steps_ok": 3, "agree_action": 3, "agree_exact": 3, "by_step": by(3),
                    "ctx_series": [100_000] * 3, "errors": 0, "stop_reason": "complete"},
        "caveman": {"steps_ok": 3, "agree_action": 2, "agree_exact": 1, "by_step": by(3),
                    "ctx_series": [100_000] * 3, "errors": 0, "stop_reason": "complete",
                    "caveman_native": native, "caveman_reverted": reverted,
                    "caveman_diverged": diverged, "caveman_stapled": stapled}}}
    buf = io.StringIO()
    # very wide: the readout must not wrap, else filtering by "caveman:" drops its continuation
    # lines and an assertion on the tail of the sentence silently passes/fails on layout
    cf.render_summary(s, Console(file=buf, width=10_000))
    return [ln for ln in buf.getvalue().splitlines() if "caveman:" in ln]


def test_caveman_readout_never_claims_a_terse_history_it_did_not_have():
    """Regression: caveman_reverted was tracked and then never rendered, so a run whose prose
    fell back to the recorded VERBOSE narration still read as '% action-aligned over a
    fully-terse history'. With native=0 that printed a red 0% over 0/0 — the incremental twin
    of the full-mode '⚠ inactive' failure, reported as a result instead of a non-measurement."""
    # nothing ran terse: say INACTIVE, never a 0% rate over an empty base
    dead = " ".join(_caveman_readout(native=0, reverted=6, diverged=0))
    assert "inactive" in dead and "0 steps ran on terse prose" in dead
    assert "6 reverted" in dead
    assert "fully-terse" not in dead and "action-aligned" not in dead

    # partly reverted: the mixture is named, not papered over as "fully-terse"
    mixed = " ".join(_caveman_readout(native=4, reverted=2, diverged=1))
    assert "75% action-aligned" in mixed          # 3 of 4 terse steps proposed the recording
    assert "2 step(s) reverted" in mixed and "fully-terse" not in mixed

    # clean run: unchanged wording
    clean = " ".join(_caveman_readout(native=4, reverted=0, diverged=1))
    assert "75% action-aligned" in clean and "over a fully-terse history" in clean
    assert "reverted" not in clean


def test_caveman_state_splits_drift_by_whether_a_frozen_action_was_there_to_clash_with():
    """`diverged` averages two step shapes that mean opposite things. On a turn with a frozen
    tool_use (~60% of real decision points) the kept terse prose ends up beside an action it was
    not written for — the residual artifact. On a prose-only turn (~9%) the whole message is
    replaced, so a divergence is clean disagreement. Only `stapled` tells the two apart."""
    def turn(prose, tool):
        c = ([{"type": "text", "text": prose}] if prose else [])
        if tool:
            c.append({"type": "tool_use", "id": tool, "name": "Read", "input": {"file_path": "a"}})
        return {"role": "assistant", "content": c}

    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "task"}]},
        turn("I'll read the file to understand the bug.", "rec_a"),   # text+tool  -> stapled
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "rec_a",
                                      "content": "A"}]},
        turn(None, "rec_b"),                                          # tool-only  -> invisible
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "rec_b",
                                      "content": "B"}]},
        turn("The bug is a division by zero on line 1.", None),       # text-only  -> clean
    ]
    st = eng.CavemanState(msgs, [1, 3, 5], "full")
    terse = [{"type": "text", "text": "Terse."}]
    st.advance(1, terse, matched=False)   # diverged ON a frozen action
    st.advance(3, terse, matched=False)   # diverged on a turn with no prose at all
    st.advance(5, terse, matched=False)   # diverged on a prose-only turn

    assert st.native == 2                 # the tool-only turn had no prose to shrink
    assert st.diverged == 2               # ...so it cannot drift either
    assert st.stapled == 1, "only the text+tool turn leaves prose beside a frozen action"
    # the stapled turn is exactly the artifact: terse prose, recorded action, unchanged result
    assert st.hist[1]["content"][0]["text"] == "Terse."
    assert st.hist[1]["content"][1]["id"] == "rec_a"
    # the prose-only turn is replaced wholesale — nothing to clash with
    assert st.hist[-1]["content"] == [{"type": "text", "text": "Terse."}]
    # agreement on an action turn is not stapled, by definition
    st2 = eng.CavemanState(msgs, [1], "full")
    st2.advance(1, terse, matched=True)
    assert st2.diverged == 0 and st2.stapled == 0


def test_caveman_readout_splits_stapled_from_clean_drift():
    """The readout must not report one drift number: a run can drift a lot with zero artifact,
    or a little with all of it stapled beside a frozen action."""
    mixed = " ".join(_caveman_readout(native=8, reverted=0, diverged=4, stapled=1))
    assert "4 drifted" in mixed
    assert "1 beside a different frozen action" in mixed
    assert "3 on a prose-only turn (clean)" in mixed
    # older artifacts (pre-split) carry no stapled count — degrade to the plain number
    old = " ".join(_caveman_readout(native=8, reverted=0, diverged=4, stapled=None))
    assert "4 drifted" in old and "frozen action" not in old


def test_caveman_readout_states_its_denominator():
    """action-aligned % is over PROSE steps; the table's 'vs original' is over every successful
    step. ~30% of real decision points are tool-only (no prose — cannot be terse, cannot drift),
    so the two legitimately disagree. Unexplained, one of them reads as wrong."""
    # helper builds a 3-successful-step arm; 2 of those steps carried prose
    line = " ".join(_caveman_readout(native=2, reverted=0, diverged=1))
    assert "2 of 3 successful steps had prose" in line
    assert "cannot drift" in line and "vs original" in line

