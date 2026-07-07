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


def test_check_arms_catches_unknown_arm_and_missing_keys():
    problems = eng.check_arms(["control", "headroom-ccr", "condense"], {})
    text = "\n".join(problems)
    assert "headroom-ccr" in text
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
