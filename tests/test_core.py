from minmax_bench.data.sample import sample_sessions
from minmax_bench.harness import simulate
from minmax_bench.models import Usage
from minmax_bench.pricing import cost_usd, rates_for
from minmax_bench.report import PairedRow, summarize


def test_simulator_makes_one_point_per_assistant_turn():
    session = sample_sessions()[0]
    points = simulate(session)
    # sample has 4 assistant turns
    assert len(points) == 4
    # each point's prefix ends right before its assistant turn and grows
    assert [len(p.prefix) for p in points] == [1, 3, 5, 7]
    assert all(p.expected_output.role.value == "assistant" for p in points)


def test_simulator_coalesces_split_assistant_turn():
    from minmax_bench.models import Block, Message, Role, Session, Usage

    # A source that split one assistant turn (text row, then tool_use row).
    session = Session(
        id="split:0",
        source="test",
        messages=[
            Message(role=Role.user, blocks=[Block.text_block("hi")]),
            Message(role=Role.assistant, blocks=[Block.text_block("thinking")],
                    recorded_usage=Usage(input_tokens=500)),
            Message(role=Role.assistant, blocks=[Block.tool_use("t1", "Read", {"file_path": "a"})],
                    recorded_usage=Usage(input_tokens=100)),
            Message(role=Role.tool, blocks=[Block.tool_result("t1", "ok")]),
            Message(role=Role.assistant, blocks=[Block.text_block("done")]),
        ],
    )
    points = simulate(session)
    # 2 real calls, not 3: the split assistant rows merge into one turn.
    assert len(points) == 2
    # merged turn keeps the larger-input recorded usage (the real call)
    assert points[0].recorded_usage.input_tokens == 500
    assert len(points[0].expected_output.blocks) == 2  # text + tool_use


def test_chain_usage_differencing_cache_model():
    from minmax_bench.chain import chain_usages

    session = sample_sessions()[0]
    points = simulate(session)

    class FakeCounter:
        # pretend the prompt grows 100 tokens per turn
        def __init__(self):
            self.t = 0

        def count(self, s, msgs):
            self.t += 100
            return self.t

    usages = chain_usages(session, points, FakeCounter(), lambda p: 0, rewrite=None)
    # first turn: all new (cache_write), no cache_read; later turns: prior is cache_read
    assert usages[0].cache_read == 0 and usages[0].cache_write == 100
    assert usages[1].cache_read == 100 and usages[1].cache_write == 100
    # newly-sent tokens are cache_write, not double-counted as input
    assert usages[1].input_tokens == 0


def test_with_test_run_stamps_and_rotates():
    from minmax_bench.harness import with_test_run

    session = sample_sessions()[0]
    a = with_test_run(session)
    b = with_test_run(session)
    # a fresh uuid every call (rotates) — independent reruns
    assert a.test_uuid and b.test_uuid and a.test_uuid != b.test_uuid
    # stamped into the system prompt (busts content-keyed proxy cache)
    assert a.test_uuid in (a.system or "")
    # original session untouched; original system still present in the copy
    assert session.test_uuid is None and (session.system or "") in (a.system or "")
    # fixed-length marker -> same-length uuids -> baseline token value stable
    assert len(a.system) == len(b.system)




def test_pricing_cache_tiers_cheaper():
    r = rates_for("claude-sonnet-4-5")
    assert r.cache_read < r.input < r.output
    u = Usage(input_tokens=1000, output_tokens=100, cache_read=5000, cache_write=1000)
    assert cost_usd("claude-sonnet-4-5", u) > 0


def test_anthropic_sanitizer_drops_orphans_and_fixes_user_first():
    from minmax_bench.models import Block, Message, Role, Session
    from minmax_bench.providers import render_anthropic

    session = Session(id="mid", source="test", messages=[])
    # prefix begins mid-conversation: orphaned tool_result, then a proper pair
    prefix = [
        Message(role=Role.tool, blocks=[Block.tool_result("orphan", "x")]),
        Message(role=Role.assistant, blocks=[Block.tool_use("t1", "Read", {"f": "a"})]),
        Message(role=Role.tool, blocks=[Block.tool_result("t1", "content")]),
    ]
    msgs = render_anthropic(session, prefix, max_tokens=1)["messages"]
    assert msgs[0]["role"] == "user"  # user-first guaranteed
    # no orphaned tool_result survives
    ids = [
        b.get("tool_use_id")
        for m in msgs
        for b in m["content"]
        if b.get("type") == "tool_result"
    ]
    assert "orphan" not in ids and "t1" in ids


def test_sanitizer_pairs_unpaired_tool_use():
    from minmax_bench.models import Block, Message, Role, Session
    from minmax_bench.providers import render_anthropic

    session = Session(id="gap", source="test", messages=[])
    # assistant tool_use whose tool_result is MISSING, then the run continues
    prefix = [
        Message(role=Role.user, blocks=[Block.text_block("go")]),
        Message(role=Role.assistant, blocks=[Block.tool_use("t1", "Read", {"f": "a"})]),
        Message(role=Role.user, blocks=[Block.text_block("next")]),  # no tool_result for t1
    ]
    msgs = render_anthropic(session, prefix, max_tokens=1)["messages"]
    # a placeholder tool_result for t1 now immediately follows the assistant turn
    for i, m in enumerate(msgs):
        if any(b.get("type") == "tool_use" and b.get("id") == "t1" for b in m["content"]):
            nxt = msgs[i + 1]
            assert nxt["role"] == "user"
            assert any(b.get("tool_use_id") == "t1" for b in nxt["content"])
            break


def test_long_context_pricing_tier():
    from minmax_bench.pricing import cost_usd

    small = Usage(input_tokens=100_000)
    big = Usage(input_tokens=300_000)  # above the 200k long-context threshold
    # per-token cost should be strictly higher in the long-context regime
    assert cost_usd("claude-sonnet-4-5", big) / 300_000 > cost_usd(
        "claude-sonnet-4-5", small
    ) / 100_000


def test_summarize_buckets_and_percentages():
    m = "claude-sonnet-4-5"
    rows = [
        PairedRow("s", 0, 2000, m, Usage(input_tokens=2000), Usage(input_tokens=1000)),
        PairedRow("s", 1, 20000, m, Usage(input_tokens=20000), Usage(input_tokens=10000)),
    ]
    buckets = summarize(rows)
    all_bucket = buckets[-1]
    assert all_bucket.label == "ALL" and all_bucket.n == 2
    assert abs(all_bucket.pct_tokens_saved_prompt - 50.0) < 1e-6


def test_cache_roundtrip_and_recompute(tmp_path):
    from minmax_bench.cache import MeasurementCache, cache_key
    from minmax_bench.executors.base import Measurement
    from minmax_bench.report import measurements_from_json, measurements_to_json, recompute_buckets

    # measurement cache: put then get returns same usage
    c = MeasurementCache(tmp_path / "c.json")
    k = cache_key("headroom", "claude-sonnet-4-5", 1)
    ms = [Measurement(0, "sess", Usage(input_tokens=100, cache_read=50), 0.0)]
    c.put(k, "sess", ms)
    c.save()
    c2 = MeasurementCache(tmp_path / "c.json")
    got = c2.get(k, "sess", 1)
    assert got and got[0].usage.input_tokens == 100 and got[0].usage.cache_read == 50

    # measurements json roundtrips and recomputes identical buckets
    rows = [PairedRow("s", 0, 20000, "claude-sonnet-4-5",
                      Usage(input_tokens=20000), Usage(input_tokens=8000))]
    j = measurements_to_json({"headroom": rows})
    back = measurements_from_json(j)["headroom"]
    b = recompute_buckets({"headroom": back})["headroom"][-1]
    assert abs(b.pct_tokens_saved_prompt - 60.0) < 1e-6


def test_parse_token_count_forms():
    from minmax_bench.tokens import parse_token_count
    assert parse_token_count("190k") == 190_000
    assert parse_token_count("1.5m") == 1_500_000
    assert parse_token_count("50,000") == 50_000
    assert parse_token_count("") is None
    assert parse_token_count(None) is None
    assert parse_token_count("garbage") is None


def test_wizard_resolve_tasks_safe_returns_error_instead_of_exiting():
    from minmax_bench.interactive import _resolve_tasks_safe
    tasks, why = _resolve_tasks_safe("random:notanumber")
    assert tasks is None and "random" in why
