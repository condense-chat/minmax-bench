# Optimization Evaluation Guide

# Part 1: Building an Optimization Benchmark

The measurement target of a cost optimization strategy is simple to phrase:

> Can an agent accomplish the same tasks more cheaply with a chosen strategy than without it?

This carries a couple of assumptions:
- There exists a good benchmark that models our problem space
- A given agentic system will predictably solve the same task (with a stable success probability)
- Identical tasks on an identical agent are comparable in cost and execution

With these assumptions, our evaluation simply becomes:

```
Agent :: "Some agentic loop system: harness + model"
Eval  :: Task -> Agent -> (Score, Cost)
vanilla_score,  vanilla_cost  = Eval(task, agent)
strategy_score, strategy_cost = Eval(task, Optimizer(agent))

vanilla_score ~= strategy_score
vanilla_cost  >= strategy_cost
```

But each assumption carries critical details that can lead
to a non-rigorous experimental setup, so we will deconstruct each one.

## Choosing a benchmark

A benchmark you can trust is the first critical part of a good experimental
setup.

You need a benchmark that tests actual agentic loops, not just a
simple Q&A that queries a model's knowledge or its ability to retrieve it.

Ideally, the benchmark should be in a domain similar to where the chosen agent will actually
be deployed.


In our case, we chose [Terminal-Bench](https://github.com/laude-institute/terminal-bench), as it is one of the most trusted agentic
benchmarks and spans a wide variety of tasks. We did not choose it because it
suits our optimizer; we chose it because it meets the criteria above:

- **It tests real agentic loops.** The agent works in a live terminal, running
  commands, reading their output and iterating, rather than answering a question.
- **It has a hard signal.** Every task ships with a verifier, so "did it solve
  the task" is decided by tests, not by opinion.
- **It varies in length.** Tasks range from a handful of steps to long sessions
  where the context grows large, which is where cost optimizations actually act.
- **It is independent of us.** We don't write or tune the tasks, and model
  providers use it to report their own results, so a good score can't be
  something we built the benchmark to produce.

We sometimes use other benchmarks, like [SWE-bench](https://www.swebench.com), as well.

When preparing for a client, we try to determine what models they primarily use,
the types of tasks their agents accomplish, and the length of those tasks.
The length of a task itself (i.e. the number of LLM calls) and the percentage
of generated vs. injected tokens largely determine how much our strategy can
reduce the cost.


Other companies, especially for coding tasks, build their own SWE-bench-like
tasks: they gather historical PRs, tasks, and bug fixes, and build
TASK -> UNIT TEST -> CODE CHANGE datasets, using unit tests as a hard signal
and code patch comparison as a soft signal for the quality of the code itself.

This allows these companies to ensure that any agentic system
actually performs well on the problems the company is actually solving.

## Agent Task Predictability

Having a good benchmark but running each task only once on each arm
will give you sham results.

In our experimentation, even a plain model system needs a significant number of
runs to stabilize. As the complexity of the problem increases, and with it
the number of turns in the loop, the more repeats you need before the
standard deviation starts to stabilize.

For minimal stability, we often see that 8 repeats is the minimum in such a
setup, but not all tasks are created equal.

An interesting example is the Terminal-Bench task "path-tracing", where the
model is tasked with looking at an image and writing code to recreate it.

If you run it multiple times, you will see a bimodal
distribution.
If you investigate the executions and trajectories (the subsequent "steps" the model takes),
you can notice that the model can partially cheat, as there is a binary version of the program.

Because of this, in some runs the model disassembles the binary and writes C code directly from it.
In others, it looks at the image, produces its code, then produces verification code
and iterates on the code until it matches.

As you can probably imagine, smarter models look great on this now: they solve it in ~6-8 steps,
while older ones sometimes take 18 and sometimes 100.

This challenges the second and third assumptions. If a task is time-limited or turn-limited,
a model that took the honest path could fail, while a "cheating" one would
succeed.

And comparing cost would be basically impossible.

This leads to an additional gate that must be considered.

## Trajectory Gating

Instead of just running hundreds of runs and wasting money and time,
it's often best to also compare the trajectories themselves after the runs,

and then compare the cost and score of similar trajectories.

A shift in the distribution of trajectories is a signal in itself, showing that
the optimizer had some impact on the model's actions.

## Conclusions

This leads to a more refined shape of evaluation:

> Is the optimizer affecting the trajectories and solve rates of the agentic system?

# Part 2: Using an Optimization Benchmark

[minmax-bench](https://github.com/condense-chat/minmax-bench) is our take on the benchmark described in Part 1.
It treats a cost optimization as a two-sided trade: **cost** is the
*minimize* target, **quality** is the *maximize* target, and neither means
anything without the other.

The Part 1 conclusion (the optimizer can change the trajectory itself) splits
the evaluation into three types. Each one answers a different question, and each
one relies on a different subset of the Part 1 assumptions.

## 1. Same-trajectory cost evaluation (the ideal case)

> If the agent takes exactly the same steps, how much cheaper is each step?

This is the ideal case from our first formula: we assume the trajectory is
fixed, and only measure cost. A recorded session is chopped into the exact
sequence of model calls a real harness would make, and each call is sent
through the strategy and compared against the uncompressed baseline for the
same `(session, turn)`.

Because the trajectory is held fixed, there is no variance to beat: the same
transcript always yields the same request points, so one pass is enough. That
makes it cheap to run on large, real datasets (e.g. 64 sessions / ~11.7k
turns), and in `--mode rewrite` it costs nothing at all.

The one detail that must not be skipped is caching. A strategy that rewrites
history can invalidate the prompt cache, trading cheap cache reads for
expensive cache writes. **Token savings are not dollar savings**, so the cost
has to be priced cache-aware, not counted in tokens.

What it can't tell you: whether the agent would actually have taken those
steps. It assumes quality is free, and the other two evaluations exist to check
that assumption.

```bash
uv run minmax-bench cost run -d swe-chat:64 -s condense-sync -s headroom --mode rewrite
```

Methodology: [docs/cost.md](https://github.com/condense-chat/minmax-bench/blob/main/docs/cost.md).

## 2. Incremental: "would the model take the same step?"

> Given the optimized history, does the model still make the same next decision?

Long trajectories are exactly where optimizers pay off, and exactly where
running full repeats becomes too expensive. The incremental evaluation avoids
this by teacher-forcing a recorded session: at every step, both the control and
the strategy arm see the same recorded history (the arm sees its optimized
version), and we compare the next action each one proposes.

This makes the comparison deterministic and paired, step by step, so it isolates
*informational* loss: did the optimization drop something the model needed for
its next decision? Agreement is read against the control's own agreement (two
control runs don't agree 100% either), so only the gap below that noise floor is
signal.

What it can't tell you: *behavioral* effects that compound. The recorded tool
results are replayed, not executed, so a diverged step never actually plays out
and its downstream effect can't propagate. It also only works on sessions long
enough for the optimizer to engage (hence the `--ctx-gate`).

```bash
uv run minmax-bench quality incremental ~/.claude/projects/<proj>/<id>.jsonl --arms condense -n 30
```

## 3. Full runs: the honest, dumb approach

> Can the agent accomplish the same tasks more cheaply with the strategy than without it?

This is the initial hypothesis from Part 1, tested directly: run the full agent
end-to-end on real tasks (Terminal-Bench through [Harbor](https://github.com/laude-institute/harbor)), for the vanilla arm and
each strategy arm, `k` times each. It is the closest to reality and the most
expensive, and it's the only evaluation that catches behavioral changes: turn
count inflation, induced planning, lost solves.

Everything from Part 1 applies here:

- **Repeats:** `--k` trials per arm (default 4), with an extra vanilla run to
  sharpen the noise floor. A verdict needs at least 2 finished runs per arm.
- **Noise floor, not identity:** the bar is not "identical to vanilla" but
  "within the vanilla-vs-vanilla spread."
- **Quality axes:** solve rate and milestone coverage decide whether quality
  held; length is a cost axis (longer is a real signal, shorter is the point
  of the optimizer); rework catches compaction amnesia.
- **Gating:** tasks whose context never grows large enough for the optimizer to
  fire are marked ⊘, since a difference there measures wiring side effects, not
  the optimization.

What it can't tell you: *why* a trajectory changed. And with a small `k`, band
overlap only rules out gross divergence; it is not proof of equivalence.

```bash
uv run minmax-bench quality run --arms condense,headroom --tasks long --k 4 --milestones
```

Methodology: [docs/quality.md](https://github.com/condense-chat/minmax-bench/blob/main/docs/quality.md).

## Reading them together

| evaluation | question | trajectory | cost to run | blind to |
|---|---|---|---|---|
| same-trajectory cost | how much cheaper per step? | fixed | low (zero in rewrite mode) | any quality change |
| incremental | same next step? | teacher-forced | medium | compounding behavior |
| full runs | same tasks, cheaper? | free-running | high (k repeats) | the cause of a change |

Start with the cost evaluation to see if there are savings worth chasing, use
incremental to check that those savings don't lose information on long
sessions, and confirm with full runs that the agent still solves the same tasks
along similar trajectories. **Cost tells you what a strategy saves; quality
tells you whether those savings are real.**
