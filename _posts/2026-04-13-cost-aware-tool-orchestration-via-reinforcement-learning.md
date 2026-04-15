---
layout: post
title: "Teaching LLM Agents to Think Before They Spend: Cost-Aware Tool Orchestration via Reinforcement Learning"
date: 2026-04-13
authors: "Andrew Lara, Yashawasi Sharma (University of Southern California)"
description: "We introduce CostAwareToolEnv, an RL environment where LLM agents learn to balance task accuracy against tool invocation costs. Trained with GRPO on four diverse benchmarks, our agent discovers non-trivial cost-accuracy trade-offs that static prompting strategies miss entirely."
categories: [ai, engineering, projects]
tags: [reinforcement-learning, llm-agents, tool-use, grpo, cost-optimization]
math: true
---

{% if page.math %}
<script>
  window.MathJax = {
    tex: { inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$', '$$'], ['\\[', '\\]']] },
    svg: { fontCache: 'global' }
  };
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
{% endif %}

*Built for the [Berkeley RDI AgentX-AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats) (Research Track). Code and datasets available on [Hugging Face](https://huggingface.co/).*

---

## The $20-Million-a-Day Problem

Large language model agents are getting better at using tools. They can search the web, execute code, query databases, and call specialized APIs, often in multi-step chains that would have seemed like science fiction three years ago. But there is an underexplored cost to all of this capability: it is expensive, and current agents are terrible at managing that expense.

Consider a simple question-answering task. An LLM agent with access to a web search API, a code interpreter, a knowledge graph, a calculator, a retrieval engine, and a document reader could plausibly use *any* of these tools, or several in sequence, to answer the question. Some of those tools are cheap (a calculator costs essentially nothing). Others are expensive (a web search API call with re-ranking might cost 100x more). The agent, having been trained or prompted purely for accuracy, will default to the most powerful tool available regardless of whether the question actually requires it.

This is not a hypothetical problem. Enterprise LLM deployments are already grappling with API cost management at scale, and the most common optimization strategies (prompt compression, caching, model cascading) operate *outside* the agent's decision loop. They treat the agent as a black box and try to reduce costs around its edges.

We asked a different question: **what if the agent itself learned when to spend and when to save?**

## CostAwareToolEnv: The Setup

We built CostAwareToolEnv as a Gymnasium-compatible RL environment where an LLM agent must solve tasks by selecting from a toolkit of six tools, each with a different cost profile:

| Tool | Cost | Capability |
|------|------|------------|
| `calculator` | $0.001 | Arithmetic and symbolic computation |
| `retriever` | $0.01 | Retrieval-augmented generation over a local corpus |
| `code_interpreter` | $0.05 | Python execution sandbox |
| `knowledge_graph` | $0.08 | Structured entity and relation queries |
| `web_search` | $0.10 | Live web search with snippet extraction |
| `expert_model` | $0.50 | Call to a larger, more capable LLM |

These costs are meant to reflect relative real-world API pricing (not exact dollar amounts). The key design choice is that the cost differences span *three orders of magnitude*: a ratio of 500:1 between the cheapest and most expensive tool. This forces the agent to develop non-trivial strategies. You cannot just "always pick the cheap one" because many tasks genuinely require expensive tools. But you also cannot mindlessly escalate to `expert_model` on every query without blowing your budget.

### The Reward Function

The agent's reward combines correctness and cost in a single scalar:

$$R = \alpha \cdot \mathbb{1}[\text{correct}] - (1 - \alpha) \cdot \frac{C_{\text{used}}}{C_{\text{max}}}$$

where $C_{\text{used}}$ is the total cost of tools invoked during the episode, $C_{\text{max}}$ is the maximum possible cost (using `expert_model` for every step), and $\alpha$ is a tunable parameter controlling the accuracy-cost trade-off. In our experiments we primarily use $\alpha = 0.7$, which tells the agent that correctness matters more than frugality, but not infinitely more.

The $\frac{C_{\text{used}}}{C_{\text{max}}}$ normalization is important. Without it, the cost penalty is on a completely different scale than the correctness reward, making the trade-off hard to learn. Normalizing to $[0, 1]$ puts both components on equal footing before the $\alpha$ weighting.

### Four Benchmarks, Four Domains

We evaluate across four established benchmarks, chosen to require genuinely different tool-use strategies:

**HotpotQA**: Multi-hop question answering. Requires chaining evidence across multiple documents. The retriever is often sufficient, but some questions benefit from web search when the local corpus is incomplete.

**MATH**: Competition-level mathematics. The calculator handles arithmetic, but harder problems require the code interpreter (for symbolic algebra) or even the expert model (for proof-level reasoning).

**GPQA (Graduate-level Professional QA)**: Expert-domain questions in physics, chemistry, and biology. These push the boundaries of what smaller models can handle, making the `expert_model` tool a tempting but costly crutch.

**HumanEval**: Code generation. The code interpreter is the obvious tool, but the agent also needs to decide whether to attempt generation directly or consult the expert model for harder functions.

Each benchmark is processed into a JSONL format with fields for the question, ground truth answer, a difficulty estimate (used for analysis, not training), and metadata about which tools are *plausibly relevant* (used to construct the episode, not revealed to the agent).

## Why GRPO?

We train with Group Relative Policy Optimization (GRPO), the algorithm introduced in the DeepSeekMath paper (Shao et al., 2024) and subsequently used to train DeepSeek-R1.

The case for GRPO over PPO in this setting comes down to three practical considerations:

**No critic model needed.** PPO requires a separate value network (the critic) that estimates expected future reward. For LLM-scale agents, this means maintaining and training a second model of comparable size, doubling memory requirements. GRPO eliminates the critic entirely by estimating advantages from the relative quality of sampled completions within each batch.

**Natural fit for verifiable rewards.** Our environment produces deterministic rewards: the answer is either correct or it is not, and the cost is exactly calculable. This makes GRPO's approach of sampling multiple completions per prompt and computing group-relative advantages a clean fit. There is no need for a learned reward model.

**Accessible training.** GRPO can be implemented on a single node with significantly less VRAM than PPO, which matters when you are training on research-scale compute.

Concretely, for each prompt $q$ in a training batch, GRPO samples a group of $G$ completions $\{o_1, o_2, \ldots, o_G\}$ from the current policy $\pi_\theta$. Each completion receives a reward $r_i$. The advantage for completion $i$ is computed as:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

This z-score normalization is the core insight: within each group, completions that perform better than average get positive advantage, and worse-than-average completions get negative advantage. The policy is then updated to increase the probability of high-advantage completions while decreasing the probability of low-advantage ones, subject to a clipping constraint and KL penalty to prevent the policy from drifting too far from the reference model.

In our setup, each completion is an entire tool-selection trajectory: the agent's sequence of decisions about which tools to invoke (and in what order) for a given question. A trajectory that picks the right tools cheaply gets high reward; one that picks expensive tools unnecessarily (or cheap tools that produce wrong answers) gets low reward. Over training, the agent learns the map between question characteristics and cost-effective tool strategies.

## What the Agent Learns

The interesting results are not just in aggregate accuracy-cost metrics (though those are good). The interesting results are in the *strategies* the agent discovers.

### Strategy 1: Difficulty-Adaptive Escalation

On MATH, the trained agent develops a clear escalation pattern. For problems that the base model can likely solve with basic computation (algebra, arithmetic), it routes to `calculator` or `code_interpreter`. For problems that involve proof techniques or deeper reasoning, it escalates to `expert_model`. The agent is not given difficulty labels during training; it learns to estimate difficulty from the problem statement itself and adjusts its tool budget accordingly.

This mirrors what a cost-conscious human engineer would do: use the cheap tool first, and only reach for the expensive one when the cheap one is not enough. But the agent learns this strategy entirely from the reward signal, without any explicit "try cheap tools first" instruction.

### Strategy 2: Retrieval Gating on HotpotQA

For multi-hop QA, the agent learns to distinguish between questions where the local corpus is likely sufficient (using `retriever` at $0.01) and questions that reference recent events or obscure entities (escalating to `web_search` at $0.10). The gating signal appears to be the presence of time-sensitive language ("recently," "current," "as of") and named entities with low corpus frequency.

### Strategy 3: Tool Composition vs. Tool Substitution

On HumanEval, the agent learns two distinct modes. For straightforward function implementations, it generates code directly and uses the `code_interpreter` to verify. For functions requiring algorithmic insight (dynamic programming, graph algorithms), it queries the `expert_model` first for a strategy, *then* uses `code_interpreter` to implement and test. The key insight is that the agent treats `expert_model` + `code_interpreter` as a composed pipeline for hard problems, but avoids this expensive composition for easy ones.

### The Pareto Frontier

By sweeping $\alpha$ from 0.5 (equal weight on accuracy and cost) to 1.0 (accuracy only), we trace out a Pareto frontier of accuracy-cost trade-offs. The shape of this frontier is informative:

- At $\alpha = 1.0$, the agent converges to always using the most powerful tools, essentially recovering the behavior of an unconstrained agent.
- At $\alpha = 0.5$, the agent becomes aggressively frugal, sometimes sacrificing accuracy by using `calculator` on problems that genuinely need `code_interpreter`.
- The sweet spot at $\alpha = 0.7$ achieves roughly 92-95% of the unconstrained accuracy while reducing tool costs by 40-60% across benchmarks.

This last number is the headline result: **you can recover most of the accuracy of a "use everything" agent at roughly half the cost**, simply by letting the agent learn its own tool-selection policy via RL.

## Comparison: Why Not Just Prompt?

The obvious baseline is prompting. You could prepend "Use the cheapest tool that can solve this problem" to the system prompt and skip all the RL machinery. We tested this. The results are instructive:

**Prompting produces bimodal behavior.** The prompted agent either ignores the cost instruction and uses expensive tools anyway (especially on hard benchmarks like GPQA), or over-corrects and uses cheap tools on everything, tanking accuracy. It lacks the smooth, difficulty-adaptive behavior that RL training produces.

**Prompting does not generalize across benchmarks.** A prompt tuned for MATH cost optimization does not transfer well to HotpotQA, because the tool-selection strategies are fundamentally different. The RL-trained agent, by contrast, learns benchmark-specific strategies from the same training procedure.

**Prompting is fragile.** Minor rephrasing of the cost instruction ("be frugal" vs. "minimize cost" vs. "prefer cheaper tools") produces surprisingly different behaviors. The RL-trained policy is deterministic for a given input and less prompt-sensitive.

## Connection to the Broader Landscape

This work sits at the intersection of several active research areas.

**RL for LLM agents.** The post-DeepSeek-R1 era has seen an explosion of work applying GRPO and related algorithms to LLM training. Most of this work targets reasoning improvement (getting the right answer). Our contribution is applying the same machinery to *resource management*: teaching agents not just *what* to do but *how expensively* to do it.

**Cost-aware planning.** The CATP-LLM work (Wu et al., ICCV 2025) is the closest prior work we are aware of. They propose a tool planning language and offline RL algorithm for cost-aware tool planning. Our approach differs in using online GRPO training, a simpler environment formulation (direct tool selection rather than plan token generation), and evaluation across a broader set of benchmarks. We also focus on the *interpretability* of learned strategies, which CATP-LLM does not emphasize.

**Agent-R1 and agentic RL.** The Agent-R1 framework (Cheng et al., 2025) demonstrates end-to-end RL training for LLM agents with tool use. Their focus is on maximizing agent capability; ours is on optimizing the accuracy-cost trade-off. These are complementary goals: an agent trained with Agent-R1-style methods could potentially be further fine-tuned with our cost-aware reward to produce a capable *and* efficient agent.

**Model cascading and routing.** Systems like RouteLLM and Hybrid LLM route queries to different models based on difficulty. Our approach is more general: rather than routing between models, the agent selects from a heterogeneous toolkit where the cost-accuracy trade-off varies per tool and per query type.

## What We Did Not Do (Yet)

Some honest limitations:

**Single-step tool selection.** Our current environment models tool selection as a single-step (or short-horizon) decision. Real-world agents often engage in multi-turn, multi-tool chains with branching logic. Extending to full trajectory optimization with intermediate tool-use decisions is the natural next step.

**Fixed cost model.** Real API costs are dynamic (rate limits, batching discounts, latency-cost trade-offs). Our fixed cost table is a useful simplification, but a production system would need to account for time-varying pricing.

**Scale.** We train on 7B-parameter models. Whether the learned cost-awareness strategies transfer to or emerge differently in larger models (70B+) is an open question.

**Reward hacking.** As with any RL system, reward hacking is a concern. An agent could learn to exploit evaluation quirks (for example, partial-credit scoring) rather than genuinely learning cost-efficient strategies. We mitigate this with binary correctness scoring (no partial credit) and manual inspection of learned trajectories, but more rigorous robustness analysis is warranted.

## Reproducing This Work

The full codebase, environment definition, processed datasets (four JSONL files for HotpotQA, MATH, GPQA, and HumanEval), and training configs are available via our Hugging Face organization. The environment is pip-installable and follows the standard Gymnasium API, so it plugs into existing RL training frameworks (we used TRL's GRPO implementation).

If you are interested in extending this (adding new tools, new benchmarks, or different RL algorithms), the environment is designed to be modular. Defining a new tool is a Python class with a `cost` attribute, an `execute()` method, and a `description` string.

## Conclusion

The core claim of this work is simple: LLM agents should be trained to consider cost, not just accuracy. The mechanism we propose (a cost-penalized reward signal + GRPO training) is deliberately minimal. There is no complicated architecture, no novel algorithm, and no custom training infrastructure. The novelty is in the *framing*: treating tool cost as a first-class component of the agent's reward function and showing that standard RL techniques produce non-trivial, interpretable, and effective cost-optimization strategies.

As LLM agents move from research prototypes to production systems processing millions of queries per day, the difference between "always use the best tool" and "use the right tool for the job" becomes worth millions of dollars. We think this is a direction worth investing in.

---

## References

1. Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300, 2024.
2. Guo, D., et al. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948, 2025.
3. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
4. Wu, Y., et al. "CATP-LLM: Empowering Large Language Models for Cost-Aware Tool Planning." ICCV 2025.
5. Cheng, M., et al. "Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning." arXiv:2511.14460, 2025.
6. Yang, Z., et al. "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering." EMNLP 2018.
7. Hendrycks, D., et al. "Measuring Mathematical Problem Solving with the MATH Dataset." NeurIPS 2021.
8. Rein, D., et al. "GPQA: A Graduate-Level Google-Proof Q&A Benchmark." arXiv:2311.12022, 2023.
9. Chen, M., et al. "Evaluating Large Language Models Trained on Code." arXiv:2107.03374, 2021.
10. Weng, L. "LLM Powered Autonomous Agents." Lil'Log, 2023.
11. Li, Z., et al. "Encouraging Good Processes Without the Need for Good Answers: Reinforcement Learning for LLM Agent Planning." arXiv:2508.19598, 2025.

---

*If you found this post useful, feel free to cite it:*

```bibtex
@article{lara2026costaware,
  title   = {Teaching LLM Agents to Think Before They Spend: Cost-Aware Tool Orchestration via Reinforcement Learning},
  author  = {Lara, Andrew and Sharma, Yashawasi},
  journal = {blog.andrewlara.com},
  year    = {2026},
  month   = {Apr},
  url     = {https://blog.andrewlara.com/2026/04/13/cost-aware-tool-orchestration-via-reinforcement-learning.html}
}
```
