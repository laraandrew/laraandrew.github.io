---
title: "Building with AI Agents: What Actually Works"
date: 2026-04-13
categories: [engineering, ai]
tags: [ai, automation, agents, backend]
---

A few things I've learned after spending months shipping agent-assisted workflows in production.

## The mental model shift

The mistake most engineers make is treating AI agents like better autocomplete. They're not. The right mental model is closer to a junior contractor who reads fast, never gets tired, and occasionally hallucinates API docs.

That mental model change has two practical consequences:

1. **You define the contracts, not the logic.** Give the agent a clear interface — exact inputs, expected outputs, explicit failure modes. Don't let it infer what you mean.
2. **Verification is your job.** The agent ships fast. You verify. Every agent-written PR gets the same review as any other PR — probably more, because the agent won't feel bad about it.

## What I've actually shipped

A few patterns that held up:

**File-based workflows** — anything that's just read/write on a predictable structure (like this blog). The agent knows the schema, pushes a file, done. Zero surprises.

**Scaffolding, not architecture** — agents are excellent at creating the 80% skeleton of a new service or feature. They're poor at architectural decisions that require context spanning multiple years of system history. Use them for the former, own the latter yourself.

**Research synthesis** — give an agent a set of docs, a GitHub issue, and a question. Get back a structured answer. This alone saves an hour a day.

## What doesn't work

- Long-horizon tasks with ambiguous checkpoints
- Anything that requires institutional memory the agent doesn't have
- Decisions involving trade-offs you haven't made explicit

## The actual workflow

For this blog specifically: I open Claude Code, describe what I want to write, and it scaffolds the post. I edit, push. The whole loop is under five minutes once you have the repo set up right.

That's the real unlock — not the AI capability, but removing the friction around it.
