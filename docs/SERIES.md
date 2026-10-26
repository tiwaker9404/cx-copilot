# LinkedIn post: cx-copilot v0.3

The most dangerous thing a support bot says isn't rude. It's "I've upgraded you to express, it'll arrive tomorrow."

Nobody offered that. The customer now expects it anyway.

So I built a guardrail into cx-copilot, my open-source support agent (LangGraph + LangSmith): every timeline, refund, credit, upgrade, "I've escalated this" and menu path in a draft has to be backed by the policy it was given. If it isn't, the draft goes back to the model with the reasons. If it fails twice, a human gets the ticket instead.

I ran it on 125 support tickets with free local models (qwen2.5:7b, llama3.2). Cost: $0.

What I found:
→ No reply that went out contained a made-up promise. The first drafts did: qwen offered to expedite, invented "24 hours", and asked customers for order numbers they'd just given.
→ A grounded prompt removed the invented offers and repeat questions, raised replies with a clear next step from 81% to 94%, and halved unnecessary escalations.
→ It did NOT stop invented timelines. 5 of 6 came from tickets where retrieval found no policy. With nothing to quote, the model made a number up. That's a retrieval fix, not a prompt fix.
→ An LLM judge was worse at spotting made-up promises than plain rules (72% vs 93% agreement with my labels). It's only trusted for tone, and the release gate never relies on it.
→ I read every flagged draft by hand. 5 of 15 flags were my own rules being wrong. Fixed, with tests.

Also new: multi-turn conversations, human review before risky replies go out, cost and latency budgets, and a loop that turns reviewer edits into new test cases.

Code and every report: <link to the repo>

#LangChain #LangGraph #LangSmith #CustomerSupport #AIEngineering #LLMEvaluation
