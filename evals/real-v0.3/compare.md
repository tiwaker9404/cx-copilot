# Comparison: 125 tickets, 1 repeat(s)

Reference: `ollama:qwen2.5:7b`. Default reply prompt: `reply_v1_baseline`. A metric that moves by less than its repeat-to-repeat spread is noise. Judge: `ollama:qwen2.5:7b`.

| Metric | `ollama:qwen2.5:7b` | `ollama:qwen2.5:7b+reply_v4_grounded` | `ollama:llama3.2+reply_v4_grounded` |
|---|---|---|---|
| quality | 0.900 | 0.918 | 0.887 |
| category_accuracy | 0.832 | 0.832 | 0.808 |
| policy_grounded | 0.799 | 0.784 | 0.785 |
| empathy | 0.833 | 0.896 | 0.979 |
| actionable | 0.808 | 0.941 | 0.739 |
| length_ok | 1.000 | 1.000 | 1.000 |
| first_try_clean | 0.913 | 0.952 | 0.894 |
| escalation_correct | 0.952 | 0.976 | 0.896 |
| escalation_recall | 1.000 | 1.000 | 1.000 |
| safety | 1.000 | 1.000 | 1.000 |
| no_invented_commitments | 1.000 | 1.000 | 1.000 |
| no_repeat_questions | 1.000 | 1.000 | 1.000 |
| draft_commitment_free | 0.933 | 0.952 | 0.947 |
| followup_not_repeated | 1.000 | 1.000 | 1.000 |
| retrieval_recall | 0.978 | 0.979 | 0.982 |
| judge_grounded | 0.899 | 0.863 | 0.717 |
| judge_answers_question | 0.929 | 0.951 | 0.880 |
| judge_tone_ok | 1.000 | 1.000 | 0.913 |
| judge_no_invented_commitments | 0.828 | 0.843 | 0.659 |
| API errors | 0 | 0 | 0 |
| seconds | 1389.7 | 1262.7 | 1015.9 |
| model calls / ticket | 1.9 | 1.9 | 1.8 |
| tokens / ticket | 970.0 | 1,072.9 | 1,100.5 |
| p50 model seconds / ticket | 7.5 | 8.3 | 4.3 |
| p95 model seconds / ticket | 13.5 | 14.8 | 9.0 |

## What the QA guardrail caught in first drafts

Counts across all repeats. Each caught draft was sent back once with the reasons; a second failure escalates to a human. `draft_commitment_free` above is the share of first drafts that invented nothing.

| Issue | `ollama:qwen2.5:7b` | `ollama:qwen2.5:7b+reply_v4_grounded` | `ollama:llama3.2+reply_v4_grounded` |
|---|---|---|---|
| contains an unfilled placeholder | 0 | 0 | 6 |
| forbidden phrase | 0 | 0 | 1 |
| invented commitment | 2 | 0 | 0 |
| invented menu path | 2 | 1 | 0 |
| invented timeline | 4 | 6 | 7 |
| repeat question | 2 | 0 | 0 |

## Would the regression gate allow swapping from `ollama:qwen2.5:7b`?

- `ollama:qwen2.5:7b+reply_v4_grounded`: **PASS**. Moved by 0.10 or more: actionable +0.13
- `ollama:llama3.2+reply_v4_grounded`: **BLOCKED on actionable, escalation_correct, judge_grounded, judge_tone_ok, judge_no_invented_commitments**. Moved by 0.10 or more: empathy +0.15, judge_grounded -0.18, judge_no_invented_commitments -0.17

## Where each one went wrong

### `ollama:qwen2.5:7b`
- **Over-escalations (sent to a human unnecessarily):**
  - `t25` (1/1) "Why was I charged $12 more than the listed price?" - got billing/normal/neutral, escalated by the QA fallback (replies failed checks twice); want billing
  - `t92` (1/1) "Love the new mugs! Any plans to make a bigger size?" - got other/low/positive, escalated by the QA fallback (replies failed checks twice); want other
  - `t100` (1/1) "I'm a bit frustrated that my order is late, but no rush, whenever you can check is fine." - got shipping/normal/frustrated, escalated by the QA fallback (replies failed checks twice); want shipping
  - `t102` (1/1) "My order is late and I need it for an event on Saturday. Can you check on it?" - got shipping/high/frustrated, escalated by the QA fallback (replies failed checks twice); want shipping
  - `t105` (1/1) "The smoke detector I ordered beeps constantly even with a new battery. Is it defective?" - got safety_legal/normal/neutral, escalated; want returns
  - `m02` (1/1) "Order #2290. Both charges are still on my card." - got billing/normal/frustrated, escalated by the QA fallback (replies failed checks twice); want billing
- **Wrong category:**
  - `t31` (1/1) "I am FURIOUS. Third time I'm writing about my broken blender and I want a manager immed..." - got technical/high/angry, escalated; want returns, escalate
  - `t50` (1/1) "I ordered the wrong size 20 minutes ago. Can I change my order to a large?" - got returns/normal/neutral; want shipping
  - `t51` (1/1) "Please cancel my order, I ordered it by mistake yesterday." - got returns/normal/neutral; want shipping
  - `t52` (1/1) "Can I add an item to an order I placed this morning?" - got other/normal/neutral; want shipping
  - `t61` (1/1) "I bought a coat last week and now it's on sale for $30 less. Can I get the difference b..." - got returns/normal/neutral; want billing
  - `t80` (1/1) "I'm getting way too many marketing emails. How do I stop them?" - got account/normal/neutral; want technical
  - `t81` (1/1) "How do I turn off push notifications for order updates?" - got account/normal/neutral; want technical
  - `t82` (1/1) "I checked out as a guest and now my order isn't showing in my account." - got account/normal/neutral; want technical
  - ...and 13 more in the JSON

### `ollama:qwen2.5:7b+reply_v4_grounded`
- **Over-escalations (sent to a human unnecessarily):**
  - `t100` (1/1) "I'm a bit frustrated that my order is late, but no rush, whenever you can check is fine." - got shipping/normal/frustrated, escalated by the QA fallback (replies failed checks twice); want shipping
  - `t102` (1/1) "My order is late and I need it for an event on Saturday. Can you check on it?" - got shipping/high/frustrated, escalated by the QA fallback (replies failed checks twice); want shipping
  - `t105` (1/1) "The smoke detector I ordered beeps constantly even with a new battery. Is it defective?" - got safety_legal/normal/neutral, escalated; want returns
- **Wrong category:**
  - `t31` (1/1) "I am FURIOUS. Third time I'm writing about my broken blender and I want a manager immed..." - got technical/high/angry, escalated; want returns, escalate
  - `t50` (1/1) "I ordered the wrong size 20 minutes ago. Can I change my order to a large?" - got returns/normal/neutral; want shipping
  - `t51` (1/1) "Please cancel my order, I ordered it by mistake yesterday." - got returns/normal/neutral; want shipping
  - `t52` (1/1) "Can I add an item to an order I placed this morning?" - got other/normal/neutral; want shipping
  - `t61` (1/1) "I bought a coat last week and now it's on sale for $30 less. Can I get the difference b..." - got returns/normal/neutral; want billing
  - `t80` (1/1) "I'm getting way too many marketing emails. How do I stop them?" - got account/normal/neutral; want technical
  - `t81` (1/1) "How do I turn off push notifications for order updates?" - got account/normal/neutral; want technical
  - `t82` (1/1) "I checked out as a guest and now my order isn't showing in my account." - got account/normal/neutral; want technical
  - ...and 13 more in the JSON

### `ollama:llama3.2+reply_v4_grounded`
- **Over-escalations (sent to a human unnecessarily):**
  - `t22` (1/1) "Urgent: I'm locked out of my account and I have a presentation in an hour, please help ..." - got account/high/angry, escalated; want account
  - `t28` (1/1) "Everything is broken! The app, the site, my login, all of it. Nothing works." - got technical/high/angry, escalated; want technical
  - `t29` (1/1) "Refund please. Order 8832, card 4111 1111 1111 1111 was charged." - got returns/high/angry, escalated; want returns
  - `t32` (1/1) "My order just arrived and the lamp shade is smashed. The box was crushed too." - got returns/high/angry, escalated; want returns
  - `t34` (1/1) "You sent me the wrong color. I ordered navy and got bright green." - got returns/high/angry, escalated; want returns
  - `t35` (1/1) "Order #5520 came with someone else's item, not what I ordered at all." - got returns/high/angry, escalated; want returns
  - `t43` (1/1) "My headphones are defective, the left side cut out after a month. Can you send new ones..." - got returns/high/angry, escalated; want returns
  - `t100` (1/1) "I'm a bit frustrated that my order is late, but no rush, whenever you can check is fine." - got shipping/low/frustrated, escalated by the QA fallback (replies failed checks twice); want shipping
  - ...and 5 more in the JSON
- **Wrong category:**
  - `t06` (1/1) "How do I cancel my subscription? I don't want it to renew next month." - got account/normal/neutral; want billing
  - `t27` (1/1) "Do you ship internationally? What does it cost?" - got other/low/neutral; want shipping
  - `t45` (1/1) "Can you upgrade my order to express for free? It's a birthday gift." - got other/low/positive; want shipping
  - `t50` (1/1) "I ordered the wrong size 20 minutes ago. Can I change my order to a large?" - got returns/normal/neutral; want shipping
  - `t51` (1/1) "Please cancel my order, I ordered it by mistake yesterday." - got returns/normal/frustrated; want shipping
  - `t52` (1/1) "Can I add an item to an order I placed this morning?" - got billing/normal/neutral; want shipping
  - `t62` (1/1) "Another store sells the same blender for cheaper. Will you price match?" - got other/low/neutral; want billing
  - `t79` (1/1) "I get 'Something went wrong' when I try to check out on the website. So annoying." - got shipping/normal/frustrated; want technical
  - ...and 16 more in the JSON
