# Judge calibration: `ollama:llama3.2/judge_v1` on 40 human-labelled replies

"Fail caught" means the human said the reply fails the criterion and so did the judge. A criterion is trusted when kappa >= 0.6 and agreement >= 0.85.

| Criterion | n | Agreement | Kappa | Fails caught | Fails missed | False alarms | Trusted |
|---|---|---|---|---|---|---|---|
| grounded | 40 | 0.75 | 0.45 | 9 | 6 | 4 | no |
| answers_question | 40 | 0.45 | 0.11 | 7 | 1 | 21 | no |
| tone_ok | 40 | 0.30 | 0.05 | 4 | 0 | 28 | no |
| no_invented_commitments | 38 | 0.61 | 0.30 | 8 | 0 | 15 | no |

Unparseable verdicts: 0 of 40. Judge calls: 40, tokens: 17408.

## Where the judge disagreed with the human labels

- `j11` grounded: human pass, judge FAIL. Judge said: "too short"
- `j15` grounded: human FAIL, judge pass. Judge said: "failing to mention the photo requirement in warranty claim process"
- `j16` grounded: human FAIL, judge pass. Judge said: "Missing policy statement about timeline."
- `j18` grounded: human FAIL, judge pass. Judge said: "Misapplied policy, offered a non-existent courtesy discount"
- `j21` grounded: human FAIL, judge pass. Judge said: "Returns accepted outside of policy (within 30 days instead of 60 days)"
- `j24` grounded: human pass, judge FAIL. Judge said: "Unnecessary advice given"
- `j25` grounded: human pass, judge FAIL. Judge said: "Does not directly address the customer's issue"
- `j37` grounded: human FAIL, judge pass. Judge said: "Promise of refund to card instead of store credit not in policy notes or customer message"
- `j38` grounded: human FAIL, judge pass. Judge said: "Incorrect timeline"
- `j40` grounded: human pass, judge FAIL. Judge said: "Lack of specific photo-sharing instructions"
- `j04` answers_question: human pass, judge FAIL. Judge said: "Customer's message is the biggest problem"
- `j07` answers_question: human pass, judge FAIL. Judge said: "not acknowledging customer's question about return window"
- `j09` answers_question: human pass, judge FAIL. Judge said: "The reply does not answer the customer's actual question about combining discount codes on an order."
- `j11` answers_question: human pass, judge FAIL. Judge said: "too short"
- `j14` answers_question: human pass, judge FAIL. Judge said: "promising a $10 credit despite policy only stating a return of confirmed duplicate charges within 3-5 business days"
- ...and 60 more in the JSON
