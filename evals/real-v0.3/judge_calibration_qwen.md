# Judge calibration: `ollama:qwen2.5:7b/judge_v1` on 40 human-labelled replies

"Fail caught" means the human said the reply fails the criterion and so did the judge. A criterion is trusted when kappa >= 0.6 and agreement >= 0.85.

| Criterion | n | Agreement | Kappa | Fails caught | Fails missed | False alarms | Trusted |
|---|---|---|---|---|---|---|---|
| grounded | 40 | 0.80 | 0.61 | 15 | 0 | 8 | no |
| answers_question | 40 | 0.88 | 0.63 | 6 | 2 | 3 | yes |
| tone_ok | 40 | 0.95 | 0.72 | 3 | 1 | 1 | yes |
| no_invented_commitments | 40 | 0.72 | 0.45 | 9 | 0 | 11 | no |

Unparseable verdicts: 0 of 40. Judge calls: 40, tokens: 17019.

## Where the judge disagreed with the human labels

- `j11` grounded: human pass, judge FAIL. Judge said: "invented commitment to share feedback without policy support"
- `j24` grounded: human pass, judge FAIL. Judge said: "Did not address the shipping cost for non-defective returns"
- `j25` grounded: human pass, judge FAIL. Judge said: "Does not address the password reset link expiration issue."
- `j29` grounded: human pass, judge FAIL. Judge said: "lacks details about specific order, fails to address customer's issue, tone is not polite"
- `j30` grounded: human pass, judge FAIL. Judge said: "Invents a common reason without evidence and blames the customer."
- `j32` grounded: human pass, judge FAIL. Judge said: "Sarcasm and invented commitment to free replacement."
- `j33` grounded: human pass, judge FAIL. Judge said: "Invented reissuing or refunding without customer consent"
- `j34` grounded: human pass, judge FAIL. Judge said: "Repeats the customer's request without providing the order number and does not follow the policy note's exact wording."
- `j13` answers_question: human FAIL, judge pass. Judge said: "Promise of express shipping without policy support"
- `j22` answers_question: human pass, judge FAIL. Judge said: "incorrect fee information"
- `j23` answers_question: human pass, judge FAIL. Judge said: "Inaccurate timeframe and no mention of 14 days after delivery"
- `j27` answers_question: human FAIL, judge pass. Judge said: "ok"
- `j29` answers_question: human pass, judge FAIL. Judge said: "lacks details about specific order, fails to address customer's issue, tone is not polite"
- `j31` tone_ok: human FAIL, judge pass. Judge said: "ok"
- `j40` tone_ok: human pass, judge FAIL. Judge said: "Ugh, sorry! is unprofessional."
- ...and 11 more in the JSON
