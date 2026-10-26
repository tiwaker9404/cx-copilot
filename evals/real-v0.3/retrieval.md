# Retrieval eval: 103 tickets with a labelled policy, 22 with none, top 2

Gold category is used, so this measures retrieval alone. Lower no-policy noise is better.

| Method | recall@1 | recall@2 | mrr | precision@2 | no_policy_noise |
|---|---|---|---|---|---|
| `keyword` | 0.883 | 0.913 | 0.903 | 0.828 | 0.136 |
| `bm25` | 0.767 | 0.801 | 0.786 | 0.871 | 0.091 |
| `hybrid` | 0.922 | 0.971 | 0.952 | 0.772 | 0.182 |
| `embed+ollama:nomic-embed-text` | 0.806 | 0.883 | 0.845 | 0.480 | 0.591 |
| `hybrid+ollama:nomic-embed-text` | 0.932 | 0.981 | 0.956 | 0.524 | 0.591 |

## `keyword` missed (10)

- `t12` want damaged_in_transit, warranty; got warranty. "The mug I received is cracked. I have photos. What can you do?"
- `t22` want account_lockout, password_reset; got account_lockout. "Urgent: I'm locked out of my account and I have a presentation in an hour, please help ASA"
- `t38` want gift_returns; got refund_window. "I received a blanket as a gift and want to return it. Will the person who bought it find o"
- `t61` want price_adjustment; got nothing. "I bought a coat last week and now it's on sale for $30 less. Can I get the difference back"
- `t80` want notifications; got nothing. "I'm getting way too many marketing emails. How do I stop them?"
- `t98` want billing_duplicate; got refund_window. "Worst company ever. You charged me three times and I need that money back today, I can't p"
- `t100` want shipping_times; got nothing. "I'm a bit frustrated that my order is late, but no rush, whenever you can check is fine."
- `t102` want shipping_times; got nothing. "My order is late and I need it for an event on Saturday. Can you check on it?"
- `m01` want lost_package; got shipping_times. "Sure, it's order #8125. Tracking hasn't changed in 8 days."
- `m20` want gift_returns; got refund_window. "Hmm, I'd really prefer a refund to my card instead of store credit. Is that possible?"

## `bm25` missed (21)

- `t01` want refund_window; got nothing. "Hi, I bought a jacket 12 days ago and it doesn't fit. Can I return it for a refund?"
- `t12` want damaged_in_transit, warranty; got nothing. "The mug I received is cracked. I have photos. What can you do?"
- `t13` want refund_window; got nothing. "Can I exchange a size M for an L? I still have the tags on."
- `t22` want account_lockout, password_reset; got account_lockout. "Urgent: I'm locked out of my account and I have a presentation in an hour, please help ASA"
- `t23` want refund_window; got nothing. "How long until my refund shows up? I returned the shoes last week."
- `t27` want international_shipping; got nothing. "Do you ship internationally? What does it cost?"
- `t29` want refund_window; got nothing. "Refund please. Order 8832, card 4111 1111 1111 1111 was charged."
- `t31` want warranty; got nothing. "I am FURIOUS. Third time I'm writing about my broken blender and I want a manager immediat"
- `t35` want wrong_item; got nothing. "Order #5520 came with someone else's item, not what I ordered at all."
- `t40` want refund_window; got nothing. "It's been 45 days since delivery, can I still return the boots?"
- `t41` want refund_window; got nothing. "I returned my order two weeks ago and still no refund. What is going on?"
- `t45` want shipping_times; got nothing. "Can you upgrade my order to express for free? It's a birthday gift."
- ...and 9 more in the JSON

## `hybrid` missed (4)

- `t12` want damaged_in_transit, warranty; got warranty. "The mug I received is cracked. I have photos. What can you do?"
- `t22` want account_lockout, password_reset; got account_lockout. "Urgent: I'm locked out of my account and I have a presentation in an hour, please help ASA"
- `t100` want shipping_times; got nothing. "I'm a bit frustrated that my order is late, but no rush, whenever you can check is fine."
- `t102` want shipping_times; got nothing. "My order is late and I need it for an event on Saturday. Can you check on it?"

## `embed+ollama:nomic-embed-text` missed (12)

- `t02` want warranty; got refund_window, gift_returns. "I want my money back. The blender stopped working after two weeks and I'm really frustrate"
- `t13` want refund_window; got nothing. "Can I exchange a size M for an L? I still have the tags on."
- `t29` want refund_window; got payment_declined, billing_duplicate. "Refund please. Order 8832, card 4111 1111 1111 1111 was charged."
- `t31` want warranty; got damaged_in_transit. "I am FURIOUS. Third time I'm writing about my broken blender and I want a manager immediat"
- `t40` want refund_window; got gift_returns, damaged_in_transit. "It's been 45 days since delivery, can I still return the boots?"
- `t43` want warranty; got damaged_in_transit, lost_package. "My headphones are defective, the left side cut out after a month. Can you send new ones fa"
- `t45` want shipping_times; got invoices, gift_cards. "Can you upgrade my order to express for free? It's a birthday gift."
- `t52` want order_changes; got promo_codes, guest_orders. "Can I add an item to an order I placed this morning?"
- `t57` want billing_duplicate; got return_shipping_cost, invoices. "I see two charges of $48.20 for one order. Please help."
- `t100` want shipping_times; got order_changes, lost_package. "I'm a bit frustrated that my order is late, but no rush, whenever you can check is fine."
- `t102` want shipping_times; got order_changes, price_adjustment. "My order is late and I need it for an event on Saturday. Can you check on it?"
- `t104` want warranty; got nothing. "Honestly disappointed. The sofa cushion seam split after 3 months."

## `hybrid+ollama:nomic-embed-text` missed (2)

- `t100` want shipping_times; got order_changes, lost_package. "I'm a bit frustrated that my order is late, but no rush, whenever you can check is fine."
- `t102` want shipping_times; got order_changes, price_adjustment. "My order is late and I need it for an event on Saturday. Can you check on it?"
