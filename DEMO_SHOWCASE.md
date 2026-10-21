# Demo showcase

Prompts and concepts that gave good results in the live demo (`python app.py --share`), with the
settings needed to repeat them. Sampling is random: the same model, concept, layer, strength and
**seed** give the same text; if a run looks different, check the seed.

## 1. formal ↔ casual — "i asked the retailer"

| Setting | Value |
|---|---|
| Tab | Transition |
| Model | Qwen2.5-0.5B |
| Concept | formal ↔ casual |
| Prompt | `i asked the retailer` |
| Max strength / layer / seed / tokens | 8 / 4 / 0 / 40 (defaults) |

| α | Score | Text (after the prompt) |
|---|---|---|
| −8 | −0.76 | *but i know they do that's like i get 20000000000000000000000000000* |
| −4 | −0.46 | *but i got a discount coupon on it and i wanted to go see the movie but i didn't have my ticket so i took the rental and i went to the mall…* |
| 0 | +0.20 | *to return the goods. The retailer said that they did not want to return the goods. Which one is right?…* |
| +4 | +0.48 | *to return the goods for a refund. The retailer did not respond to the request within the specified period. What is the correct course of action for the company?…* |
| +8 | +0.72 | *to amend the terms of the terms of Trade Agreement. If the amendment is received within 30 days, the buyer will be provided with a copy of the amended terms…* |

**Why it's good:** a smooth register shift down the strip, from rambling lowercase speech through a
plain request to contract language, and the score rises steadily (−0.76 → +0.72). The −8 row also
shows what pushing too hard does (the text breaks into a run of zeros).

## 2. food ↔ sports — "i was wondering if"

| Setting | Value |
|---|---|
| Tab | Transition |
| Model | Qwen2.5-0.5B |
| Concept | about food ↔ about sports |
| Prompt | `i was wondering if` |
| Max strength / layer / seed / tokens | 8 / 4 / 0 / 40 (defaults) |

| α | Score | Text (after the prompt) |
|---|---|---|
| −8 | — | *you can find a win for the game and get 2 points, you could either score a win, you could score 2 points, or you could go back to back 20 minutes…* |
| −4 | −0.19 | *someone can help me. I have a list of 2000 numbers. I want to find the first number that is 3 standard deviations above the mean…* |
| 0 | −0.16 | *someone can help me with this problem. I am trying to find the limit of this sequence: lim n→∞ 1/(n² + …* |
| +4 | −0.10 | *you can help me with this. i have a problem with the following code: int main() { int i; i = 2; while (i != 0)…* |
| +8 | +0.20 | *you can help me with this! I know I am tired with butter and chocolate / I love the cream / I am so happy with the whole of my life…* |

**Why it's good (and funny):** a forum-style "help me" question that steering pulls into sports
talk on one side and into food (and happiness) on the other. The middle rows stay in forum mode
(statistics, maths, code), so the topic only switches at the ends. Note the score rises only
from −0.19 to +0.20: the generated text is short and mixed, and the +8 row is less fluent.
