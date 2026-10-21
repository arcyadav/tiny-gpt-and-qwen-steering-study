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
