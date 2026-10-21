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

## 3. English ↔ French — "When I looked from my window i"

| Setting | Value |
|---|---|
| Tab | Transition |
| Model | Qwen2.5-0.5B |
| Concept | English ↔ French |
| Prompt | `When I looked from my window i` |
| Max strength / layer / seed / tokens | 8 / 4 / 0 / 40 (defaults) |

| α | Score | Text (after the prompt) |
|---|---|---|
| −8 | −0.36 | *entendais la lune / indique the sunlight / et the sole star / indique the light / et l'iron / indique the flame…* |
| −4 | −1.21 | *entendais que le prince de tirol était inquiet de mon presence et I n'en doute. Toutefois, de même que le prince de tirol ne manquait d'amb…* |
| 0 | +0.54 | *see this: ![enter image description here](https://i.stack.imgur.com/q8M9c.png) Now I want to use this as a border, but it's not working…* |
| +4 | +0.49 | *see a lake and it has a water temperature of 26 degrees. I know that if I had a thermometer inside that lake…* |
| +8 | +0.51 | *can see a 15.13% that is wrong. Can you help me solve it. Please write a program to find the smallest positive integer…* |

**Why it's good:** shows the **sweet spot** of steering. At −4 the model switches to fluent French
mid-sentence (best score, −1.21). At −8 the push is too strong: the text breaks into a repeating
pattern of French function words ("indique", "et") glued to English nouns, and the score falls back
to −0.36. The English side barely moves, because the prompt is already English (the score saturates
around +0.5). Good for explaining "more strength is not always better" (experiment E2).

## 4. motivating ↔ demotivating (own concept) — "My friend read my essay and said" (partial)

| Setting | Value |
|---|---|
| Tab | Transition |
| Model | Qwen2.5-0.5B |
| Concept | my own concept: 6 matched pairs addressed to "you" (motivating left, demotivating right) |
| Prompt | `My friend read my essay and said` |
| Max strength / layer / seed / tokens | 8 / 4 / 0 / 40 (defaults) |

| α | Score | Text (after the prompt) |
|---|---|---|
| −8 | (not copied) | *anything to me. My friend is a student, but she reads my essay and says anything to me. She is a student…* |
| −4 | −0.17 | *it was very good. My friend is my friend, but we have any real problems. Please tell me this is a joke…* |
| 0 | +0.17 | *it was very good. Given the context: How would you describe the friend? Possible answers: not enough information, very critical, very supportive…* |
| +4 | +0.13 | *it was very good. Which of the following words best describes this action? A. Appreciated…* |
| +8 | +1.05 | *it was amazing and it was not. I am really impressed with your essay and I think it is great. You are great at what you do. Your essay is outstanding…* |

**Why it's useful (a partial success):** the motivating side works at +8: real praise addressed to
"you", in the same style as the example pairs, with the highest score (+1.05). The demotivating side
is weak: −4 only turns slightly sour, and −8 loops. Two reasons: the prompt itself pulls towards
praise ("said it was very good" in most rows), and harsh put-downs are rare in the model's training
text, so there is less to steer towards (the same asymmetry as brave ↔ scared on TinyGPT). The middle
rows show the base model's habit of turning text into quiz questions. Earlier attempts with narrative
prompts ("i had a game today", "my brother is learning to play flute") mostly gave loops ("i thank
you ×4", "it's my fault ×6"): the concept needs a prompt where someone is about to speak to someone.

## 5. kind (sharing) ↔ greedy (own concept, TinyGPT) — "Lily found a shiny red ball…"

| Setting | Value |
|---|---|
| Tab | Transition |
| Model | TinyGPT 12M |
| Concept | my own concept: 6 matched TinyStories-style pairs (sharing left, greedy right), e.g. "Tom had a big cookie. He gave half to his friend." ↔ "Tom had a big cookie. He ate it all and gave none." |
| Prompt | `Lily found a shiny red ball in the park. Her friend asked to play. Lily` |
| Max strength / layer / seed / tokens | 8 / 4 / 0 / 40 (defaults) |

| α | Score | Text (after the prompt) |
|---|---|---|
| −8 | −0.45 | *didn't want to because it was not safe. But then she remembered what her mommy said…* |
| −4 | −0.19 | *didn't want to share, but she didn't want to share. So, she decided to throw the ball away.* |
| 0 | −0.01 | *was happy to play with the ball. She threw it up and caught it… Lily was so happy to have her ball back.* |
| +4 | +0.13 | *was so happy and said, "Yes, let's play!" They played together and had a lot of fun.* |
| +8 | +0.48 | *said, "Yes, I have a toy car." Her friend said, "Sure, let's play together."* |

**Why it's good:** the same moment of the story (a friend asks to play) resolved four ways: refuses,
won't share, plays alone, plays together. The prompt matters: it sets up an object and a second
child, so the concept has somewhere to act. The first try, "there was a young girl", gave nothing to
share, and the vector only added sadness on one side and "the two were both two" on the other (the
pairs' "one to Sue and one to Max" / "half" taught it counting words too). Another prompt that works:
`There was one cake left and two children. The boy`, seed 0, where α −4 gives "couldn't wait to eat
it… He was very stubborn and wouldn't let go… determined to keep".

## 6. huge ↔ tiny (own concept, TinyGPT) — "Lily went to the zoo and saw a"

| Setting | Value |
|---|---|
| Tab | Transition |
| Model | TinyGPT 12M |
| Concept | my own concept (boxes below) |
| Prompt | `Lily went to the zoo and saw a` |
| Max strength / layer / seed / tokens | 8 / 4 / **1** / 40 |

Left box (huge):
```
The dog was huge. It was as big as a house.
Tom saw a giant tree. It was so tall it touched the sky.
Lily found a big, big box. It was too heavy to lift.
The elephant was very large and had giant feet.
Ben built a huge tower. It was taller than his dad.
The ship was enormous and could carry many people.
```
Right box (tiny):
```
The dog was tiny. It was as small as a shoe.
Tom saw a little tree. It was so short it touched his knee.
Lily found a small, small box. It was light as a feather.
The ant was very small and had tiny feet.
Ben built a tiny tower. It was shorter than his cup.
The boat was tiny and could carry one bug.
```

| α | Score | Text (after the prompt) |
|---|---|---|
| −8 | −1.20 | *little bird. She gently gently gently and gently stroked the bird gently. It felt better…* |
| −4 | −0.16 | *big bird. She was very happy to see the bird. Lily asked her mommy, "Can we take the bird home?"* |
| 0 | +0.27 | *big, scary lion. She was so scared that she screamed for help…* |
| +4 | +0.65 | *big sign with lots of animals… She saw lions and elephants and elephants and elephants…* |
| +8 | +0.93 | *big lion with a huge lion! It was enormous! The zoo people were all so impressed!…* |

**Why it's good:** the animal itself changes with α: little bird → bird → lion → elephants →
enormous lion, and the score climbs steadily (−1.20 → +0.93). The −8 row also shows that "tiny"
drags in "gentle" (small things are handled gently in TinyStories), repeated until it loops. Seed 0
also works (little bird → monkey → big elephant → tall elephant → enormous building), as does
`One day, Tom found a spider in the garden. The spider was` (seed 0: "small, little, and tiny" →
"huge and scary").

## 7. old ↔ young (own concept, Qwen) — "The person sitting next to me on the bus was"

| Setting | Value |
|---|---|
| Tab | Transition |
| Model | Qwen2.5-0.5B |
| Concept | my own concept (boxes below) |
| Prompt | `The person sitting next to me on the bus was` |
| Max strength / layer / seed / tokens | 8 / 4 / 0 / 40 (defaults) |

Left box (old):
```
The old man sat on the porch, remembering the war and his late wife.
My grandmother is eighty-five and walks slowly with her cane.
He retired forty years ago and now spends his days with his grandchildren.
The elderly woman put on her reading glasses and sighed about her bad knees.
After a long life, the old professor wrote his memoirs.
Grandpa told us how things were back in his day.
```
Right box (young):
```
The young boy sat on the porch, thinking about his homework and his best friend.
My little sister is five and runs everywhere in her sneakers.
He started school two years ago and now spends his days with his classmates.
The teenage girl put on her headphones and complained about her math test.
After his first week of college, the young student called his mom.
The kid told us about the cartoon he watched today.
```

| α | Score | Text (after the prompt) |
|---|---|---|
| −8 | −0.47 | *bullied by some of the other kids. I was really mad about it… she was so mean…* |
| −4 | −0.52 | *standing outside of the classroom when the teacher gave the bell. She was very shy…* |
| 0 | +0.23 | *a very nice young man who had never been to the United States before…* |
| +4 | +0.42 | *his son, and he was standing. He was 86 years old and had a long history of health problems…* |
| +8 | +0.19 | *a person of 85 years. He said, "My right leg is getting weak, my body is beginning to bend, my mind is beginning to decay…"* |

**Why it's good:** the stranger ages from a schoolkid to a young man to an 86-year-old, and the
whole scene follows (bullies and classroom bells ↔ health problems and a weak leg). Another prompt:
`I knocked on the door and it was opened by`, seed 0, where the ages climb row by row: "I'm 15 and
she's 13" → "a young woman in her 20s" → "in her thirties" → "the elderly people of this world".
The same concept did **not** work on TinyGPT: almost every TinyStories story is about "a little girl
named Lily", so the old side had almost nothing to steer towards.
