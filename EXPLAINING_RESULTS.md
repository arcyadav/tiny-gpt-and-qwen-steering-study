# Explaining what the demo shows

A viva cheat-sheet: for each odd thing someone might see when they try their own prompt or concept,
**why it happens**, **an example we saw**, and **what to try live**. The numbered examples (#1–#14)
are in [`DEMO_SHOWCASE.md`](DEMO_SHOWCASE.md); E1–E8 are the experiments in [`REPORT.md`](REPORT.md).

**The one idea behind almost everything:** the steering vector is the *difference between the average
hidden state of the two example boxes*, added at every token. So it pushes towards **whatever differs
between the boxes** (not only the concept you meant), and it can only push towards **things the model
learned from its training data**. A push that is too big drowns out the rest of the hidden state.

## Quick diagnosis (do this live)

1. **Look at the α 0 row first.** That is what the model writes without steering. If it is already on
   one side of the concept, only the other side can move.
2. **Text loops or breaks?** Lower the strength to 4–6.
3. **Nothing changes?** Make the prompt stop right where the concept would appear (a person, a thing,
   a choice), or raise the strength.
4. **Wrong thing appears?** Check the example boxes: do the two sides differ *only* in the concept?
5. **Try a nearby layer** (3–6). Layer 4 is best for happy/sad; other concepts can prefer 3.
6. **Try another seed.** One sample is one draw; a second seed shows whether it was luck.

## Symptoms

### 1. The text repeats itself ("wand wand wand", "i thank you. i thank you.", "bad bad bad")
**Why:** the push is too strong. At large α the added vector is as big as the hidden state itself, so
the model's next-word choice collapses onto the few tokens that point most along the direction, and it
picks them again and again. Our fluency metric distinct-2 (share of unique word pairs) drops exactly here.
**Seen:** magic ↔ everyday at +8 (#10, "wand wand wand"); motivating at +8 ("i thank you" ×4); E2: Qwen
pushed to −16 loops on "I looked at the world, I looked at the world…".
**Try:** strength 4–6.

### 2. More strength gives *less* of the concept
**Why:** past the sweet spot the hidden state is pushed off the region the later layers were trained
on, so they can no longer turn it into proper text of that kind. Only fragments get through.
**Seen:** English ↔ French: fluent French at α = −4 (score −1.21), but at −8 only "indique the light /
et l'iron" (score −0.36) (#3). E2: sad steering peaks at −12 (−0.62) and weakens at −16 (−0.51).
**Say:** "There is a usable range: perplexity crosses 3× the normal value between |α| 8 and 12 (E2)."

### 3. Only one side of the concept works
Three different reasons, worth telling apart:
- **The unsteered text is already on that side** (nothing left to push). Qwen's default world is
  American, so Indian ↔ American only moves towards Indian (#9). TinyGPT's stories already end happily
  (+0.48 unsteered), so steering towards happy changes less than towards sad.
- **The training data has little of the other side.** TinyStories has almost no old characters, so
  old ↔ young fails on TinyGPT but works on Qwen (#7). Harsh put-downs are rare in Qwen's data, so
  "demotivating" barely appears (#4). TinyGPT's brave side barely changes the story.
- **The example pairs were lopsided** (see 4).

### 4. The steering produces something *related* to the concept, not the concept
**Why:** the vector is every difference between the two boxes. If one side's sentences are also
longer, end in "?", or describe different actions, the vector learns that too.
**Seen:**
- dogs ↔ cats, first version: the cat sentences said "purred", "drank some milk", "climbed up the
  tree", "slept in the window". Steering towards cats gave "per per per" (the start of *pur*red), owls
  *perched* in trees and "tea tea tea", but never the word "cat". Pairs that differ **only** in the
  animal word ("The little dog/cat played in the garden") fixed it: "Her pet was a small cat".
- confident ↔ uncertain: the uncertain sentences were longer, hedged and ended in "?". The direction
  became "hedging ↔ short flat statements", so "confident" gave textbook exercises.
- greedy (TinyGPT) also made stories *sad*, because greedy stories in TinyStories end sadly.
**Try:** write matched pairs that are identical except for the concept.

### 5. Nothing happens, or the middle rows don't change
**Why:**
- **The prompt leaves no room for the concept.** "there was a young girl" gives sharing ↔ greedy
  nothing to share (#5); "Lily found a shiny red ball in the park. Her friend asked to play. Lily" works.
  Stop the prompt right before the person, thing or decision the concept should decide.
- **Topics switch rather than slide.** Food ↔ sports at ±4 still talks about homework; only ±8 flips
  to basketball or breakfast (#14). Tone (formal ↔ casual, #1) shifts gradually.
- **TinyGPT doesn't know the words.** It has a 4,096-token vocabulary and only children's English.
  A concept like "economics ↔ politics" has nothing to attach to.
- **A long prompt** sets up its own story, which outweighs the push. Short prompts steer more.

### 6. Qwen writes quiz questions ("A. B. C. D. Answer:", "Given the context… Possible answers:")
**Why:** Qwen2.5 is a *base* model whose training mix includes a lot of exam and question-answer text.
A short third-person sentence about someone ("The strict teacher looked at my homework and said it
was…") looks like the start of a reading-comprehension question, so it continues as one. This
happens with or without steering (look at the α 0 row).
**Try:** end the prompt with an open quote (`…and said, "`) or add a sentence of story set-up. The
demo already blocks Chinese characters and "____" blanks, which come from the same exam data.

### 7. The concept score disagrees with what you read
**Why:** the score is where the *unsteered* model places the text on the same direction we steer
along (read at layer 12 for Qwen, 3 for TinyGPT; about +1 = like the left box, −1 = like the right
box). It shows movement along the direction, not whether a human would say the text has the concept.
- Text that loops on direction-heavy words scores high even when it is nonsense ("confident" +0.57 on
  textbook exercises; magic at +8 on "wand wand wand" scored +2.59).
- Short or mixed text averages out: the French at −8 is mostly English nouns, so it scores only −0.36.
- Text about the concept in a different style can score lower (old at +8 became a speech about
  aging and scored +0.19, below +4).
**Say:** "For happy/sad we also use an independent sentiment classifier; for other concepts we read
the text, and that's in the limitations."

### 8. The rows are identical for a few words, then split
**By design.** All rows share their random draws (same seed for TinyGPT; one shared Gumbel noise per
token for Qwen), so they differ *only* because of the steering. They agree until the push changes one
word choice, and from then on each row follows its own path. The coloured part shows where.

### 9. Same seed, but the Steer tab and the Transition tab give different text
**Why:** the two tabs sample differently. The Steer tab uses normal sampling over a batch of 2–3 rows;
the Transition and Mix tabs use the shared-noise sampler so that rows stay comparable. Each tab is
reproducible on its own; they are not meant to match each other.

### 10. TinyGPT shows "[end of story — the model starts a new one]" and an unrelated story
**Why:** TinyGPT writes an end-of-story token when it thinks the story is done and then starts a new
one. Happy steering makes stories end sooner: the end-of-story probability we measured goes from 0.05
(α 0) to 0.26 (α 8) and 0.36 (α 12), because TinyStories' happy endings ("…and they lived happily")
are where stories stop.

### 11. The random-vector box (control) also changes the text
**Why:** any push changes some word choices, so the text differs from the unsteered one. What matters
is that it moves the score in **no consistent direction**: random vectors stay within about ±0.35 of the
unsteered text in every experiment, while the concept vector moves it steadily one way. That is what the
control is for.

### 12. Grammar breaks, or languages mix ("i entendais", "you will eat with a little")
**Why:** grammar lives in small words (will, -ed, le/la) that must be in exactly the right place, and the
push is added at *every* token. Content concepts (wealth, age, size, topic) shift whole scenes cleanly;
past ↔ future tense mostly failed (E7). Mixed languages are a half-way steer: the function words
switched, the content words did not.

### 13. Mix tab: one concept dominates, or the text breaks at strength 8
**Why:** both vectors are scaled to the same length, but some concepts move the text more than others
(on Qwen, rich + old reads mostly as old). Two nearly independent pushes of 8 add up to √2 × 8 ≈ 11,
already past the usable range, so use 6. The cosine shown above the grid says how independent the two
directions are (−0.06 and −0.03 for our defaults; near 0 = independent).

### 14. Steering towards "sad" gives scary, loud or mean things (TinyGPT)
**Why:** the meaning of a direction comes from the data. In children's stories the negative side is a
loud dog, a mean boy or a broken toy, not grief, so that is what "sad" means to TinyGPT ("He thought
loud loud so loud mean loud noises or scary…").

### 15. A concept works on Qwen but not TinyGPT, or only at another layer
**Why:** the two models learned from very different data (web text vs 80M tokens of children's stories)
and have different depths (24 vs 6 layers). Layer 4 is the best happy/sad layer in both, but the cat
side worked best at layer 3. Very early layers act on individual words; very late layers have little
room left to change the text (E1).

### 16. Strange characters
- "â€œ" or "â€™": broken quote characters that some TinyStories texts contain; TinyGPT learned to
  write them. The demo now shows them as “ ” ’.
- No Chinese and no "____" in Qwen's output: the demo blocks them (the experiments did not).

### 17. The first run is slow
**Why:** the first click loads the model and computes the concept vectors for every layer from the
example sentences; after that they are cached. Qwen runs on a laptop CPU: about 6 s per generation,
about 12 s for a five-row transition and about 9 s for a Mix grid.

## Short answers to likely questions

| Question | Answer |
| --- | --- |
| Why does it work at all? | Concepts are roughly linear directions in the residual stream; adding one shifts every later layer's input towards that concept. |
| Why layer 4 and not where the probe is best (9–14)? | Reading a concept and pushing it differ: an early push passes through 20 more blocks that turn it into word choices; a late push has little room left (E1). |
| Is this just prompting? | No: the prompt is unchanged, and steering moved sentiment about twice as far as asking for a happy story (E3), at a cost in fluency. |
| How do you know it isn't just "any noise changes text"? | The random vector of the same length moves sentiment by at most ±0.35 in no consistent direction (E1). |
| How many examples do you need? | Five pairs give almost the full effect (E4). The quality of the pairs matters more than the number (dogs ↔ cats). |
| What's new in this project? | (1) The happy/sad direction appears abruptly in training, steps 150–250 (E6); (2) a from-scratch 12M model is steerable like a 0.5B one (E5); (3) concept vectors can be added: huge + magic → dragon, tiny + magic → fairy (E8). |
