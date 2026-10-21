"""Example concepts for the demo: name -> (positive side, negative side, tip, example pairs).

α > 0 steers towards the positive (first) side, α < 0 towards the negative side. Examples are
matched pairs where possible, so the two sides differ mainly in the concept. TinyGPT only knows
simple TinyStories English, so it gets its own, simpler set. Each was tried at the best layer
(4) with α = ±8 on CPU before being kept; the tip says what to expect. ("polite ↔ rude" was
tried for Qwen and dropped: the rude side did not get rude.)
"""

QWEN_PRESETS = {
    "formal ↔ casual": ("formal", "casual", "Works both ways at α = ±8: business letters vs 'i was like uhh umm'.", [
        ("I would be grateful if you could kindly review the attached document.", "hey can u check this thing i sent lol"),
        ("We regret to inform you that the meeting has been postponed.", "yo the meeting's off, my bad"),
        ("Please find enclosed the requested information.", "here's the stuff u wanted"),
        ("The committee will convene at your earliest convenience.", "let's hang out whenever, no rush dude"),
        ("Kindly confirm your attendance at your earliest opportunity.", "u coming or nah?"),
        ("I am writing to express my sincere appreciation for your assistance.", "thx a ton for the help!!"),
    ]),
    "about food ↔ about sports": ("food", "sports", "Topic steering, both ways at α = ±8: recipes vs match reports.", [
        ("The soup was rich and creamy, with fresh basil and garlic.", "The striker scored a goal in the final minute of the match."),
        ("She baked a chocolate cake with strawberries on top.", "She won the tennis tournament after a tough five-set final."),
        ("We ordered spicy noodles, dumplings and fried rice.", "The team trained hard for the championship game."),
        ("The chef grilled the fish with lemon and butter.", "He ran the marathon in under three hours."),
        ("Fresh bread, cheese and olives were on the table.", "The goalkeeper saved a penalty and the fans cheered."),
        ("He made pancakes with maple syrup for breakfast.", "They lost the basketball game by two points."),
    ]),
    "English ↔ French": ("English", "French", "Try α = −4: the text switches to French. At −8 the French breaks down.", [
        ("The weather is beautiful today.", "Il fait très beau aujourd'hui."),
        ("I would like a cup of coffee, please.", "Je voudrais une tasse de café, s'il vous plaît."),
        ("My brother lives in a small house near the sea.", "Mon frère habite dans une petite maison près de la mer."),
        ("We are going to the market tomorrow morning.", "Nous allons au marché demain matin."),
        ("The children are playing in the garden.", "Les enfants jouent dans le jardin."),
        ("I don't understand the question.", "Je ne comprends pas la question."),
    ]),
    "confident ↔ uncertain": ("confident", "uncertain", "A subtler tone: the effect is weaker than the others.", [
        ("I am absolutely sure this is the right answer.", "I'm not sure, maybe this is the right answer?"),
        ("This plan will definitely work.", "This plan might work, but I don't really know."),
        ("I know exactly what to do next.", "I'm not sure what to do next."),
        ("There is no doubt that we will win.", "Perhaps we will win, it's hard to say."),
        ("The results clearly prove my point.", "The results could possibly support my point."),
        ("The train leaves at five, I'm certain.", "I think the train leaves at five, but I could be wrong."),
    ]),
}

TINY_PRESETS = {
    "brave ↔ scared": ("brave", "scared", "The scared side (α = −8) works well; the brave side barely changes the story.", [
        ("Tom was brave. He walked into the dark forest.", "Tom was scared. He did not want to go into the dark forest."),
        ("Lily was not afraid. She climbed the big tree.", "Lily was afraid. She did not climb the big tree."),
        ("The little dog was brave and barked at the big bear.", "The little dog was scared and hid from the big bear."),
        ('Ben said, "I am not scared. I can do it!"', 'Ben said, "I am scared. I cannot do it!"'),
        ("Sue was brave and jumped into the cold water.", "Sue was scared of the cold water and cried."),
        ("The boy smiled and went into the dark room.", "The boy shook and ran away from the dark room."),
    ]),
    "dogs ↔ cats": ("dogs", "cats", "The dog side works (barking, wagging tails); the cat side drifts to tea parties: the model knows less about cats.", [
        ("The dog wagged its tail and barked.", "The cat purred and licked its paw."),
        ("Max the dog liked to fetch the ball.", "Kitty the cat liked to chase the yarn."),
        ("The puppy ran to the park with his leash.", "The kitten climbed up the tree and said meow."),
        ("The big dog chewed a bone.", "The big cat drank some milk."),
        ("Tim walked his dog every day.", "Tim fed his cat every day."),
        ("The dog dug a hole in the yard.", "The cat slept in the sunny window."),
    ]),
    "sunny park ↔ rainy day at home": ("sunny park", "rainy day at home", "Works both ways: parks and blue skies vs ceilings and floors.", [
        ("The sun was shining. They played in the park.", "It was raining. They stayed inside the house."),
        ("It was a warm sunny day at the beach.", "It was a cold rainy day, so they played at home."),
        ("The kids ran on the green grass under the blue sky.", "The kids sat by the window and watched the rain."),
        ("They had a picnic outside in the sun.", "They built a fort with pillows in the living room."),
        ("Birds sang in the trees in the park.", "Thunder boomed, and they hid under the blanket."),
        ("It was hot, so they went to swim in the lake.", "It was wet outside, so they read books in bed."),
    ]),
}
