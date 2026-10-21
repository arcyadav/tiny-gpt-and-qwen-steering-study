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
    "rich ↔ poor": ("rich", "poor", "Works both ways at α = ±8. In the Transition tab try 'For dinner tonight, my mother' (seed 1): not enough rice → a five-star hotel.", [
        ("We flew to Paris on our private jet for the weekend.", "We could not afford the bus fare, so we walked to town."),
        ("My father bought me a new sports car for my birthday.", "My father could not buy me shoes for my birthday."),
        ("Dinner was served by the butler in our marble dining hall.", "Dinner was a little rice that my mother shared among six of us."),
        ("She spent thousands of dollars on designer bags without a thought.", "She counted every coin before buying bread for the family."),
        ("Our mansion has a pool, a cinema and twelve bedrooms.", "Our one-room hut has a leaking roof and no electricity."),
        ("He gave his kids expensive gifts and a large allowance.", "He worked all day for a few coins and still went to bed hungry."),
    ]),
    "old ↔ young": ("old", "young", "Works both ways at α = ±8. In the Transition tab try 'The person sitting next to me on the bus was': a schoolkid → an 86-year-old.", [
        ("The old man sat on the porch, remembering the war and his late wife.", "The young boy sat on the porch, thinking about his homework and his best friend."),
        ("My grandmother is eighty-five and walks slowly with her cane.", "My little sister is five and runs everywhere in her sneakers."),
        ("He retired forty years ago and now spends his days with his grandchildren.", "He started school two years ago and now spends his days with his classmates."),
        ("The elderly woman put on her reading glasses and sighed about her bad knees.", "The teenage girl put on her headphones and complained about her math test."),
        ("After a long life, the old professor wrote his memoirs.", "After his first week of college, the young student called his mom."),
        ("Grandpa told us how things were back in his day.", "The kid told us about the cartoon he watched today."),
    ]),
    "Indian ↔ American": ("Indian", "American", "The Indian side works at α = +4 to +8 (temples, mangoes, rupees); the American side barely changes, because the model's default setting is already American.", [
        ("In the morning, my mother made chai and parathas for everyone.", "In the morning, my mother made coffee and pancakes for everyone."),
        ("We took an auto-rickshaw to the crowded market in Delhi.", "We took the subway to the crowded mall in New York."),
        ("After school, the boys played cricket in the gully.", "After school, the boys played baseball in the backyard."),
        ("For Diwali, we lit diyas and burst crackers with our cousins.", "For Thanksgiving, we carved a turkey and watched football with our cousins."),
        ("My grandmother wore a cotton saree and prayed at the temple.", "My grandmother wore a wool sweater and prayed at the church."),
        ("We ate dal, rice and roti with our hands for lunch.", "We ate burgers and fries with a fork and knife for lunch."),
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
    "huge ↔ tiny": ("huge", "tiny", "Works both ways at α = ±8. In the Transition tab try 'Lily went to the zoo and saw a' (seed 1): a little bird → an enormous lion.", [
        ("The dog was huge. It was as big as a house.", "The dog was tiny. It was as small as a shoe."),
        ("Tom saw a giant tree. It was so tall it touched the sky.", "Tom saw a little tree. It was so short it touched his knee."),
        ("Lily found a big, big box. It was too heavy to lift.", "Lily found a small, small box. It was light as a feather."),
        ("The elephant was very large and had giant feet.", "The ant was very small and had tiny feet."),
        ("Ben built a huge tower. It was taller than his dad.", "Ben built a tiny tower. It was shorter than his cup."),
        ("The ship was enormous and could carry many people.", "The boat was tiny and could carry one bug."),
    ]),
    "sharing ↔ greedy": ("sharing", "greedy", "Needs a prompt with a thing to share and a second child, e.g. 'Lily found a shiny red ball in the park. Her friend asked to play. Lily'.", [
        ("Tom had a big cookie. He gave half to his friend.", "Tom had a big cookie. He ate it all and gave none."),
        ("Lily found many toys. She shared them with everyone.", "Lily found many toys. She kept them all for herself."),
        ("Ben had three candies. He gave one to Sue and one to Max.", "Ben had three candies. He hid them so Sue and Max got none."),
        ('The little girl smiled and said, "You can play with my ball."', 'The little girl frowned and said, "No, the ball is mine!"'),
        ("Sam saw his friend was sad, so he gave him his apple.", "Sam saw his friend was sad, but he kept his apple."),
        ("Mia let her brother have the biggest piece of cake.", "Mia took the biggest piece of cake and wanted more."),
    ]),
    "magic ↔ everyday": ("magic", "everyday", "Keep |α| ≤ 6: at +8 the story loops on 'wand wand wand'. Try 'Tom went for a walk in the woods. Suddenly, he saw a': an apple on a bench → a giant dragon.", [
        ("Tom found a magic wand and turned his cat into a dragon.", "Tom found a pencil and drew a picture of his cat."),
        ("Lily met a fairy who could grant three wishes.", "Lily met a teacher who helped her read three books."),
        ("A wizard flew over the castle on his broom.", "A man rode his bike over the bridge to work."),
        ("The princess talked to a unicorn in the enchanted forest.", "The girl talked to her mom in the kitchen."),
        ("Ben drank a magic potion and became as small as a bug.", "Ben drank a glass of milk and went to bed."),
        ("A friendly giant lived in a cloud above the kingdom.", "A friendly farmer lived in a house near the town."),
    ]),
}
