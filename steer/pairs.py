"""Contrastive text pairs (differ only in the concept) and neutral prompts for steering.

Pairs come from templates x word pairs, so the two texts of a pair are identical except for the
sentiment word. Templates are split into "train" (used to build vectors) and "heldout" (used to
test probes), so no template appears in both.
"""
import random

SENTIMENT_WORDS = [  # (positive, negative)
    ("wonderful", "terrible"), ("amazing", "awful"), ("lovely", "horrible"),
    ("delightful", "miserable"), ("fantastic", "dreadful"), ("joyful", "depressing"),
    ("beautiful", "painful"), ("great", "disappointing"), ("happy", "sad"), ("perfect", "gloomy"),
]

SENTIMENT_TEMPLATES = [
    "The trip to the beach was {}.",
    "My day at work was {}.",
    "The movie we watched last night was {}.",
    "Dinner with my family was {}.",
    "Honestly, the party was {}.",
    "Our new apartment is {}.",
    "The concert yesterday was {}.",
    "The book I finished today was {}.",
    "Meeting my old friends again was {}.",
    "The train ride home was {}.",
    "This year at college has been {}.",
    "The food at the new restaurant was {}.",
    "Today felt {}.",
    "The news I got this morning was {}.",
    "Spending the weekend at home was {}.",
    # held-out templates
    "The walk in the park was {}.",
    "My birthday this year was {}.",
    "The exam results were {}.",
    "Visiting my grandparents was {}.",
    "The first day of the holidays was {}.",
]
N_TRAIN_TEMPLATES = 15

NEUTRAL_PROMPTS = [
    "I went to the market today and",
    "Yesterday my brother called me and",
    "When I opened the door,",
    "The meeting this afternoon",
    "After class, we decided to",
    "My neighbour has a dog that",
    "This morning I woke up and",
    "The bus was late again, so",
    "On Sunday we visited",
    "I looked out of the window and",
    "My teacher told us that",
    "The new phone I bought",
    "At the station, a man",
    "We cooked dinner together and",
    "The letter from my aunt said",
    "In the evening, the city",
    "My best friend and I",
    "The old house at the end of the street",
    "During the lunch break,",
    "I started reading a new book and",
    "The football match on Saturday",
    "When the rain started,",
    "Our flight to Delhi",
    "The cat jumped onto the table and",
    "My mother asked me to",
    "At the office, everyone",
    "The garden behind our house",
    "I finally finished my project and",
    "The shopkeeper looked at me and",
    "We walked along the river and",
    "The phone rang late at night and",
    "My little sister drew a picture of",
    "The first thing I noticed was",
    "On the way to school,",
    "The doctor said that",
    "Last summer, my family",
    "The library was quiet and",
    "I opened my laptop and",
    "The village where my grandfather lived",
    "The train stopped in the middle of nowhere and",
    "My roommate came home and",
    "At the wedding, the guests",
    "The weather report for tomorrow",
    "The children in the park",
    "I wrote a message to my friend saying",
    "The waiter brought our food and",
    "When the exam ended,",
    "The street outside my window",
    "My father told me a story about",
    "The new semester",
]

# Naturally written sentences for the probe. They avoid every word in SENTIMENT_WORDS, so the
# model has to infer the sentiment instead of matching a word it saw in the training pairs.
PROBE_POSITIVE = [
    "I got the scholarship I applied for!",
    "We won the final match in the last minute.",
    "My grandmother is finally home from the hospital and doing well.",
    "She said yes when I asked her to marry me.",
    "The whole class cheered when the results came out.",
    "Our puppy learned to fetch the ball today.",
    "I passed my driving test on the first try.",
    "The surprise party made my dad laugh until he cried.",
    "We spent the evening dancing under the stars.",
    "My little brother took his first steps this morning.",
    "The interview went so well that they hired me on the spot.",
    "Everyone loved the cake I baked for the festival.",
    "After months of practice, our band played to a full hall.",
    "The doctor told us the baby is healthy.",
    "I finally saw the ocean for the first time.",
    "My friends sang for me when I walked in.",
    "The garden is full of flowers this spring.",
    "We laughed together all the way home.",
    "Our team finished the project ahead of time and the client was thrilled.",
    "I hugged my mother at the airport after two years apart.",
]
PROBE_NEGATIVE = [
    "My dog died yesterday.",
    "I failed the exam I studied for all month.",
    "We lost the final match in the last minute.",
    "My grandmother is still in the hospital and getting worse.",
    "She left without saying goodbye.",
    "The flood destroyed our house and everything in it.",
    "I was fired this morning without any reason.",
    "Nobody came to my birthday dinner.",
    "The doctor said the treatment did not work.",
    "We argued all night and stopped talking to each other.",
    "My wallet was stolen on the train.",
    "The village was empty and silent after the fire.",
    "I spent the whole holiday alone in my room.",
    "Our flight was cancelled and we missed the wedding.",
    "He cried at the funeral and could not stop.",
    "The company closed and hundreds of people lost their jobs.",
    "My best friend moved away and never wrote back.",
    "The crops failed again this year and the family went hungry.",
    "I broke my leg the day before the race.",
    "The letter said my application was rejected.",
]


def sentiment_pairs(split="train"):
    """List of (positive_text, negative_text)."""
    templates = SENTIMENT_TEMPLATES[:N_TRAIN_TEMPLATES] if split == "train" else SENTIMENT_TEMPLATES[N_TRAIN_TEMPLATES:]
    return [(t.format(p), t.format(n)) for t in templates for p, n in SENTIMENT_WORDS]


def sample_pairs(pairs, n, seed=0):
    """A fixed random subset of n pairs (for the 'how many pairs are needed' experiment)."""
    return random.Random(seed).sample(pairs, min(n, len(pairs)))


# ---------------------------------------------------------------- TinyGPT (TinyStories English)

TINY_WORDS = [  # (positive, negative), words a TinyStories model knows
    ("happy", "sad"), ("glad", "upset"), ("excited", "scared"),
    ("proud", "angry"), ("cheerful", "grumpy"), ("joyful", "lonely"),
]

TINY_TEMPLATES = [
    "Lily went to the park. She felt very {}.",
    "Tom looked at his toy. He was {}.",
    "The little dog saw the ball and was {}.",
    "Mom came home. She was {}.",
    "Ben played with his friend. He felt {}.",
    "The bird sat in the tree. It was {}.",
    "Sue got a new dress. She was very {}.",
    "The boy saw the big cake. He was so {}.",
    "After school, Amy was {}.",
    "The cat looked out of the window. It felt {}.",
    "Max and Lily went to the beach. They were {}.",
    "Grandpa read a book. He was {}.",
]

TINY_PROMPTS = [
    "Once upon a time, there was a little girl named Lily. One day, she",
    "Tom had a red ball. He",
    "One day, a big dog went to the park. The dog",
    "Sam and his mom went to the store. They",
    "There was a small cat who lived in a house. Every day, the cat",
    "Ben woke up early in the morning. He",
    "Once upon a time, there was a boy named Max. Max",
    "Sue found a box in the garden. She",
    "The sun was high in the sky. Lily and Tom",
    "Grandma made soup for lunch. Then",
    "A little bird sat on a branch. The bird",
    "Amy went to school with her bag. At school,",
    "Once upon a time, a frog lived near a pond. One day,",
    "Dad took Ben to the beach. They",
    "There was a big tree in the yard. Under the tree,",
    "Lucy had a teddy bear. She",
    "One day, Tim saw a truck on the road. The truck",
    "Mia and her brother played in the snow. They",
    "Once upon a time, there was a little boat. The boat",
    "The kids went to the farm. At the farm,",
    "Jack wanted to build a tower. He",
    "Once upon a time, a mouse found a piece of cheese. The mouse",
    "Anna opened the door and saw",
    "The rabbit hopped into the forest. There,",
    "Ella's dad gave her a book. She",
    "One night, Tom looked at the stars. He",
    "The puppy ran to the gate. Then",
    "Once upon a time, there was a king. The king",
    "Lily drew a picture of her house. Then she",
    "It was raining outside. Sam",
]

# natural TinyStories-style sentences for the probe; none use a word from TINY_WORDS
TINY_PROBE_POSITIVE = [
    "Lily got a new puppy for her birthday.",
    "Tom won the race and everyone clapped.",
    "Mom gave Ben a big hug and a cookie.",
    "The sun came out and the children played all day.",
    "Sam found his lost teddy bear.",
    "They had a fun picnic with cake and juice.",
    "Grandma came to visit with many presents.",
    "The bird learned to fly high in the sky.",
    "Amy and her friend laughed and played together.",
    "Dad took them to the zoo to see the lions.",
    "The little boat sailed home safe.",
    "Everyone sang songs at the party.",
    "Ben made a new friend at the park.",
    "The flowers in the garden grew big and pretty.",
    "Lucy got a gold star from her teacher.",
]
TINY_PROBE_NEGATIVE = [
    "Lily lost her favorite doll.",
    "Tom fell down and hurt his knee.",
    "The rain ruined the picnic.",
    "Ben's balloon flew away and he could not get it.",
    "The puppy was lost in the dark forest.",
    "Mom yelled at Sam for breaking the vase.",
    "The toy car broke into many pieces.",
    "Nobody wanted to play with Amy.",
    "The bird hurt its wing and could not fly.",
    "A big storm knocked down their house.",
    "Max dropped his ice cream on the ground.",
    "The old dog was sick and would not eat.",
    "Sue was alone in the cold room.",
    "The children fought over the ball.",
    "Grandpa went away and did not come back.",
]


def tiny_sentiment_pairs():
    return [(t.format(p), t.format(n)) for t in TINY_TEMPLATES for p, n in TINY_WORDS]
