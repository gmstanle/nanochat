"""
Task intended to make nanochat better in spelling and counting, for example:

"How many r are in strawberry?" -> 3

An interesting part of this task is that we will get the assistant to
solve the problem using a combination of manual counting and Python.
This is a good problem solving "instinct" to mix into the model and RL
may further refine it to trust one over the other. If we were extra fancy
(which we could/should be) we'd add small errors here and there to allow
the model also learn recoveries. We can do this in future versions.

There are two tasks in this file:
1. SpellingBee: Counting the number of occurrences of a letter in a word
2. SimpleSpelling: Simply spelling words

(1) is the goal, but (2) exists as a highly condensed version of the part
that makes (1) difficult, which is word spelling. This is non-trivial for an
LLM because it has to learn how every token (a little semantic chunk/atom)
maps to the sequence of individual characters that make it up. Larger models
learn this eventually on their own, but if we want this capability to exist
in smaller models, we have to actively encourage it by over-representing it
in the training data. Midtraining is a good place to do this.

To preview a few example conversations, run:
python -m tasks.spellingbee
"""

import re
import random
from tasks.common import Task
from nanochat.common import download_file_with_lock

# Letters of the alphabet
LETTERS = "abcdefghijklmnopqrstuvwxyz"
# A list of 370K English words of large variety
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
# A number bigger than 370K to separate train and test random seeds
TEST_RANDOM_SEED_OFFSET = 10_000_000

# Identical to gsm8k's answer extraction
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    """
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


# User message templates for data augmentation
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]


def make_word_letters(word: str) -> str:
    return ",".join(list(word))


def misspell_word(word: str) -> str:
    """To misspell a word, correctly spell the first k letters and
    then randomly select the next (m - k) letters, where 0 < k < n and k < m < 2*n.
    """

    n = len(word)
    k = random.randrange(start=1, stop=(n - 1))
    m = random.randrange(start=1, stop=(2 * n - k))

    word_mispelled = word[:k] + "".join(random.choices(LETTERS, k=m))

    # check that mispelled word is short but nonzero length.
    assert len(word_mispelled) <= 2 * n

    return word_mispelled


class SpellingBeeConversation:

    def __init__(
        self,
        template: str,
        word,
        letter,
        letter_quote,
        word_quote,
        usr_msg_suffix,
        mistake,
    ) -> None:

        self.template = template
        self.word = word
        self.letter = letter

        # Create user message
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        self.user_msg = (
            template.format(letter=letter_wrapped, word=word_wrapped) + usr_msg_suffix
        )

        # Create "correct" assistant message.
        self.createAssistantParts(mistake=mistake)

        # Create the full conversation
        self.messages = [
            {"role": "user", "content": self.user_msg},
            {"role": "assistant", "content": self.assistant_parts},
        ]
        self.conversation = {
            "messages": self.messages,
        }

    def countPython(self):
        self.assistant_parts.append(
            {"type": "text", "text": "\n\nLet me check this using Python:\n\n"}
        )
        self.count = self.word.count(self.letter)

        # Python tool call
        python_expr = f"'{self.word}'.count('{self.letter}')"
        self.assistant_parts.append({"type": "python", "text": python_expr})

        # Python output
        self.assistant_parts.append({"type": "python_output", "text": str(self.count)})
        self.assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {self.count}.\n"})

    def countManual(self, word_letters):
        """Create the text for manually counting self's word using word_letters. word_letters
        is specified separately so that a mispelling can be generated."""

        countText = f"\nFirst spell the word out:\n\n{self.word}:{word_letters}.\n\nThen count the occurrences of '{self.letter}':\n"

        # Remove commas
        word_letters = [w for w in word_letters if w != ","]
        # Part 1: spelling the word out letter by letter and counting.
        self.manualCount = 0
        for i, char in enumerate(word_letters, 1):
            if char == self.letter:
                self.manualCount += 1
                # note: there deliberately cannot be a space here between i and char
                # because this would create a different token! (e.g. " a" and "a" are different tokens)
                countText += f"{i}:{char} hit! count={self.manualCount}\n"
            else:
                countText += f"{i}:{char}\n"

        countText += f"\nThis gives us {self.manualCount}.\n"
        self.assistant_parts.append({"type": "text", "text": countText})

    def createAssistantParts(self, mistake=False):
        # Now create the ideal assistant response - build as parts (text + tool calls)
        self.assistant_parts = []

        if mistake:
            word_letters = make_word_letters(misspell_word(self.word))
        else:
            word_letters = make_word_letters(self.word)
        
        # Check that the number of occurences of letter is actually different in misspelled word. if not,
        # we just skip the mistake part. This cleanly deals with various edge cases.
        if mistake and word_letters.count(self.letter) == self.word.count(self.letter):
            mistake = False
            word_letters = make_word_letters(self.word)

        # Part 1: Manually count letters.
        self.assistant_parts.append(
            {
                "type": "text",
                "text": f"We are asked to find the number '{self.letter}' in the word '{self.word}'. Let me try a manual approach first.",
            }
        )
        self.countManual(word_letters=word_letters)

        # Part 2: Count letters with python tool call
        self.countPython()

        # Correction, if assistant made a mistake.
        if mistake:
            self.assistant_parts.append(
                {
                    "type": "text",
                    "text": "Oops: the manual and Python counts disagree. I must have made a mistake. Let me spell the word out again:\n",
                }
            )
            word_letters = make_word_letters(self.word)
            self.countManual(word_letters=word_letters)
            self.countPython()

        # Part 5: Final answer
        if mistake:
            self.assistant_parts.append(
                {
                    "type": "text",
                    "text": f"\n\nMy final answer is:\n\n#### {self.count}",
                }
            )
        self.assistant_parts.append(
            {
                "type": "text",
                "text": f"\n\nPython gives us {self.count}.\n\nMy final answer is:\n\n#### {self.count}",
            }
        )


class SpellingBee(Task):

    def __init__(self, size=1000, split="train", use_mistakes=False, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split
        # Only use mistakes in train data, not eval.
        if split == "train":
            self.use_mistakes = use_mistakes
        else:
            self.use_mistakes = False
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f]
        self.words = words

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.size

    def get_example(self, index):
        # TODO: does this work right? I'm not seeing any determinism with random.Random calls.
        # Best test would be that train and test set of words are disjoint (?).
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        # pick a random word
        word = rng.choice(self.words)
        # pick a letter from it (90%) or a random letter (10%)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)

        # Make a mistake 30% of the time.
        # TODO: code this as a hyperparam.
        if self.use_mistakes:
            mistake = rng.random() < 0.3
        else:
            mistake = False

        # create a user message, with a bunch of variations as data augmentation
        template = rng.choice(USER_MSG_TEMPLATES)
        # 30% chance to lowercase the template (lazy people don't use shift)
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ["", "'", '"']
        letter_quote = rng.choice(quote_options)  # is the letter quoted?
        word_quote = rng.choice(quote_options)  # is the word quoted?
        usr_msg_suffix = ""
        if rng.random() < 0.5:  # 50% of people don't even use question marks
            usr_msg_suffix = "?"

        example = SpellingBeeConversation(
            template=template,
            word=word,
            letter=letter,
            word_quote=word_quote,
            usr_msg_suffix=usr_msg_suffix,
            letter_quote=letter_quote,
            mistake=mistake,
        )
        return example.conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Identical to gsm8k's evaluation.
        """
        assert isinstance(
            assistant_response, str
        ), "Assuming simple string response for now"
        # First extract the ground truth answer from the conversation
        assistant_message = conversation["messages"][-1]
        assert (
            assistant_message["role"] == "assistant"
        ), "Last message must be from the Assistant"
        assert isinstance(
            assistant_message["content"], list
        ), "This is expected to be a list of parts"
        # The last text part contains the final answer with ####
        last_text_part = assistant_message["content"][-1]["text"]
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """Use simple 0-1 reward just like gsm8k."""
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float


class SimpleSpelling(Task):
    """Much simpler task designed to get the model to just practice spelling words."""

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words)  # use a different word order than the SpellingBee task
        self.words = words

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        # pick a random word
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        # return the full conversation
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"},
        ]
        conversation = {
            "messages": messages,
        }
        return conversation


if __name__ == "__main__":

    # preview the SpellingBee task, first 10 examples
    task = SpellingBee()
    for i in range(10):
        ex = task.get_example(i)
        print("=" * 100)
        print(ex["messages"][0]["content"])
        print("-" * 100)
        # Assistant content is now a list of parts
        assistant_parts = ex["messages"][1]["content"]
        for part in assistant_parts:
            if part["type"] == "text":
                print(part["text"], end="")
            elif part["type"] == "python":
                print(f"<<{part['text']}=", end="")
            elif part["type"] == "python_output":
                print(f"{part['text']}>>", end="")
        print()
        print("-" * 100)

    # # preview the SimpleSpelling task, first 10 examples
    # task = SimpleSpelling()
    # for i in range(10):
    #     ex = task.get_example(i)
    #     print("=" * 100)
    #     print(ex['messages'][0]['content'])
    #     print("-" * 100)
    #     print(ex['messages'][1]['content'])

    # # also scrutinize the tokenization (last example only)
    # from nanochat.tokenizer import get_tokenizer
    # tokenizer = get_tokenizer()
    # ids, mask = tokenizer.render_conversation(ex)
    # print(tokenizer.visualize_tokenization(ids, mask, with_token_id=True))
