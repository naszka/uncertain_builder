import random
import re
from gridworld.tasks.task import Task


colors_to_hotbar = {
    "blue": 1,
    "green": 2,
    "red": 3,
    "orange": 4,
    "purple": 5,
    "yellow": 6,
}

color_strings = ["blue", "yellow", "green", "orange", "purple", "red"]

number_strings = [
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


class QuestionTask(Task):
    def __init__(self, *args, **kwargs):
        is_q = kwargs.pop("is_question")
        super().__init__(*args, **kwargs)
        self.is_question = is_q


def color_ambiguity(s):
    def convert(words):
        out_words = []
        found = False
        # Iterate through the list of words
        out_words.append(words[0])
        for i in range(1, len(words) - 1):
            # Check if the current word is a number
            if (
                (words[i] in color_strings)
                and (words[i + 1].strip(".") in ("block", "blocks"))
                and not found
            ):
                found = True
                continue
            out_words.append(words[i])
        out_words.append(words[-1])
        # Join the list of words back into a string and return it
        return out_words

    words = s.split()
    words = convert(words)

    return " ".join(words)


def number_ambiguity(s):
    def convert_plural(words):
        out_words = []
        found = False
        # Iterate through the list of words
        for i in range(len(words)):
            # Check if the current word is a number
            if (words[i] in number_strings) and not found:
                found = True
                continue
            out_words.append(words[i])
        # Join the list of words back into a string and return it
        return out_words

    words = s.split()
    words = convert_plural(words)

    return " ".join(words)


def direction_ambiguity(s):
    def remove_direction(words):
        out_words = []
        # Iterate through the list of words
        for i in range(len(words)):
            # Check if the current word is a number
            if words[i] in ["horizontal", "horizontally", "vertical", "vertically"]:
                continue
            out_words.append(words[i])
        # Join the list of words back into a string and return it
        return out_words

    words = s.split()
    words = remove_direction(words)

    return " ".join(words)


def orientation_ambiguity(s):
    def convert(words):
        out_words = []
        # Iterate through the list of words
        for i in range(len(words)):
            # Check if the current word is a number
            if (words[i].lower() in ["facing", "heading"]) and (
                words[i + 1].lower().strip(",") in ["north", "east", "west", "south"]
            ):
                continue
            if (
                words[i].lower().lower().strip(",")
                in ["north", "east", "west", "south"]
            ) and (words[i - 1].lower() in ["facing", "heading"]):
                continue
            out_words.append(words[i])

        # Join the list of words back into a string and return it
        return out_words

    words = s.split()
    words = convert(words)

    return " ".join(words)


def refers_to_past(s):
    pattern = r"the\s(blue|yellow|green|orange|purple|red)"
    first = s.split(".")[0]

    # Compile the pattern
    regex = re.compile(pattern)
    return True if regex.search(first) else False


def overwrite_referenced_color(
    initial_world_blocks, last_instruction, history_utterances
):
    pattern = r"the\s(blue|yellow|green|orange|purple|red)"
    # Compile the pattern
    regex = re.compile(pattern)
    color = regex.search(last_instruction).group(0).split()[-1]
    rest_of_colors = set(color_strings) - set([color])
    color_id = colors_to_hotbar[color]
    replacement_color = random.choice(list(rest_of_colors))
    color_overwritten = []
    history_overwritten = []
    for block in initial_world_blocks:
        if block[-1] == color_id:
            new_block = (
                block[0],
                block[1],
                block[2],
                colors_to_hotbar[replacement_color],
            )
            color_overwritten.append(new_block)
        else:
            color_overwritten.append(block)
    for utt in history_utterances:
        history_overwritten.append(utt.replace(color, replacement_color))
    return color_overwritten, history_overwritten


def remove_referenced_color(initial_world_blocks, last_instruction, history_utterances):
    pattern = r"the\s(blue|yellow|green|orange|purple|red)"
    # Compile the pattern
    regex = re.compile(pattern)
    color = regex.search(last_instruction).group(0).split()[-1]
    color_id = colors_to_hotbar[color]
    blocks_overwritten = []
    history_overwritten = []
    for block in initial_world_blocks:
        if block[-1] != color_id:
            blocks_overwritten.append(block)
    for utt in history_utterances:
        if color not in utt:
            history_overwritten.append(utt)

    return blocks_overwritten, history_overwritten
