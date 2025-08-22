import json
import os
from collections import defaultdict

# Lets set the threshold for the number of sentences per character
threshold_sent_count = 50

with open("../data/new_morality_check_gpt4omini_structured.json", "r") as f:
    morality_check = json.load(f)

with open("../data/structured_data.json", "r") as f:
    structured_data = json.load(f)

def nested_dict():
    return defaultdict(lambda: defaultdict(list))

# Only check characters with 100+ sentences
moral_dialogue_100 = defaultdict(dict)
for movie, data in structured_data.items():
    characters = data.get("characters", {})
    for character, char_data in characters.items():
        sentences = char_data.get("sentences", [])
        if len(sentences) >= threshold_sent_count:   # minimum threshold
            moral_dialogue_100[movie][character] = sentences

moral_only_sentences = defaultdict(lambda: defaultdict(list))
non_moral_sentences = defaultdict(lambda: defaultdict(list))

# Match the checking results with the original data
for movie, characters in morality_check.items():
    for character, sentences in characters.items():
        if character in morality_check[movie]:
            for i, sentence in enumerate(morality_check[movie][character]):
                if sentence == "yes":
                    try:
                        moral_only_sentences[movie][character].append(moral_dialogue_100[movie][character][i])
                    except IndexError:
                        print(f"Index error for {movie}, {character}, index {i}.")

                else:
                    try:
                        non_moral_sentences[movie][character].append(moral_dialogue_100[movie][character][i])
                    except IndexError:
                        print(f"Index error for {movie}, {character}, index {i}.")


# Save the moral only and non-moral sentences
with open("../data/new_moral_dialogue_structured.json", "w") as f:
    json.dump(moral_only_sentences, f, indent=2)

with open("../data/new_non_moral_dialogue_structured.json", "w") as f:
    json.dump(non_moral_sentences, f, indent=2)


