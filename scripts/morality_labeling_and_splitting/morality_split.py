import json
import os
from collections import defaultdict

# Lets set the threshold for the number of sentences per character
threshold_sent_count = 50

with open("../data/new_morality_check_gpt4omini.json", "r") as f:
    morality_check = json.load(f)

with open("../data/new_moral_data_full.json", "r") as f:
    moral_data = json.load(f)

# Only check characters with 100+ sentences
moral_dialogue_100 = defaultdict(dict)
for movie, characters in moral_data["moral_dialogue"].items():
    for character, sentences in characters.items():
        if len(sentences) >= threshold_sent_count:
            moral_dialogue_100[movie][character] = sentences

moral_masked_100 = defaultdict(dict)
for movie, characters in moral_data["moral_dialogue_masked"].items():
    for character, sentences in characters.items():
        if len(sentences) >= threshold_sent_count:
            moral_masked_100[movie][character] = sentences

ground_truth_100 = defaultdict(dict)
for movie, characters in moral_data["ground_truths"].items():
    for character, sentences in characters.items():
        if len(sentences) >= threshold_sent_count:
            ground_truth_100[movie][character] = sentences


def nested_dict():
    return defaultdict(lambda: defaultdict(list))

moral_only_sentences = defaultdict(nested_dict)
non_moral_sentences = defaultdict(nested_dict)

# Match the checking results with the original data
for movie, characters in morality_check.items():
    for character, sentences in characters.items():
        if character in morality_check[movie]:
            for i, sentence in enumerate(morality_check[movie][character]):
                if sentence == "yes":
                    try:
                        moral_only_sentences["moral_dialogue"][movie][character].append(moral_dialogue_100[movie][character][i])
                        moral_only_sentences["moral_dialogue_masked"][movie][character].append(moral_masked_100[movie][character][i])
                        moral_only_sentences["ground_truth"][movie][character].append(ground_truth_100[movie][character][i])
                    except IndexError:
                        print(f"Index error for {movie}, {character}, index {i}.")

                else:
                    try:
                        non_moral_sentences["moral_dialogue"][movie][character].append(moral_dialogue_100[movie][character][i])
                        non_moral_sentences["moral_dialogue_masked"][movie][character].append(moral_masked_100[movie][character][i])
                        non_moral_sentences["ground_truth"][movie][character].append(ground_truth_100[movie][character][i])
                    except IndexError:
                        print(f"Index error for {movie}, {character}, index {i}.")


# Save the moral only and non-moral sentences
with open("../data/new_moral_only_data.json", "w") as f:
    json.dump(moral_only_sentences, f, indent=2)

with open("../data/new_non_moral_data.json", "w") as f:
    json.dump(non_moral_sentences, f, indent=2)


