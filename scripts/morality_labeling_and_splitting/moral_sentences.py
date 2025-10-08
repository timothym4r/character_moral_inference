### The purpose of this file is to find sentences that are potentially morally relevant. That is, sentences
### that contain moral words or phrases that could be considered morally relevant. In this case we will be using
### words from the following files to determine moral relevance:
### - hte_words_valence.csv , source : https://osf.io/me9y6
### - mfd_v2.csv


### The sentences will be extracted from the dialogue.json file and then checked for the existence of potential moral words

import pandas as pd
import numpy as np
import json
from collections import defaultdict
import re

# source_file_path = "../data/dialogue.json"
source_file_path = "../data/new_dialogue.json"

# Read the JSON file of character dialogues
with open(source_file_path, "r") as file:
    raw_dialogue = json.load(file)

# Create a nested dictionary structure
dialogue = {}
for movie, data in raw_dialogue.items():
    # Initialize movie dictionary if not exists
    if movie not in dialogue:
        dialogue[movie] = {}
        
    # If data is a dictionary (contains character information)
    if isinstance(data, dict):
        for character, lines in data.items():
            if isinstance(lines, list):
                dialogue[movie][character] = lines
    # If data is a list (direct dialogues without character info)
    elif isinstance(data, list):
        dialogue[movie]['unknown'] = data

# Loading moral foundation words from the mfd_v2.csv file
moral_df_1 = pd.read_csv("../data/mfd_v2.csv")
moral_words_1 = set(moral_df_1['word'].str.lower())

# Loading moral foundation words from hte_words_valence.csv file
moral_df_2 = pd.read_csv("../data/hte_words_valence.csv")
moral_words_2 = set(moral_df_2['word'].str.lower())

non_moral_df = pd.read_csv("../data/morally_irrelevant_words.csv")
non_moral_words = set(non_moral_df['word'].str.lower())

print(f"Loaded {len(non_moral_words)} non moral words from mfd_v2.csv")

# Combine the two sets of moral words
moral_words = moral_words_1.union(moral_words_2)
moral_words = moral_words - non_moral_words
# Optimized storage using defaultdict for nested dictionary
moral_dialogue = defaultdict(lambda: defaultdict(list))
moral_dialogue_masked = defaultdict(lambda: defaultdict(list))
ground_truths = defaultdict(lambda: defaultdict(list))
neutral_dialogue = defaultdict(lambda: defaultdict(list))

# Precompile regex patterns for all moral words (faster lookup)
moral_word_patterns = {word: re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE) for word in moral_words}

# Counter for progress tracking
movie_index = 0

# Efficient processing of dataset
for movie, characters in dialogue.items():
    movie_index += 1
    for character, lines in characters.items():
        for line in lines:
            lowered_line = line.lower()

            # Fast membership checking using set intersection
            words_in_sentence = set(lowered_line.split())
            matching_words = words_in_sentence & moral_words
            
            if matching_words:
                # Process each matching word (stop after first match for efficiency)
                for moral_word in matching_words:
                    pattern = moral_word_patterns[moral_word]

                    # Check if it's a standalone word using regex
                    if pattern.search(lowered_line):
                        # Store original sentence
                        moral_dialogue[movie][character].append(line)

                        # Mask only the first occurrence of the standalone moral word
                        masked_line = pattern.sub("[MASK]", lowered_line, count=1)
                        moral_dialogue_masked[movie][character].append(masked_line)
                        ground_truths[movie][character].append(moral_word)

                        # Stop after the first match for efficiency
                        break 

            else:
                # If no moral words found, store the line in neutral dialogue
                neutral_dialogue[movie][character].append(line)

    print(f"Finished processing movie {movie_index}")

# Summary of results
print(f"Processed {movie_index} movies")
print(f"Total morally relevant sentences: {sum(len(v) for m in moral_dialogue.values() for v in m.values())}")

# Convert defaultdict to a normal dictionary for saving

# Prepare data for saving

# Data 1 is the full dataset with moral dialogue, masked dialogue, and ground truths
# This will come in handy for training models that require masked language modeling
data_to_save_1 = {
    "moral_dialogue": moral_dialogue,
    "moral_dialogue_masked": moral_dialogue_masked,
    "ground_truths": ground_truths
}

# Data 2 is the original moral dialogue and neutral dialogue
data_to_save_2 = {
    "moral_dialogue": moral_dialogue,
    "neutral_dialogue": neutral_dialogue
}

# Save as JSON
with open("..//data//new_moral_data_full.json", "w") as f:
    json.dump(data_to_save_1, f, indent=4)

# Save as JSON
with open("..//data//new_moral_data_original.json", "w") as f:
    json.dump(data_to_save_2, f, indent=4)

