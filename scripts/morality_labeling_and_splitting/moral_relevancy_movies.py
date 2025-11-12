import openai
import os
import json
import random
from tqdm import tqdm
import time
from collections import defaultdict
import re

# Step 1: Load JSON file
with open('../data/moral_data_original.json', 'r') as f:
    data = json.load(f)

# Step 2: Randomly select N movies
n_movies = 20
all_movies = list(data["moral_dialogue"].keys())
selected_movies = random.sample(all_movies, n_movies)

selected_movies = movie_titles = [
    "Wolf of Wall Street, The",
    "Beauty and the Beast",
    "Willow",
    "Air Force One",
    "Fantastic Four",
    "Analyze This",
    "Devil's Advocate",
    "Observe and Report",
    "Hot Tub Time Machine",
    "White Jazz",
    "Judge Dredd",
    "Danish Girl, The",
    "Wild Bunch, The",
    "Imaginarium of Doctor Parnassus, The",
    "Fault in Our Stars, The",
    "Star Wars: Attack of the Clones",
    "Assassins",
    "Game, The",
    "I Still Know What You Did Last Summer",
    "Thor Ragnarok",
]


flattened_data = []

for movie in selected_movies:
    for character, sentences in data["moral_dialogue"][movie].items():
        for sentence in sentences:
            flattened_data.append((movie, character, sentence))

print(f"Randomly selected {n_movies} movies:")
for movie in selected_movies:
    print(f"- {movie}")

# Step 3: Shuffle sentences from selected movies
random.shuffle(flattened_data)

# Step 4: Function to evaluate moral relevance
def check_moral_relevance(sentences, model="gpt-4o-mini", batch_size=5):
    results = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]
        prompt = (
            "The following are sentences spoken by characters in movies.\n\n"
            "We would like to check whether each sentence is morally relevant or not. "
            "A morally relevant sentence is one that reflects ethical or moral issues, values, or principles. "
            "Specifically, a morally relevant sentence will reflect one or more of the following moral foundations:\n\n"
            "1. Care/Harm – Discussing the quality of caring, helping, or discussing possible harms.\n"
            "2. Fairness/Cheating – Concern with fairness, equity, inclusion, or discussions of cheating or opposing cheating.\n"
            "3. Authority/Subversion – Respect for or rejection of authority (e.g., parents, leaders, figures of power).\n"
            "4. Loyalty/Betrayal – Discussions about loyalty, disloyalty, or the importance of sticking together.\n"
            "5. Purity/Degradation – Discussions about religious beliefs, spiritual or physical cleanliness, or degradation.\n"
            "6. Liberty/Oppression – Talking about freedom, oppression, or related concepts.\n"
            "7. General Morality – Any claim stating that something is morally right or wrong, even if not directly tied to the categories above.\n\n"
            "### Examples:\n"
            "- 'Stealing is wrong.' → Yes (General Morality)\n"
            "- 'Helping others is the right thing to do.' → Yes (Care/Harm)\n"
            "- 'You should listen to your father.' → Yes (Authority)\n"
            "- 'We stand together no matter what happens.' → Yes (Loyalty)\n"
            "- 'I need to finish my homework.' → No\n"
            "- 'I'll have the soup, please.' → No\n\n"
            "### Instructions:\n"
            "1. Respond **with 'Yes' or 'No' only** — no additional text or numbering.\n"
            "2. Provide a response for **every sentence**. Do **not** skip any sentences.\n"
            "3. If the sentence is ambiguous, respond with your best judgment based on the moral foundations listed above.\n\n"
            "### Sentences:\n" +
            "\n".join([f"{j + 1}. {sentence}" for j, sentence in enumerate(batch)])
        )

        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20 * len(batch),
                temperature=0
            )
            outputs = response.choices[0].message.content.strip().split('\n')

            # Clean up numbering artifacts like "1. Yes" using regex
            outputs = [re.sub(r"^\d+\.\s*", "", output.strip().lower()) for output in outputs]
            # Keep only the first word ("yes" or "no")
            outputs = [output.split()[0] if output.split() else "" for output in outputs]

            results.extend(outputs)

        except Exception as e:
            print(f"Error processing batch: {e}")
            time.sleep(5)

    return results

# Step 5: Run the evaluation
sentences_to_evaluate = [sentence for _, _, sentence in flattened_data]
results = check_moral_relevance(sentences_to_evaluate)

# Step 6: Rebuild the dictionary with only relevant sentences
filtered_data = defaultdict(lambda: defaultdict(list))

for (movie, character, sentence), result in zip(flattened_data, results):
    if result == "yes":
        filtered_data[movie][character].append(sentence)

# Step 7: Save the filtered dictionary to a new file
filtered_dict = {"moral_dialogue": filtered_data}

with open('../data/moral_data_by_movies_filtered_4o_mini.json', 'w') as f:
    json.dump(filtered_dict, f, indent=4)

print(f"Filtered data saved to '../data/moral_data_by_movies_filtered.json'")