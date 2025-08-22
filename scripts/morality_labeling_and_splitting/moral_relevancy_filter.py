import openai
import os
import json
import re
import pandas as pd
import time
from tqdm import tqdm
from collections import defaultdict

threshold_sent_count = 50  # Minimum sentences per character to process

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load sentences (limit to 100 for testing)
with open('../data/new_moral_data_original.json', 'r') as f:
    data = json.load(f)

log_file = open("../data/moral_failures.txt", "a")

# Function to evaluate moral relevance using different models
def check_moral_relevance(sentences, model="gpt-4o-mini", batch_size=20):
    results = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]
        for attempt in range(3): # Max retries = 3
            prompt = (
                "The following are sentences spoken by characters in movies.\n\n"
                "We would like to check whether each sentence is morally relevant or not. "
                "A morally relevant sentence is one that reflects ethical or moral issues, values, or principles. "
                "Specifically, a morally relevant sentence will reflect one or more of the following moral foundations:\n\n"
                "1. Care/Harm : Discussing the quality of caring, helping, or discussing possible harms.\n"
                "2. Fairness/Cheating : Concern with fairness, equity, inclusion, or discussions of cheating or opposing cheating.\n"
                "3. Authority/Subversion : Respect for or rejection of authority (e.g., parents, leaders, figures of power).\n"
                "4. Loyalty/Betrayal : Discussions about loyalty, disloyalty, or the importance of sticking together.\n"
                "5. Purity/Degradation : Discussions about religious beliefs, spiritual or physical cleanliness, or degradation.\n"
                "6. Liberty/Oppression : Talking about freedom, oppression, or related concepts.\n"
                "7. General Morality : Any claim stating that something is morally right or wrong, even if not directly tied to the categories above.\n\n"
                "### Examples:\n"
                "- 'Stealing is wrong.' → Yes (General Morality)\n"
                "- 'Helping others is the right thing to do.' → Yes (Care/Harm)\n"
                "- 'You should listen to your father.' → Yes (Authority)\n"
                "- 'We stand together no matter what happens.' → Yes (Loyalty)\n"
                "- 'I need to finish my homework.' → No\n"
                "- 'I'll have the soup, please.' → No\n\n"
                "### Instructions:\n"
                "1. Respond **with 'Yes' or 'No' only**, one per line."
                "2. Provide **exactly one response per line of input below** — do not skip or combine any."
                "3. **Do not include sentence numbers in your answers**."
                "4. ⚠️ Your output must contain the same number of lines as the input block below."
                "5. No explanations. Just 'Yes' or 'No' per line."

                "### Sentences:\n" +
                "\n".join([f"{j + 1}. {sentence}" for j, sentence in enumerate(batch)])
            )

            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20 * len(batch),
                    temperature=0,
                )

                outputs = response.choices[0].message.content.strip().split('\n')

                # Clean
                outputs = [re.sub(r"^\d+\.\s*", "", o.strip().lower()) for o in outputs]
                outputs = [o.split()[0] if o.split() and o.split()[0] in {"yes", "no"} else "" for o in outputs]

                if len(outputs) == len(batch):
                    results.extend(outputs)
                    break  # success
                else:
                    print(f"⚠️ Mismatch on attempt {attempt+1}: expected {len(batch)}, got {len(outputs)}")
                    log_file.write(f"[FAILURE] {movie} / {character} — Expected {len(batch)} responses, got {len(outputs)}\n")
                    log_file.flush()  # ensures it's written immediately

                    time.sleep(1)

            except Exception as e:
                print(f"Error processing batch: {e}")
                time.sleep(5)
        else:
            print(f"Failed to get valid response after {3} attempts. Padding.")
            log_file.write(f"[GAVE UP] {movie} / {character} — Failed after 3 attempts\n")
            log_file.flush()

            results.extend(outputs)

    return results

json_path = "../data/new_morality_check_gpt4omini.json"

if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
    try:
        with open(json_path, "r") as f:
            morality_check = json.load(f)
    except json.JSONDecodeError:
        print("JSON file exists but is invalid or incomplete. Starting fresh.")
        morality_check = {}
else:
    morality_check = {}

# Main loop
for movie, characters in tqdm(data["moral_dialogue"].items(), desc="Processing movies"):
    for character, sentences in characters.items():
        if len(sentences) >= threshold_sent_count:
            # Skip already-processed characters
            if movie in morality_check and character in morality_check[movie]:
                print(f"Skipping {movie} / {character} (already processed)")
                continue

            print(f"Processing {movie} / {character} ({len(sentences)} sentences)")
            results = check_moral_relevance(sentences, model="gpt-4o-mini", batch_size=10)
            print(f"Finished processing {movie} / {character} : {len(results)} results")
            # Write result for this character
            if movie not in morality_check:
                morality_check[movie] = {}
            morality_check[movie][character] = results

            # Save immediately after each character
            with open(json_path, 'w') as f:
                json.dump(morality_check, f, indent=2)

            print(f"Saved results for {character} in {movie}")

log_file.close()