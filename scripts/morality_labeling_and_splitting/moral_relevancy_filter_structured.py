import openai
import os
import json
import re
import pandas as pd
import time
from tqdm import tqdm
from collections import defaultdict

openai.api_key = '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'

threshold_sent_count = 50  # Minimum sentences per character to process

system_prompt = """You are a precise text classifier. Your task is to decide whether a sentence is morally relevant or morally irrelevant.
A sentence is morally relevant if it explicitly or clearly implies a judgment about fairness, justice, duty, honesty, loyalty, harm, help, kindness, cruelty, purity, sin, morality, betrayal, promises, or similar concepts — even if expressed informally.
If the sentence lacks any moral evaluation or implication, it is morally irrelevant.

Treat as morality cues (even informally):
- unfair, fair, justice, honest, dishonest, lie, cheat, steal, deserve, owe, duty, right/wrong (moral sense), immoral, evil, kind, cruel, harm, help, betray, loyal, freedom, oppress, pure/impure, sin, “least they can do”, “stole my time/life/years”, promise, bail/bailing, deserve, “should(n’t) have” (in moral sense).

Not moral by themselves:
- Bare questions, logistics/plans, agreement words like “right” meaning “correct/okay”, statements of fact without moral judgment.

Return only "Yes" if the sentence is morally relevant, "No" if not.

Formatting rules:
- Output EXACTLY one line per input, in the form: <id>\t<label>
- <label> is "Yes" or "No" only.
- Do NOT add explanations or extra lines.
- If unsure, output "No"
"""

user_prompt_template = """Classify the following sentences. One output line per input.

EXAMPLE INPUT:
1\t"That's disgusting."
2\t"Text your brother."
3\t"It's not fair. I did all the work."
4\t"We’ll always be friends. 'Cause we love each other."
5\t"If you don’t buy the alcohol, I will kill you."

EXAMPLE OUTPUT:
1\tNo
2\tNo
3\tYes
4\tNo
5\tYes

END OF EXAMPLE

Now produce the output for the following INPUT only:

INPUT:
"""

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load sentences (limit to 100 for testing)
with open('../data/dump/structured_data.json', 'r') as f:
    data = json.load(f)

log_file = open("../data/moral_failures_structured.txt", "a")

def make_tsv_input(sentences):
    """
    Returns the INPUT block for the prompt, enumerated as:
        <id>\t"<sentence>"
    """
    lines = ["INPUT:"]
    for i, s in enumerate(sentences, 1):
        # Keep quotes to minimize the model splitting/joining lines
        safe = s.replace('\n', ' ').strip()
        lines.append(f'{i}\t"{safe}"')
    return "\n".join(lines)

import re

VALID_LABELS = {"yes": "Yes", "no": "No"}

def parse_tsv_labels(tsv_text: str, expected_ids):
    """
    Parse a TSV response of the form:
        <id>\t<label>
    where <label> is Yes/No (case-insensitive).

    Returns:
        labels: dict[int, str]  # {id: "Yes"/"No"}
    Raises:
        ValueError with a clear message if any validation fails.
    """
    line_re = re.compile(r'^\s*(\d+)\s*\t\s*([Yy][Ee][Ss]|[Nn][Oo])\s*$')
    labels = {}
    seen_ids = set()
    errors = []

    lines = [ln for ln in tsv_text.strip().splitlines() if ln.strip()]

    for i, ln in enumerate(lines, 1):
        m = line_re.match(ln)
        if not m:
            errors.append(f"Line {i} not in '<id>\\t<label>' format or invalid label: {ln!r}")
            continue

        id_str, lab = m.groups()
        ex_id = int(id_str)
        norm_lab = VALID_LABELS[lab.lower()]

        if ex_id in seen_ids:
            errors.append(f"Duplicate id {ex_id} at line {i}.")
        else:
            seen_ids.add(ex_id)
            labels[ex_id] = norm_lab

    # Validation against expected ids
    expected_ids = list(expected_ids)
    missing = [i for i in expected_ids if i not in labels]
    unexpected = [i for i in labels.keys() if i not in expected_ids]

    if missing:
        errors.append(f"Missing ids: {sorted(missing)}.")
    if unexpected:
        errors.append(f"Unexpected ids in output: {sorted(unexpected)}.")

    if errors:
        raise ValueError("TSV parsing/validation failed:\n- " + "\n- ".join(errors))

    return labels

# Function to evaluate moral relevance using different models
# def check_moral_relevance(sentences, model="gpt-4o-mini", batch_size=10):
#     results = []
#     for i in tqdm(range(0, len(sentences), batch_size)):
#         batch = sentences[i:i + batch_size]
#         for attempt in range(3): # Max retries = 3
#             prompt = (    
#                 user_prompt_template +
#                 make_tsv_input(batch))

#             try:
#                 response = openai.chat.completions.create(
#                     model=model,
#                      messages=[{"role":"system","content":system_prompt}, {"role": "user", "content": prompt}],
#                     max_tokens=20 * len(batch),
#                     temperature=0,
#                 )

#                 outputs = response.choices[0].message.content.strip()

#                 outputs = parse_tsv_labels(outputs, expected_ids=range(1, len(batch) + 1))

#                 if len(outputs) == len(batch):
#                     results.extend(outputs)
#                     break  # success
#                 else:
#                     print(f"⚠️ Mismatch on attempt {attempt+1}: expected {len(batch)}, got {len(outputs)}")
#                     log_file.write(f"[FAILURE] {movie} / {character} — Expected {len(batch)} responses, got {len(outputs)}\n")
#                     log_file.flush()  # ensures it's written immediately

#                     time.sleep(1)

#             except Exception as e:
#                 print(f"Error processing batch: {e}")
#                 time.sleep(5)
#         else:
#             print(f"Failed to get valid response after {3} attempts. Padding.")
#             log_file.write(f"[GAVE UP] {movie} / {character} — Failed after 3 attempts\n")
#             log_file.flush()

#             results.extend(outputs)

#     return results

# assumes: make_tsv_input, user_prompt_template, system_prompt, parse_tsv_labels, openai, tqdm, time are available

def check_moral_relevance(sentences, model="gpt-4o-mini", batch_size=10):
    results = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]
        outputs = None  # ensure defined even if first attempt raises

        for attempt in range(3):  # Max retries = 3
            prompt = (user_prompt_template + make_tsv_input(batch))

            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": prompt}],
                    max_tokens=20 * len(batch),
                    temperature=0,
                )

                raw = response.choices[0].message.content.strip()
                outputs = parse_tsv_labels(raw, expected_ids=range(1, len(batch) + 1))  # dict {id: "Yes"/"No"}

                if len(outputs) == len(batch):
                    # FIX: extend results with labels IN ORDER 1..n
                    results.extend([outputs[j] for j in range(1, len(batch) + 1)])
                    break  # success
                else:
                    print(f"⚠️ Mismatch on attempt {attempt+1}: expected {len(batch)}, got {len(outputs)}")
                    try:
                        log_file.write(f"[FAILURE] {movie} / {character} — Expected {len(batch)} responses, got {len(outputs)}\n")
                        log_file.flush()
                    except Exception:
                        pass
                    time.sleep(1)

            except Exception as e:
                print(f"Error processing batch: {e}")
                time.sleep(5)
        else:
            print(f"Failed to get valid response after {3} attempts. Padding.")
            try:
                log_file.write(f"[GAVE UP] {movie} / {character} — Failed after 3 attempts\n")
                log_file.flush()
            except Exception:
                pass

            # FIX: pad deterministically in ID order; use any partial outputs if present
            results.extend([outputs.get(j, "No") if isinstance(outputs, dict) else "No"
                            for j in range(1, len(batch) + 1)])

    return results

json_path = "../data/dump/new_morality_check_gpt4omini_structured_2.json"

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
for movie, data in tqdm(data.items(), desc="Processing movies"):
    characters = data.get("characters", {})
    for character, char_data in characters.items():
        sentences = char_data.get("sentences", [])
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
