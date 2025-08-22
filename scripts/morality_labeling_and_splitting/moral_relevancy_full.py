import openai
import os
import json
import re
import pandas as pd
import time
from tqdm import tqdm

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load sentences (limit to 100 for testing)
with open('../data/moral_data_original.json', 'r') as f:
    data = json.load(f)

# Flatten into (movie, character, sentence)
flattened_data = []
for movie, characters in data["moral_dialogue"].items():
    for character, sentences in characters.items():
        for sentence in sentences:
            flattened_data.append(sentence)

# Take a sample of 100 unique sentences
sample_sentences = list(set(flattened_data))[:100]
print(f"Loaded {len(sample_sentences)} unique sentences for evaluation.")

# Function to evaluate moral relevance using different models
def check_moral_relevance(sentences, model="gpt-4o", batch_size=5):
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

            # Clean up numbering artifacts like "1. Yes"
            outputs = [re.sub(r"^\d+\.\s*", "", output.strip().lower()) for output in outputs]
            # Keep only the first word ("yes" or "no")
            outputs = [output.split()[0] if output.split() else "" for output in outputs]

            results.extend(outputs)

        except Exception as e:
            print(f"Error processing batch: {e}")
            time.sleep(5)

    return results

# Run the evaluation using different models
results_gpt4o = check_moral_relevance(sample_sentences, model="gpt-4o")
results_gpt35 = check_moral_relevance(sample_sentences, model="gpt-3.5-turbo")
results_gpt4o_mini = check_moral_relevance(sample_sentences, model="gpt-4o-mini")

# Create a DataFrame with the results
df = pd.DataFrame({
    'Sentence': sample_sentences,
    'GPT-4o': results_gpt4o,
    'GPT-3.5': results_gpt35,
    'GPT-4o-mini': results_gpt4o_mini
})

# Save to CSV — wrap fields with quotes and escape inner quotes
csv_path = '../data/moral_relevance_comparison.csv'
df.to_csv(csv_path, index=False, quotechar='"')

print(f"Results saved to '{csv_path}'")
