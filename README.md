# Character Moral Inference

## 🧠 Hypothesis
The moral words uttered by someone are correlated with their moral character.

## ⚠ Limitations
- Some words (even if moral) might not reflect the speaker's moral character.
- The framework may perform better on virtuous ("good") characters compared to villainous ("bad") characters.

---

## 🔎 Analysis Performed So Far

### 1. **Data Preparation**
- Collected **780,000+ utterances** from **12500+** characters across **1000+** movies. 
- Each utterance is tagged with the speaker (character name) and movie title.
- Moral Foundation Dictionary (MFD) compiled to identify potential moral words.

### 2. **BERT Embedding Strategy**
- For moral **words** → Extracted embeddings from the **first layer** of BERT.
- For **utterances** → Extracted embeddings from the **last layer** of BERT.
- This separation preserves word-level generalization and utterance-level context.

### 3. **Utterance Filtering**
- Used an LLM (OpenAI API) to label whether each utterance is *morally relevant*.
- Non-moral utterances excluded from training the moral representation model.

### 4. **Character Representation**
- Averaged BERT embeddings of morally relevant utterances for each character.
- These serve as the **base character embeddings**.

### 5. **Representation Learning (Model H)**
- Developed an **Autoencoder (AE)** to compress base character embeddings into **10–20D latent moral vectors**.
- Trained AE to reconstruct the original 768D BERT embeddings.

### 6. **Masked Word Prediction (Model M)**
- Injected the reconstructed embeddings into the **last layer** of BERT.
- Task: Predict missing moral words in new utterances (masked language modeling).
- Loss function combines:
  - **Reconstruction loss** (MSE)
  - **Cross-entropy loss** for the masked prediction task.

### 7. **Model Variants**
- Trained both:
  - **Autoencoder (AE)**
  - **Variational Autoencoder (VAE)**
- Performance compared to assess meaningfulness of learned moral vectors.

---

## 🔨 Current Focus
- Finalize comparison between AE and VAE models.
- Evaluate how well the learned representations improve moral word prediction.

---

## 🗂 File Structure
```plaintext
|-- data/            # Dialogue data, character metadata, moral word dictionary
|-- models/          # AE/VAE checkpoints and BERT adapters
|-- scripts/         # Preprocessing, embedding extraction, model training
|-- results/         # Evaluation metrics, plots, and analyses
|-- README.md        # Project overview (this file)

