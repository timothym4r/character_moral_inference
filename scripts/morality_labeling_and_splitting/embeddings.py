# S-BERT embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

sbert_model = SentenceTransformer('all-mpnet-base-v2')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def character_embedding_sbert(dialogue, movie="", minimum_total_speech=0):
    """Generate embeddings for characters in a movie using SBERT."""

    character_embeddings = []
    character_info = []

    if movie not in dialogue:
        raise ValueError(f"Movie '{movie}' not found in dialogue data")
    
    characters = dialogue[movie]
    for character, speeches in characters.items():
        if len(speeches) >= minimum_total_speech:  # Minimum dialogue threshold
            # Process all speeches at once (SBERT is optimized for batch processing)
            speech_embeddings = sbert_model.encode(speeches)
            
            # Average embeddings for the character
            character_embedding = np.mean(speech_embeddings, axis=0)
            character_embeddings.append(character_embedding)
            character_info.append((movie, character))

    print(f"Total characters: {len(character_embeddings)}")
    
    return character_embeddings, character_info


def character_embedding_bert(dialogue, movie="", minimum_total_speech=0):
    """Generate embeddings for characters in a movie using BERT."""

    character_embeddings = []
    character_info = []

    if movie not in dialogue:
        raise ValueError(f"Movie '{movie}' not found in dialogue data")
    
    characters = dialogue[movie]
    for character, speeches in characters.items():
        if len(speeches) >= minimum_total_speech:  # Minimum dialogue threshold
            # Process all speeches at once (BERT is optimized for batch processing)
            inputs = bert_tokenizer(speeches, return_tensors='pt', padding=True, truncation=True)
            outputs = bert_model(**inputs)
            speech_embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
            
            # Average embeddings for the character
            character_embedding = np.mean(speech_embeddings, axis=0)
            character_embeddings.append(character_embedding)
            character_info.append((movie, character))

    print(f"Total characters: {len(character_embeddings)}")
    
    return character_embeddings, character_info



