import torch
import numpy as np

def normalize_mask_token(data, tokenizer):
    """
    Replace [MASK] with the tokenizer's mask token in the masked sentences.

    Currently, it is assumed that the masked token in the data is represented as [MASK].
    """

    ms = tokenizer.mask_token
    for row in data:
        row["masked_sentence"] = (
            row["masked_sentence"]
            .replace("[MASK]", ms)
        )
    return data


def exponential_smoothing(data, alpha=0.3):
    """
    Apply exponential smoothing to a 2D numpy array.
    """
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]
    
    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t - 1]
    
    return smoothed_data

def moving_average(data, window_size=5):
    pass



