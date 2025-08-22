import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

MORAL_TRAIT_SET = [14, 25, 28, 31, 38, 39, 79, 81, 84, 101, 121, 154, 195, 222, 227, 396, 434, 448, 450, 489, 494]
MORAL_TRAIT_COLS = [f"trait_{i}" for i in MORAL_TRAIT_SET]

# These are the traits corresponding to the indices in MORAL_TRAIT_SET
# NOTE: We have more but these are the ones that have moral valence.
trait_dict = {
    14: "cunning-honorable",
    25: "forgiving-vengeful",
    28: "loyal-traitorous",
    31: "rude-respectful",
    38: "arrogant-humble",
    39: "heroic-villainous",
    79: "selfish-altruistic",
    81: "angelic-demonic",
    84: "cruel-kind",
    101: "biased-impartial",
    121: "sarcastic-genuine",
    154: "judgemental-accepting",
    195: "complimentary-insulting",
    222: "wholesome-salacious",
    227: "racist-egalitarian",
    396: "innocent-jaded",
    434: "resentful-euphoric",
    448: "fake-real",
    450: "catty-supportive",
    489: "sincere-irreverent",
    494: "hopeful-fearful"
}

def get_strong_correlations(corr_df, threshold=0.4, save_path=None):
    """
    Return [(latent, trait, correlation)] where |correlation| > threshold.
    Optionally writes a CSV/TXT. Works across pandas versions.
    """
    # Optional: restrict columns if available
    try:
        corr_df = corr_df[MORAL_TRAIT_COLS]
    except NameError:
        pass

    # --- FIX: make columns after stack portable across pandas versions ---
    long = (
        corr_df.stack()                 # Series with MultiIndex (latent, trait)
               .rename("correlation")   # name the values column
               .rename_axis(["latent", "trait"])  # name the index levels
               .reset_index()           # -> DataFrame with columns latent, trait, correlation
    )

    # Filter by threshold
    long = long[long["correlation"].notna() & (long["correlation"].abs() > threshold)]

    # Optional trait mapping (safe)
    if not long.empty:
        try:
            idx = long["trait"].astype(str).str.replace("trait_", "", regex=False).astype(int)
            long["trait"] = idx.map(trait_dict).fillna(long["trait"])
        except Exception:
            pass

    strong_pairs = list(long[["latent", "trait", "correlation"]].itertuples(index=False, name=None))

    # Save if requested
    if save_path and strong_pairs:
        p = Path(save_path)

        # If caller accidentally joined to a data file path like ".../latent_embeddings.pkl/foo.txt",
        # fix the parent to be the file's directory.
        if p.parent.suffix.lower() in {".pkl", ".pt", ".npy", ".csv", ".txt"} or (
            p.parent.exists() and p.parent.is_file()
        ):
            p = p.parent.parent / p.name

        if p.suffix == "":               # treat as directory
            p = p / "strong_correlations.txt"

        p.parent.mkdir(parents=True, exist_ok=True)
        # Write as plain txt (fast). Change to .csv if you prefer.
        with open(p, "w") as f:
            f.write("latent\ttrait\tcorrelation\n")
            for a, b, c in strong_pairs:
                f.write(f"{a}\t{b}\t{float(c):.6f}\n")

    return strong_pairs


def plot_r2_scores(result_df, top_n=None, figsize=(12, 6), title="Trait-wise R² Scores", save_path=None):
    """
    Plots a bar chart of R² scores from the result of method_2.
    Optionally saves the plot to a file.

    Parameters:
        result_df (pd.DataFrame): Output of method_2
        top_n (int, optional): Plot only the top N traits by R². If None, plot all.
        figsize (tuple): Figure size
        title (str): Title of the plot
        save_path (str, optional): Path to save plot image (e.g., 'plot.png')
    """
    # Drop NaN values and sort by R² score
    result_df = result_df[result_df["trait_index"].isin(MORAL_TRAIT_COLS)]
    df = result_df.dropna(subset=["r2_score"]).sort_values("r2_score", ascending=False)

    if top_n:
        df = df.head(top_n)

    # Extract index from 'trait_<index>' and map to trait name
    df["trait_label"] = df["trait_index"].apply(
        lambda x: trait_dict.get(int(x.replace("trait_", "")), x)
    )

    plt.figure(figsize=figsize)
    plt.bar(df["trait_label"], df["r2_score"], color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Trait")
    plt.ylabel("R² Score")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        
        p = Path(save_path)

        # If no file extension is given, treat as a directory and use a default filename
        if p.suffix == "":
            p = p / "plot.png"

        # Ensure the parent directory exists
        p.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(p, dpi=300, bbox_inches="tight")


def find_files_with_key_words(folder_path: str, key_word:str) -> str:
    """
    Searches for files in the specified folder that contain the given key word in their name.

    Parameters:
        folder_path (str): Path to the folder where files are searched.
        key_word (str): Key word to search for in file names.

    Returns:
        str: The path of the first file found with the key word, or None if no file is found.
    """

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if key_word in file:
                return os.path.join(root, file)
    
    return None
