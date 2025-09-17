"""
Compound-word coherence check
─────────────────────────────
(1) Load multiple coinage datasets.
(2) Train Morfessor on the full slang vocabulary.
(3) Keep only words whose segmentation is classified as “Compound”.
(4) For each compound word, embed its definition and compute the mean
    Euclidean distance to the non-slang dictionary senses of *each*
    of its Morfessor segments (exact headword match only).
(5) Print per-source summary statistics.
"""

# ── imports ──────────────────────────────────────────────────────────
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import morfessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import iqr, kurtosis

# ── Wiktionary helpers ───────────────────────────────────────────────
class Wiki_Word:
    def __init__(self, word):
        self.word = word
        self.pos_tags: set[str] = set()
        self.definitions: list[dict] = []

    def attach_def(self, word_def, pos, sentences, tags):
        self.pos_tags.add(pos)
        self.definitions.append(
            {"def": word_def, "pos": pos, "sents": sentences, "tags": tags}
        )


# <<< Replace with your own path ------------------------------------------------->
WIKT_PATH = "/path/to/wikt-en.npy"
wiki_data = np.load(WIKT_PATH, allow_pickle=True)
# -------------------------------------------------------------------------------

word_dic = {e.word: i for i, e in enumerate(wiki_data)}


def get_nonslang_definition(word: str) -> list[str]:
    """Return all non-slang senses for *exact* headword `word`."""
    idx = word_dic.get(word)
    if idx is None:
        return []
    out = []
    for d in wiki_data[idx].definitions:
        tags = [t.lower() for t in d.get("tags", [])]
        if "slang" not in tags and "informal" not in tags:
            out.append(d["def"])
    return out


def pick_def_col(df: pd.DataFrame) -> str:
    """Return the column that holds definition text; create blank if absent."""
    for c in ["definition", "Definition", "OSD_definition"]:
        if c in df.columns:
            return c
    warnings.warn("No definition column found; inserting blanks")
    df["Definition"] = ""
    return "Definition"


# ── Morfessor segment classifier (given) ─────────────────────────────
def classify_segments(segs):
    if len(segs) >= 2 and len(set(segs)) == 1:
        return "Reduplication"
    if all(s in word_dic for s in segs):
        return "Compound"
    for s in segs:
        if any(s in word for word in word_dic):
            return "Blend"
    return "Other"


# ── Load coinage datasets (single-word only) ─────────────────────────
def load_coinage_sources_single_word_only():
    # <<< Replace with your own paths --------------------------------------->
    EXT_FILE = "/path/to/slang_dictionary_external_coinage_v1_valid.csv"
    OSD_FILE = "/path/to/online_slang_dictionary_refined.csv"
    # -----------------------------------------------------------------------

    gpt = (
        pd.read_csv(EXT_FILE)
        .query("is_slang == 'coinage'")
        .loc[lambda d: ~d["word"].astype(str).str.startswith('None_')]
    )
    osd = pd.read_csv(OSD_FILE).query("is_slang == 'coinage'")

    gpt["Definition"] = gpt[pick_def_col(gpt)]
    osd["Definition"] = osd[pick_def_col(osd)]

    # sample to equalise sizes
    gpt = gpt.sample(1500, random_state=42)
    osd = osd.sample(2000, random_state=42)

    extra_paths = {
        "GPT-4o U-C": "/path/to/slang_dictionary_internal_coinage_valid_v1.csv",
        "LLaMA-8B-INT": "/path/to/slang_dictionary_llama8bit_internal_coinage_valid_v1.csv",
        "LLaMA-8B-INT F.t GPT-4o C-C":
            "/path/to/slang_dictionary_llama8bit_gpt4o-external-coinage_finetune_internal_coinage_valid_v1.csv",
        "LLaMA-8B-INT F.t GPT-4o U-C":
            "/path/to/slang_dictionary_llama8bit_gpt4o-internal-coinage_finetune_internal_coinage_valid_v1.csv",
        "LLaMA-8B F.t. OSD C":
            "/path/to/slang_dictionary_llama8bit_OSD-coinage_finetune_internal_coinage_valid_v1.csv",
    }

    datasets = {
        "GPT-4o C-C": gpt[["word", "Definition"]],
        "OSD C": osd[["word", "Definition"]],
    }

    for name, path in extra_paths.items():
        tmp = pd.read_csv(path)
        tmp["Definition"] = tmp[pick_def_col(tmp)]
        datasets[name] = tmp[["word", "Definition"]]

    # keep only single-token entries
    for name in list(datasets):
        datasets[name] = datasets[name][
            ~datasets[name]["word"].astype(str).str.contains(r"\s")
        ].reset_index(drop=True)

    return datasets


datasets = load_coinage_sources_single_word_only()

# ── Morfessor: train on union vocabulary ──────────────────────────────
vocab = np.unique(np.concatenate([d["word"].values for d in datasets.values()]))
mf = morfessor.BaselineModel()
mf.load_data([(1, w) for w in vocab])
mf.train_batch()


def segments(word: str) -> list[str]:
    try:
        return mf.viterbi_segment(word)[0]
    except Exception:
        return []


# ── SBERT encoder (GPU if available) ─────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

# pre-encode definition vectors
def_vec_cache = {
    src: embed.encode(
        df["Definition"].tolist(),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    for src, df in datasets.items()
}

# ── Build exact-match segment-sense cache ────────────────────────────
segment_cache: dict[str, np.ndarray] = {}
for df in datasets.values():
    for w in df["word"]:
        for seg in segments(w):
            if seg in segment_cache or seg not in word_dic:
                continue
            defs = get_nonslang_definition(seg)
            if defs:
                segment_cache[seg] = embed.encode(
                    defs,
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

# ── Compute mean distance for COMPOUND words only ───────────────────
results = []
for src, df in datasets.items():
    for (word, defin), vec in zip(df.itertuples(index=False), def_vec_cache[src]):
        segs = segments(word)
        if classify_segments(segs) != "Compound":
            continue
        if not segs or not all(s in segment_cache for s in segs):
            continue
        sense_vecs = np.vstack([segment_cache[s] for s in segs])
        dist = float(euclidean_distances(vec.reshape(1, -1), sense_vecs).mean())
        results.append({"Source": src, "Word": word, "AvgSegSenseDist": dist})

master = pd.DataFrame(results)

# ── Summary stats ────────────────────────────────────────────────────
stats = (
    master.groupby("Source")["AvgSegSenseDist"]
    .agg(
        Count="size",
        Mean="mean",
        Std="std",
        IQR=lambda x: iqr(x, nan_policy="omit"),
        Kurtosis=lambda x: kurtosis(x, nan_policy="omit", fisher=True),
    )
    .round(4)
)

print("\n── Compound-word → Segment-sense coherence (Euclidean) ──")
print(stats.to_string())
