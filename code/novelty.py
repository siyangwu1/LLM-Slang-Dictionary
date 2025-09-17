# --- Imports -------------------------------------------------------------
import io
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import trange

# ------------------------------------------------------------------------
# Wiktionary helpers
# ------------------------------------------------------------------------
class Wiki_Word:
    def __init__(self, word):
        self.word = word
        self.pos_tags = set()
        self.definitions = []

    def attach_def(self, word_def, pos, sentences, tags):
        entry = {"def": word_def, "pos": pos, "sents": sentences, "tags": tags}
        self.pos_tags.add(pos)
        self.definitions.append(entry)


# <<< Replace with your own path ---------------------------------------->
WIKT_PATH = "/path/to/wikt-en.npy"
wiki_data = np.load(WIKT_PATH, allow_pickle=True)
# ------------------------------------------------------------------------

word_dic = {entry.word: idx for idx, entry in enumerate(wiki_data)}

# ------------------------------------------------------------------------
# SBERT model (loaded once, GPU if available)
# ------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)


def get_definition(word):
    idx = word_dic.get(word)
    if idx is not None:
        return [d["def"] for d in wiki_data[idx].definitions]
    return []


def get_nonslang_definition(word):
    idx = word_dic.get(word)
    if idx is not None:
        defs = wiki_data[idx].definitions
        return [
            d["def"]
            for d in defs
            if not {"slang", "informal"}.intersection({t.lower() for t in d.get("tags", [])})
        ]
    return []


def compute_mean_pairwise_euclidean_distance(text, texts, model):
    emb1 = model.encode(text, convert_to_tensor=True, device=model.device)
    emb2 = model.encode(texts, convert_to_tensor=True, device=model.device)
    return torch.norm(emb2 - emb1, p=2, dim=1).mean().item()


def measure_novelty(word, definition, is_human=False):
    ref_defs = get_nonslang_definition(word) if is_human else get_definition(word)
    return compute_mean_pairwise_euclidean_distance(definition, ref_defs, sbert_model)


# ------------------------------------------------------------------------
# Data loading (paths anonymised)
# ------------------------------------------------------------------------
# <<< Replace with your own path ---------------------------------------->
EXTERNAL_REUSE_CSV = "/path/to/slang_dictionary_external_reuse_v4_valid.csv"
OSD_REFINED_CSV = "/path/to/online_slang_dictionary_refined.csv"

GPT_REUSE_UNCOND_CSV = "/path/to/slang_dictionary_internal_reuse_valid_v1.csv"
LLAMA_REUSE_UNCOND_CSV = "/path/to/slang_dictionary_llama8bit_internal_reuse_valid_v1.csv"
LLAMA_OSD_FT_REUSE_UNCOND_CSV = "/path/to/slang_dictionary_llama8bit_osd_reuse_finetune_internal_reuse_valid_v1.csv"
LLAMA_GPT4O_EXT_FT_REUSE_CSV = "/path/to/slang_dictionary_llama8bit_gpt4o-external-reuse_finetune_internal_reuse_valid_v1.csv"
LLAMA_GPT4O_INT_FT_REUSE_CSV = "/path/to/slang_dictionary_llama8bit_gpt4o-internal-reuse_finetune_internal_reuse_valid_v1.csv"
# ------------------------------------------------------------------------

df = pd.read_csv(EXTERNAL_REUSE_CSV)
external_reuse_gpt = df[~df["word"].astype(str).str.startswith("None_")]

osd_refined = pd.read_csv(OSD_REFINED_CSV)
osd_refined = osd_refined[osd_refined["is_slang"] == "reuse"]

common_ids = set(external_reuse_gpt["OSD_cluster_idx"]).intersection(osd_refined["cluster_idx"])
sampled_ids = pd.Series(list(common_ids)).sample(n=1000, random_state=42).tolist()

gpt_reuse_condition = (
    external_reuse_gpt[external_reuse_gpt["OSD_cluster_idx"].isin(sampled_ids)]
    .groupby("OSD_cluster_idx", group_keys=False)
    .sample(n=1, random_state=42)
)
osd_reuse = (
    osd_refined[osd_refined["cluster_idx"].isin(sampled_ids)]
    .groupby("cluster_idx", group_keys=False)
    .sample(n=1, random_state=42)
)

# ------------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------------
def safe_str(x):
    return str(x) if isinstance(x, str) else None


# ------------------------------------------------------------------------
# Load more corpora (paths anonymised)
# ------------------------------------------------------------------------
gpt_reuse_uncond = pd.read_csv(GPT_REUSE_UNCOND_CSV)
llama_reuse_uncond = pd.read_csv(LLAMA_REUSE_UNCOND_CSV)
llama_osd_ft_reuse_uncond = pd.read_csv(LLAMA_OSD_FT_REUSE_UNCOND_CSV)
llama_gpt4o_ext_ft_reuse = pd.read_csv(LLAMA_GPT4O_EXT_FT_REUSE_CSV)
llama_gpt4o_int_ft_reuse = pd.read_csv(LLAMA_GPT4O_INT_FT_REUSE_CSV)

# ------------------------------------------------------------------------
# Novelty computation
# ------------------------------------------------------------------------
def compute_novelties(df_iter, definition_field="definition", human=False, tag=""):
    novelties = []
    print(f"[INFO] Computing novelty for {tag} ...")
    for idx, row in df_iter.iterrows():
        word = safe_str(row["word"])
        definition = safe_str(row[definition_field])
        if None in (word, definition):
            print(f"[SKIP][{tag}] idx={idx} | word={word} | def={definition}")
            novelties.append(None)
            continue
        try:
            novelties.append(measure_novelty(word, definition, human))
        except Exception as e:
            print(f"[ERROR][{tag}] idx={idx} | word={word} | {e}")
            novelties.append(None)
    return novelties


osd_novelties = compute_novelties(osd_reuse, human=True, tag="osd")
gpt_uncond_novelties = compute_novelties(gpt_reuse_uncond, tag="gpt-uncond")
gpt_cond_novelties = compute_novelties(
    gpt_reuse_condition, definition_field="OSD_definition", tag="gpt-cond"
)
llama_uncond_novelties = compute_novelties(llama_reuse_uncond, tag="llama-uncond")
llama_osd_ft_novelties = compute_novelties(llama_osd_ft_reuse_uncond, tag="llama-osd-ft")
llama_gpt4o_ext_novelties = compute_novelties(llama_gpt4o_ext_ft_reuse, tag="llama-gpt4o-ext")
llama_gpt4o_int_novelties = compute_novelties(llama_gpt4o_int_ft_reuse, tag="llama-gpt4o-int")
