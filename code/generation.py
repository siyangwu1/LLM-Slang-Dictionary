# --- Imports --------------------------------------------------------------
from openai import OpenAI
import io
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm, trange

# --- Helper classes -------------------------------------------------------
class Wiki_Word:
    def __init__(self, word):
        self.word = word
        self.pos_tags = set()
        self.definitions = []

    def attach_def(self, word_def, pos, sentences, tags):
        entry = {"def": word_def, "pos": pos, "sents": sentences, "tags": tags}
        self.pos_tags.add(pos)
        self.definitions.append(entry)

# --- Load resources -------------------------------------------------------
# <<< Replace with your own path ------------------------------------------------>
WIKT_PATH = "/path/to/wikt-en.npy"
wiki_data = np.load(WIKT_PATH, allow_pickle=True)
# --------------------------------------------------------------------------------
word_dic = {entry.word: idx for idx, entry in enumerate(wiki_data)}

# --- Utility functions ----------------------------------------------------
def number_of_slang_definition(definitions):
    """Count slang / informal definitions in a Wiktionary entry."""
    num_slang = sum(
        1
        for d in definitions
        if {"slang", "informal"}.intersection(set(d["tags"]))
    )
    return num_slang, len(definitions)


def label_is_coinage(word):
    idx = word_dic.get(word)
    if idx is not None:
        defs = wiki_data[idx].definitions
        n_slang, n_total = number_of_slang_definition(defs)
        if (n_total - n_slang) >= 1:
            return "reuse"
    return "coinage"


def label_list(df):
    df["is_slang"] = [label_is_coinage(w) for w in df["word"]]
    return df


# --- OpenAI client (key taken from env-var) --------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def gpt_chat(content):
    response = client.chat.completions.create(
        messages=[{"role": "developer", "content": content}],
        model="gpt-4o",
        temperature=1.2,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"},
        max_tokens=16383,
    )
    return response.choices[0].message.content


# --- Prompt builders ------------------------------------------------------
def build_prompt_general(existing_words, n, definition):
    existing = ", ".join(existing_words)
    return f"""
      You are a creative slang dictionary generator …
      Generate {n} novel slang usages in English that express the definition: {definition}

      {{
        "word": [], "definition": [], "usage_context": []
      }}

      Do not repeat: [{existing}]
    """


def build_prompt_reuse(existing_words, n, definition):
    existing = ", ".join(existing_words)
    return f"""
      You are a creative slang dictionary generator …
      Reuse existing English words (do NOT coin new forms) to convey: {definition}

      {{
        "word": [], "definition": [], "usage_context": []
      }}

      Do not repeat: [{existing}]
      Generate {n} entries.
    """


def build_prompt_coinage(existing_words, n, definition):
    existing = ", ".join(existing_words)
    return f"""
      You are a creative slang dictionary generator …
      Coin entirely new words to convey: {definition}

      {{
        "word": [], "definition": [], "usage_context": []
      }}

      Do not repeat: [{existing}]
      Generate {n} entries.
    """


# --- Validation / deduplication ------------------------------------------
def valid_word_count(new_df, existing_df, mode):
    if mode != "general":
        new_df = new_df[new_df["is_slang"] == mode]

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    final = pd.DataFrame(columns=combined.columns)
    for word in combined["word"].unique():
        group = combined[combined["word"] == word]
        if len(group) == 1:
            final = pd.concat([final, group])
            continue

        unique_defs = []
        embeddings = model.encode(group["definition"].tolist(), convert_to_tensor=True)
        for idx, definition in enumerate(group["definition"]):
            duplicate = any(
                util.cos_sim(embeddings[idx], embeddings[j]).item() > 0.8
                for j in range(len(unique_defs))
            )
            if not duplicate:
                unique_defs.append(definition)
                final = pd.concat([final, group.iloc[[idx]]])

    return len(final), final


# --- CSV helpers ----------------------------------------------------------
def check_and_read_csv(path):
    cols = ["word", "definition", "usage_context", "is_slang"]
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False)
    return pd.read_csv(path)


# --- Main generation routine ---------------------------------------------
def generate_slang_and_save(
    n_target,
    prompt_func,
    frequent_def,
    existing_words,
    mode,
    cluster_idx,
    csv_out,
):
    collected = set()
    frames = []
    left = n_target
    attempts, max_attempts = 0, 6

    while left > 0 and attempts < max_attempts:
        attempts += 1
        prompt = prompt_func(existing_words, n_target + 4, frequent_def)
        raw = gpt_chat(prompt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[Attempt {attempts}] Invalid JSON.")
            continue

        if not isinstance(data, dict):
            print(f"[Attempt {attempts}] JSON is not an object.")
            continue

        words, defs, usages = (
            data.get("word", []),
            data.get("definition", []),
            data.get("usage_context", []),
        )
        if not (len(words) == len(defs) == len(usages)):
            print(f"[Attempt {attempts}] Length mismatch.")
            continue

        rows = []
        for w, d, u in zip(words, defs, usages):
            if not isinstance(u, list):
                u = [str(u)]
            rows.append(
                {
                    "word": w,
                    "LLM_definition": d,
                    "OSD_definition": frequent_def,
                    "usage_context": " | ".join(u),
                    "OSD_cluster_idx": cluster_idx,
                }
            )

        df_new = pd.DataFrame(rows)
        df_new = label_list(df_new)  # add 'is_slang'

        existing_words.extend(df_new["word"].astype(str).tolist())
        if mode != "general":
            df_new = df_new[df_new["is_slang"] == mode]

        new_unique = set(df_new["word"].astype(str)) - collected
        collected |= new_unique
        left -= len(new_unique)
        frames.append(df_new)

    # Consolidate
    if frames:
        final_df = pd.concat(frames, ignore_index=True).drop_duplicates("word")
    else:
        final_df = pd.DataFrame()

    # If still short, pad with placeholders
    if len(final_df) < n_target:
        pad = n_target - len(final_df)
        placeholders = pd.DataFrame(
            {
                "word": [f"None_{i}" for i in range(pad)],
                "LLM_definition": [f"None_{i}" for i in range(pad)],
                "OSD_definition": frequent_def,
                "usage_context": [f"None_{i}" for i in range(pad)],
                "OSD_cluster_idx": cluster_idx,
                "is_slang": [f"None_{i}" for i in range(pad)],
            }
        )
        final_df = pd.concat([final_df, placeholders], ignore_index=True)

    # Merge with any existing CSV
    if os.path.isfile(csv_out):
        existing = pd.read_csv(csv_out)
        final_df = pd.concat([existing, final_df], ignore_index=True)

    final_df.to_csv(csv_out, index=False)


# --- Experiment setup -----------------------------------------------------
# <<< Replace with your own paths ----------------------------------------->
CLUSTER_CSV = "/path/to/OSD_definition_clusterIdx.csv"
OSD_CSV = "/path/to/online_slang_dict.csv"
RESULT_DIR = "result/"
# -------------------------------------------------------------------------

cluster_df = pd.read_csv(CLUSTER_CSV, low_memory=False)
osd_df = pd.read_csv(OSD_CSV, low_memory=False)

# Divide clusters (example: 4 equal parts) -------------------------------
grouped = list(cluster_df.groupby("cluster_idx"))
q = len(grouped) // 4
cluster_part_3 = grouped[2 * q : 3 * q]  # third quarter
cluster_group = cluster_part_3
part_tag = "part3"

# Map generation modes to prompt builders ---------------------------------
PROMPT_FUNCS = {
    "external_reuse": build_prompt_reuse,
    # "external_general": build_prompt_general,
    # "external_coinage": build_prompt_coinage,
}

# --- Main loop ------------------------------------------------------------
for key, prompt_builder in PROMPT_FUNCS.items():
    mode = key.split("_")[-1]
    for cluster_idx, cluster_rows in tqdm(cluster_group, desc=f"Processing {key}"):
        n_entries = len(cluster_rows)
        defs_series = osd_df.loc[cluster_rows.index, "definition"]
        most_freq_def = defs_series.value_counts().idxmax() if not defs_series.empty else ""
        existing_words = osd_df.loc[cluster_rows.index, "word"].astype(str).tolist()

        out_csv = f"{RESULT_DIR}slang_dictionary_{key}_v1_{part_tag}.csv"
        try:
            generate_slang_and_save(
                n_entries,
                prompt_builder,
                most_freq_def,
                existing_words,
                mode,
                cluster_idx,
                out_csv,
            )
        except Exception as exc:
            print(f"Cluster {cluster_idx}: {exc}")
