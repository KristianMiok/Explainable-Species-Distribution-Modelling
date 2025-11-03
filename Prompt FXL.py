#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build improved LLM prompts for FXL (presence vs true absence) using ONLY:
RWQ, ALT, FFP, BIO1.

What this script does
- Reads NETWORK.xlsx
- Builds FXL_BIN from FXL_PREZ / FXL_TRUEABS on labeled rows (drops ambiguous 1/1)
- Computes SAFE aggregate stats + rich quantiles for the 4 predictors
- Computes class balance of FXL_BIN for the labeled set
- Creates an improved generation prompt (over-generate trees, quantile anchoring,
  soft priors, strict diversity, schema & validation checklist)
- Creates a second "referee" prompt to clean/repair LLM JSON trees
- Saves both prompts under outputs/:
    - paper_llm_trees_FXL_prompt.txt
    - paper_llm_trees_FXL_referee.txt

Edit N_TREES / MAX_DEPTH / SPECIES fields if needed.
"""

import os, json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
EXCEL_PATH   = "/Users/kristianmiok/Desktop/Lucian/LLM/RF/Data/NETWORK.xlsx"
OUT_DIR      = None  # if None, writes next to EXCEL under outputs/
N_TREES      = 50    # over-generate; you'll later select the best K
MAX_DEPTH    = 2     # try 2 for stricter interpretability; 3 if needed
SPECIES_CODE = "FXL"
SPECIES_LONG = "FXL crayfish (code FXL)"  # edit to the full scientific name if desired

# Only these predictors are allowed
KEEP_PREDICTORS = ["RWQ", "ALT", "FFP", "BIO1"]

# Human-friendly names (shown in prompt)
FEATURE_RENAMES = {
    "RWQ":  "River width/flow proxy (dimensionless; ~0–4)",
    "ALT":  "Altitude (m)",
    "FFP":  "Flood frequency / flow persistence proxy",
    "BIO1": "Mean annual temperature (°C)",
}

# Ecological priors (soft). Tweak based on your species knowledge.
# (These are intentionally light-touch; edit freely if you know FXL ecology well.)
DOMAIN_PRIORS = [
    "Presence often relates to accessible, perennial habitats; extreme hydrologic instability can reduce suitability.",
    "ALT: low to mid elevations are typically less limiting than very high elevations.",
    "RWQ: moderate-to-higher throughflow can be favorable relative to very small/stagnant channels.",
    "BIO1: very warm extremes may reduce suitability; avoid relying on ultra-tight temperature bands.",
]

# Quantiles to report (for threshold anchoring hints)
Q_LIST = [0.05, 0.20, 0.40, 0.50, 0.60, 0.80, 0.95]

# =========================
# UTILS
# =========================
def outputs_dir(excel_path: str) -> Path:
    outd = Path(excel_path).expanduser().resolve().parent / "outputs"
    outd.mkdir(parents=True, exist_ok=True)
    return outd

def load_excel_any(path: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except Exception:
        return pd.read_excel(path, engine="openpyxl")

def clean_decimal_commas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.replace("\u00A0", "", regex=False).str.strip()
            # replace only commas that are decimal separators (not thousands)
            coerced = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
            if coerced.notna().sum() >= 0.5 * len(df[c]):
                df[c] = coerced
    return df

def build_fxl_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["FXL_PREZ", "FXL_TRUEABS"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")
    prez = pd.to_numeric(df["FXL_PREZ"], errors="coerce").fillna(0).astype(int)
    tabs = pd.to_numeric(df["FXL_TRUEABS"], errors="coerce").fillna(0).astype(int)
    mask_labeled = (prez == 1) | (tabs == 1)
    data = df.loc[mask_labeled].copy()

    # drop ambiguous 1/1
    both = (data["FXL_PREZ"].astype(int) == 1) & (data["FXL_TRUEABS"].astype(int) == 1)
    if both.any():
        data = data.loc[~both].copy()

    data[f"{SPECIES_CODE}_BIN"] = (data["FXL_PREZ"].astype(int) == 1).astype(int)
    return data

def safe_feature_stats(df_num: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats = {}
    for c in df_num.columns:
        s = pd.to_numeric(df_num[c], errors="coerce").dropna()
        if s.empty:
            continue
        stats[c] = {
            "min":     float(s.min()),
            "max":     float(s.max()),
            "median":  float(s.quantile(0.5)),
            "q": {f"q{int(q*100):02d}": float(s.quantile(q)) for q in Q_LIST},
            "missing": int(df_num[c].isna().sum()),
        }
    return stats

def format_stats_block(stats: Dict[str, Dict[str, float]]) -> str:
    lines = []
    for c in KEEP_PREDICTORS:
        st = stats.get(c)
        if not st:
            continue
        pretty = FEATURE_RENAMES.get(c, c)
        qparts = [f"{k}={v:.3g}" for k, v in st["q"].items()]
        lines.append(
            f"- {c} ({pretty}): min={st['min']:.3g}, "
            + ", ".join(qparts)
            + f", median={st['median']:.3g}, max={st['max']:.3g} "
            f"(missing={st['missing']})"
        )
    return "\n".join(lines)

def format_priors_block() -> str:
    return "\n".join([f"- {p}" for p in DOMAIN_PRIORS])

def make_generation_prompt(X: pd.DataFrame, n_trees: int, max_depth: int, class_ratio: str) -> str:
    cols_keep = [c for c in KEEP_PREDICTORS if c in X.columns]
    stats = safe_feature_stats(X[cols_keep])
    stats_block = format_stats_block(stats)
    priors_block = format_priors_block()
    roots_hint = ", ".join(cols_keep)

    prompt = f"""
You are an aquatic ecologist. Produce EXACTLY {n_trees} shallow, human-auditable decision trees
(depth ≤ {max_depth}) to predict {SPECIES_LONG} presence (1) vs true absence (0) for Romanian rivers.

OPTIMIZE FOR
- **Macro-F1** (not accuracy). Class balance on labeled data is: {class_ratio}.
- Prefer thresholds near **empirical quantiles** to improve stability.

ALLOWED FEATURES (ONLY): RWQ, ALT, FFP, BIO1

ECOLOGICAL PRIORS (soft, can be violated if stats strongly suggest otherwise):
{priors_block}

SAFE AGGREGATE STATS (choose thresholds within these ranges; anchor near quantiles):
{stats_block}

DIVERSITY & SIMPLICITY CONSTRAINTS
- Each tree MUST use a DIFFERENT root feature (cycle across [{roots_hint}]).
- Per tree, do not test the same feature more than **twice** total (root may reappear once).
- Avoid tiny numeric boxes; keep splits coarse and interpretable.

THRESHOLD GUIDELINES
- Use thresholds close to quantiles (e.g., q20, q40, q60, q80).
- **Include a "quantile_hint" field** at each non-leaf node to indicate your intended quantile.

OUTPUT FORMAT (STRICT JSON array of EXACTLY {n_trees} trees; NO commentary outside JSON):
[
  {{
    "tree_id": 1,
    "target": "{SPECIES_CODE}_BIN",
    "max_depth": {max_depth},
    "root": {{
      "feature": "<RWQ|ALT|FFP|BIO1>", "op": "<=", "value": <number>, "quantile_hint": "<q20|q40|q60|q80>",
      "left":  {{ "leaf": <0_or_1> }},
      "right": {{
        "feature": "<RWQ|ALT|FFP|BIO1>", "op": ">", "value": <number>, "quantile_hint": "<q20|q40|q60|q80>",
        "left":  {{ "leaf": <0_or_1> }},
        "right": {{ "leaf": <0_or_1> }}
      }}
    }}
  }}
  ... repeat until EXACTLY {n_trees} trees ...
]

ALLOWED ops: "<=", ">", "<", ">=".

VALIDATION CHECKLIST (ensure your JSON satisfies all):
- depth ≤ {max_depth}
- roots all different across trees
- thresholds within SAFE [min, max]
- quantile_hint present where a split uses "feature"/"value"
- no micro-rectangles; respect monotone tendencies unless clearly contradicted by stats
""".strip("\n")
    return prompt

def make_referee_prompt() -> str:
    # A second prompt to clean/repair after the LLM outputs its JSON
    return f"""
You are a JSON referee. You will be given a JSON array of shallow decision trees (depth ≤ 3)
for {SPECIES_CODE}_BIN using ONLY features: RWQ, ALT, FFP, BIO1. Your task:

1) **Validate & Repair** schema strictly:
   - JSON must parse as a list of objects.
   - Each object must have a "root" with fields: feature/op/value and children ("left", "right").
   - ALLOWED ops: "<=", ">", "<", ">=".
   - Leaves are of the form {{"leaf": 0 or 1}}.
   - If a node has feature/op/value but no child, create the missing child as a leaf with the **other** class.

2) **Enforce constraints**:
   - depth ≤ 3, features only in {{RWQ, ALT, FFP, BIO1}}.
   - If thresholds are outside SAFE min–max (from the companion prompt), clip them to the closest bound.
   - Ensure each tree's root feature is present; if duplicate trees are identical in (root, thresholds, leaves),
     keep only the first occurrence.

3) **Quantile hints**:
   - If "quantile_hint" is missing on a non-leaf node, add a best-guess hint (q20/q40/q60/q80) based on the threshold location
     within the known SAFE range.
   - Do not add free-text commentary.

4) **Output**:
   - Return ONLY the repaired JSON array (no extra text).
""".strip("\n")

# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(EXCEL_PATH)

    df_raw = load_excel_any(EXCEL_PATH)
    df = clean_decimal_commas(df_raw)
    data = build_fxl_frame(df)

    # Restrict to only the four predictors
    missing = [c for c in KEEP_PREDICTORS if c not in data.columns]
    if missing:
        raise RuntimeError(f"Missing expected predictors: {missing}")
    X = data[KEEP_PREDICTORS].copy()

    # class balance string for the prompt
    y = data[f"{SPECIES_CODE}_BIN"].astype(int)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    class_ratio = f"presence= {n1}  |  true_absence= {n0}  (total labeled = {int(len(y))})"

    # Build prompts
    gen_prompt = make_generation_prompt(X, n_trees=N_TREES, max_depth=MAX_DEPTH, class_ratio=class_ratio)
    ref_prompt = make_referee_prompt()

    # Save
    outd = Path(OUT_DIR) if OUT_DIR else outputs_dir(EXCEL_PATH)
    out_gen = outd / "paper_llm_trees_FXL_prompt.txt"
    out_ref = outd / "paper_llm_trees_FXL_referee.txt"
    out_gen.write_text(gen_prompt, encoding="utf-8")
    out_ref.write_text(ref_prompt, encoding="utf-8")

    # Print the main prompt to console (referee prompt is saved next to it)
    print("\n" + "="*90)
    print("[PROMPT FOR LLM — copy everything below]")
    print("="*90 + "\n")
    print(gen_prompt)
    print("\n" + "="*90)
    print(f"[i] Prompt saved to: {out_gen}")
    print(f"[i] Referee/repair prompt saved to: {out_ref}")
    print("[tip] Generate a few batches with moderate temperature (e.g., 0.7–0.9), then score & select the best trees.")
    print("[note] Consider refining DOMAIN_PRIORS if you have stronger, species-specific knowledge for FXL.")

if __name__ == "__main__":
    main()