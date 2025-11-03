#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUB: 5-fold CV with DT, LLM, and tuned hybrids (aligned to AUT protocol)
- Predictors: RWQ, ALT, FFP, BIO1
- Target: AUB_BIN (1 if AUB_PREZ==1, else 0 if AUB_TRUEABS==1)
- Metrics: Accuracy & Macro-F1 (mean ± std across 5 folds)
- Hybrids: AND/OR, k-veto, soft-veto (grid), soft-blend (grid), stacked logistic
- NOTE: No train-time under-sampling (to mirror AUT)
"""

import os, re, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# =========================
# CONFIG (AUB)
# =========================
EXCEL_PATH = "/Users/kristianmiok/Desktop/Lucian/LLM/RF/Data/NETWORK.xlsx"
LLM_TREES_PATH = str(Path(EXCEL_PATH).expanduser().resolve().parent / "outputs" / "paper_llm_trees_AUB.json")

SPECIES = "AUB"
PREZ_COL = "AUB_PREZ"
TRUEABS_COL = "AUB_TRUEABS"
TARGET = f"{SPECIES}_BIN"  # 1 if *_PREZ==1, 0 if *_TRUEABS==1

PREDICTORS = ["RWQ", "ALT", "FFP", "BIO1"]
DT_DEPTH = 2
KNN_K = 10
N_SPLITS = 5
RANDOM_STATE = 42

# Hybrid grids (match AUT)
AND_TAU_LIST      = [0.50, 0.60, 0.70, 0.80]
SOFT_VETO_THETAS  = [0.5, 0.6, 0.7]
SOFT_VETO_ALPHAS  = [0.25, 0.5, 0.75]
SOFT_BLEND_WEIGHTS = [0.50, 0.55, 0.60, 0.625, 0.65, 0.675, 0.70]

# =========================
# Utils
# =========================
def banner(msg: str):
    print("\n" + "="*90)
    print(msg)
    print("="*90)

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
            coerced = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
            if coerced.notna().sum() >= 0.5 * len(df[c]):
                df[c] = coerced
    return df

def id_like_cols(cols: List[str]) -> List[str]:
    return [c for c in cols if re.search(r"(^id$|_id$|^fid$|cellid$)", str(c), flags=re.I)]

# ----- LLM trees loader & evaluator -----
def load_llm_trees(path: str) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    def _clean_quotes(t: str) -> str:
        return (t.replace("\u201c", '"').replace("\u201d", '"')
                 .replace("\u201e", '"').replace("\u201f", '"')
                 .replace("\u2018", "'").replace("\u2019", "'")
                 .replace("\u2032", "'").replace("\u2033", '"'))
    def _strip_code_fences(t: str) -> str:
        t = t.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t
            if t.lower().startswith("json\n"):
                t = t.split("\n", 1)[1] if "\n" in t else t
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return t
    def _remove_js_comments(t: str) -> str:
        t = re.sub(r"(^|\s)//.*$", "", t, flags=re.MULTILINE)
        t = re.sub(r"/\*.*?\*/", "", t, flags=re.DOTALL)
        return t
    def _remove_trailing_commas(t: str) -> str:
        return re.sub(r",(\s*[\]\}])", r"\1", t)

    cleaned = _remove_trailing_commas(_remove_js_comments(_strip_code_fences(_clean_quotes(text))))

    try:
        obj = json.loads(cleaned)
    except Exception:
        m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if m:
            obj = json.loads(_remove_trailing_commas(m.group(0)))
        else:
            m2 = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if m2:
                obj = json.loads(_remove_trailing_commas(m2.group(0)))
            else:
                dbg = Path(path).with_name("bad_llm_json_AUB_debug.txt")
                dbg.write_text(cleaned, encoding="utf-8")
                raise ValueError(f"No JSON found. See {dbg}")

    if isinstance(obj, dict) and "trees" in obj:
        obj = obj["trees"]
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list) or len(obj) == 0:
        raise ValueError("LLM trees must be a non-empty list.")
    for i, t in enumerate(obj, 1):
        if not isinstance(t, dict) or "root" not in t:
            raise ValueError(f"Tree #{i} has no 'root'.")
    return obj

OPS = {"<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b, ">=": lambda a, b: a >= b}

def _eval_tree_node(node: Dict[str, Any], row: pd.Series) -> int:
    if "leaf" in node:
        return int(node["leaf"])
    feat = node.get("feature"); op = node.get("op"); val = node.get("value")
    if feat not in row.index or op not in OPS:
        return 0
    x = row[feat]
    if pd.isna(x):
        return int(node.get("majority", 0)) if "majority" in node else 0
    child = node.get("left") if OPS[op](x, val) else node.get("right")
    if child is None:
        return 0
    return _eval_tree_node(child, row)

def predict_with_llm_tree_json(X: pd.DataFrame, tree: Dict[str, Any]) -> np.ndarray:
    preds = np.zeros(len(X), dtype=int)
    root = tree.get("root", {})
    for i, (_, r) in enumerate(X.iterrows()):
        preds[i] = _eval_tree_node(root, r)
    return preds

def llm_vote_matrix_and_prob(X: pd.DataFrame, trees: List[Dict[str,Any]]) -> Tuple[np.ndarray, np.ndarray]:
    mat = np.column_stack([predict_with_llm_tree_json(X, t) for t in trees])
    p = mat.mean(axis=1)  # fraction of trees predicting presence
    return mat, p

# =========================
# 5-fold CV runner (Accuracy & Macro-F1)
# =========================
def run_cv(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Build AUB target and restrict predictors
    for col in [PREZ_COL, TRUEABS_COL]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column: {col}")

    prez = pd.to_numeric(df[PREZ_COL], errors="coerce").fillna(0).astype(int)
    tabs = pd.to_numeric(df[TRUEABS_COL], errors="coerce").fillna(0).astype(int)
    mask_labeled = (prez == 1) | (tabs == 1)
    data = df.loc[mask_labeled].copy()

    # remove ambiguous rows with both labels
    both = (data[PREZ_COL].astype(int)==1) & (data[TRUEABS_COL].astype(int)==1)
    if both.any():
        data = data.loc[~both].copy()

    data[TARGET] = (data[PREZ_COL].astype(int) == 1).astype(int)

    # Leakage drop; keep just predictors
    all_cols = list(data.columns)
    drop_cols = set(id_like_cols(all_cols) + [PREZ_COL, TRUEABS_COL, TARGET])
    drop_cols.update([c for c in all_cols if c.endswith("_PREZ")])
    X_all = data.drop(columns=list(drop_cols), errors="ignore").copy()
    y_all = data[TARGET].astype(int).values

    missing = [c for c in PREDICTORS if c not in X_all.columns]
    if missing:
        raise RuntimeError(f"Missing expected predictors: {missing}")
    X_all = X_all[PREDICTORS].copy()

    # Load LLM trees (impute inside CV)
    trees = load_llm_trees(LLM_TREES_PATH)
    T = len(trees)
    ensemble_name = f"LLM ({T}-tree ensemble)"
    print(f"[i] Loaded {T} LLM trees from: {LLM_TREES_PATH}")

    # Prepare CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    per_fold: Dict[str, List[Tuple[float,float]]] = {}
    per_fold_rows: List[List[Any]] = []

    def record(name: str, y_true, y_pred, fold_id: int):
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        per_fold.setdefault(name, []).append((acc, f1m))
        per_fold_rows.append([fold_id, name, acc, f1m])

    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all), start=1):
        X_tr_raw, X_te_raw = X_all.iloc[tr_idx].copy(), X_all.iloc[te_idx].copy()
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # Impute on TRAIN only → transform TEST
        imputer = KNNImputer(n_neighbors=KNN_K)
        X_tr = pd.DataFrame(imputer.fit_transform(X_tr_raw), columns=PREDICTORS, index=X_tr_raw.index)
        X_te = pd.DataFrame(imputer.transform(X_te_raw),   columns=PREDICTORS, index=X_te_raw.index)

        # DT (depth=2)
        dt = DecisionTreeClassifier(max_depth=DT_DEPTH, min_samples_leaf=5, random_state=RANDOM_STATE)
        dt.fit(X_tr, y_tr)
        p_dt_tr = dt.predict_proba(X_tr)[:,1]
        p_dt_te = dt.predict_proba(X_te)[:,1]
        y_dt_te = (p_dt_te >= 0.5).astype(int)
        record(f"DT(d={DT_DEPTH})", y_te, y_dt_te, fold_id)

        # LLM ensemble (probability = mean vote)
        mat_tr, p_llm_tr = llm_vote_matrix_and_prob(X_tr, trees)
        mat_te, p_llm_te = llm_vote_matrix_and_prob(X_te, trees)
        y_llm_maj_te = (p_llm_te >= 0.5).astype(int)
        record(ensemble_name, y_te, y_llm_maj_te, fold_id)

        # ====== Hybrids (match AUT) ======
        # AND-presence with tau
        for tau in AND_TAU_LIST:
            y_and = ((y_dt_te == 1) & (p_llm_te >= tau)).astype(int)
            record(f"DT+LLM AND-presence (τ={tau:.2f})", y_te, y_and, fold_id)

        # OR-presence (for reference)
        y_or = ((y_dt_te==1) | (y_llm_maj_te==1)).astype(int)
        record("DT+LLM OR-presence", y_te, y_or, fold_id)

        # k-veto (rarely helpful, kept for parity)
        absence_votes_te = (mat_te == 0).sum(axis=1)
        for k in [3, 4]:
            veto_mask = absence_votes_te >= k
            y_kveto = y_dt_te.copy()
            y_kveto[veto_mask] = 0
            record(f"DT+LLM k-veto (k={k}/{T})", y_te, y_kveto, fold_id)

        # Soft-veto grid
        absence_frac = 1.0 - p_llm_te
        for theta in SOFT_VETO_THETAS:
            for alpha in SOFT_VETO_ALPHAS:
                p_softv = p_dt_te.copy()
                p_softv[absence_frac >= theta] *= alpha
                y_softv = (p_softv >= 0.5).astype(int)
                record(f"DT+LLM soft-veto (θ={theta:.2f}, α={alpha:.2f})", y_te, y_softv, fold_id)

        # Soft-blend fine sweep
        for w in SOFT_BLEND_WEIGHTS:
            p_blend = w*p_dt_te + (1.0-w)*p_llm_te
            y_blend = (p_blend >= 0.5).astype(int)
            record(f"DT+LLM soft blend (w={w:.3f})", y_te, y_blend, fold_id)

        # Stacked meta-model on [p_dt, p_llm]
        meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        X_meta_tr = np.column_stack([p_dt_tr, p_llm_tr])
        X_meta_te = np.column_stack([p_dt_te, p_llm_te])
        meta.fit(X_meta_tr, y_tr)
        p_meta_te = meta.predict_proba(X_meta_te)[:,1]
        y_meta_te = (p_meta_te >= 0.5).astype(int)
        record("DT+LLM stacked (logistic)", y_te, y_meta_te, fold_id)

    # Aggregate mean ± std
    rows = []
    for name, vals in per_fold.items():
        accs = np.array([a for a,_ in vals], dtype=float)
        f1ms = np.array([f for _,f in vals], dtype=float)
        rows.append([name, accs.mean(), accs.std(ddof=1), f1ms.mean(), f1ms.std(ddof=1)])

    df_sum = pd.DataFrame(rows, columns=["Model", "Accuracy_mean", "Accuracy_std", "MacroF1_mean", "MacroF1_std"]) \
             .sort_values(["MacroF1_mean","Accuracy_mean"], ascending=[False, False])

    df_folds = pd.DataFrame(per_fold_rows, columns=["Fold","Model","Accuracy","MacroF1"]) \
               .sort_values(["Model","Fold"])

    return df_sum, df_folds

# =========================
# MAIN
# =========================
def main():
    banner("[1] Load & clean")
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(EXCEL_PATH)
    df_raw = load_excel_any(EXCEL_PATH)
    df = clean_decimal_commas(df_raw)

    banner("[2] 5-fold CV (Accuracy & Macro-F1) + tuned hybrids (AUT-aligned)")
    df_sum, df_folds = run_cv(df)
    print("\nTop results (sorted by MacroF1 then Accuracy):\n")
    print(df_sum.head(20).to_string(index=False))

    out_base = Path(EXCEL_PATH).expanduser().resolve().parent / "outputs"
    out_base.mkdir(exist_ok=True, parents=True)
    sum_path   = out_base / "AUB_cv5_accuracy_f1_tuned_AUTaligned.csv"
    folds_path = out_base / "AUB_cv5_perfold_metrics_AUTaligned.csv"
    df_sum.to_csv(sum_path, index=False)
    df_folds.to_csv(folds_path, index=False)
    print(f"\n[i] Saved summary to: {sum_path}")
    print(f"[i] Saved per-fold metrics to: {folds_path}")

if __name__ == "__main__":
    main()


import pandas as pd

# Load your data (Excel or CSV)
df = pd.read_excel(EXCEL_PATH)   # or pd.read_csv("NETWORK.csv")

def count_sites(data: pd.DataFrame, sp: str):
    prez_col = f"{sp}_PREZ"
    trueabs_col = f"{sp}_TRUEABS"
    assert prez_col in data.columns and trueabs_col in data.columns, f"Missing {sp} cols"

    d = data.copy()
    d["prez"] = pd.to_numeric(d[prez_col], errors="coerce").fillna(0).astype(int)
    d["trueabs"] = pd.to_numeric(d[trueabs_col], errors="coerce").fillna(0).astype(int)

    # keep rows with either presence or true absence
    d = d[(d["prez"] == 1) | (d["trueabs"] == 1)]
    # drop ambiguous rows (both marked as 1)
    d = d[~((d["prez"] == 1) & (d["trueabs"] == 1))]

    return {
        "species": sp,
        "N_sites": len(d),
        "N_presence": int((d["prez"] == 1).sum()),
        "N_true_abs": int((d["trueabs"] == 1).sum())
    }

# Collect results for all species
res = pd.DataFrame([
    count_sites(df, "AUT"),
    count_sites(df, "AUB"),
    count_sites(df, "FXL")
])

print(res)

# Also get just the site counts for manuscript text
N_AUT = res.loc[res.species=="AUT", "N_sites"].values[0]
N_AUB = res.loc[res.species=="AUB", "N_sites"].values[0]
N_FXL = res.loc[res.species=="FXL", "N_sites"].values[0]
print(f"AUT: {N_AUT} sites | AUB: {N_AUB} sites | FXL: {N_FXL} sites")