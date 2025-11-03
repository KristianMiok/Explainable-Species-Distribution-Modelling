#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUT: 5-fold CV with DT, LLM, and tuned hybrids
- Predictors: RWQ, ALT, FFP, BIO1
- Metrics: Accuracy & Macro-F1 (mean ± std across 5 folds)
- Adds grids:
    * AND-presence: LLM vote threshold tau in {0.50, 0.60, 0.70, 0.80}
    * Soft-veto: theta in {0.5, 0.6, 0.7}, alpha in {0.25, 0.5, 0.75}
    * Soft-blend: w in {0.50, 0.55, 0.60, 0.625, 0.65, 0.675, 0.70}
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
# CONFIG (AUT)
# =========================
EXCEL_PATH = "/Users/kristianmiok/Desktop/Lucian/LLM/RF/Data/NETWORK.xlsx"
LLM_TREES_PATH = str(
    Path(EXCEL_PATH).expanduser().resolve().parent / "outputs" / "paper_llm_trees_AUT.json"
)

SPECIES = "AUT"
PREZ_COL = "AUT_PREZ"
TRUEABS_COL = "AUT_TRUEABS"
TARGET = f"{SPECIES}_BIN"      # 1 if *_PREZ==1, 0 if *_TRUEABS==1

PREDICTORS = ["RWQ", "ALT", "FFP", "BIO1"]
DT_DEPTH = 2
KNN_K = 10
N_SPLITS = 5
RANDOM_STATE = 42

# Hybrid grids
AND_TAU_LIST = [0.50, 0.60, 0.70, 0.80]          # LLM vote threshold for AND-presence
SOFT_VETO_THETAS = [0.5, 0.6, 0.7]               # absence vote fraction threshold
SOFT_VETO_ALPHAS = [0.25, 0.5, 0.75]             # damping factor on DT prob
SOFT_BLEND_WEIGHTS = [0.50, 0.55, 0.60, 0.625, 0.65, 0.675, 0.70]  # DT weight w
BEST_SOFT_BLEND_W = 0.55  # used for per-instance explanation logging (top AUT setting)

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
                dbg = Path(LLM_TREES_PATH).with_name("bad_llm_json_AUT_debug.txt")
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

# ----- Explanation helpers (DT path + LLM rule examples) -----
def extract_dt_path_for_row(dt: DecisionTreeClassifier, feature_names: List[str], x_row: np.ndarray) -> str:
    """Return a human-readable path 'FEAT<=thr -> FEAT>thr -> leaf=…' for one row."""
    tree = dt.tree_
    feat_idx = tree.feature
    thresh = tree.threshold
    node = 0
    parts = []
    while feat_idx[node] != -2:  # not a leaf
        f = feature_names[feat_idx[node]]
        t = thresh[node]
        if x_row[feat_idx[node]] <= t:
            parts.append(f"{f}<= {t:.3f}")
            node = tree.children_left[node]
        else:
            parts.append(f"{f}> {t:.3f}")
            node = tree.children_right[node]
    # majority class at leaf:
    leaf_counts = tree.value[node][0]
    leaf_class = int(np.argmax(leaf_counts))
    parts.append(f"leaf={leaf_class}")
    return " → ".join(parts)

def explain_llm_tree_for_row(tree_json: Dict[str, Any], row: pd.Series) -> Tuple[int, str]:
    """Traverse one LLM rule tree on a row; return (leaf, 'cond1 & cond2 …')."""
    conds = []
    node = tree_json.get("root", {})
    while "leaf" not in node:
        f = node.get("feature"); op = node.get("op"); v = node.get("value")
        if f is None or op is None:
            return 0, "invalid"
        x = row.get(f, np.nan)
        if pd.isna(x):
            if "majority" in node:
                return int(node["majority"]), "missing→majority"
            else:
                return 0, "missing→0"
        if op == "<=":
            go_left = (x <= v)
        elif op == ">":
            go_left = (x > v)
        elif op == "<":
            go_left = (x < v)
        elif op == ">=":
            go_left = (x >= v)
        else:
            return 0, "badop"
        conds.append(f"{f}{op}{v}")
        node = node["left"] if go_left else node["right"]
        if node is None:
            return 0, "incomplete"
    return int(node["leaf"]), " & ".join(conds)

def llm_vote_and_examples(trees: List[Dict[str, Any]], row: pd.Series, top_k: int = 3) -> Tuple[float, List[str]]:
    """Return vote fraction for presence and up to 3 example rule strings that voted with the majority."""
    votes = []
    examples = {1: [], 0: []}
    for t in trees:
        y, path_str = explain_llm_tree_for_row(t, row)
        votes.append(y)
        if len(examples[y]) < top_k:
            ex_id = t.get("tree_id", "?")
            examples[y].append(f"tree#{ex_id}: {path_str} → {y}")
    votes = np.array(votes, dtype=int)
    maj = int(votes.mean() >= 0.5)
    return votes.mean(), examples[maj]

# =========================
# 5-fold CV runner (Accuracy & Macro-F1)
# =========================
def run_cv(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Build AUT target and restrict predictors
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

    # 10-NN imputation (numeric)
    imputer = KNNImputer(n_neighbors=KNN_K)
    X_all[PREDICTORS] = imputer.fit_transform(X_all[PREDICTORS])

    # Load LLM trees
    trees = load_llm_trees(LLM_TREES_PATH)
    T = len(trees)
    ensemble_name = f"LLM ({T}-tree ensemble)"
    print(f"[i] Loaded {T} LLM trees from: {LLM_TREES_PATH}")

    # Prepare CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    per_fold: Dict[str, List[Tuple[float,float]]] = {}
    # Optional: store per-fold metrics for top models
    per_fold_rows: List[List[Any]] = []
    # Collectors for explanation completeness & stability
    per_instance_rows: List[Dict[str, Any]] = []
    dt_split_counter: List[Tuple[str,int,str,float]] = []
    llm_rule_fires: List[Tuple[str,int,int,int]] = []
    feature_names = PREDICTORS

    def record(name: str, y_true, y_pred, fold_id: int):
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        per_fold.setdefault(name, []).append((acc, f1m))
        per_fold_rows.append([fold_id, name, acc, f1m])

    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all), start=1):
        X_tr, X_te = X_all.iloc[tr_idx].copy(), X_all.iloc[te_idx].copy()
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # DT (depth=2)
        dt = DecisionTreeClassifier(max_depth=DT_DEPTH, min_samples_leaf=5, random_state=RANDOM_STATE)
        dt.fit(X_tr, y_tr)
        p_dt_tr = dt.predict_proba(X_tr)[:,1]
        p_dt_te = dt.predict_proba(X_te)[:,1]
        y_dt_te = (p_dt_te >= 0.5).astype(int)
        # Log DT splits (stability across folds)
        for node_idx in range(dt.tree_.node_count):
            feat_id = dt.tree_.feature[node_idx]
            if feat_id != -2:
                dt_split_counter.append((SPECIES, fold_id, feature_names[feat_id], float(dt.tree_.threshold[node_idx])))
        record(f"DT(d={DT_DEPTH})", y_te, y_dt_te, fold_id)

        # LLM ensemble (probability = mean vote)
        mat_tr, p_llm_tr = llm_vote_matrix_and_prob(X_tr, trees)
        mat_te, p_llm_te = llm_vote_matrix_and_prob(X_te, trees)
        y_llm_maj_te = (p_llm_te >= 0.5).astype(int)
        record(ensemble_name, y_te, y_llm_maj_te, fold_id)

        # ====== Hybrids ======
        # AND-presence with tau
        for tau in AND_TAU_LIST:
            y_and = ((y_dt_te == 1) & (p_llm_te >= tau)).astype(int)
            record(f"DT+LLM AND-presence (τ={tau:.2f})", y_te, y_and, fold_id)

        # OR-presence (kept for reference; usually harms precision)
        y_or = ((y_dt_te==1) | (y_llm_maj_te==1)).astype(int)
        record("DT+LLM OR-presence", y_te, y_or, fold_id)

        # k-veto (rarely helpful here, but reported)
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

        # Compute best soft-blend for explanations
        p_blend_best = BEST_SOFT_BLEND_W * p_dt_te + (1.0 - BEST_SOFT_BLEND_W) * p_llm_te

        # Stacked meta-model (kept for completeness)
        meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        X_meta_tr = np.column_stack([p_dt_tr, p_llm_tr])
        X_meta_te = np.column_stack([p_dt_te, p_llm_te])
        meta.fit(X_meta_tr, y_tr)
        p_meta_te = meta.predict_proba(X_meta_te)[:,1]
        y_meta_te = (p_meta_te >= 0.5).astype(int)
        record("DT+LLM stacked (logistic)", y_te, y_meta_te, fold_id)

        # ---- Per-instance explanations on the test set ----
        X_te_np = X_te.values
        for i, (row_idx, row) in enumerate(X_te.iterrows()):
            dt_path = extract_dt_path_for_row(dt, feature_names, X_te_np[i])
            p_llm_i = float(p_llm_te[i])
            y_llm_i = int(p_llm_i >= 0.5)
            p_h_i = float(p_blend_best[i])
            y_h_i = int(p_h_i >= 0.5)
            vote_frac, examples = llm_vote_and_examples(trees, row, top_k=3)

            per_instance_rows.append({
                "species": SPECIES,
                "fold": fold_id,
                "row_id": int(row_idx),
                "y_true": int(y_te[i]),
                "p_dt": float(p_dt_te[i]),
                "y_dt": int(y_dt_te[i]),
                "dt_path": dt_path,
                "p_llm": p_llm_i,
                "y_llm": y_llm_i,
                "p_hybrid": p_h_i,
                "y_hybrid": y_h_i,
                "hybrid_name": f"DT+LLM soft blend (w={BEST_SOFT_BLEND_W:.2f})",
                "llm_top3_rules": " | ".join(examples)
            })

            # LLM rule firing (stability)
            for t in trees:
                leaf, _ = explain_llm_tree_for_row(t, row)
                llm_rule_fires.append((SPECIES, fold_id, int(t.get("tree_id", -1)), int(leaf)))

    # Aggregate mean ± std
    rows = []
    for name, vals in per_fold.items():
        accs = np.array([a for a,_ in vals], dtype=float)
        f1ms = np.array([f for _,f in vals], dtype=float)
        rows.append([name, accs.mean(), accs.std(ddof=1), f1ms.mean(), f1ms.std(ddof=1)])

    # Build explanation/stability DataFrames
    df_expl = pd.DataFrame(per_instance_rows).sort_values(["species","fold","row_id"])
    df_dt_stab = (
        pd.DataFrame(dt_split_counter, columns=["species","fold","feature","threshold"])
          .groupby(["species","feature"], as_index=False)
          .agg(n_splits=("threshold","count"),
               mean_threshold=("threshold","mean"),
               sd_threshold=("threshold","std"))
          .sort_values(["species","n_splits"], ascending=[True, False])
    )
    df_llm_fire = pd.DataFrame(llm_rule_fires, columns=["species","fold","tree_id","vote"])
    df_llm_stab = (
        df_llm_fire
          .groupby(["species","tree_id"], as_index=False)
          .agg(fired_presence=("vote", lambda v: int(np.sum(v==1))),
               fired_absence=("vote", lambda v: int(np.sum(v==0))),
               support_rate=("vote", "mean"))
          .sort_values(["species","support_rate"], ascending=[True, False])
    )

    df_sum = pd.DataFrame(rows, columns=["Model", "Accuracy_mean", "Accuracy_std", "MacroF1_mean", "MacroF1_std"]) \
        .sort_values(["MacroF1_mean","Accuracy_mean"], ascending=[False, False])

    df_folds = pd.DataFrame(per_fold_rows, columns=["Fold","Model","Accuracy","MacroF1"]) \
        .sort_values(["Model","Fold"])

    return df_sum, df_folds, df_expl, df_dt_stab, df_llm_stab

# =========================
# MAIN
# =========================
def main():
    banner("[1] Load & clean")
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(EXCEL_PATH)
    df_raw = load_excel_any(EXCEL_PATH)
    df = clean_decimal_commas(df_raw)

    banner("[2] 5-fold CV (Accuracy & Macro-F1) + tuned hybrids")
    df_sum, df_folds, df_expl, df_dt_stab, df_llm_stab = run_cv(df)
    print("\nTop results (sorted by MacroF1 then Accuracy):\n")
    print(df_sum.head(20).to_string(index=False))

    out_base = Path(EXCEL_PATH).expanduser().resolve().parent / "outputs"
    out_base.mkdir(exist_ok=True, parents=True)
    sum_path = out_base / "AUT_cv5_accuracy_f1_tuned.csv"
    folds_path = out_base / "AUT_cv5_perfold_metrics.csv"
    df_sum.to_csv(sum_path, index=False)
    df_folds.to_csv(folds_path, index=False)
    print(f"\n[i] Saved summary to: {sum_path}")
    print(f"[i] Saved per-fold metrics to: {folds_path}")
    expl_path = out_base / "AUT_cv5_per_instance_explanations.csv"
    dtstab_path = out_base / "AUT_cv5_dt_split_stability.csv"
    llmstab_path = out_base / "AUT_cv5_llm_rule_stability.csv"
    df_expl.to_csv(expl_path, index=False)
    df_dt_stab.to_csv(dtstab_path, index=False)
    df_llm_stab.to_csv(llmstab_path, index=False)
    print(f"[i] Saved per-instance explanations to: {expl_path}")
    print(f"[i] Saved DT split stability to: {dtstab_path}")
    print(f"[i] Saved LLM rule stability to: {llmstab_path}")

if __name__ == "__main__":
    main()