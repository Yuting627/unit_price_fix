"""Per-period ANOVA monitor of the base_cell choice: partial eta^2 per factor on log1p(sales).

Main-effects OLS ``y ~ C(PLATFORMNAME) + C(SHOPTYPE) + C(PROVINCE) + C(category)``.
SS_f is the Type II sum of squares (RSS without f minus RSS of the full model), which
matches ``statsmodels.anova_lm(typ=2)`` but is solved through sparse normal equations
so millions of rows fit in memory.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

FACTORS = ("PLATFORMNAME", "SHOPTYPE", "PROVINCE", "category")
BASE_CELL_CANDIDATES = ("PLATFORMNAME", "SHOPTYPE")


def _dummies(values: pd.Series) -> sparse.csr_matrix | None:
    codes, uniques = pd.factorize(values, sort=True)
    k = len(uniques)
    if k < 2:
        return None
    rows = np.nonzero(codes > 0)[0]
    cols = codes[rows] - 1
    data = np.ones(rows.size)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(values), k - 1))


def _rss(blocks: list, y: np.ndarray) -> tuple[float, int]:
    X = sparse.hstack(blocks, format="csr")
    xtx = (X.T @ X).toarray()
    xty = X.T @ y
    beta, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), int(np.linalg.matrix_rank(xtx))


def partial_eta2(df: pd.DataFrame, y_col: str = "y", factors=FACTORS) -> tuple[pd.DataFrame, float]:
    """Return (table with factor/df/ss/partial_eta2, R^2 of full model)."""
    data = df.dropna(subset=[y_col, *factors])
    y = data[y_col].to_numpy(dtype=float)
    intercept = sparse.csr_matrix(np.ones((len(y), 1)))
    blocks = {f: _dummies(data[f].astype(str)) for f in factors}
    active = [f for f in factors if blocks[f] is not None]
    rss_full, rank_full = _rss([intercept] + [blocks[f] for f in active], y)
    tss = float(((y - y.mean()) ** 2).sum())
    rows = []
    for f in active:
        rss_red, rank_red = _rss([intercept] + [blocks[g] for g in active if g != f], y)
        ss = max(rss_red - rss_full, 0.0)
        rows.append(
            {
                "factor": f,
                "df": rank_full - rank_red,
                "ss": ss,
                "partial_eta2": ss / (ss + rss_full) if ss + rss_full > 0 else np.nan,
            }
        )
    rows.append({"factor": "Residual", "df": len(y) - rank_full, "ss": rss_full, "partial_eta2": np.nan})
    r2 = 1 - rss_full / tss if tss > 0 else np.nan
    return pd.DataFrame(rows), r2


def classify_strength(eta2: float, strong: float, medium: float) -> str:
    if pd.isna(eta2):
        return ""
    if eta2 >= strong:
        return "strong"
    if eta2 >= medium:
        return "medium"
    return "weak"


def interaction_eta2(df: pd.DataFrame, y_col: str, a: str, b: str, factors=FACTORS) -> dict:
    """Partial eta^2 of the a:b interaction on top of the main-effects model.

    A main effect can look weak when it flips sign across levels of another factor; the interaction catches that.
    """
    data = df.dropna(subset=[y_col, *factors])
    y = data[y_col].to_numpy(dtype=float)
    intercept = sparse.csr_matrix(np.ones((len(y), 1)))
    mains = [m for m in (_dummies(data[f].astype(str)) for f in factors) if m is not None]
    cell = _dummies(data[a].astype(str) + "|" + data[b].astype(str))
    rss_main, rank_main = _rss([intercept, *mains], y)
    if cell is None:
        return {"factor": f"{a}:{b}", "df": 0, "ss": 0.0, "partial_eta2": np.nan}
    rss_int, rank_int = _rss([intercept, *mains, cell], y)
    ss = max(rss_main - rss_int, 0.0)
    return {"factor": f"{a}:{b}", "df": rank_int - rank_main, "ss": ss,
            "partial_eta2": ss / (ss + rss_int) if ss + rss_int > 0 else np.nan}


def run_stratify(sc: pd.DataFrame, strong: float, medium: float, factors=FACTORS) -> tuple[pd.DataFrame, float]:
    """ANOVA on log1p(sales_value) of store x category x period rows, plus the base_cell candidates' interaction."""
    data = sc[list(factors)].copy()
    data["y"] = np.log1p(sc["sales_value"].clip(lower=0))
    table, r2 = partial_eta2(data, "y", factors)
    inter = interaction_eta2(data, "y", *BASE_CELL_CANDIDATES, factors=factors)
    table = pd.concat([table.iloc[:-1], pd.DataFrame([inter]), table.iloc[-1:]], ignore_index=True)
    table["strength"] = [classify_strength(e, strong, medium) for e in table["partial_eta2"]]
    table["n_obs"] = len(data)
    table["r2_model"] = r2
    return table, r2


def suggest_base_cell(table: pd.DataFrame, current: tuple[str, ...], candidates=BASE_CELL_CANDIDATES) -> list[str]:
    """Hard strata from the base_cell candidates, with hysteresis so the suggestion does not flip-flop.

    A candidate is added when its main effect or the candidates' interaction is strong; a candidate already
    in ``current`` is only dropped when both are weak. Keeps ``current`` when nothing qualifies.
    """
    strength = dict(zip(table["factor"], table["strength"]))
    inter = strength.get(":".join(candidates), "")

    def keep(f):
        if strength.get(f) == "strong" or inter == "strong":
            return True
        return f in current and ("medium" in (strength.get(f), inter))

    chosen = [f for f in candidates if keep(f)]
    return chosen or list(current)


def check_base_cell(sc: pd.DataFrame, current, stratify, logger=None) -> tuple[pd.DataFrame, list[str]]:
    """ANOVA on ``sc``; warn when the suggested base_cell differs from ``current``. Never changes config."""
    table, r2 = run_stratify(sc, stratify.eta2_strong, stratify.eta2_medium)
    suggested = suggest_base_cell(table, tuple(current))
    table["base_cell_current"] = " x ".join(current)
    table["base_cell_suggested"] = " x ".join(suggested)
    if logger is not None:
        eff = table[table["factor"] != "Residual"]
        logger.info(
            "[STRATIFY] %s; R2=%.3f",
            "; ".join(f"{r.factor} eta2={r.partial_eta2:.3f} {r.strength}" for r in eff.itertuples()), r2,
        )
        if list(suggested) != list(current):
            logger.warning("[STRATIFY] WARNING 建议 base_cell=%s, 配置为 %s -> 人工确认后修改 peer_params.json",
                           " x ".join(suggested), " x ".join(current))
        else:
            logger.info("[STRATIFY] base_cell=%s 与建议一致", " x ".join(current))
        if dict(zip(eff["factor"], eff["strength"])).get("PROVINCE") == "strong":
            logger.warning("[STRATIFY] PROVINCE eta2 为 strong: 省份合并需谨慎, 关注 ibd_define 的 VI/KS/PSI 与 forced_nearest")
    return table, suggested
