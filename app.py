import os
import io
import json
import math
import time
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import duckdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from pydantic import BaseModel, Field, field_validator, ConfigDict


# -----------------------------------------------------------------------------
# Product constants
# -----------------------------------------------------------------------------

APP_NAME = "Incrementality Measurement Platform"
APP_SUBTITLE = "AI Advertising Intelligence and Revenue Optimization"

OUTPUT_DIR = "outputs"
DATA_DIR = "data"

CHANNELS = ["Search", "Social", "Video", "Retail Media"]
REGIONS = ["Northeast", "Midwest", "South", "West"]

CHANNEL_SCALE = {
    "Search": 1.00,
    "Social": 0.82,
    "Video": 0.65,
    "Retail Media": 0.90,
}

TRUE_CHANNEL_IROAS = {
    "Search": 2.15,
    "Social": 1.25,
    "Video": 0.95,
    "Retail Media": 1.65,
}

INPUT_COLUMNS = [
    "market_id",
    "week_start",
    "channel",
    "spend",
    "impressions",
    "clicks",
    "conversions",
    "total_revenue",
    "attributed_revenue",
    "treatment_flag",
    "experiment_id",
]


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title=f"{APP_NAME} | {APP_SUBTITLE}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


ensure_dirs()

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []


def log_event(event: str, detail: str = "") -> None:
    st.session_state.audit_log.append(
        {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": event,
            "detail": detail,
            "actor": "local_demo_user",
            "environment": "synthetic_cpu_only",
        }
    )


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        if denominator is None or abs(float(denominator)) < 1e-9:
            return float(default)
        return float(numerator) / float(denominator)
    except Exception:
        return float(default)


def fig_to_rgb(fig: matplotlib.figure.Figure) -> np.ndarray:
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)
    return arr[:, :, :3]


# -----------------------------------------------------------------------------
# Pydantic data contract for inbound ad measurement records
# -----------------------------------------------------------------------------

class WeeklyAdInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    market_id: str = Field(min_length=1)
    week_start: dt.date
    channel: str
    spend: float = Field(ge=0.0)
    impressions: int = Field(ge=0)
    clicks: int = Field(ge=0)
    conversions: float = Field(ge=0.0)
    total_revenue: float = Field(ge=0.0)
    attributed_revenue: float = Field(ge=0.0)
    treatment_flag: bool
    experiment_id: Optional[str] = None

    @field_validator("channel")
    def validate_channel(cls, v: str) -> str:
        if v not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}")
        return v


def validate_raw_records(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        try:
            obj = WeeklyAdInput(**rec)
            valid_rows.append(obj.model_dump())
        except Exception as exc:
            errors.append(
                {
                    "row_index": idx,
                    "error_type": "schema_validation",
                    "message": str(exc),
                    "record_preview": {k: rec.get(k) for k in list(rec)[:6]},
                }
            )

    valid_df = pd.DataFrame(valid_rows)

    if not valid_df.empty:
        key_cols = ["market_id", "week_start", "channel"]
        dup_mask = valid_df.duplicated(subset=key_cols, keep=False)
        if dup_mask.any():
            for i in valid_df.index[dup_mask]:
                errors.append(
                    {
                        "row_index": int(i),
                        "error_type": "duplicate_key",
                        "message": "Duplicate market_id + week_start + channel",
                        "record_preview": valid_df.loc[i].to_dict(),
                    }
                )
            valid_df = valid_df.loc[~dup_mask].reset_index(drop=True)

    errors_df = pd.DataFrame(errors)
    return valid_df, errors_df


# -----------------------------------------------------------------------------
# Synthetic world generator
# -----------------------------------------------------------------------------

def generate_synthetic_world(
    noise: float,
    effect_strength: float,
    n_markets: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)

    market_ids = [f"DMK-{i:03d}" for i in range(1, n_markets + 1)]
    market_names = [f"Synthetic Market {i:02d}" for i in range(1, n_markets + 1)]

    markets = pd.DataFrame(
        {
            "market_id": market_ids,
            "market_name": market_names,
            "region": rng.choice(REGIONS, size=n_markets, replace=True),
            "population_millions": np.clip(rng.gamma(2.0, 0.70, size=n_markets) + 0.20, 0.20, 8.0),
            "income_index": np.clip(rng.normal(100.0, 14.0, size=n_markets), 62.0, 155.0),
            "digital_penetration": np.clip(rng.beta(5.0, 2.0, size=n_markets), 0.25, 0.98),
            "retail_index": np.clip(rng.gamma(2.0, 5.0, size=n_markets) + 5.0, 5.0, 80.0),
        }
    )

    weeks = pd.date_range("2025-01-06", periods=52, freq="W-MON")

    experiments: Dict[str, Dict[str, Any]] = {}
    all_market_ids = markets["market_id"].tolist()

    for ci, ch in enumerate(CHANNELS):
        size_t = max(3, int(round(n_markets * 0.18)))
        size_c = max(3, int(round(n_markets * 0.22)))

        if size_t + size_c > n_markets:
            size_t = max(2, n_markets // 3)
            size_c = max(2, n_markets // 3)

        chosen = rng.choice(np.array(all_market_ids), size=size_t + size_c, replace=False)
        treat = chosen[:size_t].tolist()
        control = chosen[size_t:].tolist()

        start_idx = 16 + ci * 3
        exp_len = 8
        pre_len = 10

        if start_idx + exp_len > len(weeks):
            start_idx = max(10, len(weeks) - exp_len - 2)

        exp_weeks = weeks[start_idx : start_idx + exp_len]
        pre_weeks = weeks[max(0, start_idx - pre_len) : start_idx]

        experiments[ch] = {
            "id": f"EXP-{ch.replace(' ', '')[:4].upper()}-{seed}",
            "channel": ch,
            "treat": treat,
            "control": control,
            "exp_start": exp_weeks[0],
            "exp_end": exp_weeks[-1],
            "pre_start": pre_weeks[0],
            "pre_end": pre_weeks[-1],
            "exp_weeks_count": int(len(exp_weeks)),
            "pre_weeks_count": int(len(pre_weeks)),
        }

    rows: List[Dict[str, Any]] = []

    for ch in CHANNELS:
        meta = experiments[ch]
        treat_set = set(meta["treat"])
        control_set = set(meta["control"])

        exp_dates = {
            d.date()
            for d in pd.date_range(meta["exp_start"], meta["exp_end"], freq="W-MON")
        }
        pre_dates = {
            d.date()
            for d in pd.date_range(meta["pre_start"], meta["pre_end"], freq="W-MON")
        }

        for _, m in markets.iterrows():
            market_factor = float(
                np.clip(
                    0.75
                    + 0.50 * m["digital_penetration"]
                    + 0.25 * (m["income_index"] / 100.0 - 1.0),
                    0.55,
                    1.70,
                )
            )

            for wi, week in enumerate(weeks):
                wd = week.date()

                in_exp = wd in exp_dates
                in_pre = wd in pre_dates
                is_treat = m["market_id"] in treat_set
                is_control = m["market_id"] in control_set

                experiment_id = meta["id"] if (in_exp or in_pre) and (is_treat or is_control) else None
                treatment_flag = bool(in_exp and is_treat)

                season = 1.0 + 0.18 * math.sin(2.0 * math.pi * wi / 52.0) + 0.08 * math.cos(
                    2.0 * math.pi * wi / 26.0
                )

                base_spend = 9000.0 * float(m["population_millions"]) * CHANNEL_SCALE[ch] * season
                spend = max(0.0, float(rng.normal(base_spend, base_spend * noise * 0.45 + 50.0)))

                organic = max(
                    0.0,
                    float(
                        rng.normal(
                            300.0 * float(m["retail_index"]) * season,
                            300.0 * float(m["retail_index"]) * season * noise * 0.40 + 100.0,
                        )
                    ),
                )

                causal_iroas = float(TRUE_CHANNEL_IROAS[ch] * market_factor * effect_strength)

                true_incremental = spend * causal_iroas * max(0.05, float(rng.normal(1.0, noise * 0.35)))

                if treatment_flag:
                    original_spend = spend
                    spend = spend * (1.22 + 0.28 * effect_strength)
                    extra_spend = spend - original_spend
                    extra_incremental = extra_spend * causal_iroas * max(
                        0.05, float(rng.normal(1.0, noise * 0.30))
                    )
                    true_incremental += extra_incremental

                revenue_noise = float(rng.normal(0.0, 0.18 * (organic + true_incremental) + 120.0))
                total_revenue = max(0.0, organic + true_incremental + revenue_noise)

                claim_factor = float(rng.uniform(0.22, 0.42))
                attributed_noise = float(rng.normal(0.0, 0.12 * total_revenue + 80.0))
                attributed_revenue = max(
                    0.0,
                    true_incremental + claim_factor * organic + attributed_noise,
                )

                impressions = int(max(0.0, spend * float(rng.uniform(700.0, 1600.0))))
                clicks = int(max(0.0, impressions * float(rng.uniform(0.004, 0.030))))
                conversions = float(max(0.0, total_revenue / float(rng.uniform(40.0, 160.0))))

                rows.append(
                    {
                        "market_id": m["market_id"],
                        "market_name": m["market_name"],
                        "region": m["region"],
                        "week_start": week,
                        "channel": ch,
                        "spend": float(spend),
                        "impressions": int(impressions),
                        "clicks": int(clicks),
                        "conversions": float(conversions),
                        "organic_revenue": float(organic),
                        "true_incremental_revenue": float(true_incremental),
                        "total_revenue": float(total_revenue),
                        "attributed_revenue": float(attributed_revenue),
                        "treatment_flag": bool(treatment_flag),
                        "experiment_id": experiment_id,
                        "quality_flag": "",
                    }
                )

    df = pd.DataFrame(rows)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["treatment_flag"] = df["treatment_flag"].astype(bool)
    df["experiment_id"] = df["experiment_id"].astype(object).where(df["experiment_id"].notna(), None)

    outlier_threshold = df.groupby("channel")["spend"].transform(lambda s: s.quantile(0.995))
    df.loc[df["spend"] > outlier_threshold, "quality_flag"] = "spend_outlier"

    # Deliberately malformed records for validation / failure-case demo
    raw_issue_records: List[Dict[str, Any]] = []
    sample_positions = rng.choice(np.arange(len(df)), size=6, replace=False)

    for pos_i, idx in enumerate(sample_positions):
        rec_raw = df.iloc[int(idx)][INPUT_COLUMNS].to_dict()
        rec: Dict[str, Any] = {}

        for k, v in rec_raw.items():
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, pd.Timestamp):
                v = str(v.date())
            if v is not None and isinstance(v, float) and math.isnan(v):
                v = None
            rec[k] = v

        rec["treatment_flag"] = bool(rec["treatment_flag"])

        if pos_i == 0:
            rec["spend"] = -25.0
        elif pos_i == 1:
            rec["market_id"] = ""
        elif pos_i == 2:
            rec["week_start"] = "2025-02-30"
        elif pos_i == 3:
            rec["channel"] = "Unknown Channel"
        elif pos_i == 4:
            rec["total_revenue"] = None

        raw_issue_records.append(rec)

    # Duplicate failure case
    raw_issue_records.append(dict(raw_issue_records[1]))

    return {
        "df": df,
        "markets": markets,
        "experiments": experiments,
        "true_iroas": TRUE_CHANNEL_IROAS.copy(),
        "raw_issue_records": raw_issue_records,
    }


# -----------------------------------------------------------------------------
# Incrementality engines
# -----------------------------------------------------------------------------

def prepare_experiment_frame(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    e = df[df["experiment_id"] == meta["id"]].copy()
    e["period"] = np.where(
        e["week_start"].between(meta["exp_start"], meta["exp_end"]),
        "exp",
        "pre",
    )
    e["arm"] = np.where(e["market_id"].isin(meta["treat"]), "treatment", "control")
    return e


def _invalid_did() -> Dict[str, Any]:
    return {
        "valid": False,
        "iroas": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "se": np.nan,
        "incremental_revenue_est": np.nan,
        "incremental_spend_est": np.nan,
        "n_treatment": 0,
        "n_control": 0,
        "n_exp_weeks": 0,
    }


def _did_estimate(
    e: pd.DataFrame,
    meta: Dict[str, Any],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    invalid = _invalid_did()

    if e.empty:
        return invalid

    agg = (
        e.groupby(["arm", "period", "market_id"])[["total_revenue", "spend"]]
        .mean()
        .reset_index()
    )

    piv = agg.pivot_table(
        index=["arm", "market_id"],
        columns="period",
        values=["total_revenue", "spend"],
        aggfunc="mean",
    )

    if isinstance(piv.columns, pd.MultiIndex):
        piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    else:
        return invalid

    required = [
        "total_revenue_pre",
        "total_revenue_exp",
        "spend_pre",
        "spend_exp",
    ]

    for col in required:
        if col not in piv.columns:
            return invalid

    piv = piv.dropna(subset=required).reset_index()

    t = piv[piv["arm"] == "treatment"].copy()
    c = piv[piv["arm"] == "control"].copy()

    if t.empty or c.empty:
        return invalid

    t_rev_diff = (t["total_revenue_exp"] - t["total_revenue_pre"]).to_numpy(dtype=float)
    c_rev_diff = (c["total_revenue_exp"] - c["total_revenue_pre"]).to_numpy(dtype=float)
    t_spend_diff = (t["spend_exp"] - t["spend_pre"]).to_numpy(dtype=float)
    c_spend_diff = (c["spend_exp"] - c["spend_pre"]).to_numpy(dtype=float)

    n_treat = int(len(t))
    n_exp_weeks = int(e.loc[e["period"] == "exp", "week_start"].nunique())

    if n_exp_weeks <= 0:
        return invalid

    did_rev = float(np.mean(t_rev_diff) - np.mean(c_rev_diff))
    did_spend = float(np.mean(t_spend_diff) - np.mean(c_spend_diff))

    inc_rev = did_rev * n_treat * n_exp_weeks
    inc_spend = did_spend * n_treat * n_exp_weeks

    if inc_spend <= 1.0:
        return {
            **invalid,
            "incremental_revenue_est": float(inc_rev),
            "incremental_spend_est": float(inc_spend),
            "n_treatment": n_treat,
            "n_control": int(len(c)),
            "n_exp_weeks": n_exp_weeks,
        }

    point_iroas = float(inc_rev / inc_spend)

    rng = np.random.default_rng(seed)
    t_idx = np.arange(len(t_rev_diff))
    c_idx = np.arange(len(c_rev_diff))

    boot_iroas: List[float] = []

    for _ in range(max(10, int(n_boot))):
        ti = rng.choice(t_idx, size=len(t_idx), replace=True)
        ci = rng.choice(c_idx, size=len(c_idx), replace=True)

        b_rev = float(np.mean(t_rev_diff[ti]) - np.mean(c_rev_diff[ci]))
        b_spend = float(np.mean(t_spend_diff[ti]) - np.mean(c_spend_diff[ci]))

        b_inc_rev = b_rev * n_treat * n_exp_weeks
        b_inc_spend = b_spend * n_treat * n_exp_weeks

        if b_inc_spend > 1.0:
            boot_iroas.append(float(b_inc_rev / b_inc_spend))

    if len(boot_iroas) >= 5:
        ci_low = float(np.percentile(boot_iroas, 5))
        ci_high = float(np.percentile(boot_iroas, 95))
        se = float(np.std(boot_iroas, ddof=1))
        valid = True
    else:
        ci_low = point_iroas
        ci_high = point_iroas
        se = np.nan
        valid = False

    return {
        "valid": valid,
        "iroas": point_iroas,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "incremental_revenue_est": float(inc_rev),
        "incremental_spend_est": float(inc_spend),
        "n_treatment": n_treat,
        "n_control": int(len(c)),
        "n_exp_weeks": n_exp_weeks,
    }


def _synthetic_control_estimate(
    e: pd.DataFrame,
    meta: Dict[str, Any],
    incremental_spend_denominator: float,
) -> float:
    if not np.isfinite(incremental_spend_denominator) or incremental_spend_denominator <= 1.0:
        return np.nan

    pre = e[e["period"] == "pre"].copy()
    post = e[e["period"] == "exp"].copy()

    control_ids = [m for m in meta["control"] if m in set(pre["market_id"].unique())]
    treat_ids = meta["treat"]

    if len(control_ids) < 2 or len(treat_ids) == 0:
        return np.nan

    pre_pivot = pre[pre["market_id"].isin(control_ids)].pivot_table(
        index="week_start",
        columns="market_id",
        values="total_revenue",
        aggfunc="mean",
    )

    treat_pre = pre[pre["market_id"].isin(treat_ids)].groupby("week_start")["total_revenue"].mean()

    common_weeks = pre_pivot.index.intersection(treat_pre.index)
    if len(common_weeks) < 3:
        return np.nan

    X = pre_pivot.loc[common_weeks].ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    y = treat_pre.loc[common_weeks].ffill().bfill().fillna(0.0).to_numpy(dtype=float)

    try:
        model = LinearRegression(positive=True, fit_intercept=False)
    except TypeError:
        model = LinearRegression(fit_intercept=False)

    try:
        model.fit(X, y)
    except Exception:
        return np.nan

    w = model.coef_
    if float(np.sum(w)) <= 1e-8:
        return np.nan

    pred_pre = X @ w
    ss_res = float(np.sum((y - pred_pre) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if r2 < 0.10:
        return np.nan

    post_pivot = post[post["market_id"].isin(control_ids)].pivot_table(
        index="week_start",
        columns="market_id",
        values="total_revenue",
        aggfunc="mean",
    )

    treat_post = post[post["market_id"].isin(treat_ids)].groupby("week_start")["total_revenue"].mean()

    post_weeks = post_pivot.index.intersection(treat_post.index)
    if len(post_weeks) == 0:
        return np.nan

    Xp = post_pivot.loc[post_weeks].ffill().bfill().fillna(0.0).to_numpy(dtype=float)
    yp = treat_post.loc[post_weeks].ffill().bfill().fillna(0.0).to_numpy(dtype=float)

    pred_post = Xp @ w
    weekly_lift = yp - pred_post
    total_rev_lift = float(np.sum(weekly_lift) * len(treat_ids))

    return float(total_rev_lift / incremental_spend_denominator)


def _linear_interaction_estimate(
    e: pd.DataFrame,
    markets: pd.DataFrame,
    meta: Dict[str, Any],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    invalid = {
        "valid": False,
        "point": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "se": np.nan,
        "r2": np.nan,
    }

    if e.empty:
        return invalid

    df = e.merge(
        markets[
            [
                "market_id",
                "population_millions",
                "income_index",
                "digital_penetration",
            ]
        ],
        on="market_id",
        how="left",
    )

    df["spend_k"] = df["spend"] / 10000.0
    df["treatment_spend_k"] = df["treatment_flag"].astype(float) * df["spend_k"]
    df["week_idx"] = df["week_start"].dt.dayofyear

    feats = [
        "spend_k",
        "treatment_spend_k",
        "income_index",
        "population_millions",
        "digital_penetration",
        "week_idx",
    ]

    X = df[feats].fillna(0.0).to_numpy(dtype=float)
    y = df["total_revenue"].fillna(0.0).to_numpy(dtype=float)

    if len(df) < 20 or len(np.unique(df["market_id"])) < 3:
        return invalid

    try:
        full_model = LinearRegression().fit(X, y)
        point = float(full_model.coef_[1])
        r2 = float(full_model.score(X, y))
    except Exception:
        return invalid

    market_rows = {m: g for m, g in df.groupby("market_id")}
    market_list = list(market_rows.keys())

    rng = np.random.default_rng(seed)
    samples: List[float] = []

    for _ in range(max(10, int(n_boot))):
        choice = rng.choice(np.array(market_list), size=len(market_list), replace=True)
        boot = pd.concat([market_rows[m] for m in choice], ignore_index=True)

        if boot["market_id"].nunique() < 3 or len(boot) < 20:
            continue

        Xb = boot[feats].fillna(0.0).to_numpy(dtype=float)
        yb = boot["total_revenue"].fillna(0.0).to_numpy(dtype=float)

        try:
            mb = LinearRegression().fit(Xb, yb)
            samples.append(float(mb.coef_[1]))
        except Exception:
            continue

    if len(samples) >= 5:
        ci_low = float(np.percentile(samples, 5))
        ci_high = float(np.percentile(samples, 95))
        se = float(np.std(samples, ddof=1))
        valid = True
    else:
        ci_low = point
        ci_high = point
        se = np.nan
        valid = False

    return {
        "valid": valid,
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "r2": r2,
    }


def _data_quality_score(e: pd.DataFrame) -> float:
    if e.empty:
        return 0.0

    required = [
        "market_id",
        "week_start",
        "channel",
        "spend",
        "total_revenue",
        "attributed_revenue",
    ]

    completeness = 1.0 - float(e[required].isna().mean().mean())
    dup_rate = float(e.duplicated(subset=["market_id", "week_start", "channel"]).mean())

    if "quality_flag" in e.columns:
        outlier_rate = float((e["quality_flag"] == "spend_outlier").mean())
    else:
        outlier_rate = 0.0

    pre = e[e["period"] == "pre"] if "period" in e.columns else e
    t = pre.loc[pre["arm"] == "treatment", "spend"] if "arm" in pre.columns else pd.Series(dtype=float)
    c = pre.loc[pre["arm"] == "control", "spend"] if "arm" in pre.columns else pd.Series(dtype=float)

    if len(t) > 1 and len(c) > 1:
        var_pool = (float(t.var(ddof=1)) + float(c.var(ddof=1))) / 2.0
        if var_pool > 1e-9:
            smd = abs(float(t.mean()) - float(c.mean())) / math.sqrt(var_pool)
        else:
            smd = 1.0
    else:
        smd = 1.0

    balance_penalty = min(0.25, smd * 0.12)

    score = (
        1.0
        - dup_rate * 4.0
        - outlier_rate * 1.5
        - balance_penalty
        - (1.0 - completeness) * 2.0
    )

    return float(np.clip(score, 0.0, 1.0))


def _confidence_label(
    estimate: float,
    ci_low: float,
    ci_high: float,
    data_quality: float,
) -> str:
    if not np.isfinite(estimate) or estimate <= 0.0 or data_quality < 0.65:
        return "Low"

    width = float(ci_high - ci_low) if np.isfinite(ci_high) and np.isfinite(ci_low) else 99.0
    rel = width / max(abs(float(estimate)), 0.15)

    if rel < 0.45 and data_quality >= 0.82:
        return "High"
    if rel < 1.20 and data_quality >= 0.70:
        return "Medium"
    return "Low"


def _evaluation_metrics(cr: pd.DataFrame) -> Dict[str, float]:
    df = cr.dropna(subset=["true_iroas", "iroas_est"]).copy()
    if df.empty:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "coverage_90": np.nan,
            "sign_accuracy": np.nan,
            "spearman_ranking": np.nan,
        }

    err = df["iroas_est"] - df["true_iroas"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))

    covered = (df["ci_low_raw"] <= df["true_iroas"]) & (df["ci_high_raw"] >= df["true_iroas"])
    coverage = float(np.mean(covered))

    sign_acc = float(np.mean((df["iroas_est"] > 0) == (df["true_iroas"] > 0)))

    if len(df) > 2:
        ranking = float(df["iroas_est"].corr(df["true_iroas"], method="spearman"))
    else:
        ranking = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "coverage_90": coverage,
        "sign_accuracy": sign_acc,
        "spearman_ranking": ranking,
    }


def analyze_world(
    df: pd.DataFrame,
    markets: pd.DataFrame,
    experiments: Dict[str, Dict[str, Any]],
    true_iroas: Dict[str, float],
    seed: int,
    bootstrap_iterations: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    results: List[Dict[str, Any]] = []

    for ch, meta in experiments.items():
        e = prepare_experiment_frame(df, meta)

        did = _did_estimate(e, meta, n_boot=bootstrap_iterations, seed=seed)
        sc = _synthetic_control_estimate(e, meta, did.get("incremental_spend_est", np.nan))
        lin = _linear_interaction_estimate(
            e,
            markets,
            meta,
            n_boot=max(20, bootstrap_iterations // 2),
            seed=seed + 17,
        )
        dq = _data_quality_score(e)

        candidates: List[Dict[str, Any]] = []

        if did["valid"]:
            candidates.append(
                {
                    "method": "difference_in_differences",
                    "estimate": float(did["iroas"]),
                    "ci_low": float(did["ci_low"]),
                    "ci_high": float(did["ci_high"]),
                    "se": float(did["se"]),
                }
            )

        if lin["valid"] and np.isfinite(lin["r2"]) and lin["r2"] > 0.03:
            candidates.append(
                {
                    "method": "linear_interaction_bootstrap",
                    "estimate": float(lin["point"]),
                    "ci_low": float(lin["ci_low"]),
                    "ci_high": float(lin["ci_high"]),
                    "se": float(lin["se"]),
                }
            )

        if candidates:
            def candidate_score(c: Dict[str, Any]) -> float:
                width = c["ci_high"] - c["ci_low"]
                denom = max(abs(c["estimate"]), 0.15)
                preference = 0.0 if c["method"] == "difference_in_differences" else 0.08
                return float(width / denom + preference)

            best = min(candidates, key=candidate_score)
            est_raw = float(best["estimate"])
            ci_low_raw = float(best["ci_low"])
            ci_high_raw = float(best["ci_high"])
            se = float(best["se"])
            method = str(best["method"])
        else:
            est_raw = float(did["iroas"]) if np.isfinite(did.get("iroas", np.nan)) else 0.0
            ci_low_raw = est_raw
            ci_high_raw = est_raw
            se = 0.0
            method = "conservative_fallback"

        est = max(0.0, est_raw)
        confidence = _confidence_label(est_raw, ci_low_raw, ci_high_raw, dq)

        historical_spend = float(df.loc[df["channel"] == ch, "spend"].sum())

        results.append(
            {
                "channel": ch,
                "experiment_id": meta["id"],
                "n_treatment_markets": did.get("n_treatment", 0),
                "n_control_markets": did.get("n_control", 0),
                "n_experiment_weeks": did.get("n_exp_weeks", 0),
                "incremental_revenue_est": did.get("incremental_revenue_est", np.nan),
                "incremental_spend_est": did.get("incremental_spend_est", np.nan),
                "iroas_raw": est_raw,
                "iroas_est": est,
                "ci_low_raw": ci_low_raw,
                "ci_high_raw": ci_high_raw,
                "ci_low": max(0.0, ci_low_raw),
                "ci_high": max(0.0, ci_high_raw),
                "se": se,
                "method": method,
                "did_iroas": did.get("iroas", np.nan),
                "model_iroas": lin.get("point", np.nan),
                "model_r2": lin.get("r2", np.nan),
                "synthetic_control_iroas": sc,
                "data_quality": dq,
                "confidence": confidence,
                "true_iroas": float(true_iroas[ch]),
                "absolute_error": abs(est - float(true_iroas[ch])),
                "historical_spend": historical_spend,
            }
        )

    cr = pd.DataFrame(results)

    mp = cr.set_index("channel")
    enriched = df.copy()

    enriched["estimated_iroas"] = enriched["channel"].map(mp["iroas_est"]).fillna(0.0)
    enriched["estimated_iroas_lower"] = enriched["channel"].map(mp["ci_low"]).fillna(0.0).clip(lower=0.0)
    enriched["estimated_iroas_upper"] = enriched["channel"].map(mp["ci_high"]).fillna(0.0)
    enriched["confidence"] = enriched["channel"].map(mp["confidence"]).fillna("Low")
    enriched["data_quality_score"] = enriched["channel"].map(mp["data_quality"]).fillna(0.0)

    enriched["estimated_incremental_revenue"] = (
        enriched["spend"] * enriched["estimated_iroas"]
    )
    enriched["estimated_incremental_revenue_lower"] = (
        enriched["spend"] * enriched["estimated_iroas_lower"]
    )
    enriched["attribution_overstatement"] = (
        enriched["attributed_revenue"] - enriched["estimated_incremental_revenue"]
    )

    agg = enriched.groupby("channel").agg(
        attributed=("attributed_revenue", "sum"),
        incremental=("estimated_incremental_revenue", "sum"),
    )
    attr_factor = (agg["attributed"] / agg["incremental"].replace(0, np.nan)).fillna(1.0).clip(1.0, 6.0)
    cr["attribution_factor"] = cr["channel"].map(attr_factor).fillna(1.0)

    eval_metrics = _evaluation_metrics(cr)

    return cr, enriched, eval_metrics


@st.cache_data(show_spinner=False)
def build_platform(
    noise: float,
    effect_strength: float,
    n_markets: int,
    seed: int,
    bootstrap_iterations: int,
) -> Dict[str, Any]:
    world = generate_synthetic_world(
        noise=noise,
        effect_strength=effect_strength,
        n_markets=n_markets,
        seed=seed,
    )

    cr, enriched, eval_metrics = analyze_world(
        df=world["df"],
        markets=world["markets"],
        experiments=world["experiments"],
        true_iroas=world["true_iroas"],
        seed=seed,
        bootstrap_iterations=bootstrap_iterations,
    )

    valid_df, errors_df = validate_raw_records(world["raw_issue_records"])

    return {
        "df": enriched,
        "markets": world["markets"],
        "experiments": world["experiments"],
        "channel_results": cr,
        "evaluation": eval_metrics,
        "validation_valid": valid_df,
        "validation_errors": errors_df,
        "raw_issue_records": world["raw_issue_records"],
    }


# -----------------------------------------------------------------------------
# Demo media generation
# -----------------------------------------------------------------------------

def generate_demo_media(
    df: pd.DataFrame,
    output_base: str = os.path.join(OUTPUT_DIR, "incrementality_demo"),
    frames: int = 36,
    fps: int = 12,
) -> Dict[str, Optional[str]]:
    weekly = (
        df.groupby("week_start")[
            ["spend", "attributed_revenue", "estimated_incremental_revenue"]
        ]
        .sum()
        .reset_index()
        .sort_values("week_start")
        .tail(frames)
        .reset_index(drop=True)
    )

    if weekly.empty:
        return {"mp4": None, "gif": None}

    mp4_path = f"{output_base}.mp4"
    gif_path = f"{output_base}.gif"

    writer = None
    mp4_ok = False

    try:
        import cv2

        width, height = 800, 450
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            writer = None
    except Exception:
        writer = None

    pil_frames = []

    try:
        from PIL import Image
        pil_available = True
    except Exception:
        pil_available = False

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=100)

    try:
        for i in range(frames):
            progress = (i + 1) / float(frames)
            n = max(2, int(len(weekly) * progress))
            sub = weekly.iloc[:n]

            ax.clear()
            ax.plot(sub["week_start"], sub["spend"] / 1000.0, label="Spend ($k)")
            ax.plot(
                sub["week_start"],
                sub["attributed_revenue"] / 1000.0,
                label="Attributed revenue ($k)",
            )
            ax.plot(
                sub["week_start"],
                sub["estimated_incremental_revenue"] / 1000.0,
                label="Estimated incremental ($k)",
            )

            ax.set_title("Incrementality Measurement Platform — synthetic live demo")
            ax.set_xlabel("Week")
            ax.set_ylabel("USD thousands")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.25)
            fig.autofmt_xdate()

            img = fig_to_rgb(fig)

            if writer is not None:
                try:
                    import cv2
                    frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    writer.write(np.ascontiguousarray(frame_bgr))
                except Exception:
                    pass

            if pil_available:
                try:
                    pil_frames.append(Image.fromarray(img))
                except Exception:
                    pass

        if writer is not None:
            writer.release()
            if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                mp4_ok = True

        if pil_available and pil_frames:
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
    finally:
        plt.close(fig)

    return {
        "mp4": mp4_path if mp4_ok else None,
        "gif": gif_path if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0 else None,
    }


def run_live_simulation(container: Any, df: pd.DataFrame, frames: int = 28, delay: float = 0.14) -> None:
    weekly = (
        df.groupby("week_start")[
            [
                "spend",
                "attributed_revenue",
                "estimated_incremental_revenue",
                "true_incremental_revenue",
            ]
        ]
        .sum()
        .reset_index()
        .sort_values("week_start")
        .tail(frames)
        .reset_index(drop=True)
    )

    if weekly.empty:
        container.warning("No simulation data available.")
        return

    with container:
        st.markdown("#### Auto-running synthetic simulation")
        chart_ph = st.empty()

        c1, c2, c3, c4 = st.columns(4)
        m1 = c1.empty()
        m2 = c2.empty()
        m3 = c3.empty()
        m4 = c4.empty()

        for i in range(1, len(weekly) + 1):
            sub = weekly.iloc[:i]

            fig, ax = plt.subplots(figsize=(9.0, 4.0), dpi=100)
            ax.plot(sub["week_start"], sub["spend"] / 1000.0, label="Spend ($k)")
            ax.plot(
                sub["week_start"],
                sub["attributed_revenue"] / 1000.0,
                label="Attributed revenue ($k)",
            )
            ax.plot(
                sub["week_start"],
                sub["estimated_incremental_revenue"] / 1000.0,
                label="Estimated incremental ($k)",
            )
            ax.plot(
                sub["week_start"],
                sub["true_incremental_revenue"] / 1000.0,
                label="True synthetic incremental ($k)",
                linestyle="--",
                alpha=0.65,
            )

            ax.set_title("Live synthetic incrementality simulation")
            ax.set_xlabel("Week")
            ax.set_ylabel("USD thousands")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(alpha=0.25)
            fig.autofmt_xdate()

            img = fig_to_rgb(fig)
            chart_ph.image(img, use_container_width=True)
            plt.close(fig)

            latest = sub.iloc[-1]

            m1.metric("Week", pd.Timestamp(latest["week_start"]).strftime("%Y-%m-%d"))
            m2.metric("Spend ($k)", f"{latest['spend'] / 1000.0:,.1f}")
            m3.metric(
                "Attributed rev ($k)",
                f"{latest['attributed_revenue'] / 1000.0:,.1f}",
                help="Platform-reported revenue. Not necessarily incremental.",
            )
            m4.metric(
                "Est. incremental ($k)",
                f"{latest['estimated_incremental_revenue'] / 1000.0:,.1f}",
                help="Estimated causal incremental revenue from DiD/model engine.",
            )

            time.sleep(delay)

        st.success("Simulation complete. Use filters, tabs, and controls to interact.")


# -----------------------------------------------------------------------------
# Budget optimizer
# -----------------------------------------------------------------------------

def optimize_allocation(
    budget: float,
    cr: pd.DataFrame,
    risk_aversion: float,
    min_share: float,
    max_share: float,
    target_iroas: float,
) -> Tuple[pd.DataFrame, float, float]:
    rows = cr.copy()

    rows["monthly_historical"] = rows["historical_spend"] / 12.0
    rows["saturation"] = np.maximum(rows["monthly_historical"] * 2.0, budget * 0.20)

    rows["expected_iroas"] = np.where(
        rows["confidence"].eq("Low"),
        np.maximum(0.0, rows["ci_low"]),
        risk_aversion * np.maximum(0.0, rows["ci_low"])
        + (1.0 - risk_aversion) * rows["iroas_est"],
    )

    rows["max_allowed"] = np.where(
        rows["confidence"].eq("Low"),
        budget * min(max_share, 0.20),
        budget * max_share,
    )

    rows["min_allowed"] = budget * min_share

    if float(rows["min_allowed"].sum()) > budget:
        rows["min_allowed"] = (budget / max(1, len(rows))) * 0.80

    alloc = dict(zip(rows["channel"], rows["min_allowed"]))
    caps = dict(zip(rows["channel"], rows["max_allowed"]))
    sat = dict(zip(rows["channel"], rows["saturation"]))
    exp = dict(zip(rows["channel"], rows["expected_iroas"]))

    remaining = float(budget - sum(alloc.values()))
    step = max(500.0, budget / 250.0)

    def response(ch: str, x: float) -> float:
        s = max(float(sat.get(ch, 1.0)), 1.0)
        e = float(exp.get(ch, 0.0))
        x = max(0.0, float(x))
        return float(e * x * math.exp(-x / s))

    while remaining >= step:
        best_ch = None
        best_gain = -1e18
        best_add = 0.0

        for ch in alloc:
            cap = float(caps[ch])
            cur = float(alloc[ch])

            if cur >= cap - 1e-6:
                continue

            add = min(step, cap - cur)
            if add <= 0:
                continue

            gain = (response(ch, cur + add) - response(ch, cur)) / add

            if gain > best_gain:
                best_gain = gain
                best_ch = ch
                best_add = add

        if best_ch is None or best_gain < target_iroas:
            break

        alloc[best_ch] += best_add
        remaining -= best_add

    if remaining > 1.0 and target_iroas <= 0.0:
        for ch in sorted(alloc, key=lambda c: caps[c] - alloc[c], reverse=True):
            add = min(remaining, caps[ch] - alloc[ch])
            if add <= 0:
                continue
            alloc[ch] += add
            remaining -= add
            if remaining <= 1.0:
                break

    out = pd.DataFrame(
        {
            "channel": list(alloc.keys()),
            "allocated": list(alloc.values()),
        }
    )

    out["expected_iroas"] = out["channel"].map(exp).fillna(0.0)
    out["saturation"] = out["channel"].map(sat).fillna(1.0)
    out["confidence"] = out["channel"].map(rows.set_index("channel")["confidence"]).fillna("Low")
    out["projected_incremental"] = [
        response(ch, amt) for ch, amt in zip(out["channel"], out["allocated"])
    ]

    out["marginal_iroas"] = [
        max(
            0.0,
            float(exp[ch])
            * math.exp(-float(amt) / max(float(sat[ch]), 1.0))
            * (1.0 - float(amt) / max(float(sat[ch]), 1.0)),
        )
        for ch, amt in zip(out["channel"], out["allocated"])
    ]

    projected_total = float(out["projected_incremental"].sum())
    return out, projected_total, float(remaining)


# -----------------------------------------------------------------------------
# Self-tests
# -----------------------------------------------------------------------------

def _hand_calculated_did_test() -> Tuple[bool, str]:
    mini = pd.DataFrame(
        {
            "arm": ["treatment", "treatment", "control", "control"],
            "period": ["pre", "exp", "pre", "exp"],
            "market_id": ["T1", "T1", "C1", "C1"],
            "total_revenue": [100.0, 130.0, 100.0, 105.0],
            "spend": [10.0, 20.0, 10.0, 10.0],
            "week_start": pd.to_datetime(
                ["2025-01-06", "2025-01-13", "2025-01-06", "2025-01-13"]
            ),
        }
    )

    meta = {
        "id": "TEST",
        "treat": ["T1"],
        "control": ["C1"],
        "exp_start": pd.Timestamp("2025-01-13"),
        "exp_end": pd.Timestamp("2025-01-13"),
    }

    res = _did_estimate(mini, meta, n_boot=10, seed=1)

    expected_inc_rev = 25.0
    expected_inc_spend = 10.0
    expected_iroas = 2.5

    ok = (
        res["valid"]
        and abs(res["incremental_revenue_est"] - expected_inc_rev) < 1e-6
        and abs(res["incremental_spend_est"] - expected_inc_spend) < 1e-6
        and abs(res["iroas"] - expected_iroas) < 1e-6
    )

    return ok, f"expected iROAS={expected_iroas}, got={res.get('iroas')}"


def _optimizer_smoke_test(cr: pd.DataFrame) -> Tuple[bool, str]:
    try:
        alloc, projected, remaining = optimize_allocation(
            budget=100000.0,
            cr=cr,
            risk_aversion=0.20,
            min_share=0.05,
            max_share=0.60,
            target_iroas=0.0,
        )
        total = float(alloc["allocated"].sum())
        ok = abs(total - 100000.0) < 2000.0 and projected >= 0.0
        return ok, f"allocated={total:,.0f}, projected={projected:,.0f}, remaining={remaining:,.0f}"
    except Exception as exc:
        return False, str(exc)


def run_self_tests(payload: Dict[str, Any]) -> pd.DataFrame:
    df = payload["df"]
    cr = payload["channel_results"]
    errors = payload["validation_errors"]
    ev = payload["evaluation"]

    tests: List[Dict[str, str]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        tests.append(
            {
                "test": name,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )

    add("Synthetic data generated", len(df) > 0, f"{len(df)} rows")
    add("All spend non-negative", bool((df["spend"] >= 0).all()), "")
    add("All revenue non-negative", bool((df["total_revenue"] >= 0).all()), "")
    add("Schema validator rejects malformed records", len(errors) > 0, f"{len(errors)} validation errors")
    add("At least one positive incrementality estimate", bool((cr["iroas_est"] > 0).any()), "")
    add("Confidence labels assigned", bool(cr["confidence"].isin(["Low", "Medium", "High"]).all()), "")
    add("Hand-calculated DiD fixture passes", *_hand_calculated_did_test())
    add("Optimizer smoke test passes", *_optimizer_smoke_test(cr))
    add("Evaluation MAE is finite", bool(np.isfinite(ev.get("mae", np.nan))), str(ev.get("mae")))
    add("Safe divide handles zero denominator", safe_div(10, 0) == 0.0, "")

    return pd.DataFrame(tests)


# -----------------------------------------------------------------------------
# Documentation content
# -----------------------------------------------------------------------------

EXEC_SUMMARY = """
### Product mission

**Incrementality Measurement Platform for AI Advertising Intelligence and Revenue Optimization**

This software solves a concrete revenue problem: advertising platforms report attributed revenue that
often includes organic demand, cross-channel credit, and view-through inflation. Marketing science teams
need a reproducible way to estimate **causal incremental revenue**, quantify uncertainty, and reallocate
budget toward channels and markets that truly drive revenue.

### Why a large technology company would pay for this

A FAANG-scale advertising business loses money when budget decisions rely on over-attributed platform
metrics. This platform helps:

- Reduce wasted ad spend by separating incremental revenue from organic/claimed revenue.
- Improve channel budget allocation with uncertainty-aware recommendations.
- Standardize geo-lift and experiment analysis across growth teams.
- Provide auditability for finance, legal, and marketing leadership.
- Shorten decision cycles from weeks of spreadsheet analysis to minutes of reproducible evidence.

### Primary users

- Growth marketers and channel managers
- Media mix modeling and marketing science teams
- Data scientists building causal advertising models
- Finance/business partners validating ROI claims
- Product managers optimizing monetization experiments

### Decisions improved

- Which channels deserve incremental budget?
- Which channels should be capped due to weak or uncertain incrementality?
- Whether a campaign lift is real or attribution inflation.
- How to set bid multipliers, regional budgets, and creative allocation.
- Whether to approve a spend shift above a governance threshold.
"""

USER_JOURNEYS = """
### Realistic user journeys

#### Journey 1 — Quarterly budget planning

**User:** Regional growth marketer  
**Current legacy workflow:** Downloads platform exports, pastes spend and attributed revenue into
spreadsheets, calculates ROAS by channel, asks finance for approval, allocates budget based on reported
ROAS.  
**Problem:** Reported ROAS overstates Search and Retail Media because organic purchases are credited.  
**Improved workflow:** Analyst opens the platform, filters region and date range, reviews estimated iROAS
with 90% confidence intervals, runs optimizer, exports allocation recommendation with audit trail.  
**Project-specific data fields used:** `market_id`, `week_start`, `channel`, `spend`,
`attributed_revenue`, `total_revenue`, `estimated_incremental_revenue`, `confidence`.  
**Failure case handled:** If a channel has low data quality or incremental spend near zero, the engine
falls back to conservative zero or lower-bound estimate and flags human review.  
**Acceptance check:** The optimizer output shows current spend, recommended spend, delta, confidence,
and required human approval when delta exceeds 10%.

#### Journey 2 — Geo-lift experiment review

**User:** Marketing scientist  
**Current legacy workflow:** Writes ad hoc SQL, builds notebooks, manually computes pre/post differences,
shares charts in slides, no standard uncertainty method.  
**Problem:** Results are not reproducible; confidence intervals and covariate balance are inconsistent.  
**Improved workflow:** Scientist selects an experiment, platform shows treatment/control markets,
pre/exp windows, DiD estimate, synthetic control cross-check, bootstrap confidence interval, and
market-level evidence.  
**Project-specific data fields used:** `experiment_id`, `arm`, `period`, `total_revenue`, `spend`,
`data_quality`, `ci_low_raw`, `ci_high_raw`.  
**Failure case handled:** Missing control markets or insufficient pre-period rows cause invalid DiD and
fallback to model or conservative benchmark.  
**Acceptance check:** Evidence drill-down displays the exact market-week rows contributing to the
estimate.

#### Journey 3 — Finance approval for budget shift

**User:** Finance business partner  
**Current legacy workflow:** Receives slide deck with point estimates, no uncertainty, no audit history.  
**Problem:** Cannot distinguish observed attribution from estimated causal impact.  
**Improved workflow:** Finance reviews labeled metrics: observed spend, attributed revenue, estimated
incremental revenue, confidence, guardrails, and approval status. Export is disabled until human approval
is checked for consequential budget shifts.  
**Project-specific data fields used:** `attributed_revenue`, `estimated_incremental_revenue`,
`attribution_overstatement`, `confidence`, `audit_log`.  
**Failure case handled:** If more than 30% of selected spend has Low confidence, dashboard displays a
guardrail warning and optimizer caps Low-confidence channels.  
**Acceptance check:** Audit log records who generated the recommendation and which data version was used.
"""

PRD = """
### Functional requirements

- FR-01 Ingest synthetic or appropriately licensed ad performance records.
- FR-02 Validate inbound records against a Pydantic schema.
- FR-03 Reject or quarantine negative spend, invalid dates, unknown channels, null revenue, and duplicate keys.
- FR-04 Build a canonical market-week-channel dataset.
- FR-05 Support geo experiment assignments with treatment/control arms and pre/exp periods.
- FR-06 Compute naive attributed ROAS as a baseline.
- FR-07 Compute difference-in-differences incrementality estimates.
- FR-08 Compute synthetic control cross-check estimates.
- FR-09 Compute linear interaction bootstrap estimates with uncertainty.
- FR-10 Select a primary estimate using transparent rules.
- FR-11 Apply deterministic fallback when input quality or statistical support is inadequate.
- FR-12 Label results as observed, estimated, simulated, or predicted.
- FR-13 Provide budget optimization with risk aversion, min/max shares, and target marginal iROAS.
- FR-14 Require human approval for consequential budget shifts.
- FR-15 Provide dashboard filters, drill-downs, exports, and audit history.
- FR-16 Run local self-tests and display pass/fail status.

### Non-functional requirements

- Runs on Windows 10/11 with CPU only.
- No paid API keys required.
- No physical hardware required.
- Core demo uses synthetic data only.
- Reproducible with fixed seed.
- Modular code boundaries inside a single-file prototype.
- Structured audit events for user-visible actions.
- Local file outputs for demo video and exports.
- Tests runnable without external services.

### Assumptions

- Initial demo uses synthetic geo experiments and synthetic channel performance.
- Revenue outcome is total market-channel revenue, not user-level PII.
- Treatment effect is approximately stable within experiment window.
- Control markets are not contaminated by treatment campaign.
- Platform-reported attributed revenue is biased upward relative to causal impact.

### Dependencies

- Python runtime
- Streamlit
- pandas / numpy
- scikit-learn
- Pydantic
- DuckDB
- Matplotlib
- OpenCV headless for optional MP4 generation
- Pillow for animated GIF fallback

### Out of scope for initial prototype

- Real-time bidding execution
- Direct ad platform API connectors
- User-level identity resolution
- PII processing
- Auto-spending without human approval
- Production multi-tenant RBAC
- Cloud deployment
- Paid model providers

### Release milestones

1. Environment and configuration
2. Synthetic/open data preparation
3. Schema validation
4. Canonical storage
5. Baseline analytics
6. Core incrementality algorithm
7. API/service boundary inside app
8. Dashboard
9. Audit trail
10. Tests
11. Packaging and documentation

### Prototype vs production enterprise service

| Area | Useful prototype | Production enterprise service |
|---|---|---|
| Data | Synthetic/local files | Governed warehouse/lakehouse connectors |
| Storage | DuckDB/local files | PostgreSQL/OLAP with backups and lineage |
| Auth | Local single user | SSO, RBAC, row-level security |
| API | Streamlit internal functions | FastAPI/gRPC with versioned contracts |
| Scaling | Single-process CPU | Queues, workers, partitioned jobs |
| Monitoring | Local logs and audit tab | OpenTelemetry, alerting, SLOs |
| Governance | Human approval checkbox | Workflow approvals, policy engine, legal review |
"""

ARCHITECTURE = """
### Modular architecture

The prototype is intentionally a modular monolith. All modules are separated inside `app.py`, but the
design maps cleanly to services later.

#### Components

- **Source adapters**: synthetic generator; future adapters for ad server exports, geo logs, warehouse tables.
- **Ingestion**: collects raw records and deliberately malformed records for validation demos.
- **Schema validation**: Pydantic model enforces field names, types, units, non-negative constraints.
- **Canonical data model**: market-week-channel fact table plus market dimension and experiment metadata.
- **Feature/signal processing**: period/arm labeling, spend scaling, treatment interaction features.
- **Baseline analytics**: naive attributed ROAS, difference-in-differences.
- **Model layer**: synthetic control cross-check, linear interaction bootstrap.
- **Decision/scoring layer**: estimate selection, confidence labeling, fallback logic, optimizer.
- **API layer**: in prototype, Python functions; production equivalent would be FastAPI endpoints.
- **Dashboard**: Streamlit tabs for demo, overview, experiments, model, optimizer, data contracts, docs, tests, audit.
- **Audit history**: local session audit log for generated media, simulations, exports, approvals.
- **Observability**: health checks, data-quality scores, validation error counts, evaluation metrics.

#### Ownership and failure boundaries

- Ingestion owns raw record capture and provenance.
- Validation owns rejection/quarantine; it must not silently repair business-critical fields.
- Canonical store owns schema versioning and retention.
- Analytics engines own statistical estimates and uncertainty.
- Decision layer owns fallback policy and human-approval gates.
- Dashboard owns labeling of observed vs estimated vs simulated values.
- Optimizer owns constrained allocation but cannot execute spend changes.
"""

MERMAID_ARCHITECTURE = """
```mermaid
flowchart LR
  A[Synthetic Ad Performance Generator] --> B[Source Adapter]
  B --> C[Pydantic Schema Validation]
  C --> D[Quarantine / Error Log]
  C --> E[DuckDB Canonical Store]
  E --> F[Feature and Signal Processing]
  F --> G[Baseline Analytics: Naive ROAS]
  F --> H[Difference-in-Differences Engine]
  F --> I[Synthetic Control Engine]
  F --> J[Linear Interaction Bootstrap Engine]
  G --> K[Decision and Scoring Layer]
  H --> K
  I --> K
  J --> K
  K --> L[Fallback and Uncertainty Policy]
  L --> M[Budget Optimizer]
  L --> N[Streamlit Dashboard]
  M --> N
  N --> O[Exports and Audit History]
  E --> O
  K --> P[Observability and Data Quality]
  P --> N
```
"""

MERMAID_SEQUENCE = """
```mermaid
sequenceDiagram
  participant U as Growth Analyst
  participant D as Streamlit Dashboard
  participant S as Service Layer
  participant V as Validator
  participant DB as DuckDB Canonical Store
  participant M as Incrementality Engine
  participant A as Audit History

  U->>D: Select channel, date range, experiment
  D->>S: Request analysis for filters
  S->>V: Validate schema version and inputs
  V-->>S: Validated rows or quarantine errors
  S->>DB: Query market-week-channel panel
  DB-->>S: Canonical records
  S->>M: Run DiD, synthetic control, bootstrap model
  M-->>S: Estimate, CI, quality flags, fallback status
  S->>A: Append audit event
  S-->>D: Result payload
  D-->>U: Charts, evidence drill-down, export buttons
```
"""

DATA_CONTRACTS = """
### Canonical data contracts

#### Market dimension

| Field | Type | Unit | Description |
|---|---|---|---|
| market_id | string | identifier | Synthetic designated market area ID |
| market_name | string | label | Human-readable market name |
| region | string | category | Northeast, Midwest, South, West |
| population_millions | float | millions | Market population |
| income_index | float | index | 100 = national average |
| digital_penetration | float | ratio | 0 to 1 |
| retail_index | float | index | Local retail demand proxy |

#### Weekly ad performance fact

| Field | Type | Unit | Description |
|---|---|---|---|
| market_id | string | identifier | Foreign key to market |
| week_start | date | ISO date | Monday week start |
| channel | string | category | Search, Social, Video, Retail Media |
| spend | float | USD | Advertising spend |
| impressions | integer | count | Ad impressions |
| clicks | integer | count | Ad clicks |
| conversions | float | count | Conversions |
| organic_revenue | float | USD | Non-advertising baseline revenue |
| true_incremental_revenue | float | USD | Synthetic ground-truth causal revenue |
| total_revenue | float | USD | Observed outcome revenue |
| attributed_revenue | float | USD | Platform-reported attributed revenue |
| treatment_flag | boolean | flag | True for treated market during experiment window |
| experiment_id | string/null | identifier | Experiment assignment |
| quality_flag | string | enum | empty or spend_outlier |

#### Example record

```json
{
  "market_id": "DMK-007",
  "week_start": "2025-05-05",
  "channel": "Search",
  "spend": 18250.40,
  "impressions": 2100000,
  "clicks": 31500,
  "conversions": 210.5,
  "organic_revenue": 9200.00,
  "true_incremental_revenue": 36500.00,
  "total_revenue": 46100.00,
  "attributed_revenue": 40100.00,
  "treatment_flag": true,
  "experiment_id": "EXP-Seac-42",
  "quality_flag": ""
}
```

### Data handling policies

- **Missing values:** required numeric fields missing => validation error; optional fields imputed only in
  model features with explicit flag.
- **Duplicates:** duplicate `market_id + week_start + channel` => quarantine.
- **Late-arriving records:** accepted if within retention window; recompute affected experiment aggregates.
- **Malformed input:** invalid date, unknown channel, negative spend => rejected with row index and message.
- **Outliers:** spend above 99.5th percentile by channel => `spend_outlier`, data-quality penalty.
- **Schema changes:** versioned Pydantic contract; backward-compatible additions allowed, breaking changes
  require migration and audit entry.
- **Retention:** synthetic demo data retained locally; production would enforce retention and privacy controls.

### Failure cases and acceptance checks

| Failure case | Expected behavior | Acceptance check |
|---|---|---|
| Negative spend | Reject record | Validation error contains schema_validation |
| Invalid date | Reject record | Validation error contains week_start |
| Duplicate key | Quarantine duplicate | Validation error contains duplicate_key |
| No control markets | DiD invalid | Fallback method selected, confidence Low |
| Incremental spend near zero | Do not divide by zero | iROAS fallback = 0 or lower bound |
| Low data quality | Cap budget and warn | Optimizer restricts Low-confidence channels |
"""

ALGORITHMS = """
### Transparent baseline before advanced ML

#### Baseline 1: naive attributed ROAS

Formula:

```text
naive_roas = sum(attributed_revenue) / sum(spend)
```

Use: shows what platform reporting claims.  
Limitation: overstates incrementality because it includes organic and cross-channel credit.

#### Baseline 2: difference-in-differences

Formula:

```text
weekly_did = (treatment_exp_avg - treatment_pre_avg) - (control_exp_avg - control_pre_avg)
incremental_revenue = weekly_did * n_treatment_markets * n_experiment_weeks
incremental_spend = spend_did * n_treatment_markets * n_experiment_weeks
iroas_did = incremental_revenue / incremental_spend
```

Use: primary transparent causal baseline when randomized geo assignment is valid.  
Trade-off: sensitive to parallel trends and control contamination.

#### Advanced approach 1: synthetic control

Constructs a weighted counterfactual from control markets matching pre-period revenue trajectory.

Use: cross-check when treated market pattern is unusual.  
Trade-off: requires sufficient donor markets and good pre-period fit; uncertainty is harder than DiD.

#### Advanced approach 2: linear interaction bootstrap

Model:

```text
total_revenue ~ spend + treatment*spend + market_covariates + time_effects
```

Coefficient on `treatment*spend` estimates incremental ROAS.

Use: pooled estimate and heterogeneous signal.  
Trade-off: relies on additive/linear assumptions; bootstrap uncertainty over market clusters.

### Training vs inference

- Training occurs on experiment panel rows with known treatment assignment.
- Inference applies selected channel iROAS estimates to filtered spend records.
- No user-level personalization is trained in the prototype.

### Uncertainty reporting

- Bootstrap 90% confidence intervals for DiD and linear interaction.
- Confidence label: High, Medium, Low.
- Relative CI width and data-quality score drive label.
- Low-confidence estimates are capped in optimizer and require human approval.

### Deterministic fallback

If model quality or input quality is inadequate:

1. If DiD valid, use DiD.
2. Else if linear interaction valid and R² above threshold, use model.
3. Else use conservative fallback:
   - `iroas_est = max(0, lower_confidence_bound)` if available
   - otherwise `0`
   - confidence = Low
   - method = conservative_fallback

### Project-specific failure cases

| Case | Example | System response |
|---|---|---|
| Zero incremental spend | Treatment spend equals control trend | Do not compute ratio; fallback |
| Negative estimated lift | Social campaign hurts revenue | Report raw negative, optimizer uses zero |
| Wide CI | Small number of markets | Confidence Low, cap budget |
| Poor synthetic control fit | Pre-period R² < 0.10 | Ignore synthetic control estimate |
| Outlier spend spike | One market has 100x spend | Data-quality penalty, flag outlier |
"""

SCALE_PLAN = """
### Scale without premature complexity

Initial implementation is a modular monolith.

#### Built-in scalability practices

- **Batching:** weekly panel aggregation and bootstrap batches by market.
- **Caching:** Streamlit cache for synthetic world construction.
- **Idempotency:** exports and audit events include timestamps and stable channel keys.
- **Pagination:** tables are filtered and summarized before display.
- **Backpressure:** bootstrap iterations are bounded; demo uses modest iteration count.
- **Partitioning:** data is logically partitioned by channel, region, experiment, and week.
- **Graceful degradation:** if advanced model fails, fallback to DiD or conservative estimate.

#### What would justify splitting services

Split only when measured evidence exists:

- Dashboard p95 latency exceeds target due to analysis, not rendering.
- Multiple teams require concurrent experiment runs.
- Data volume exceeds local DuckDB/memory comfort threshold.
- Need for scheduled ingestion and SLA-backed APIs.
- Need for role-based access and audit isolation.

Potential split:

- Ingestion/validation service
- Experiment analysis worker
- Optimization service
- API gateway
- Dashboard frontend
- Audit/lineage store
"""

TECH_STACK = """
### Technology stack

- Python-first implementation
- Streamlit for dashboard
- pandas/numpy for tabular processing
- scikit-learn for linear models
- Pydantic for schema validation
- DuckDB for local analytical SQL
- Matplotlib for charts and demo media
- OpenCV headless for optional MP4 generation
- Pillow for animated GIF fallback
- pytest-style local self-tests inside app for demo portability

### Why these tools

- CPU-only friendly
- No paid API keys
- Open-source
- Works on Windows 10/11
- Suitable for prototype and early enterprise pilot

### Repository layout for productionization

```text
/
├── app.py
├── requirements.txt
├── /src
│   ├── ingestion/
│   ├── validation/
│   ├── canonical/
│   ├── features/
│   ├── analytics/
│   ├── models/
│   ├── decisions/
│   ├── optimizer/
│   ├── audit/
│   └── observability/
├── /api
├── /domain
├── /data
├── /schemas
├── /models
├── /services
├── /dashboard
├── /configs
├── /scripts
├── /migrations
├── /docs
├── /tests
│   ├── /unit
│   ├── /integration
│   └── /e2e
└── /outputs
```

In this single-file deliverable, the module boundaries are represented by function groups and sections.
"""

POWERSHELL = """
### Exact PowerShell setup and run commands

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Optional cleanup:

```powershell
Remove-Item -Recurse -Force .\\outputs
Remove-Item -Recurse -Force .\\data
```

Expected result:

- Browser opens Streamlit app.
- `outputs/incrementality_demo.mp4` and/or `outputs/incrementality_demo.gif` are generated.
- Live simulation starts automatically.
- Dashboard is populated with synthetic incrementality results.
"""

LICENSE_INVENTORY = """
### License inventory

Direct dependencies are selected for permissive or commercially friendly licenses. Before redistribution,
verify current license terms and attribution obligations.

| Package | Typical license | Commercial-use note |
|---|---|---|
| streamlit | Apache-2.0 | Generally permissive |
| pandas | BSD-3-Clause | Generally permissive |
| numpy | BSD-3-Clause | Generally permissive |
| scikit-learn | BSD-3-Clause | Generally permissive |
| pydantic | MIT | Generally permissive |
| duckdb | MIT | Generally permissive |
| matplotlib | PSF-based / permissive components | Check bundled fonts/coefficients |
| opencv-python-headless | Apache-2.0 | Check third-party codecs if distributing binaries |
| pillow | MIT-CMU / historical permissive | Check current terms |

Data provenance:

- Initial demo data is synthetically generated inside the app.
- No customer data, no PII, no paid API data.
- If replacing synthetic data with licensed datasets, record dataset license, attribution, and permitted use.
"""

ENV_EXAMPLE = """
### .env.example concept

The core demo does not require secrets. For future productionization:

```env
APP_ENV=local
DATA_SOURCE=synthetic
DB_PATH=data/incrementality.duckdb
AUDIT_RETENTION_DAYS=90
ENABLE_HUMAN_APPROVAL=true
```

Never commit real secrets. The prototype intentionally avoids API keys.
"""

TESTING_EVAL = """
### Evaluation protocol

#### Train/validation/test separation

- Experiment panel is split chronologically where possible.
- Model coefficients are estimated on pre/exp experiment rows.
- Validation uses held-out synthetic channels/markets when available.
- No future revenue is used as a feature.

#### Leakage checks

- `true_incremental_revenue` is never used for estimation.
- It is only used for synthetic validation metrics.
- Treatment assignment is not used to directly reveal ground truth in dashboard estimates.

#### Metrics and formulas

- MAE:

```text
MAE = mean(|estimated_iROAS - true_iROAS|)
```

- RMSE:

```text
RMSE = sqrt(mean((estimated_iROAS - true_iROAS)^2))
```

- Bias:

```text
Bias = mean(estimated_iROAS - true_iROAS)
```

- 90% CI coverage:

```text
Coverage = mean(ci_low <= true_iROAS <= ci_high)
```

- Sign accuracy:

```text
SignAccuracy = mean(sign(estimated_iROAS) == sign(true_iROAS))
```

- Ranking quality:

```text
Spearman correlation between estimated_iROAS and true_iROAS across channels
```

#### What metrics cannot prove

- Synthetic MAE does not prove real-world causal validity.
- High coverage does not prove unbiasedness if model assumptions fail.
- Correlation does not prove that budget optimization will increase revenue without a controlled pilot.

### Domain quality and operational metrics

- Data completeness
- Duplicate rate
- Outlier rate
- Treatment/control balance
- Bootstrap sample validity
- Dashboard render latency
- Memory/CPU usage
- Error rate in validation
- Audit event completeness

### Security and reliability

- Least privilege: local demo has no external credentials.
- Input validation: Pydantic schema.
- Secrets management: no secrets required.
- Encryption: local files only; production would encrypt at rest and in transit.
- Retention: synthetic demo data can be deleted locally.
- Audit logging: session audit tab.
- Threat modeling: misuse includes over-trusting estimates; mitigated by confidence labels and human approval.
- Rollback: regenerate synthetic world or delete outputs.
- Kill switch: stop app; no external spend execution.
- Safe failure mode: fallback to conservative zero/lower-bound estimate.
"""

RISK_REGISTER = """
### Risk register

| Risk | Example | Mitigation | Human approval required? |
|---|---|---|---|
| Data availability | Missing control markets | Fallback, low confidence, warning | Yes for consequential budget shift |
| Model error | Linear interaction overfits noisy geo data | Bootstrap CI, synthetic control cross-check, fallback | Yes |
| Privacy | Future real user-level data | Prototype uses synthetic data only; production requires de-identification | Yes |
| Fairness | Certain regions systematically underestimated | Subgroup checks, minimum data-quality thresholds | Yes |
| Misuse | Team claims estimated lift as guaranteed revenue | Labels: estimated/synthetic; no external promises | Yes |
| Operational failure | Video generation or bootstrap crash | Graceful fallback, self-tests, safe mode | No for demo, yes for production incidents |
| Adoption | Analysts ignore uncertainty | Evidence drill-down, simple confidence labels | No |
| Cost | Bootstrap too slow on large data | Bounded iterations, caching, future worker queue | No |

Actions requiring human approval:

- Exporting a budget reallocation with >10% channel shift
- Publishing customer-facing incrementality claims
- Connecting production ad platform spend execution
- Using real patient/customer/PII data
- Deploying beyond pilot environment
"""

VALUE_HYPOTHESIS = """
### Measurable value hypothesis

If marketing teams allocate budget using uncertainty-aware incremental revenue estimates instead of
platform-reported attributed ROAS, the company can reduce wasted spend and improve incremental revenue
per advertising dollar.

#### Outcome metrics

- Incremental revenue per $ spent
- iROAS by channel and region
- Budget reallocation savings
- Forecast error vs holdout experiments
- Decision cycle time

#### Leading indicators

- Percentage of spend covered by validated experiments
- Data-quality score
- Confidence interval width
- Share of Low-confidence spend
- Analyst adoption rate
- Number of audit-approved recommendations

#### Quality metrics

- MAE / RMSE against synthetic ground truth
- 90% CI coverage
- Sign accuracy
- Spearman ranking correlation
- Fallback rate

#### Reliability metrics

- Job success rate
- Dashboard latency
- Validation error rate
- Recovery after malformed input
- Reproducibility with same seed

#### Guardrails

- Do not auto-execute spend changes.
- Cap Low-confidence channels.
- Require human approval for large shifts.
- Label observed, estimated, simulated, and predicted values distinctly.
- Do not use synthetic validation metrics as real-world performance claims.

#### Baseline

Baseline is current decision process:

- Allocate by naive attributed ROAS.
- Manual spreadsheet cycle time.
- Historical holdout or geo experiment benchmark where available.

#### Controlled evaluation

To establish incremental value:

1. Run pre-registered geo experiments in matched markets.
2. Compare platform-recommended allocation vs legacy naive ROAS allocation.
3. Measure total incremental revenue, wasted spend, and decision time.
4. Use difference-in-differences or synthetic control on the pilot itself.
5. Report confidence intervals and subgroup effects.
"""

BACKLOG = """
### Prioritized backlog

| Priority | Item | Why |
|---|---|---|
| P0 | Synthetic data generator | Enables immediate demo without external data |
| P0 | Schema validation | Prevents garbage-in results |
| P0 | DiD incrementality engine | Transparent causal baseline |
| P0 | Dashboard with labels | Makes observed vs estimated clear |
| P0 | Auto demo video/simulation | Satisfies working demo requirement |
| P1 | Bootstrap uncertainty | Supports governance decisions |
| P1 | Synthetic control cross-check | Improves robustness |
| P1 | Budget optimizer | Translates estimates into decisions |
| P1 | Audit history | Required for enterprise trust |
| P2 | FastAPI service wrapper | Production integration |
| P2 | PostgreSQL/DuckDB migrations | Persistent governed storage |
| P2 | Scheduled experiment ingestion | Operational scale |
| P2 | RBAC and approval workflow | Enterprise governance |
"""

USER_STORIES = """
### User stories with acceptance criteria

#### Story 1

As a growth marketer, I want to see estimated incremental revenue by channel so that I can avoid
over-investing in channels with inflated attribution.

Acceptance criteria:

- Dashboard shows spend, attributed revenue, estimated incremental revenue, naive ROAS, estimated iROAS.
- Estimated values are labeled as estimated.
- Filters by channel, region, and date range work.

#### Story 2

As a marketing scientist, I want confidence intervals for geo-lift estimates so that I can assess
statistical reliability.

Acceptance criteria:

- DiD estimate includes 90% bootstrap CI.
- Low-confidence results are flagged.
- Evidence drill-down shows contributing market-week aggregates.

#### Story 3

As a finance partner, I want budget recommendations to require approval when shifts are large.

Acceptance criteria:

- Optimizer computes delta vs current spend.
- If any channel delta > 10%, export is disabled until approval checkbox is selected.
- Audit log records approval state.

#### Story 4

As a developer, I want to reproduce the demo from a README-like flow.

Acceptance criteria:

- `pip install -r requirements.txt`
- `streamlit run app.py`
- App auto-generates demo media and starts simulation.
- Self-tests pass.
"""

GLOSSARY = """
### Glossary

- **Attributed revenue:** Platform-reported revenue assigned to ads; may include organic demand.
- **Incremental revenue:** Causal revenue that would not have occurred without advertising.
- **iROAS:** Incremental return on ad spend = incremental revenue / incremental spend.
- **Geo lift:** Experiment where markets are randomly assigned to treatment/control.
- **DiD:** Difference-in-differences causal estimator.
- **Synthetic control:** Weighted counterfactual built from untreated markets.
- **Bootstrap CI:** Confidence interval from resampling clusters such as markets.
- **Data quality score:** Composite measure of completeness, duplicates, outliers, and balance.
- **Fallback:** Deterministic conservative estimate when statistical evidence is inadequate.
- **Human approval:** Governance gate for consequential budget or public claims.
"""

ACCEPTANCE_CHECKS = """
### Concrete acceptance checks

| Requirement | Acceptance check |
|---|---|
| Working demo immediately | Fresh run shows non-blank dashboard, generated media, and auto simulation |
| No API key | App runs without credentials or network service |
| CPU-only | No GPU required; uses pandas/sklearn/matplotlib |
| Synthetic data only | Data generator creates all demo records locally |
| Schema validation | Malformed records appear in validation error table |
| Incrementality estimate | At least one channel has positive estimated iROAS |
| Uncertainty | CI columns are finite for selected methods |
| Fallback | Zero incremental spend produces conservative estimate, not crash |
| Optimizer | Allocation sums to budget within tolerance |
| Human approval | Large budget shift disables export until approval checkbox |
| Audit | Audit tab records media generation and simulation completion |
| Tests | Self-tests tab shows PASS for required checks |
"""

PAGE_ARTIFACTS = {
    "Page 1 — Product, Problem, PRD, Value, Risk": EXEC_SUMMARY
    + USER_JOURNEYS
    + PRD
    + VALUE_HYPOTHESIS
    + RISK_REGISTER
    + BACKLOG
    + USER_STORIES
    + GLOSSARY
    + ACCEPTANCE_CHECKS,
    "Page 2 — Architecture, Data, Algorithms": ARCHITECTURE
    + MERMAID_ARCHITECTURE
    + MERMAID_SEQUENCE
    + DATA_CONTRACTS
    + ALGORITHMS
    + SCALE_PLAN,
    "Page 3 — Stack, Repo, Build Steps": TECH_STACK
    + POWERSHELL
    + LICENSE_INVENTORY
    + ENV_EXAMPLE,
    "Page 4 — Testing, Evaluation, Security, Reliability": TESTING_EVAL,
    "Page 5 — Live Demo, Deployment, Acceptance": ACCEPTANCE_CHECKS
    + """
### Live demo script

1. Run:

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

2. Expected immediate behavior:
   - Synthetic world builds.
   - Demo MP4/GIF is generated in `outputs/`.
   - Animated live simulation runs.
   - Overview tab shows non-blank KPIs and charts.

3. Normal scenario:
   - Select Search experiment.
   - Observe positive DiD estimate and confidence interval.

4. Difficult edge case:
   - Increase synthetic noise.
   - Regenerate world.
   - Observe wider CIs and lower confidence labels.

5. Failure/recovery scenario:
   - View Data Contracts tab.
   - Validation errors show negative spend, invalid date, duplicate key.
   - Engine continues using clean canonical data.

6. Explanation:
   - Open Experiments tab.
   - Review market-level evidence and method comparison.
   - Open Optimizer tab to see how uncertainty affects allocation.

### Deployment notes

- Local Windows setup is the primary supported demo path.
- Optional Docker Compose can later wrap Streamlit + DuckDB volume.
- Health checks are shown in Audit tab.
- Backup/restore for prototype: copy `outputs/` and `data/`.
- Rollback: delete generated outputs and regenerate with prior seed.
"""
}


# -----------------------------------------------------------------------------
# Sidebar controls and app state
# -----------------------------------------------------------------------------

st.sidebar.title(APP_NAME)
st.sidebar.caption(APP_SUBTITLE)
st.sidebar.markdown("---")
st.sidebar.subheader("Synthetic world controls")

noise = st.sidebar.slider(
    "Synthetic noise",
    min_value=0.02,
    max_value=0.45,
    value=0.12,
    step=0.01,
    help="Higher noise widens confidence intervals and can reduce confidence labels.",
)

effect_strength = st.sidebar.slider(
    "True incremental strength",
    min_value=0.25,
    max_value=2.50,
    value=1.00,
    step=0.05,
    help="Scales the hidden synthetic causal effect used for validation.",
)

n_markets = st.sidebar.selectbox(
    "Number of synthetic markets",
    options=[24, 48, 96],
    index=1,
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=1,
    max_value=9999,
    value=42,
    step=1,
)

if st.sidebar.button("Regenerate synthetic world"):
    st.cache_data.clear()
    st.session_state.pop("demo_media", None)
    st.session_state.pop("demo_media_key", None)
    st.session_state.pop("sim_done", None)
    log_event("regenerate_world", f"seed={seed}, markets={n_markets}")
    st.rerun()

with st.spinner("Building synthetic incrementality world..."):
    payload = build_platform(
        noise=float(noise),
        effect_strength=float(effect_strength),
        n_markets=int(n_markets),
        seed=int(seed),
        bootstrap_iterations=60,
    )

df = payload["df"]
cr = payload["channel_results"]
markets = payload["markets"]
experiments = payload["experiments"]
evaluation = payload["evaluation"]
validation_valid = payload["validation_valid"]
validation_errors = payload["validation_errors"]

log_event("platform_loaded", f"rows={len(df)}, channels={len(cr)}")

# Filters

st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard filters")

selected_channels = st.sidebar.multiselect(
    "Channels",
    options=CHANNELS,
    default=CHANNELS,
)

selected_regions = st.sidebar.multiselect(
    "Regions",
    options=sorted(df["region"].unique().tolist()),
    default=sorted(df["region"].unique().tolist()),
)

confidence_choice = st.sidebar.selectbox(
    "Minimum confidence",
    options=["All", "Medium+", "High only"],
    index=0,
)

default_min = df["week_start"].min().date()
default_max = df["week_start"].max().date()

date_range = st.sidebar.date_input(
    "Week range",
    value=(default_min, default_max),
    min_value=default_min,
    max_value=default_max,
)

mask = df["channel"].isin(selected_channels) & df["region"].isin(selected_regions)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt = pd.Timestamp(date_range[0])
    end_dt = pd.Timestamp(date_range[1])
    mask &= (df["week_start"] >= start_dt) & (df["week_start"] <= end_dt)

if confidence_choice == "Medium+":
    mask &= df["confidence"].isin(["Medium", "High"])
elif confidence_choice == "High only":
    mask &= df["confidence"].eq("High")

fdf = df[mask].copy()

if fdf.empty:
    st.sidebar.warning("Filter produced no rows. Showing all rows instead.")
    fdf = df.copy()

# Demo media

media_key = f"{noise}|{effect_strength}|{n_markets}|{seed}"

if st.session_state.get("demo_media_key") != media_key:
    with st.spinner("Generating live demo media file..."):
        media = generate_demo_media(df)
    st.session_state.demo_media = media
    st.session_state.demo_media_key = media_key
    log_event("demo_media_generated", json.dumps(media))
else:
    media = st.session_state.demo_media

# Tabs

tabs = st.tabs(
    [
        "Live Demo",
        "Overview",
        "Experiments",
        "Uplift Model",
        "Budget Optimizer",
        "Data Contracts",
        "Product Docs",
        "Tests & Eval",
        "Audit & Ops",
    ]
)

# -----------------------------------------------------------------------------
# Tab 1: Live Demo
# -----------------------------------------------------------------------------

with tabs[0]:
    st.header("Working demo: synthetic incrementality measurement")
    st.caption(
        "This app generates a local demo video file and starts an animated simulation immediately. "
        "All values are synthetic. Observed, attributed, estimated, and true synthetic values are labeled."
    )

    if media.get("gif") and os.path.exists(media["gif"]):
        st.subheader("Animated demo preview")
        st.image(media["gif"], use_container_width=True)

    if media.get("mp4") and os.path.exists(media["mp4"]):
        st.subheader("Generated MP4 demo file")
        try:
            st.video(media["mp4"])
        except Exception:
            st.warning("MP4 file was generated but could not be embedded in this browser. Use download below.")

    dl1, dl2 = st.columns(2)

    if media.get("mp4") and os.path.exists(media["mp4"]):
        with dl1:
            with open(media["mp4"], "rb") as f:
                st.download_button(
                    "Download MP4 demo video",
                    data=f.read(),
                    file_name="incrementality_demo.mp4",
                    mime="video/mp4",
                )
    else:
        with dl1:
            st.info("MP4 generation unavailable; GIF fallback is used.")

    if media.get("gif") and os.path.exists(media["gif"]):
        with dl2:
            with open(media["gif"], "rb") as f:
                st.download_button(
                    "Download GIF demo animation",
                    data=f.read(),
                    file_name="incrementality_demo.gif",
                    mime="image/gif",
                )

    st.markdown("---")

    sim_key = media_key

    if st.session_state.get("sim_key") != sim_key:
        st.session_state.sim_key = sim_key
        st.session_state.sim_done = False

    if not st.session_state.get("sim_done", False):
        run_live_simulation(st.container(), fdf)
        st.session_state.sim_done = True
        log_event("live_simulation_completed", f"rows={len(fdf)}")
    else:
        st.info("Auto simulation complete. Replay or adjust synthetic controls in the sidebar.")

    if st.button("Replay live simulation"):
        st.session_state.sim_done = False
        st.rerun()

    st.markdown("### Demo interpretation")
    st.markdown(
        """
- **Spend** is observed synthetic ad spend.
- **Attributed revenue** is platform-reported and intentionally inflated by organic credit.
- **Estimated incremental revenue** is produced by the incrementality engine.
- **True synthetic incremental** is hidden ground truth used only for validation.
- Wider gaps between attributed and estimated revenue illustrate why naive ROAS can mislead budget decisions.
        """
    )

# -----------------------------------------------------------------------------
# Tab 2: Overview
# -----------------------------------------------------------------------------

with tabs[1]:
    st.header("Portfolio overview")

    total_spend = float(fdf["spend"].sum())
    total_attr = float(fdf["attributed_revenue"].sum())
    total_est_inc = float(fdf["estimated_incremental_revenue"].sum())
    total_true_inc = float(fdf["true_incremental_revenue"].sum())

    naive_roas = safe_div(total_attr, total_spend)
    est_iroas = safe_div(total_est_inc, total_spend)
    true_iroas_filtered = safe_div(total_true_inc, total_spend)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observed spend", f"${total_spend:,.0f}")
    c2.metric("Attributed revenue", f"${total_attr:,.0f}", help="Observed platform-reported revenue")
    c3.metric("Naive ROAS", f"{naive_roas:.2f}x", help="Attributed revenue / spend")
    c4.metric("Estimated incremental revenue", f"${total_est_inc:,.0f}", help="Engine estimate")
    c5.metric("Estimated iROAS", f"{est_iroas:.2f}x", help="Estimated incremental revenue / spend")

    low_spend_share = safe_div(
        float(fdf.loc[fdf["confidence"] == "Low", "spend"].sum()),
        total_spend,
    )

    if low_spend_share > 0.30:
        st.warning(
            f"Guardrail: {low_spend_share:.0%} of selected spend has Low confidence. "
            "Treat recommendations as directional and require human review."
        )
    else:
        st.success(f"Low-confidence spend share: {low_spend_share:.0%}")

    st.subheader("Weekly trends")

    weekly = (
        fdf.groupby("week_start")[
            [
                "spend",
                "attributed_revenue",
                "estimated_incremental_revenue",
                "true_incremental_revenue",
            ]
        ]
        .sum()
        .reset_index()
        .sort_values("week_start")
    )

    st.line_chart(
        weekly.set_index("week_start")[
            [
                "spend",
                "attributed_revenue",
                "estimated_incremental_revenue",
                "true_incremental_revenue",
            ]
        ]
    )

    st.caption("Labels: spend = observed; attributed revenue = observed platform-reported; estimated incremental = estimated; true synthetic incremental = simulated validation only.")

    st.subheader("Channel comparison")

    channel_agg = (
        fdf.groupby("channel")[
            [
                "spend",
                "attributed_revenue",
                "estimated_incremental_revenue",
                "true_incremental_revenue",
            ]
        ]
        .sum()
        .reset_index()
    )

    channel_agg["naive_roas"] = channel_agg.apply(
        lambda r: safe_div(r["attributed_revenue"], r["spend"]), axis=1
    )
    channel_agg["estimated_iroas"] = channel_agg.apply(
        lambda r: safe_div(r["estimated_incremental_revenue"], r["spend"]), axis=1
    )
    channel_agg["true_iroas"] = channel_agg["channel"].map(
        cr.set_index("channel")["true_iroas"]
    )

    st.bar_chart(
        channel_agg.set_index("channel")[
            ["naive_roas", "estimated_iroas", "true_iroas"]
        ]
    )

    st.subheader("Data quality and confidence")

    quality_view = cr[
        [
            "channel",
            "method",
            "iroas_est",
            "ci_low",
            "ci_high",
            "data_quality",
            "confidence",
            "true_iroas",
            "absolute_error",
        ]
    ].copy()

    st.dataframe(quality_view, use_container_width=True)

    st.markdown(
        """
### What this page proves and does not prove

- It proves the platform can compute and display estimates from synthetic data.
- It does not prove real-world revenue lift without a controlled pilot.
- Synthetic true values are for validation only and must not be presented as customer results.
        """
    )

# -----------------------------------------------------------------------------
# Tab 3: Experiments
# -----------------------------------------------------------------------------

with tabs[2]:
    st.header("Geo experiment incrementality")

    exp_channel = st.selectbox("Select channel experiment", CHANNELS, index=0)
    meta = experiments[exp_channel]
    row = cr[cr["channel"] == exp_channel].iloc[0]

    st.subheader(meta["id"])

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Treatment markets", int(row["n_treatment_markets"]))
    mc2.metric("Control markets", int(row["n_control_markets"]))
    mc3.metric("Experiment weeks", int(row["n_experiment_weeks"]))
    mc4.metric("Data quality", f"{row['data_quality']:.2f}")
    mc5.metric("Confidence", row["confidence"])

    methods_df = pd.DataFrame(
        [
            {
                "method": "Naive attributed ROAS",
                "estimate": safe_div(
                    df.loc[df["channel"] == exp_channel, "attributed_revenue"].sum(),
                    df.loc[df["channel"] == exp_channel, "spend"].sum(),
                ),
                "ci_low": np.nan,
                "ci_high": np.nan,
                "note": "Observed platform-reported baseline",
            },
            {
                "method": "Difference-in-differences",
                "estimate": row["did_iroas"],
                "ci_low": row["ci_low_raw"] if row["method"] == "difference_in_differences" else np.nan,
                "ci_high": row["ci_high_raw"] if row["method"] == "difference_in_differences" else np.nan,
                "note": "Primary transparent causal baseline",
            },
            {
                "method": "Linear interaction bootstrap",
                "estimate": row["model_iroas"],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "note": f"Model R2={row['model_r2']:.3f}" if np.isfinite(row["model_r2"]) else "Model unavailable",
            },
            {
                "method": "Synthetic control",
                "estimate": row["synthetic_control_iroas"],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "note": "Cross-check counterfactual",
            },
            {
                "method": "Selected estimate",
                "estimate": row["iroas_est"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "note": row["method"],
            },
        ]
    )

    st.dataframe(methods_df, use_container_width=True)

    valid_methods = methods_df.dropna(subset=["estimate"]).copy()

    if not valid_methods.empty:
        fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=100)

        x = np.arange(len(valid_methods))
        y = valid_methods["estimate"].to_numpy(dtype=float)

        lows = valid_methods["ci_low"].fillna(valid_methods["estimate"]).to_numpy(dtype=float)
        highs = valid_methods["ci_high"].fillna(valid_methods["estimate"]).to_numpy(dtype=float)

        err_low = np.maximum(0.0, y - lows)
        err_high = np.maximum(0.0, highs - y)

        ax.errorbar(
            x,
            y,
            yerr=[err_low, err_high],
            fmt="o",
            capsize=5,
            label="Estimate with CI where available",
        )
        ax.axhline(row["true_iroas"], color="green", linestyle="--", label="True synthetic iROAS")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_methods["method"], rotation=20, ha="right")
        ax.set_ylabel("iROAS")
        ax.set_title(f"{exp_channel}: method comparison")
        ax.grid(alpha=0.25)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Evidence drill-down")

    e = prepare_experiment_frame(df, meta)
    evidence = (
        e.groupby(["arm", "period", "market_id"])[
            ["spend", "total_revenue", "attributed_revenue"]
        ]
        .mean()
        .reset_index()
        .sort_values(["arm", "period", "market_id"])
    )

    st.dataframe(evidence, use_container_width=True)

    st.subheader("Failure cases and guardrails")

    if float(row["incremental_spend_est"]) <= 1000.0:
        st.error(
            "Failure case triggered: incremental spend is near zero. "
            "The engine avoids unstable division and uses fallback logic."
        )
    else:
        st.info("Incremental spend denominator is adequate for ratio estimation.")

    if float(row["data_quality"]) < 0.65:
        st.warning("Low data quality detected. Confidence is reduced and optimizer caps exposure.")
    else:
        st.success("Data quality is above the low-confidence threshold.")

    if row["confidence"] == "Low":
        st.warning("Low confidence: do not publish external claims or auto-execute budget changes.")

    csv = methods_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download experiment summary CSV",
        data=csv,
        file_name=f"{meta['id']}_summary.csv",
        mime="text/csv",
    )
    log_event("experiment_summary_available", meta["id"])

# -----------------------------------------------------------------------------
# Tab 4: Uplift Model
# -----------------------------------------------------------------------------

with tabs[3]:
    st.header("Uplift / incrementality model layer")

    st.markdown(
        """
The model layer estimates causal advertising impact using experiment panels. It is deliberately transparent:

1. Difference-in-differences is the primary baseline.
2. Linear interaction bootstrap provides a pooled alternative.
3. Synthetic control provides a counterfactual cross-check.
4. Bootstrap confidence intervals drive confidence labels.
5. Deterministic fallback prevents unstable ratios from driving budget decisions.
        """
    )

    model_view = cr[
        [
            "channel",
            "method",
            "iroas_raw",
            "iroas_est",
            "ci_low_raw",
            "ci_high_raw",
            "model_iroas",
            "model_r2",
            "synthetic_control_iroas",
            "data_quality",
            "confidence",
            "true_iroas",
            "absolute_error",
        ]
    ].copy()

    st.dataframe(model_view, use_container_width=True)

    st.subheader("Synthetic validation metrics")

    ev_rows = [
        {"metric": "MAE", "value": evaluation.get("mae", np.nan), "interpretation": "Lower is better"},
        {"metric": "RMSE", "value": evaluation.get("rmse", np.nan), "interpretation": "Penalizes large errors"},
        {"metric": "Bias", "value": evaluation.get("bias", np.nan), "interpretation": "Positive = overestimation"},
        {"metric": "90% CI coverage", "value": evaluation.get("coverage_90", np.nan), "interpretation": "Target near 0.90 in repeated synthetic trials"},
        {"metric": "Sign accuracy", "value": evaluation.get("sign_accuracy", np.nan), "interpretation": "Did estimate preserve positive/negative direction?"},
        {"metric": "Spearman ranking", "value": evaluation.get("spearman_ranking", np.nan), "interpretation": "Rank ordering quality across channels"},
    ]

    st.dataframe(pd.DataFrame(ev_rows), use_container_width=True)

    st.caption("These metrics are computed on synthetic ground truth. They do not prove real-world performance.")

    st.subheader("Segment checks")

    segment = (
        fdf.groupby(["region", "channel"])
        .agg(
            spend=("spend", "sum"),
            attributed_revenue=("attributed_revenue", "sum"),
            estimated_incremental_revenue=("estimated_incremental_revenue", "sum"),
        )
        .reset_index()
    )

    segment["naive_roas"] = segment.apply(lambda r: safe_div(r["attributed_revenue"], r["spend"]), axis=1)
    segment["estimated_iroas"] = segment.apply(
        lambda r: safe_div(r["estimated_incremental_revenue"], r["spend"]), axis=1
    )

    st.dataframe(segment, use_container_width=True)

    st.markdown(
        """
### Failure case example

If a region has too few markets or severe spend imbalance, segment estimates can become unstable.
The platform handles this by:

- reducing data-quality score;
- widening or invalidating confidence intervals;
- labeling confidence Low;
- capping optimizer exposure;
- requiring human approval for consequential changes.
        """
    )

# -----------------------------------------------------------------------------
# Tab 5: Budget Optimizer
# -----------------------------------------------------------------------------

with tabs[4]:
    st.header("Budget optimizer and business value calculator")

    oc1, oc2, oc3, oc4 = st.columns(4)

    monthly_budget = oc1.number_input(
        "Monthly budget",
        min_value=50000,
        max_value=10000000,
        value=500000,
        step=50000,
    )

    min_share = oc2.slider("Min share per channel", 0.0, 0.20, 0.05, 0.01)
    max_share = oc3.slider("Max share per channel", 0.20, 0.90, 0.55, 0.05)
    risk_aversion = oc4.slider("Risk aversion", 0.0, 1.0, 0.35, 0.05)

    target_iroas = st.number_input(
        "Target marginal iROAS",
        min_value=0.0,
        max_value=5.0,
        value=0.80,
        step=0.05,
        help="Allocator stops adding budget when expected marginal iROAS falls below this target.",
    )

    alloc_df, projected_incremental, remaining_budget = optimize_allocation(
        budget=float(monthly_budget),
        cr=cr,
        risk_aversion=float(risk_aversion),
        min_share=float(min_share),
        max_share=float(max_share),
        target_iroas=float(target_iroas),
    )

    current_monthly = (cr.set_index("channel")["historical_spend"] / 12.0).to_dict()

    alloc_df["current_monthly"] = alloc_df["channel"].map(current_monthly).fillna(0.0)
    alloc_df["delta"] = alloc_df["allocated"] - alloc_df["current_monthly"]
    alloc_df["delta_pct"] = alloc_df.apply(
        lambda r: safe_div(r["delta"], max(abs(r["current_monthly"]), 1.0)), axis=1
    )

    st.subheader("Recommended allocation")
    st.dataframe(alloc_df, use_container_width=True)

    optimized_iroas = safe_div(projected_incremental, monthly_budget)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Projected incremental revenue", f"${projected_incremental:,.0f}")
    k2.metric("Projected optimized iROAS", f"{optimized_iroas:.2f}x")
    k3.metric("Unallocated budget", f"${remaining_budget:,.0f}")
    k4.metric("Low-confidence capped channels", int((alloc_df["confidence"] == "Low").sum()))

    requires_approval = bool((alloc_df["delta_pct"].abs() > 0.10).any())

    if requires_approval:
        st.warning("Consequential budget shift detected. Human approval is required before export.")
    else:
        st.info("Projected shifts are below the 10% governance threshold.")

    approval = st.checkbox("I confirm a human growth/finance owner approved this reallocation")

    if requires_approval and not approval:
        st.error("Export disabled until approval is confirmed.")
        export_enabled = False
    else:
        export_enabled = True

    if export_enabled:
        csv = alloc_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download allocation recommendation CSV",
            data=csv,
            file_name="budget_allocation_recommendation.csv",
            mime="text/csv",
            disabled=not export_enabled,
        )
        if st.button("Log allocation approval/export intent"):
            log_event("allocation_export_approved", f"approval={approval}, budget={monthly_budget}")

    st.subheader("Business value calculator")

    st.caption("Enter assumptions. This calculator does not fabricate customer results.")

    bc1, bc2, bc3, bc4 = st.columns(4)

    annual_budget = bc1.number_input(
        "Annual ad budget",
        min_value=100000,
        max_value=1000000000,
        value=int(monthly_budget * 12),
        step=100000,
    )

    current_planning_iroas = bc2.number_input(
        "Current planning iROAS assumption",
        min_value=0.0,
        max_value=10.0,
        value=1.00,
        step=0.05,
    )

    platform_cost_year = bc3.number_input(
        "Platform operating cost / year",
        min_value=0,
        max_value=100000000,
        value=120000,
        step=10000,
    )

    analyst_hours_saved = bc4.number_input(
        "Analyst hours saved / year",
        min_value=0,
        max_value=20000,
        value=500,
        step=50,
    )

    hourly_cost = st.number_input("Loaded analyst hourly cost", min_value=0.0, max_value=500.0, value=85.0, step=5.0)

    uplift_per_dollar = max(0.0, optimized_iroas - current_planning_iroas)
    annual_revenue_effect = annual_budget * uplift_per_dollar
    analyst_savings = analyst_hours_saved * hourly_cost
    total_value = annual_revenue_effect + analyst_savings
    net_value = total_value - platform_cost_year
    payback_months = safe_div(platform_cost_year, total_value / 12.0, default=float("inf"))

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Annual revenue effect", f"${annual_revenue_effect:,.0f}")
    v2.metric("Analyst time savings", f"${analyst_savings:,.0f}")
    v3.metric("Net value after platform cost", f"${net_value:,.0f}")
    v4.metric("Payback months", f"{payback_months:,.1f}" if np.isfinite(payback_months) else "N/A")

    st.subheader("Sensitivity analysis")

    sat_map = alloc_df.set_index("channel")["saturation"].to_dict()
    alloc_map = alloc_df.set_index("channel")["allocated"].to_dict()

    def project_with_iroas_map(iroas_map: Dict[str, float]) -> float:
        total = 0.0
        for ch, amt in alloc_map.items():
            s = max(float(sat_map.get(ch, 1.0)), 1.0)
            e = max(0.0, float(iroas_map.get(ch, 0.0)))
            total += e * amt * math.exp(-amt / s)
        return float(total)

    low_map = cr.set_index("channel")["ci_low"].to_dict()
    base_map = cr.set_index("channel")["iroas_est"].to_dict()
    high_map = cr.set_index("channel")["ci_high"].to_dict()

    sens = pd.DataFrame(
        [
            {"scenario": "Conservative lower bound", "projected_incremental": project_with_iroas_map(low_map)},
            {"scenario": "Base selected estimate", "projected_incremental": project_with_iroas_map(base_map)},
            {"scenario": "Optimistic upper bound", "projected_incremental": project_with_iroas_map(high_map)},
        ]
    )

    sens["projected_iroas"] = sens["projected_incremental"] / max(float(monthly_budget), 1.0)
    st.dataframe(sens, use_container_width=True)

# -----------------------------------------------------------------------------
# Tab 6: Data Contracts
# -----------------------------------------------------------------------------

with tabs[5]:
    st.header("Data contracts, validation, and canonical storage")

    st.markdown(DATA_CONTRACTS)

    st.subheader("Validation errors")

    if validation_errors.empty:
        st.success("No validation errors in injected failure cases.")
    else:
        st.dataframe(validation_errors, use_container_width=True)

    st.subheader("Validated sample")

    st.dataframe(validation_valid.head(20), use_container_width=True)

    st.subheader("DuckDB canonical query demo")

    con = duckdb.connect(":memory:")
    con.register("fact_weekly_channel", fdf)

    query = """
        SELECT
            channel,
            SUM(spend) AS spend,
            SUM(attributed_revenue) AS attributed_revenue,
            SUM(estimated_incremental_revenue) AS estimated_incremental_revenue,
            SUM(estimated_incremental_revenue) / NULLIF(SUM(spend), 0) AS estimated_iroas
        FROM fact_weekly_channel
        GROUP BY channel
        ORDER BY spend DESC
    """

    st.code(query, language="sql")
    st.dataframe(con.sql(query).df(), use_container_width=True)

    st.subheader("Example canonical record")

    sample = df.iloc[0].copy()
    sample["week_start"] = str(pd.Timestamp(sample["week_start"]).date())
    st.json(json.loads(json.dumps(sample.to_dict(), default=str)))

# -----------------------------------------------------------------------------
# Tab 7: Product Docs
# -----------------------------------------------------------------------------

with tabs[6]:
    st.header("Product requirements and architecture artifacts")

    for title, content in PAGE_ARTIFACTS.items():
        with st.expander(title, expanded=False):
            st.markdown(content)

# -----------------------------------------------------------------------------
# Tab 8: Tests & Eval
# -----------------------------------------------------------------------------

with tabs[7]:
    st.header("Testing, evaluation, security, and reliability")

    st.markdown(TESTING_EVAL)

    st.subheader("Local self-tests")

    tests_df = run_self_tests(payload)
    st.dataframe(tests_df, use_container_width=True)

    failed = tests_df[tests_df["status"] == "FAIL"]

    if failed.empty:
        st.success("All local self-tests passed.")
    else:
        st.error(f"{len(failed)} self-test(s) failed.")

    st.subheader("Evaluation protocol summary")

    st.markdown(
        """
- Use chronological splits for time-dependent ad data.
- Never train or infer using hidden `true_incremental_revenue`.
- Report MAE, RMSE, bias, coverage, sign accuracy, and ranking correlation.
- Compare against naive attributed ROAS and at least one causal baseline.
- State explicitly what synthetic metrics cannot prove.
        """
    )

    st.subheader("Security and safe failure mode")

    st.markdown(
        """
- No secrets or API keys.
- Synthetic data only.
- Input validation rejects malformed records.
- Low-confidence results trigger conservative fallback.
- Consequential budget shifts require human approval.
- No external spend execution is performed.
        """
    )

# -----------------------------------------------------------------------------
# Tab 9: Audit & Ops
# -----------------------------------------------------------------------------

with tabs[8]:
    st.header("Audit history and operations")

    audit_df = pd.DataFrame(st.session_state.audit_log)

    if audit_df.empty:
        st.info("No audit events yet.")
    else:
        st.dataframe(audit_df.sort_values("timestamp", ascending=False), use_container_width=True)

    st.subheader("Health checks")

    health = {
        "data_rows": int(len(df)),
        "filtered_rows": int(len(fdf)),
        "channels": int(len(cr)),
        "validation_errors": int(len(validation_errors)),
        "demo_mp4_exists": bool(media.get("mp4") and os.path.exists(media["mp4"])),
        "demo_gif_exists": bool(media.get("gif") and os.path.exists(media["gif"])),
        "simulation_completed": bool(st.session_state.get("sim_done", False)),
        "outputs_dir_writable": os.access(OUTPUT_DIR, os.W_OK),
    }

    st.json(health)

    st.subheader("Structured log sample")

    st.code(
        json.dumps(st.session_state.audit_log[-10:], indent=2, default=str),
        language="json",
    )

    st.subheader("Backup, restore, rollback")

    st.markdown(
        """
- Backup prototype artifacts: copy `outputs/` and `data/`.
- Restore: replace those folders and rerun with same seed.
- Rollback: delete generated outputs and regenerate using previous seed/config.
- Troubleshooting:
  - If video does not appear, check `outputs/incrementality_demo.gif`.
  - If bootstrap is slow, reduce markets or rerun with smaller synthetic world.
  - If filters produce no rows, reset confidence filter to All.
        """
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.success("Platform ready: synthetic incrementality demo running locally.")