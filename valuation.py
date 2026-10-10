"""Valuation logic for the Residual Income Valuation App.

Pure functions only (no Streamlit code), so they can be tested on their own.
All money amounts are in $ millions unless a name ends in "_ps" (per share).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Course base case for NIKE (problem set, Phase I and II A)
# Net income path from the three-statement forecast (FY2027E-FY2031E).
# ---------------------------------------------------------------------------
NIKE_COURSE_CASE = {
    "fiscal_year": 2026,
    "revenue0": 46_398.0,          # FY2026 revenue, $m (10-K)
    "book_value0": 14_865.0,       # FY2026 total shareholders' equity, $m (10-K)
    "shares": 1_479.8,             # FY2026 basic weighted-average shares, m (10-K)
    "growth": [-0.0159, 0.047, 0.047, 0.0326, 0.0326],
    "net_income": [3342.7963, 4124.0325, 4393.8559, 4697.1027, 5011.9149],
    "payout": 0.6232,
    "rf": 0.0444,
    "beta": 1.11,
    "erp": 0.0418,
    "terminal_growth": 0.0215,
}


def course_margins() -> list[float]:
    """Net income margins implied by the course forecast (NI / revenue)."""
    rev = NIKE_COURSE_CASE["revenue0"]
    out = []
    for g, ni in zip(NIKE_COURSE_CASE["growth"], NIKE_COURSE_CASE["net_income"]):
        rev *= 1 + g
        out.append(ni / rev)
    return out


def capm(rf: float, beta: float, erp: float) -> float:
    """Cost of equity = risk-free rate + beta x equity risk premium."""
    return rf + beta * erp


def residual_income_valuation(revenue0: float, book_value0: float, shares: float,
                              growth: list[float], margin: list[float], payout: float,
                              cost_of_equity: float, terminal_growth: float,
                              start_year: int) -> dict:
    """Simplified residual income model.

    Revenue(t) = Revenue(t-1) x (1 + growth)
    Net income = Revenue x net margin
    Dividends  = Net income x payout ratio
    Book value(t) = Book value(t-1) + Net income - Dividends   (clean surplus)
    Residual income = Net income - cost of equity x beginning book value
    Equity value = BV(0) + sum PV(RI) + PV(terminal value)
    Terminal value = RI(N) x (1 + g) / (ke - g), discounted N years.
    """
    if len(growth) != len(margin):
        raise ValueError("growth and margin need one value per forecast year")
    if cost_of_equity <= terminal_growth:
        raise ValueError("Cost of equity must be higher than the terminal growth rate")
    if shares <= 0:
        raise ValueError("Shares outstanding must be positive")

    rows, rev, bv, pv_sum = [], revenue0, book_value0, 0.0
    for t, (g, m) in enumerate(zip(growth, margin), start=1):
        rev = rev * (1 + g)
        ni = rev * m
        div = ni * payout
        charge = cost_of_equity * bv
        ri = ni - charge
        df = 1 / (1 + cost_of_equity) ** t
        rows.append({
            "Fiscal year": f"FY{start_year + t}E",
            "Revenue": rev, "Revenue growth": g, "Net income": ni, "Net margin": m,
            "Dividends": div, "Beginning book value": bv, "ROE (on beginning BV)": ni / bv if bv else np.nan,
            "Equity charge": charge, "Residual income": ri, "Discount factor": df, "PV of residual income": ri * df,
            "Ending book value": bv + ni - div,
        })
        bv = bv + ni - div
        pv_sum += ri * df

    n = len(rows)
    ri_last = rows[-1]["Residual income"]
    tv = ri_last * (1 + terminal_growth) / (cost_of_equity - terminal_growth)
    pv_tv = tv / (1 + cost_of_equity) ** n
    equity = book_value0 + pv_sum + pv_tv
    return {
        "table": pd.DataFrame(rows),
        "book_value0": book_value0,
        "pv_ri": pv_sum,
        "terminal_value": tv,
        "pv_tv": pv_tv,
        "equity_value": equity,
        "value_ps": equity / shares,
    }


def sensitivity_grid(base_kwargs: dict, ke_values: list[float], g_values: list[float]) -> pd.DataFrame:
    """Value per share for each (cost of equity, terminal growth) pair."""
    out = pd.DataFrame(index=[f"{k:.2%}" for k in ke_values], columns=[f"{g:.2%}" for g in g_values], dtype=float)
    for k in ke_values:
        for g in g_values:
            kw = dict(base_kwargs, cost_of_equity=k, terminal_growth=g)
            try:
                out.loc[f"{k:.2%}", f"{g:.2%}"] = residual_income_valuation(**kw)["value_ps"]
            except ValueError:
                out.loc[f"{k:.2%}", f"{g:.2%}"] = np.nan
    out.index.name = "Cost of equity"
    out.columns.name = "Terminal growth"
    return out


# ---------------------------------------------------------------------------
# Trading comparables
# ---------------------------------------------------------------------------
MULTIPLES = ["EV/Revenue", "EV/EBITDA", "P/E", "P/Book"]


def _pos(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x if math.isfinite(x) and x > 0 else np.nan


def multiples_from_info(info: dict) -> dict:
    """Read valuation multiples from a Yahoo Finance info dict.

    Uses Yahoo's own ratios first (they handle currency differences between
    price and reported financials), and falls back to computing them.
    """
    price = _pos(info.get("currentPrice") or info.get("regularMarketPrice"))
    ev = _pos(info.get("enterpriseValue"))
    rev = _pos(info.get("totalRevenue"))
    ebitda = _pos(info.get("ebitda"))
    eps = _pos(info.get("trailingEps"))
    bvps = _pos(info.get("bookValue"))
    ev_rev = _pos(info.get("enterpriseToRevenue"))
    ev_ebitda = _pos(info.get("enterpriseToEbitda"))
    pe = _pos(info.get("trailingPE"))
    pb = _pos(info.get("priceToBook"))
    if np.isnan(ev_rev) and ev and rev:
        ev_rev = _pos(ev / rev)
    if np.isnan(ev_ebitda) and ev and ebitda:
        ev_ebitda = _pos(ev / ebitda)
    if np.isnan(pe) and price and eps:
        pe = _pos(price / eps)
    if np.isnan(pb) and price and bvps:
        pb = _pos(price / bvps)
    # drop extreme outliers that distort a small peer set
    if ev_ebitda > 100:
        ev_ebitda = np.nan
    if pe > 150:
        pe = np.nan
    return {"EV/Revenue": ev_rev, "EV/EBITDA": ev_ebitda, "P/E": pe, "P/Book": pb}


def implied_prices(multiple_table: pd.DataFrame, subject: dict) -> pd.DataFrame:
    """Implied value per share for the subject at the peer 25th / median / 75th percentile.

    subject keys ($ in units of the subject's reporting currency, shares in units):
      revenue, ebitda, eps, bvps, net_debt, shares
    """
    rows = []
    for m in MULTIPLES:
        vals = pd.to_numeric(multiple_table[m], errors="coerce").dropna() if m in multiple_table else pd.Series(dtype=float)
        if vals.empty:
            continue
        q = {"Low (25th pct)": vals.quantile(0.25), "Median": vals.median(), "High (75th pct)": vals.quantile(0.75)}
        prices = {}
        for label, mult in q.items():
            if m == "EV/Revenue":
                p = (mult * subject["revenue"] - subject["net_debt"]) / subject["shares"] if subject.get("revenue") else np.nan
            elif m == "EV/EBITDA":
                p = (mult * subject["ebitda"] - subject["net_debt"]) / subject["shares"] if subject.get("ebitda", 0) and subject["ebitda"] > 0 else np.nan
            elif m == "P/E":
                p = mult * subject["eps"] if subject.get("eps", 0) and subject["eps"] > 0 else np.nan
            else:
                p = mult * subject["bvps"] if subject.get("bvps", 0) and subject["bvps"] > 0 else np.nan
            prices[label] = p
        rows.append({"Method": m, "Peers used": int(vals.size), "Median multiple": q["Median"], **prices})
    return pd.DataFrame(rows)


def recommendation(value_ps: float, price: float, band: float = 0.10) -> str:
    """Simple rating rule: Buy above +band upside, Sell below -band, otherwise Hold."""
    if not price or not math.isfinite(price):
        return "n/a"
    up = value_ps / price - 1
    return "BUY" if up > band else ("SELL" if up < -band else "HOLD")
