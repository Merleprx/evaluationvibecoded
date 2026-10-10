"""Residual Income Valuation App (Phase III B)

Run with:   streamlit run app.py
Type any stock ticker (NKE, AAPL, MSFT, ...). The app pulls financial and market
data from Yahoo Finance (yfinance), forecasts revenue and net income, values the
equity with a residual income model, compares it with trading comparables and
the market price, and shows all of it on one value ladder.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf

from valuation import (MULTIPLES, NIKE_COURSE_CASE, capm, course_margins, implied_prices,
                       multiples_from_info, recommendation, residual_income_valuation, sensitivity_grid)

st.set_page_config(page_title="Residual income valuation", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens (palette validated for colour-vision deficiency, see README)
# ---------------------------------------------------------------------------
INK, MUTED, HAIRLINE, PANEL = "#16202b", "#5b6675", "#e1e5ea", "#f3f5f8"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"        # categorical slots 1-3
BLUE_LIGHT, GREY_BAR = "#9ec5f4", "#c9ced6"
UP, DOWN, MID = "#2a78d6", "#e34948", "#f0efec"              # diverging pair + neutral midpoint
GOOD_TXT, BAD_TXT = "#0b6e0b", "#b42318"

pio.templates["ledger"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Public Sans, Helvetica, Arial, sans-serif", size=13, color=INK),
    paper_bgcolor="white", plot_bgcolor="white",
    colorway=[BLUE, ORANGE, AQUA],
    xaxis=dict(showgrid=False, linecolor=HAIRLINE, ticks="outside", tickcolor=HAIRLINE, zeroline=False,
               title=dict(font=dict(color=MUTED, size=12))),
    yaxis=dict(gridcolor=HAIRLINE, gridwidth=1, linecolor=HAIRLINE, zeroline=False,
               title=dict(font=dict(color=MUTED, size=12))),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=MUTED)),
    hoverlabel=dict(bgcolor="white", bordercolor=HAIRLINE, font=dict(color=INK, family="Public Sans, sans-serif")),
    margin=dict(l=10, r=10, t=40, b=10), bargap=0.35, title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
))
pio.templates.default = "ledger"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }}
  [data-testid="stMetricValue"], .num, table td {{ font-variant-numeric: tabular-nums; }}
  .company-line {{ color: {MUTED}; margin: -0.6rem 0 1.4rem 0; font-size: 0.95rem; }}
  .verdict {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 1.55rem; line-height: 1.42;
             max-width: 46rem; color: {INK}; margin: 0 0 0.4rem 0; }}
  .verdict b {{ font-weight: 600; }}
  .rating {{ display: inline-block; padding: 0.05rem 0.55rem; border: 1.5px solid currentColor;
            border-radius: 0.3rem; font-weight: 600; font-size: 0.95rem; margin-right: 0.5rem;
            font-family: 'Public Sans', sans-serif; vertical-align: 0.2rem; }}
  .rating.buy {{ color: {GOOD_TXT}; }} .rating.sell {{ color: {BAD_TXT}; }} .rating.hold {{ color: {INK}; }}
  .rule-note {{ color: {MUTED}; font-size: 0.9rem; max-width: 46rem; }}
  .figures {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); border-top: 1px solid {HAIRLINE};
             border-bottom: 1px solid {HAIRLINE}; margin: 0.6rem 0 1.6rem 0; }}
  .figures > div {{ padding: 0.8rem 1rem 0.8rem 0; }}
  .figures > div + div {{ padding-left: 1rem; border-left: 1px solid {HAIRLINE}; }}
  .figures .v {{ font-size: 1.35rem; font-weight: 600; font-variant-numeric: tabular-nums; color: {INK}; }}
  .figures .l {{ font-size: 0.82rem; color: {MUTED}; margin-top: 0.15rem; }}
  @media (max-width: 900px) {{
    .figures {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .figures > div + div {{ border-left: none; padding-left: 0; }}
    .verdict {{ font-size: 1.3rem; }}
  }}
  section[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; margin-top: 1.1rem; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {HAIRLINE}; }}
  .stTabs [data-baseweb="tab"] {{ padding-left: 0; padding-right: 0; }}
</style>
""", unsafe_allow_html=True)

DEFAULT_PEERS = {
    "NKE": "ADDYY, LULU, DECK, ONON, CROX, UAA",
    "AAPL": "MSFT, GOOGL, AMZN, META, SONY",
    "MSFT": "AAPL, GOOGL, AMZN, ORCL, CRM",
}
INFO_KEYS = ["shortName", "longName", "sector", "industry", "currency", "financialCurrency", "currentPrice",
             "regularMarketPrice", "previousClose", "marketCap", "enterpriseValue", "totalRevenue", "ebitda",
             "trailingEps", "bookValue", "trailingPE", "priceToBook", "enterpriseToRevenue", "enterpriseToEbitda",
             "sharesOutstanding", "beta", "totalDebt", "totalCash", "payoutRatio", "dividendRate",
             "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "targetMeanPrice", "recommendationKey", "longBusinessSummary"]


# ---------------------------------------------------------------------------
# Yahoo Finance data (cached for one hour)
# ---------------------------------------------------------------------------
def _num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _row(df: pd.DataFrame, names: list[str]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").dropna()
            s.index = [pd.Timestamp(i).year for i in s.index]
            return s.sort_index()
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:  # network error, rate limit, unknown ticker
        return {"_error": str(e)}
    return {k: info.get(k) for k in INFO_KEYS}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company(ticker: str) -> dict:
    out = {"ticker": ticker, "errors": []}
    t = yf.Ticker(ticker)
    info = fetch_info(ticker)
    if "_error" in info:
        out["errors"].append(f"company profile ({info['_error'][:60]})")
        info = {}
    out["info"] = info
    try:
        inc = t.income_stmt
    except Exception:
        inc = pd.DataFrame(); out["errors"].append("income statement")
    try:
        bs = t.balance_sheet
    except Exception:
        bs = pd.DataFrame(); out["errors"].append("balance sheet")
    try:
        hist = t.history(period="1y")["Close"]
        hist.index = hist.index.tz_localize(None) if hist.index.tz is not None else hist.index
    except Exception:
        hist = pd.Series(dtype=float); out["errors"].append("price history")
    out["revenue"] = _row(inc, ["Total Revenue", "Operating Revenue"]) / 1e6
    out["net_income"] = _row(inc, ["Net Income", "Net Income Common Stockholders"]) / 1e6
    out["equity"] = _row(bs, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]) / 1e6
    out["price_history"] = hist
    price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    if price is None and not hist.empty:
        price = float(hist.iloc[-1])
    out["price"] = price
    out["shares_m"] = (_num(info.get("sharesOutstanding")) or 0) / 1e6 or None
    return out


def history_defaults(d: dict) -> dict:
    rev, ni = d["revenue"], d["net_income"]
    growth = 0.04
    if len(rev) >= 2:
        yrs = min(3, len(rev) - 1)
        growth = (rev.iloc[-1] / rev.iloc[-1 - yrs]) ** (1 / yrs) - 1
    margin = float(ni.iloc[-1] / rev.iloc[-1]) if len(ni) and len(rev) and rev.iloc[-1] else 0.08
    payout = _num(d["info"].get("payoutRatio"))
    if payout is None:
        dr, sh = _num(d["info"].get("dividendRate")), d.get("shares_m")
        payout = dr * sh / ni.iloc[-1] if (dr and sh and len(ni) and ni.iloc[-1] > 0) else 0.3
    return {
        "growth": float(np.clip(growth, -0.10, 0.25)),
        "margin": float(np.clip(margin, -0.20, 0.60)),
        "payout": float(np.clip(payout, 0.0, 1.0)),
        "beta": _num(d["info"].get("beta")) or 1.0,
        "revenue0": float(rev.iloc[-1]) if len(rev) else 0.0,
        "book_value0": float(d["equity"].iloc[-1]) if len(d["equity"]) else 0.0,
        "fiscal_year": int(rev.index[-1]) if len(rev) else datetime.now().year - 1,
    }


def money(x, dec=0):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:,.{dec}f}"


def pct(x, dec=1):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:.{dec}f}%"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Company")
ticker = st.sidebar.text_input("Ticker", value="NKE", help="Yahoo Finance symbol, e.g. NKE, AAPL, MSFT").strip().upper() or "NKE"
with st.spinner(f"Loading {ticker} from Yahoo Finance"):
    data = fetch_company(ticker)
info = data["info"]
hd = history_defaults(data)

is_nke = ticker == "NKE"
preset = st.sidebar.toggle("Use the course base case", value=is_nke, disabled=not is_nke,
                           help="Nike only. Loads the problem-set assumptions: FY2026 10-K starting point, the Phase I "
                                "revenue path and net income margins, CAPM inputs and 2.15% terminal growth.")
preset = preset and is_nke
k = f"{ticker}_{preset}"  # widget keys change with ticker/preset so defaults refresh

st.sidebar.markdown("## Forecast")
n_years = st.sidebar.slider("Years to forecast", 3, 10, 5, key=f"n_{k}")
c_growth = st.sidebar.number_input("Revenue growth, % a year", -50.0, 100.0,
                                   round((NIKE_COURSE_CASE["growth"][-1] if preset else hd["growth"]) * 100, 2), 0.25, key=f"g_{k}") / 100
c_margin = st.sidebar.number_input("Net income margin, % of revenue", -50.0, 80.0,
                                   round((course_margins()[-1] if preset else hd["margin"]) * 100, 2), 0.25, key=f"m_{k}") / 100
payout = st.sidebar.number_input("Dividend payout, % of net income", 0.0, 150.0,
                                 round((NIKE_COURSE_CASE["payout"] if preset else hd["payout"]) * 100, 2), 1.0, key=f"p_{k}") / 100
with st.sidebar.expander("Set growth and margin year by year", expanded=preset):
    custom = st.checkbox("Use a different value each year", value=preset, key=f"cust_{k}")
    if preset:
        g_path = list(NIKE_COURSE_CASE["growth"]) + [c_growth] * max(0, n_years - 5)
        m_path = course_margins() + [c_margin] * max(0, n_years - 5)
    else:
        g_path, m_path = [c_growth] * n_years, [c_margin] * n_years
    g_path, m_path = g_path[:n_years], m_path[:n_years]
    if custom:
        editor = st.data_editor(pd.DataFrame({"Yr": [i + 1 for i in range(n_years)],
                                              "Growth %": np.round(np.array(g_path) * 100, 2),
                                              "Margin %": np.round(np.array(m_path) * 100, 4)}),
                                hide_index=True, disabled=["Yr"], key=f"ed_{k}_{n_years}")
        g_path = (pd.to_numeric(editor["Growth %"], errors="coerce").fillna(0) / 100).tolist()
        m_path = (pd.to_numeric(editor["Margin %"], errors="coerce").fillna(0) / 100).tolist()
    else:
        g_path, m_path = [c_growth] * n_years, [c_margin] * n_years
        st.caption("Off: the growth and margin above apply to every year.")

st.sidebar.markdown("## Cost of equity")
ke_mode = st.sidebar.radio("Method", ["CAPM", "Type it in"], horizontal=True, key=f"kem_{k}", label_visibility="collapsed")
rf = st.sidebar.number_input("Risk-free rate, %", 0.0, 20.0, NIKE_COURSE_CASE["rf"] * 100, 0.05, key=f"rf_{k}") / 100
beta = st.sidebar.number_input("Beta", -1.0, 4.0, round(NIKE_COURSE_CASE["beta"] if preset else hd["beta"], 2), 0.01, key=f"b_{k}")
erp = st.sidebar.number_input("Equity risk premium, %", 0.0, 20.0, NIKE_COURSE_CASE["erp"] * 100, 0.05, key=f"erp_{k}") / 100
ke_capm = capm(rf, beta, erp)
if ke_mode == "CAPM":
    ke = ke_capm
    st.sidebar.caption(f"{rf:.2%} + {beta:.2f} × {erp:.2%} = **{ke:.3%}**")
else:
    ke = st.sidebar.number_input("Cost of equity, %", 0.0, 50.0, round(ke_capm * 100, 3), 0.1, key=f"ke_{k}") / 100
tg = st.sidebar.number_input("Growth after the forecast, %", -5.0, 10.0, NIKE_COURSE_CASE["terminal_growth"] * 100, 0.05,
                             key=f"tg_{k}", help="Terminal growth rate of residual income") / 100

st.sidebar.markdown("## Starting point")
st.sidebar.caption("Filled in from Yahoo Finance in $ millions. Change anything that is missing or looks wrong.")
fy = st.sidebar.number_input("Last reported fiscal year", 1990, 2100,
                             NIKE_COURSE_CASE["fiscal_year"] if preset else hd["fiscal_year"], key=f"fy_{k}")
revenue0 = st.sidebar.number_input("Revenue", 0.0, None, NIKE_COURSE_CASE["revenue0"] if preset else hd["revenue0"], 100.0, key=f"r0_{k}")
bv0 = st.sidebar.number_input("Book value of equity", None, None,
                              NIKE_COURSE_CASE["book_value0"] if preset else hd["book_value0"], 100.0, key=f"bv_{k}")
shares = st.sidebar.number_input("Shares, millions", 0.0, None,
                                 NIKE_COURSE_CASE["shares"] if preset else float(data["shares_m"] or 0.0), 1.0, key=f"sh_{k}")
fallback_px = 34.74 if preset else 0.0  # NKE close on Oct 8, 2026, used only if Yahoo returns no price
price = st.sidebar.number_input("Share price today", 0.0, None, float(data["price"] or fallback_px), 0.01, key=f"px_{k}")

st.sidebar.markdown("## Peers")
peers_text = st.sidebar.text_input("Peer tickers, separated by commas", DEFAULT_PEERS.get(ticker, ""), key=f"peers_{ticker}")
peers = [p.strip().upper() for p in peers_text.replace(";", ",").split(",") if p.strip() and p.strip().upper() != ticker]

# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------
name = info.get("longName") or info.get("shortName") or ticker
ready = revenue0 > 0 and shares > 0 and bv0 != 0
if not ready:
    st.title(name)
    st.error("The valuation needs revenue, book value of equity and shares. Yahoo Finance did not return them, "
             "so type them into the sidebar under Starting point.")
    st.stop()

base_kwargs = dict(revenue0=revenue0, book_value0=bv0, shares=shares, growth=g_path, margin=m_path,
                   payout=payout, cost_of_equity=ke, terminal_growth=tg, start_year=int(fy))
try:
    res = residual_income_valuation(**base_kwargs)
except ValueError as e:
    st.title(name)
    st.error(f"{e}. Lower the growth after the forecast or raise the cost of equity in the sidebar.")
    st.stop()
fc = res["table"]
value_ps = res["value_ps"]
upside = value_ps / price - 1 if price else float("nan")
rating = recommendation(value_ps, price)

# trading comps
comps = pd.DataFrame()
comp_res = pd.DataFrame()
subject = {}
if peers:
    rows = []
    with st.spinner("Loading peers"):
        for p in peers:
            pi = fetch_info(p)
            if "_error" in pi or not (pi.get("marketCap") or pi.get("currentPrice")):
                rows.append({"Ticker": p, "Company": "not available", **{m: np.nan for m in MULTIPLES}})
                continue
            rows.append({"Ticker": p, "Company": pi.get("shortName") or p,
                         "Market cap, $bn": (_num(pi.get("marketCap")) or np.nan) / 1e9, **multiples_from_info(pi)})
    comps = pd.DataFrame(rows)
    subject = {"revenue": _num(info.get("totalRevenue")) or 0, "ebitda": _num(info.get("ebitda")) or 0,
               "eps": _num(info.get("trailingEps")) or 0, "bvps": _num(info.get("bookValue")) or 0,
               "net_debt": (_num(info.get("totalDebt")) or 0) - (_num(info.get("totalCash")) or 0),
               "shares": (_num(info.get("sharesOutstanding")) or shares * 1e6)}
    comp_res = implied_prices(comps, subject)

# ladder ranges
ladder = []
ri_grid = sensitivity_grid(base_kwargs, [ke - 0.01, ke, ke + 0.01], [tg - 0.005, tg, tg + 0.005]).values
ri_vals = ri_grid[np.isfinite(ri_grid)]
if ri_vals.size:
    ladder.append(("Residual income", "cost of equity ±1 pt, growth ±0.5 pt", ri_vals.min(), ri_vals.max(), value_ps, BLUE))
for _, r in comp_res.iterrows():
    lo, hi = sorted([r["Low (25th pct)"], r["High (75th pct)"]])
    if pd.notna(lo) and pd.notna(hi):
        ladder.append((f"Peers on {r['Method']}", "25th to 75th percentile of peers", lo, hi, r["Median"], BLUE_LIGHT))
lo52, hi52 = _num(info.get("fiftyTwoWeekLow")), _num(info.get("fiftyTwoWeekHigh"))
if lo52 and hi52:
    ladder.append(("52-week trading range", "lowest to highest close", lo52, hi52, None, GREY_BAR))
analyst_target = _num(info.get("targetMeanPrice"))

# ---------------------------------------------------------------------------
# Header: verdict, value ladder, key figures
# ---------------------------------------------------------------------------
st.title(name)
sub = [ticker]
for x in (info.get("industry"), info.get("currency")):
    if x:
        sub.append(x)
st.markdown(f"<div class='company-line'>{', '.join(sub)}. Data from Yahoo Finance, loaded "
            f"{datetime.now():%d %b %Y, %H:%M}.{' Course base case is on.' if preset else ''}</div>",
            unsafe_allow_html=True)

if data["errors"]:
    st.warning("Yahoo Finance did not return the " + ", ".join(data["errors"]) +
               ". The valuation still runs on the numbers in the sidebar; check them under Starting point.")
if info.get("financialCurrency") and info.get("currency") and info["financialCurrency"] != info["currency"]:
    st.info(f"{ticker} reports in {info['financialCurrency']} but trades in {info['currency']}. Convert the starting "
            "point to the trading currency before comparing value and price.")

if price:
    direction = "above" if upside >= 0 else "below"
    cls, icon = {"BUY": ("buy", "▲ Buy"), "SELL": ("sell", "▼ Sell"), "HOLD": ("hold", "● Hold")}.get(rating, ("hold", rating))
    st.markdown(f"<p class='verdict'><span class='rating {cls}'>{icon}</span>Residual income puts {ticker} at "
                f"<b>${value_ps:,.2f}</b> a share, {abs(upside):.1%} {direction} today's price of <b>${price:,.2f}</b>.</p>"
                f"<p class='rule-note'>Rating rule: buy above +10%, sell below −10%, hold in between.</p>",
                unsafe_allow_html=True)
else:
    st.markdown(f"<p class='verdict'>Residual income puts {ticker} at <b>${value_ps:,.2f}</b> a share. "
                "Enter today's share price in the sidebar to compare.</p>", unsafe_allow_html=True)

if ladder:
    fig = go.Figure()
    labels = [f"<b>{a}</b><br><span style='color:{MUTED};font-size:11px'>{b}</span>" for a, b, *_ in ladder]
    for (a, b, lo, hi, mid, col), lab in zip(ladder, labels):
        fig.add_bar(y=[lab], x=[hi - lo], base=[lo], orientation="h", marker=dict(color=col, line=dict(width=0)),
                    width=0.42, showlegend=False,
                    hovertemplate=f"{a}<br>{lo:,.2f} to {hi:,.2f}<extra></extra>")
        fig.add_annotation(y=lab, x=lo, text=f"{lo:,.2f}", xanchor="right", showarrow=False, xshift=-6,
                           font=dict(color=MUTED, size=11))
        fig.add_annotation(y=lab, x=hi, text=f"{hi:,.2f}", xanchor="left", showarrow=False, xshift=6,
                           font=dict(color=MUTED, size=11))
        if mid is not None and np.isfinite(mid):
            fig.add_scatter(y=[lab], x=[mid], mode="markers", showlegend=False,
                            marker=dict(symbol="line-ns", size=18, line=dict(width=3, color=INK)),
                            hovertemplate=f"{a} point estimate {mid:,.2f}<extra></extra>")
    if price:
        fig.add_vline(x=price, line=dict(color=ORANGE, width=2))
        fig.add_annotation(x=price, y=1.0, yref="paper", yanchor="bottom", xanchor="right", xshift=-4,
                           text=f"Today ${price:,.2f}", showarrow=False, font=dict(color=INK, size=12))
    if analyst_target:
        fig.add_vline(x=analyst_target, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=analyst_target, y=1.0, yref="paper", yanchor="bottom", xanchor="left", xshift=4,
                           text=f"Analyst mean target ${analyst_target:,.2f}", showarrow=False, font=dict(color=MUTED, size=11))
    xs = [v for _, _, lo, hi, *_ in ladder for v in (lo, hi)] + [v for v in (price, analyst_target) if v]
    span_x = max(xs) - min(xs)
    fig.update_layout(height=90 + 62 * len(ladder), margin=dict(l=10, r=40, t=34, b=34),
                      xaxis=dict(title="Value per share", showgrid=True, gridcolor=HAIRLINE, tickprefix="$",
                                 range=[max(0, min(xs) - 0.12 * span_x), max(xs) + 0.10 * span_x]),
                      yaxis=dict(autorange="reversed", showgrid=False, linecolor="white"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Bars show each method's range; the dark tick is its point estimate (base case or peer median). "
               "The orange line is today's share price.")

peer_median = float(np.nanmedian(comp_res["Median"])) if not comp_res.empty and comp_res["Median"].notna().any() else None
figs = [
    (f"{ke:.2%}", "Cost of equity"),
    (f"${res['equity_value'] / 1000:,.1f}bn", "Equity value"),
    (f"{res['pv_tv'] / res['equity_value']:.0%}", "of value from after the forecast"),
    (f"{res['equity_value'] / bv0:.2f}x", "Value to book value"),
    (f"${peer_median:,.2f}" if peer_median else "n/a", "Peer-implied median per share"),
]
st.markdown("<div class='figures'>" + "".join(f"<div><div class='v'>{v}</div><div class='l'>{l}</div></div>" for v, l in figs)
            + "</div>", unsafe_allow_html=True)

tabs = st.tabs(["Forecast", "Valuation", "Peers", "Price history", "Data and method"])

# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
with tabs[0]:
    x = fc["Fiscal year"]
    c1, c2, c3 = st.columns(3)
    for col, (title, y, fmt, colr) in zip((c1, c2, c3), [
            ("Revenue, $m", fc["Revenue"], ",.0f", BLUE),
            ("Net income, $m", fc["Net income"], ",.0f", BLUE),
            ("Residual income, $m", fc["Residual income"], ",.0f", BLUE)]):
        f = go.Figure(go.Bar(x=x, y=y, marker_color=colr, hovertemplate="%{x}: %{y:" + fmt + "}<extra></extra>"))
        f.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(tickformat=","))
        col.markdown(f"##### {title}")
        col.plotly_chart(f, width="stretch", config={"displayModeBar": False})
    c4, c5 = st.columns(2)
    f = go.Figure()
    f.add_scatter(x=x, y=fc["Net margin"], mode="lines+markers", name="Net margin", line=dict(color=BLUE, width=2),
                  marker=dict(size=8), hovertemplate="%{x}: %{y:.2%}<extra>Net margin</extra>")
    f.add_scatter(x=x, y=fc["ROE (on beginning BV)"], mode="lines+markers", name="Return on equity",
                  line=dict(color=ORANGE, width=2), marker=dict(size=8), hovertemplate="%{x}: %{y:.2%}<extra>ROE</extra>")
    f.add_hline(y=ke, line=dict(color=MUTED, width=1))
    f.add_annotation(x=x.iloc[-1], y=ke, text=f"Cost of equity {ke:.2%}", showarrow=False, yshift=10,
                     xanchor="right", font=dict(color=MUTED, size=11))
    f.update_layout(height=300, yaxis=dict(tickformat=".0%"))
    c4.markdown("##### Return on equity stays above the cost of equity, so residual income is positive")
    c4.plotly_chart(f, width="stretch", config={"displayModeBar": False})
    f = go.Figure()
    f.add_bar(x=x, y=fc["Equity charge"], name="Equity charge (cost of equity × book value)", marker_color=ORANGE,
              hovertemplate="%{x}: %{y:,.0f}<extra>Equity charge</extra>")
    f.add_bar(x=x, y=fc["Residual income"], name="Residual income", marker_color=BLUE,
              hovertemplate="%{x}: %{y:,.0f}<extra>Residual income</extra>")
    f.update_layout(barmode="stack", height=300, yaxis=dict(tickformat=","))
    c5.markdown("##### How net income splits, $m")
    c5.plotly_chart(f, width="stretch", config={"displayModeBar": False})

    show = fc.copy()
    for c in ["Revenue", "Net income", "Dividends", "Beginning book value", "Equity charge", "Residual income",
              "PV of residual income", "Ending book value"]:
        show[c] = show[c].map(money)
    for c in ["Revenue growth", "Net margin", "ROE (on beginning BV)"]:
        show[c] = show[c].map(lambda v: pct(v, 2))
    show["Discount factor"] = show["Discount factor"].map(lambda v: f"{v:.4f}")
    st.markdown("#### Forecast table, $ millions")
    st.dataframe(show.set_index("Fiscal year").T, width="stretch")
    st.download_button("Download the forecast as CSV", fc.to_csv(index=False).encode(), f"{ticker}_ri_forecast.csv", "text/csv")

# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------
with tabs[1]:
    a, b = st.columns(2)
    with a:
        st.markdown("#### From book value to equity value, $m")
        wf = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "total"],
            x=[f"Book value<br>FY{fy}", "Residual income<br>(discounted)", "After the forecast<br>(discounted)", "Equity<br>value"],
            y=[res["book_value0"], res["pv_ri"], res["pv_tv"], 0],
            text=[money(res["book_value0"]), money(res["pv_ri"]), money(res["pv_tv"]), money(res["equity_value"])],
            textposition="outside", connector=dict(line=dict(color=HAIRLINE)),
            increasing=dict(marker=dict(color=BLUE)), decreasing=dict(marker=dict(color=DOWN)),
            totals=dict(marker=dict(color=INK)), hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
        wf.update_layout(height=360, showlegend=False, yaxis=dict(tickformat=","))
        st.plotly_chart(wf, width="stretch", config={"displayModeBar": False})
        tbl = pd.DataFrame({"": ["Book value of equity", "Residual income, discounted", "Value after the forecast (undiscounted)",
                                 "Value after the forecast, discounted", "Equity value", "Shares, millions",
                                 "Value per share", "Share price today", "Upside or downside"],
                            "Value": [money(res["book_value0"]), money(res["pv_ri"]), money(res["terminal_value"]),
                                      money(res["pv_tv"]), money(res["equity_value"]), money(shares, 1),
                                      f"${value_ps:,.2f}", f"${price:,.2f}", pct(upside)]}).set_index("")
        st.table(tbl)
    with b:
        st.markdown("#### Value per share as cost of equity and growth change")
        ke_vals = [ke + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
        g_vals = [tg + d for d in (-0.01, -0.005, 0, 0.005, 0.01)]
        grid = sensitivity_grid(base_kwargs, ke_vals, g_vals)
        z = grid.values.astype(float)
        mid = price if price else value_ps
        span = np.nanmax(np.abs(z - mid)) if np.isfinite(z).any() else 1
        hm = go.Figure(go.Heatmap(
            z=z, x=list(grid.columns), y=list(grid.index),
            zmin=mid - span, zmax=mid + span, colorscale=[[0, DOWN], [0.5, MID], [1, UP]],
            text=[[f"{v:,.2f}" if np.isfinite(v) else "n/a" for v in row] for row in z], texttemplate="%{text}",
            textfont=dict(color=INK), xgap=2, ygap=2, colorbar=dict(title="$/share", thickness=10),
            hovertemplate="Cost of equity %{y}, growth %{x}: $%{z:,.2f}<extra></extra>"))
        hm.update_layout(height=360, yaxis=dict(autorange="reversed", title="Cost of equity", type="category"),
                         xaxis=dict(title="Growth after the forecast", type="category"))
        st.plotly_chart(hm, width="stretch", config={"displayModeBar": False})
        st.caption(f"Blue cells are above today's price, red cells below; grey is close to ${mid:,.2f}. "
                   "The centre cell is your current case.")

    st.markdown("#### Which assumption moves the value most")
    drivers = {"Revenue growth ±1 pt": ("growth", 0.01), "Net margin ±1 pt": ("margin", 0.01),
               "Payout ratio ±10 pts": ("payout", 0.10), "Cost of equity ±1 pt": ("cost_of_equity", 0.01),
               "Growth after the forecast ±0.5 pt": ("terminal_growth", 0.005)}
    rows = []
    for label, (key, step) in drivers.items():
        vals = []
        for sgn in (+1, -1):
            kw = dict(base_kwargs)
            kw[key] = [v + sgn * step for v in kw[key]] if isinstance(kw[key], list) else kw[key] + sgn * step
            try:
                vals.append(residual_income_valuation(**kw)["value_ps"])
            except ValueError:
                vals.append(np.nan)
        rows.append({"Driver": label, "Higher": vals[0], "Lower": vals[1]})
    tor = pd.DataFrame(rows)
    tor["Swing"] = (tor["Higher"] - tor["Lower"]).abs()
    tor = tor.sort_values("Swing")
    tf = go.Figure()
    tf.add_bar(y=tor["Driver"], x=tor["Higher"] - value_ps, orientation="h", name="Assumption higher", marker_color=UP,
               hovertemplate="%{y}: %{x:+,.2f} per share<extra>higher</extra>")
    tf.add_bar(y=tor["Driver"], x=tor["Lower"] - value_ps, orientation="h", name="Assumption lower", marker_color=DOWN,
               hovertemplate="%{y}: %{x:+,.2f} per share<extra>lower</extra>")
    tf.update_layout(barmode="overlay", height=300, xaxis=dict(title=f"Change from ${value_ps:,.2f} per share, $", tickformat="+,.0f",
                                                               showgrid=True, gridcolor=HAIRLINE, zeroline=True, zerolinecolor=MUTED))
    st.plotly_chart(tf, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Peers
# ---------------------------------------------------------------------------
with tabs[2]:
    if not peers:
        st.info("Add peer tickers in the sidebar under Peers to compare multiples.")
    else:
        st.markdown("#### Peer multiples, trailing twelve months")
        disp = comps.copy()
        for m in MULTIPLES:
            disp[m] = disp[m].map(lambda v: f"{v:.1f}x" if pd.notna(v) else "n/a")
        if "Market cap, $bn" in disp:
            disp["Market cap, $bn"] = disp["Market cap, $bn"].map(lambda v: f"{v:,.1f}" if pd.notna(v) else "n/a")
        subj = multiples_from_info(info) if info else {m: np.nan for m in MULTIPLES}
        stats = pd.DataFrame({m: [comps[m].median(), comps[m].quantile(.25), comps[m].quantile(.75), comps[m].mean(), subj[m]]
                              for m in MULTIPLES}, index=["Peer median", "Peer 25th percentile", "Peer 75th percentile",
                                                          "Peer mean", f"{ticker} today"])
        st.dataframe(disp.set_index("Ticker"), width="stretch")
        st.dataframe(stats.map(lambda v: f"{v:.1f}x" if pd.notna(v) else "n/a"), width="stretch")
        if comp_res.empty:
            st.warning("None of the peers returned usable multiples. Check the tickers or try again in a minute.")
        else:
            st.markdown(f"#### What the peer multiples imply for {ticker}, per share")
            cr = comp_res.copy()
            cr["Median multiple"] = cr["Median multiple"].map(lambda v: f"{v:.1f}x")
            for c in ["Low (25th pct)", "Median", "High (75th pct)"]:
                cr[c] = cr[c].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "n/a")
            st.dataframe(cr.set_index("Method"), width="stretch")
            nd = subject['net_debt'] / 1e6
            nd_txt = f"net debt of ${nd:,.0f}m" if nd >= 0 else f"net cash of ${-nd:,.0f}m, added back"
            st.caption(f"Enterprise-value multiples: (multiple × {ticker}'s metric − net debt) ÷ shares, with {nd_txt}. "
                       "P/E and P/Book: multiple × EPS or book value per share. "
                       "P/E above 150x and EV/EBITDA above 100x are left out as outliers.")

# ---------------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------------
with tabs[3]:
    hist = data["price_history"]
    if hist is None or hist.empty:
        st.info("Yahoo Finance returned no price history for this ticker.")
    else:
        pf = go.Figure()
        pf.add_scatter(x=hist.index, y=hist.values, name="Closing price", line=dict(color=INK, width=2),
                       hovertemplate="%{x|%d %b %Y}: $%{y:,.2f}<extra></extra>")
        pf.add_hline(y=value_ps, line=dict(color=BLUE, width=2))
        pf.add_annotation(x=hist.index[0], y=value_ps, text=f"Residual income value ${value_ps:,.2f}", showarrow=False,
                          yshift=11, xanchor="left", font=dict(color=BLUE, size=12))
        if peer_median:
            pf.add_hline(y=peer_median, line=dict(color=ORANGE, width=2))
            pf.add_annotation(x=hist.index[0], y=peer_median, text=f"Peer-implied median ${peer_median:,.2f}",
                              showarrow=False, yshift=-11, xanchor="left", font=dict(color=ORANGE, size=12))
        pf.update_layout(height=380, yaxis=dict(tickprefix="$"))
        st.markdown("##### Share price over the last year")
        st.plotly_chart(pf, width="stretch", config={"displayModeBar": False})
    implied_g = None
    try:
        lo_g, hi_g = -0.05, ke - 0.0005
        fz = lambda gg: residual_income_valuation(**dict(base_kwargs, terminal_growth=gg))["value_ps"] - price
        if price and fz(lo_g) * fz(hi_g) < 0:
            for _ in range(60):
                m_ = (lo_g + hi_g) / 2
                lo_g, hi_g = (m_, hi_g) if fz(lo_g) * fz(m_) > 0 else (lo_g, m_)
            implied_g = (lo_g + hi_g) / 2
    except ValueError:
        pass
    if implied_g is not None:
        st.markdown(f"**What the market is pricing in.** Holding every other assumption fixed, today's price of "
                    f"${price:,.2f} matches a growth rate after the forecast of **{implied_g:.2%}**, against your {tg:.2%}.")

# ---------------------------------------------------------------------------
# Data and method
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("#### Reported history from Yahoo Finance, $ millions")
    hist_df = pd.DataFrame({"Revenue": data["revenue"], "Net income": data["net_income"],
                            "Book value of equity": data["equity"]})
    if hist_df.empty:
        st.info("Yahoo Finance returned no annual statements for this ticker.")
    else:
        hist_df["Net margin"] = hist_df["Net income"] / hist_df["Revenue"]
        hist_df["Revenue growth"] = hist_df["Revenue"].pct_change()
        hist_df["Return on equity"] = hist_df["Net income"] / hist_df["Book value of equity"]
        show_h = hist_df.copy()
        for c in ["Revenue", "Net income", "Book value of equity"]:
            show_h[c] = show_h[c].map(money)
        for c in ["Net margin", "Revenue growth", "Return on equity"]:
            show_h[c] = show_h[c].map(pct)
        show_h.index = [f"FY{i}" for i in show_h.index]
        st.dataframe(show_h.T, width="stretch")
    st.markdown(
        """
#### Method
1. Revenue grows each year by the growth rate: Revenue(t) = Revenue(t−1) × (1 + growth).
2. Net income = revenue × net income margin. Dividends = net income × payout ratio.
3. Book value rolls forward: BV(t) = BV(t−1) + net income − dividends (no buybacks or new shares).
4. Residual income = net income − cost of equity × book value at the start of the year, discounted at the cost of equity.
5. Value after the forecast = last residual income × (1 + g) ÷ (cost of equity − g), discounted to today.
6. Equity value = today's book value + discounted residual income + discounted value after the forecast.

For Nike with the course base case on, the app gives **$36.97** per share, the same as the Excel model (Phase II A).
Yahoo Finance data is free and unofficial: figures can be missing, restated or delayed, so check key inputs against the 10-K.
        """)
