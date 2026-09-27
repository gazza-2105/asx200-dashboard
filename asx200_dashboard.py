#!/usr/bin/env python3
"""
ASX 200 market dashboard
========================

Pulls key data for every S&P/ASX 200 constituent from Yahoo Finance
(price, 52-week high/low, market cap, trailing & forward P/E, EPS,
dividend yield, 1-year return, sector) and builds a single interactive
HTML dashboard designed to show all ~200 stocks at once without turning
into a wall of bars.

Charts
  1. Market map          - treemap: sector > company, sized by market cap,
                           colour switchable (52-wk position / 1-yr return /
                           earnings yield)
  2. 52-week position    - beeswarm: every stock placed on 0-100% of its
                           52-week range, one row per sector
  3. Valuation spread    - beeswarm of P/E by sector (log scale) with medians
  4. Value vs momentum   - earnings yield vs drawdown from 52-wk high,
                           bubbles sized by market cap, four labelled quadrants
  5. Earnings outlook    - trailing vs forward P/E (below the diagonal =
                           earnings expected to grow)
  6. Concentration curve - cumulative share of index market cap by rank
  7. Full table          - sortable / searchable, with inline 52-week range bars

Usage
  pip install -r requirements.txt
  python asx200_dashboard.py                 # fetch fresh data + build dashboard
  python asx200_dashboard.py --cache         # rebuild charts from the latest saved CSV
  python asx200_dashboard.py --tickers my_list.csv   # use your own ticker list
  python asx200_dashboard.py --workers 4     # slower but gentler on Yahoo's rate limits

Outputs (in ./output):
  asx200_data_YYYY-MM-DD.csv / .xlsx   - raw data, one row per stock
  asx200_dashboard.html                - the dashboard (open in any browser)
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import random
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import numpy as np
import pandas as pd

try:  # Use certifi's CA bundle so python.org installs on macOS don't hit SSL errors
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import yfinance as yf
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs, get_plotlyjs_version

# --------------------------------------------------------------------------------------
# 1. Constituents
# --------------------------------------------------------------------------------------

WIKI_URL = "https://en.wikipedia.org/wiki/S%26P/ASX_200"

# Fallback list (used only if the live Wikipedia list can't be fetched). Index membership
# changes every quarter, so tickers that have since left the index or delisted are simply
# skipped. Pass --tickers to use your own list.
FALLBACK_TICKERS = """
360 A2M AAI ABB ABG ALD ALL ALQ ALX AMC AMP ANN ANZ APA APE ARB ARF ASB ASX AUB AZJ
BAP BEN BGA BGL BHP BKW BOE BOQ BPT BRG BSL BWP BXB CAR CBA CCP CGF CHC CHN CIA CIP
CKF CLW CMM CNI CNU COH COL CPU CQR CRN CSC CSL CTD CU6 CWY CYL DBI DEG DGT DMP DOW
DRO DRR DTL DXI DXS DYL EBO EDV ELD EMR EVN EVT FBU FLT FMG FPH GDG GGP GMD GMG GNC
GOZ GPT GQG HDN HLI HMC HUB HVN IAG IEL IFL IFT IGO ILU INA ING IPH IPL IPX IRE JBH
JDO JHX KAR KLS LLC LNW LOV LTR LYC MAQ MFG MGH MGR MIN MND MP1 MPL MQG MSB MTS NAB
NAN NEC NEM NEU NHC NHF NIC NSR NST NUF NWH NWL NWS NXG NXT OBM ORA ORG ORI PDN PLS
PME PMV PNI PNR PNV PPT PRN PRU PXA QAN QBE QUB RED REA REH RGN RHC RIO RMD RMS RRL
RSG RWC S32 SCG SDF SDR SEK SFR SGH SGM SGP SGR SHL SIG SIQ SLC SLX SMR SNZ SOL SPK
STO SUL SUN SVW SX2 TAH TCL TLC TLS TLX TNE TPG TPW TWE TYR VAU VCX VEA VNT WA1 WAF
WBC WBT WDS WEB WES WGX WHC WOR WOW WPR WTC XRO YAL ZIP
""".split()

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def get_constituents(tickers_file: str | None = None) -> list[str]:
    """Return Yahoo-style tickers (e.g. 'BHP.AX')."""
    codes: list[str] = []
    if tickers_file:
        df = pd.read_csv(tickers_file)
        col = next((c for c in df.columns if c.lower() in ("ticker", "code", "symbol")), df.columns[0])
        codes = df[col].astype(str).tolist()
        print(f"Loaded {len(codes)} tickers from {tickers_file}")
    else:
        try:
            import requests

            html = requests.get(WIKI_URL, headers=UA, timeout=20).text
            for t in pd.read_html(io.StringIO(html)):
                cols = {str(c).strip().lower(): c for c in t.columns}
                key = next((cols[k] for k in ("code", "asx code", "ticker", "symbol") if k in cols), None)
                if key is not None and len(t) >= 150:
                    codes = t[key].astype(str).tolist()
                    break
            if codes:
                print(f"Fetched {len(codes)} current ASX 200 constituents from Wikipedia")
        except Exception as e:  # network / parse failure -> fallback
            print(f"Couldn't fetch live constituent list ({e.__class__.__name__}); using built-in list")
        if not codes:
            codes = FALLBACK_TICKERS
            print(f"Using built-in list of {len(codes)} tickers")

    out = []
    for c in codes:
        c = c.strip().upper().replace("ASX:", "")
        if not c or c == "NAN":
            continue
        out.append(c if c.endswith(".AX") else f"{c}.AX")
    return list(dict.fromkeys(out))  # de-dupe, keep order


# --------------------------------------------------------------------------------------
# 2. Data fetch
# --------------------------------------------------------------------------------------

def fetch_history(tickers: list[str], batch: int = 50) -> dict[str, dict]:
    """Bulk-download 1y of daily prices. Gives price, 52-wk hi/lo and 1-yr return even
    when the per-stock .info endpoint is rate-limited."""
    out: dict[str, dict] = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        print(f"  price history {i + 1}-{i + len(chunk)} of {len(tickers)}")
        for attempt in range(3):
            try:
                data = yf.download(chunk, period="1y", interval="1d", group_by="ticker",
                                   auto_adjust=False, threads=True, progress=False)
                break
            except Exception as e:
                wait = 5 * (attempt + 1)
                print(f"    download failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        else:
            continue
        for t in chunk:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                close = df["Close"].dropna()
                if close.empty:
                    continue
                out[t] = {
                    "price_hist": float(close.iloc[-1]),
                    "high52_hist": float(df["High"].max()),
                    "low52_hist": float(df["Low"].min()),
                    "ret_1y": float(close.iloc[-1] / close.iloc[0] - 1) * 100 if len(close) > 200 else np.nan,
                }
            except Exception:
                continue
        time.sleep(1)
    return out


INFO_FIELDS = {
    "name": ("longName", "shortName"),
    "sector": ("sector",),
    "industry": ("industry",),
    "market_cap": ("marketCap",),
    "price_info": ("currentPrice", "regularMarketPrice"),
    "high52_info": ("fiftyTwoWeekHigh",),
    "low52_info": ("fiftyTwoWeekLow",),
    "pe_trailing": ("trailingPE",),
    "pe_forward": ("forwardPE",),
    "eps_trailing": ("trailingEps", "epsTrailingTwelveMonths"),
    "eps_forward": ("forwardEps",),
    "dividend_rate": ("dividendRate", "trailingAnnualDividendRate"),
    "price_to_book": ("priceToBook",),
    "beta": ("beta",),
    "fin_currency": ("financialCurrency",),
}


def fetch_info_one(ticker: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            row = {}
            for k, keys in INFO_FIELDS.items():
                row[k] = next((info[x] for x in keys if info.get(x) not in (None, "", "Infinity")), None)
            if row["market_cap"] is None:  # lighter endpoint as a fallback
                try:
                    row["market_cap"] = tk.fast_info.get("market_cap")
                except Exception:
                    pass
            return row
        except Exception as e:
            msg = str(e).lower()
            wait = (2 ** attempt) * (3 if ("rate" in msg or "429" in msg or "too many" in msg) else 1)
            time.sleep(wait + random.random())
    return {}


def fetch_info(tickers: list[str], workers: int = 6) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_info_one, t): t for t in tickers}
        for n, f in enumerate(as_completed(futs), 1):
            out[futs[f]] = f.result()
            if n % 20 == 0 or n == len(tickers):
                ok = sum(1 for v in out.values() if v.get("market_cap"))
                print(f"  fundamentals {n}/{len(tickers)} ({ok} with market cap)")
    return out


def num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def build_dataset(tickers: list[str], workers: int) -> pd.DataFrame:
    print("Downloading price history...")
    hist = fetch_history(tickers)
    print("Downloading fundamentals (this is the slow part, ~1-3 min)...")
    info = fetch_info(tickers, workers)

    rows = []
    for t in tickers:
        h, i = hist.get(t, {}), info.get(t, {})
        if not h and not i.get("price_info"):
            continue
        price = num(i.get("price_info")) if not math.isnan(num(i.get("price_info"))) else num(h.get("price_hist"))
        hi = num(i.get("high52_info")) if not math.isnan(num(i.get("high52_info"))) else num(h.get("high52_hist"))
        lo = num(i.get("low52_info")) if not math.isnan(num(i.get("low52_info"))) else num(h.get("low52_hist"))
        rows.append({
            "ticker": t.replace(".AX", ""),
            "name": i.get("name") or t.replace(".AX", ""),
            "sector": i.get("sector") or "Unclassified",
            "industry": i.get("industry") or "",
            "price": price,
            "low_52w": lo,
            "high_52w": hi,
            "market_cap": num(i.get("market_cap")),
            "pe_trailing": num(i.get("pe_trailing")),
            "pe_forward": num(i.get("pe_forward")),
            "eps_trailing": num(i.get("eps_trailing")),
            "eps_forward": num(i.get("eps_forward")),
            "dividend_rate": num(i.get("dividend_rate")),
            "price_to_book": num(i.get("price_to_book")),
            "beta": num(i.get("beta")),
            "fin_currency": i.get("fin_currency") or "",
            "ret_1y": num(h.get("ret_1y")),
        })
    df = pd.DataFrame(rows)
    return add_derived(df)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rng = (df["high_52w"] - df["low_52w"]).replace(0, np.nan)
    df["range_pos"] = ((df["price"] - df["low_52w"]) / rng * 100).clip(0, 100)
    df["pct_from_high"] = (df["price"] / df["high_52w"] - 1) * 100
    df["pct_above_low"] = (df["price"] / df["low_52w"] - 1) * 100
    # Fill P/E from price/EPS when Yahoo omits it (only meaningful for positive EPS)
    missing = df["pe_trailing"].isna() & (df["eps_trailing"] > 0) & (df["fin_currency"].isin(["AUD", ""]))
    df.loc[missing, "pe_trailing"] = df.loc[missing, "price"] / df.loc[missing, "eps_trailing"]
    # Earnings yield = E/P (%). Works for loss-makers (negative) where P/E breaks down.
    df["earnings_yield"] = np.where(df["pe_trailing"] > 0, 100 / df["pe_trailing"],
                                    df["eps_trailing"] / df["price"] * 100)
    df.loc[df["earnings_yield"].abs() > 100, "earnings_yield"] = np.nan
    df["div_yield"] = df["dividend_rate"] / df["price"] * 100
    df.loc[df["div_yield"] > 30, "div_yield"] = np.nan
    return df


def save_dataset(df: pd.DataFrame, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"asx200_data_{date.today().isoformat()}")
    df.to_csv(stem + ".csv", index=False)
    try:
        df.to_excel(stem + ".xlsx", index=False)
    except Exception:
        pass  # openpyxl not installed - CSV is enough
    return stem + ".csv"


def load_latest(out_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(out_dir, "asx200_data_*.csv")))
    if not files:
        sys.exit(f"No cached data in {out_dir}/ - run without --cache first.")
    print(f"Using cached data: {files[-1]}")
    return add_derived(pd.read_csv(files[-1]))


# --------------------------------------------------------------------------------------
# 3. Visual system
# --------------------------------------------------------------------------------------

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e8e7e3"
BLUE, BLUE_LIGHT = "#2a78d6", "#b7d3f6"
# Diverging red <-> gray <-> blue (red = weak / near lows, blue = strong / near highs)
DIVERGING = [[0.0, "#9c2b2b"], [0.2, "#e34948"], [0.4, "#f2b8b5"], [0.5, "#ecebe7"],
             [0.6, "#b7d3f6"], [0.8, "#3987e5"], [1.0, "#104281"]]
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

HOVER = ("<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
         "%{customdata[2]}<br>"
         "Price $%{customdata[3]:.2f}  (52w $%{customdata[4]:.2f} – $%{customdata[5]:.2f})<br>"
         "Position in 52w range: %{customdata[6]:.0f}%<br>"
         "Mkt cap: %{customdata[7]}<br>"
         "P/E %{customdata[8]} · Fwd P/E %{customdata[9]} · EPS %{customdata[10]}<br>"
         "1-yr return: %{customdata[11]}<extra></extra>")


def fmt_cap(v):
    if pd.isna(v):
        return "n/a"
    return f"A${v / 1e9:,.1f}b" if v >= 1e9 else f"A${v / 1e6:,.0f}m"


def fmt(v, d=1, suffix=""):
    return "n/a" if pd.isna(v) else f"{v:,.{d}f}{suffix}"


def customdata(d: pd.DataFrame) -> np.ndarray:
    return np.stack([
        d["ticker"], d["name"], d["sector"], d["price"], d["low_52w"], d["high_52w"],
        d["range_pos"].fillna(-1), d["market_cap"].map(fmt_cap),
        d["pe_trailing"].map(lambda v: fmt(v)), d["pe_forward"].map(lambda v: fmt(v)),
        d["eps_trailing"].map(lambda v: fmt(v, 2)),
        d["ret_1y"].map(lambda v: fmt(v, 1, "%")),
    ], axis=-1)


def base_layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK2),
        margin=kw.pop("margin", dict(l=10, r=20, t=10, b=40)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=12, color=INK)),
        showlegend=False, **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", title_font=dict(color=INK2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="", title_font=dict(color=INK2))
    return fig


def beeswarm(values: np.ndarray, px_per_unit: float, dot_px: float, max_off_px: float) -> np.ndarray:
    """Greedy beeswarm: returns a vertical pixel offset per value so dots don't overlap."""
    order = np.argsort(values)
    placed: list[tuple[float, float]] = []
    offsets = np.zeros(len(values))
    for idx in order:
        x = values[idx] * px_per_unit
        near = [(px, py) for px, py in placed if abs(px - x) < dot_px]
        for k in range(0, 200):
            y = ((k + 1) // 2) * dot_px * (1 if k % 2 else -1) * 0.95
            if abs(y) > max_off_px:
                y = random.uniform(-max_off_px, max_off_px)
                break
            if all((px - x) ** 2 + (py - y) ** 2 >= (dot_px * 0.95) ** 2 for px, py in near):
                break
        placed.append((x, y))
        offsets[idx] = y
    return offsets


# --------------------------------------------------------------------------------------
# 4. Charts
# --------------------------------------------------------------------------------------

TREEMAP_STATES: dict = {}


def chart_treemap(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["market_cap"]).copy()
    d = d[d["market_cap"] > 0]
    sectors = d.groupby("sector")["market_cap"].sum().sort_values(ascending=False)

    def wavg(s, col):
        g = d[d["sector"] == s].dropna(subset=[col])
        return float(np.average(g[col], weights=g["market_cap"])) if len(g) else np.nan

    signed = lambda v: "n/a" if pd.isna(v) else f"{v:+.0f}%"
    metrics = {  # key: (column, cmin, cmax, cmid, label formatter)
        "range_pos": ("range_pos", 0, 100, 50, lambda v: fmt(v, 0, "%")),
        "ret_1y": ("ret_1y", -50, 50, 0, signed),
        "earnings_yield": ("earnings_yield", -8, 12, 0, lambda v: fmt(v, 1, "%")),
    }
    ids = ["ASX 200"] + [f"s:{s}" for s in sectors.index] + list(d["ticker"])
    labels = ["ASX 200"] + list(sectors.index) + list(d["ticker"])
    parents = [""] + ["ASX 200"] * len(sectors) + [f"s:{s}" for s in d["sector"]]
    values = [sectors.sum()] + list(sectors.values) + list(d["market_cap"])

    def colors_text(col, fmtf, mid):
        c = [np.nan] + [wavg(s, col) for s in sectors.index] + list(d[col])
        c = [mid if pd.isna(v) else v for v in c]
        t = [""] + [f"{fmtf(v)} (cap-wtd)" for v in c[1:1 + len(sectors)]] + [fmtf(v) for v in d[col]]
        return c, t

    hover_names = ["ASX 200"] + list(sectors.index) + [f"{r.name}<br>{r.industry}" for r in d.itertuples()]
    caps = [fmt_cap(v) for v in values]
    extra = [""] * (1 + len(sectors)) + [
        f"P/E {fmt(r.pe_trailing)} · 52w pos {fmt(r.range_pos, 0, '%')} · 1-yr {fmt(r.ret_1y, 0, '%')}"
        for r in d.itertuples()]

    first = list(metrics.values())[0]
    c0, t0 = colors_text(first[0], first[4], first[3])
    fig = go.Figure(go.Treemap(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        text=t0, texttemplate="<b>%{label}</b><br>%{text}", textposition="middle center",
        customdata=np.stack([hover_names, caps, extra], axis=-1),
        hovertemplate="<b>%{label}</b> · %{customdata[0]}<br>Mkt cap %{customdata[1]} "
                      "(%{percentRoot:.1%} of index)<br>%{text}<br>%{customdata[2]}<extra></extra>",
        marker=dict(colors=c0, colorscale=DIVERGING, cmin=first[1], cmax=first[2], cmid=first[3],
                    line=dict(color=SURFACE, width=2), pad=dict(t=22, l=2, r=2, b=2),
                    colorbar=dict(thickness=10, len=0.6, outlinewidth=0, ticksuffix="%",
                                  tickfont=dict(color=INK2))),
        tiling=dict(pad=2), maxdepth=3, root=dict(color=SURFACE),
        pathbar=dict(visible=True, thickness=22),
        insidetextfont=dict(family=FONT, color=INK),
    ))
    # Colour/label states for each factor; the page's own factor buttons switch between them
    TREEMAP_STATES.clear()
    for key, (col, lo, hi, mid, f) in metrics.items():
        c, t = colors_text(col, f, mid)
        TREEMAP_STATES[key] = {"colors": [None if pd.isna(v) else float(v) for v in c], "text": t,
                               "cmin": lo, "cmax": hi, "cmid": mid}
    base_layout(fig, 720, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def sector_rows(d: pd.DataFrame, col: str, ascending=True) -> list[str]:
    return list(d.groupby("sector")[col].median().sort_values(ascending=ascending).index)


def chart_range_swarm(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["range_pos"]).copy()
    rows = sector_rows(d, "range_pos")
    row_px, dot, plot_w = 64, 9, 1050
    d["y"] = 0.0
    for i, s in enumerate(rows):
        m = d["sector"] == s
        off = beeswarm(d.loc[m, "range_pos"].to_numpy(), plot_w / 100, dot + 1, row_px * 0.46)
        d.loc[m, "y"] = i + off / row_px
    fig = go.Figure()
    for x, lab in [(10, "near 52-wk low"), (90, "near 52-wk high")]:
        fig.add_vrect(x0=0 if x == 10 else 90, x1=10 if x == 10 else 100, fillcolor="#f3f2ee",
                      line_width=0, layer="below")
        fig.add_annotation(x=x - 5 if x == 10 else x + 5, y=len(rows) - 0.35, text=lab, showarrow=False,
                           font=dict(size=11, color=MUTED))
    fig.add_trace(go.Scatter(
        x=d["range_pos"], y=d["y"], mode="markers", customdata=customdata(d), hovertemplate=HOVER,
        marker=dict(size=dot, color=d["range_pos"], colorscale=DIVERGING, cmin=0, cmax=100,
                    line=dict(color=SURFACE, width=1.5))))
    medians = d.groupby("sector")["range_pos"].median()
    fig.add_trace(go.Scatter(x=[medians[s] for s in rows], y=list(range(len(rows))), mode="markers",
                             marker=dict(symbol="line-ns", size=row_px * 0.8, line=dict(width=2, color=INK)),
                             hovertemplate="%{text} median: %{x:.0f}%<extra></extra>", text=rows))
    counts = d["sector"].value_counts()
    base_layout(fig, row_px * len(rows) + 90, margin=dict(l=10, r=20, t=24, b=50))
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=[f"<b>{s}</b>  ({counts[s]})" for s in rows],
                     range=[-0.6, len(rows) - 0.15], showgrid=False, tickfont=dict(color=INK))
    fig.update_xaxes(range=[-2, 102], ticksuffix="%", dtick=10,
                     title="Where the price sits between its 52-week low (0%) and high (100%)")
    for i in range(len(rows)):
        fig.add_hline(y=i + 0.5, line=dict(color=GRID, width=1))
    return fig


def loss_mask(df: pd.DataFrame) -> pd.Series:
    """Companies with no positive P/E because they lost money over the last 12 months."""
    return ~(df["pe_trailing"] > 0) & ((df["eps_trailing"] < 0) | (df["earnings_yield"] < 0))


def loss_makers_html(df: pd.DataFrame) -> str:
    """Expandable list of the companies left off the P/E chart, grouped by sector."""
    from html import escape
    loss = df[loss_mask(df)].copy()
    nodata = df[~(df["pe_trailing"] > 0) & ~loss_mask(df)]
    if loss.empty and nodata.empty:
        return ""
    loss["_cap"] = loss["market_cap"].fillna(0)
    order = loss.groupby("sector")["_cap"].sum().sort_values(ascending=False).index
    groups = []
    for sec in order:
        g = loss[loss["sector"] == sec].sort_values("_cap", ascending=False)
        items = "".join(
            f'<li title="{escape(str(r.industry))}"><b>{escape(r.ticker)}</b><span class="nm">{escape(str(r.name))}</span>'
            f'<span class="v">EPS {fmt(r.eps_trailing, 2)} · {fmt_cap(r.market_cap)} · '
            f'1-yr <span class="{"neg" if r.ret_1y < 0 else "pos"}">{fmt(r.ret_1y, 0, "%")}</span></span></li>'
            for r in g.itertuples())
        groups.append(f'<div class="lmg"><h4>{escape(sec)} <span>({len(g)})</span></h4><ul>{items}</ul></div>')
    note = ""
    if len(nodata):
        tick = ", ".join(escape(t) for t in nodata.sort_values("market_cap", ascending=False)["ticker"])
        note = (f'<p class="rnote">{len(nodata)} more have no earnings figure from Yahoo, so they are also left off: '
                f'{tick}.</p>')
    return (f'<details class="lm"><summary>Show the {len(loss)} loss-making companies not plotted</summary>'
            f'<p class="rnote">Negative earnings per share over the last 12 months, grouped by sector and sorted by '
            f'market cap. Hover a name for its industry.</p><div class="lmgrid">{"".join(groups)}</div>{note}</details>')


def chart_pe_swarm(df: pd.DataFrame) -> go.Figure:
    d = df[(df["pe_trailing"] > 0)].copy()
    d["pe_plot"] = d["pe_trailing"].clip(2, 150)
    rows = sector_rows(d, "pe_trailing", ascending=False)
    row_px, dot, plot_w = 58, 9, 1050
    lo, hi = math.log10(2), math.log10(150)
    d["y"] = 0.0
    for i, s in enumerate(rows):
        m = d["sector"] == s
        off = beeswarm(np.log10(d.loc[m, "pe_plot"].to_numpy()), plot_w / (hi - lo), dot + 1, row_px * 0.46)
        d.loc[m, "y"] = i + off / row_px
    neg = df[loss_mask(df)].groupby("sector").size()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["pe_plot"], y=d["y"], mode="markers", customdata=customdata(d),
                             hovertemplate=HOVER,
                             marker=dict(size=dot, color=BLUE, opacity=0.85, line=dict(color=SURFACE, width=1.5))))
    med = d.groupby("sector")["pe_trailing"].median()
    fig.add_trace(go.Scatter(x=[med[s] for s in rows], y=list(range(len(rows))), mode="markers",
                             marker=dict(symbol="line-ns", size=row_px * 0.8, line=dict(width=2, color=INK)),
                             text=rows, hovertemplate="%{text} median P/E: %{x:.1f}<extra></extra>"))
    idx_med = d["pe_trailing"].median()
    fig.add_vline(x=idx_med, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=math.log10(idx_med), y=len(rows) - 0.3, text=f"index median {idx_med:.1f}x",
                       showarrow=False, font=dict(size=11, color=MUTED), xanchor="left", xshift=4)
    counts = d["sector"].value_counts()
    ticktext = [f"<b>{s}</b>  ({counts[s]}" + (f", {int(neg.get(s, 0))} loss-making)" if neg.get(s, 0) else ")")
                for s in rows]
    base_layout(fig, row_px * len(rows) + 90, margin=dict(l=10, r=20, t=24, b=50))
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=ticktext, range=[-0.6, len(rows) - 0.1],
                     showgrid=False, tickfont=dict(color=INK))
    fig.update_xaxes(type="log", range=[lo, hi], tickvals=[2, 5, 10, 15, 20, 30, 50, 100, 150],
                     ticksuffix="x", title="Trailing P/E (log scale; values above 150x pinned to the edge)")
    for i in range(len(rows)):
        fig.add_hline(y=i + 0.5, line=dict(color=GRID, width=1))
    return fig


def bubble_size(cap: pd.Series, lo=6, hi=46) -> np.ndarray:
    s = np.sqrt(cap.fillna(cap.min()).to_numpy())
    return lo + (s - s.min()) / (s.max() - s.min() + 1e-9) * (hi - lo)


def label_top(fig, d, x, y, n=12):
    for r in d.nlargest(n, "market_cap").itertuples():
        fig.add_annotation(x=getattr(r, x), y=getattr(r, y), text=r.ticker, showarrow=False,
                           font=dict(size=10, color=INK), yshift=0)


def chart_value_momentum(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["pct_from_high", "earnings_yield", "market_cap"]).copy()
    d["ey_plot"] = d["earnings_yield"].clip(-20, 20)
    d = d.sort_values("market_cap", ascending=False)  # big bubbles drawn first, small on top
    mx, my = d["pct_from_high"].median(), d["earnings_yield"].median()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["pct_from_high"], y=d["ey_plot"], mode="markers", customdata=customdata(d), hovertemplate=HOVER,
        marker=dict(size=bubble_size(d["market_cap"]), color=BLUE, opacity=0.55,
                    line=dict(color=SURFACE, width=1.5))))
    fig.add_hline(y=my, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_vline(x=mx, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color=INK2, width=1))
    xmin = max(d["pct_from_high"].min() - 3, -90)
    quad = [(xmin, 19, "left", "top", "Cheap & beaten down"), (0, 19, "right", "top", "Cheap & near highs"),
            (xmin, -19, "left", "bottom", "Pricey / loss-making & beaten down"),
            (0, -19, "right", "bottom", "Pricey / loss-making & near highs")]
    for x, y, xa, ya, t in quad:
        fig.add_annotation(x=x, y=y, text=f"<b>{t}</b>", showarrow=False, xanchor=xa, yanchor=ya,
                           font=dict(size=11, color=MUTED))
    label_top(fig, d, "pct_from_high", "ey_plot")
    base_layout(fig, 640)
    fig.update_xaxes(title="Distance below 52-week high", ticksuffix="%", range=[xmin, 2])
    fig.update_yaxes(title="Earnings yield (E/P, %)", ticksuffix="%", range=[-21, 21])
    return fig


def chart_pe_outlook(df: pd.DataFrame) -> go.Figure:
    d = df[(df["pe_trailing"] > 0) & (df["pe_forward"] > 0)].dropna(subset=["market_cap"]).copy()
    d["t"] = d["pe_trailing"].clip(3, 150)
    d["f"] = d["pe_forward"].clip(3, 150)
    d = d.sort_values("market_cap", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[3, 150], y=[3, 150], mode="lines", hoverinfo="skip",
                             line=dict(color=MUTED, width=1, dash="dot")))
    fig.add_trace(go.Scatter(
        x=d["t"], y=d["f"], mode="markers", customdata=customdata(d), hovertemplate=HOVER,
        marker=dict(size=bubble_size(d["market_cap"], 6, 38), color=BLUE, opacity=0.55,
                    line=dict(color=SURFACE, width=1.5))))
    label_top(fig, d, "t", "f", 10)
    fig.add_annotation(x=math.log10(120), y=math.log10(8), text="<b>Earnings expected to grow</b><br>"
                       "(forward P/E below trailing)", showarrow=False, xanchor="right",
                       font=dict(size=11, color=MUTED))
    fig.add_annotation(x=math.log10(4), y=math.log10(100), text="<b>Earnings expected to fall</b>",
                       showarrow=False, xanchor="left", font=dict(size=11, color=MUTED))
    base_layout(fig, 560)
    tv = [3, 5, 10, 20, 50, 100, 150]
    fig.update_xaxes(type="log", range=[math.log10(3), math.log10(150)], tickvals=tv, ticksuffix="x",
                     title="Trailing P/E")
    fig.update_yaxes(type="log", range=[math.log10(3), math.log10(150)], tickvals=tv, ticksuffix="x",
                     title="Forward P/E")
    return fig


def chart_concentration(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["market_cap"]).sort_values("market_cap", ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1
    d["cum"] = d["market_cap"].cumsum() / d["market_cap"].sum() * 100
    fig = go.Figure(go.Scatter(
        x=d["rank"], y=d["cum"], mode="lines", fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
        line=dict(color=BLUE, width=2), customdata=np.stack([d["ticker"], d["name"], d["market_cap"].map(fmt_cap)], -1),
        hovertemplate="#%{x} %{customdata[0]} · %{customdata[1]} (%{customdata[2]})<br>"
                      "Top %{x} = %{y:.1f}% of total cap<extra></extra>"))
    for n in (10, 20, 50):
        if n <= len(d):
            y = d.loc[n - 1, "cum"]
            fig.add_trace(go.Scatter(x=[n], y=[y], mode="markers", hoverinfo="skip",
                                     marker=dict(size=8, color=BLUE, line=dict(color=SURFACE, width=2))))
            fig.add_annotation(x=n, y=y, text=f"Top {n}: <b>{y:.0f}%</b>", showarrow=False, xanchor="left",
                               yanchor="top", xshift=6, yshift=-2, font=dict(size=12, color=INK))
    base_layout(fig, 400)
    fig.update_xaxes(title="Companies ranked by market cap", range=[0, len(d) + 2])
    fig.update_yaxes(title="Cumulative share of total market cap", ticksuffix="%", range=[0, 102])
    return fig


# --------------------------------------------------------------------------------------
# 5. HTML dashboard
# --------------------------------------------------------------------------------------

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--surface:#fcfcfb;--page:#f5f4f1;--line:#e8e7e3;--blue:#2a78d6}
*{box-sizing:border-box}html{color-scheme:light}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:28px 16px 60px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}.sub{color:var(--ink2);margin:0 0 22px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:22px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font-size:24px;font-weight:650;letter-spacing:-.01em}.kpi .l{color:var(--ink2);font-size:12.5px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 18px 8px;margin-bottom:18px}
.card h2{font-size:16px;margin:0 0 2px}.card p.d{color:var(--ink2);margin:0 0 8px;font-size:13px;max-width:880px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:980px){.grid2{grid-template-columns:1fr}}
.controls{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 12px}
.controls input,.controls select{font:inherit;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink)}
.controls input{flex:1;min-width:200px}
.tbl{overflow:auto;max-height:720px;border-top:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th{position:sticky;top:0;background:var(--surface);text-align:right;font-weight:600;color:var(--ink2);padding:8px 10px;border-bottom:1px solid var(--line);cursor:pointer;white-space:nowrap;user-select:none}
th.l,td.l{text-align:left}th:hover{color:var(--ink)}th.s::after{content:' ▾'}th.s.asc::after{content:' ▴'}
td{padding:6px 10px;border-bottom:1px solid #f0efeb;text-align:right;white-space:nowrap}
tr:hover td{background:#f7f6f3}td.nm{max-width:220px;overflow:hidden;text-overflow:ellipsis;color:var(--ink2)}
.rb{position:relative;width:120px;height:6px;border-radius:3px;background:linear-gradient(90deg,#f2b8b5,#ecebe7 50%,#b7d3f6);display:inline-block;vertical-align:middle}
.rb i{position:absolute;top:-3px;width:4px;height:12px;border-radius:2px;background:var(--ink);transform:translateX(-2px)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:9px;padding:3px;background:#fff;gap:2px;flex-wrap:wrap}
.seg button{font:inherit;font-size:13px;border:0;background:none;color:var(--ink2);padding:6px 12px;border-radius:6px;cursor:pointer}
.seg button:hover{color:var(--ink)}.seg button[aria-selected=true]{background:#eaf2fc;color:#184f95;font-weight:600}
.seg.sm button{font-size:12px;padding:4px 9px}
.fdesc{margin:10px 0 12px;color:var(--ink);font-size:13.5px;max-width:880px}
.mapwrap{display:flex;gap:16px;align-items:stretch}.map{flex:1;min-width:0}
.rank{width:330px;flex:none;display:flex;flex-direction:column;border-left:1px solid var(--line);padding-left:16px;height:720px}
@media(max-width:980px){.mapwrap{flex-direction:column}.rank{width:auto;border-left:0;padding-left:0;height:520px}}
.rhead{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.rhead span{font-weight:600;font-size:13px}
.rlist{overflow:auto;flex:1;padding-right:4px}
.rk{display:grid;grid-template-columns:30px 46px 1fr 58px;column-gap:6px;align-items:center;padding:5px 2px 6px;border-bottom:1px solid #f0efeb;font-size:12.5px;font-variant-numeric:tabular-nums}
.rk .n{color:var(--muted);text-align:right}.rk .nm{color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rk .v{text-align:right;font-weight:600}.rk .bt{grid-column:2/5;height:5px;position:relative;background:#f3f2ee;border-radius:3px;margin-top:4px}
.rk .bt i{position:absolute;top:0;bottom:0;border-radius:3px}.rk .bt b{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--muted)}
.rnote{color:var(--muted);font-size:12px;padding-top:6px}
.lm{margin:4px 0 12px;border-top:1px solid var(--line);padding-top:10px}
.lm summary{cursor:pointer;font-weight:600;font-size:13.5px;color:#184f95;list-style:none;display:inline-flex;align-items:center;gap:6px}
.lm summary::-webkit-details-marker{display:none}.lm summary::before{content:'▸';transition:transform .15s}
.lm[open] summary::before{transform:rotate(90deg)}
.lmgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px 22px;margin-top:6px}
.lmg h4{margin:6px 0 4px;font-size:13px}.lmg h4 span{color:var(--muted);font-weight:400}
.lmg ul{list-style:none;margin:0;padding:0}
.lmg li{display:grid;grid-template-columns:44px 1fr auto;gap:8px;align-items:baseline;padding:4px 0;border-bottom:1px solid #f0efeb;font-size:12.5px;font-variant-numeric:tabular-nums}
.lmg li .nm{color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.lmg li .v{color:var(--ink2);white-space:nowrap}
@media(max-width:520px){.lmgrid{grid-template-columns:1fr}.lmg li{grid-template-columns:44px 1fr}.lmg li .v{grid-column:2}}
.neg{color:#b3261e}.pos{color:#1c5cab}.foot{color:var(--muted);font-size:12px;margin-top:18px}
"""

FACTOR_JS = """
const TM = __TM__, STOPS = __DIV__;
const FACTORS = {
 range_pos:{label:'52-week position', title:'52-week position',
   desc:"Where today's price sits between the stock's lowest (0%) and highest (100%) price of the past 52 weeks.",
   asc:'Nearest 52-wk low', desc2:'Nearest 52-wk high', f:v=>v.toFixed(0)+'%'},
 ret_1y:{label:'1-year return', title:'1-year price return',
   desc:'How much the share price has risen or fallen over the past 12 months (price only, excluding dividends).',
   asc:'Worst first', desc2:'Best first', f:v=>(v>0?'+':'')+v.toFixed(1)+'%'},
 earnings_yield:{label:'Earnings yield', title:'Earnings yield',
   desc:'Annual profit per share as a percentage of the share price (the inverse of P/E): higher means cheaper for each dollar of earnings, negative means loss-making.',
   asc:'Lowest first', desc2:'Highest first', f:v=>v.toFixed(1)+'%'}};
let factor='range_pos', dir='asc';
function hex(h){return [1,3,5].map(i=>parseInt(h.slice(i,i+2),16))}
function colorAt(t){t=Math.max(0,Math.min(1,t));for(let i=1;i<STOPS.length;i++){const[a,ca]=STOPS[i-1],[b,cb]=STOPS[i];
  if(t<=b){const k=(t-a)/(b-a),x=hex(ca),y=hex(cb);return `rgb(${x.map((v,j)=>Math.round(v+(y[j]-v)*k)).join(',')})`}}return STOPS.at(-1)[1]}
function norm(v,s){return v<s.cmid?0.5*(v-s.cmin)/(s.cmid-s.cmin):0.5+0.5*(v-s.cmid)/(s.cmax-s.cmid)}
const segF=document.getElementById('factors');
segF.innerHTML=Object.entries(FACTORS).map(([k,F])=>`<button role="tab" data-f="${k}">${F.label}</button>`).join('');
segF.addEventListener('click',e=>{const b=e.target.closest('button');if(b){factor=b.dataset.f;update(true)}});
document.getElementById('rdir').addEventListener('click',e=>{const b=e.target.closest('button');if(b){dir=b.dataset.d;update(false)}});
function update(restyle){
  const F=FACTORS[factor],S=TM[factor];
  segF.querySelectorAll('button').forEach(b=>b.setAttribute('aria-selected',b.dataset.f===factor));
  document.querySelectorAll('#rdir button').forEach(b=>{b.textContent=b.dataset.d==='asc'?F.asc:F.desc2;b.setAttribute('aria-selected',b.dataset.d===dir)});
  document.getElementById('fdesc').innerHTML=`<b>${F.title}:</b> ${F.desc}`;
  document.getElementById('rtitle').textContent='Ranked by '+F.label.toLowerCase();
  const el=document.getElementById('fig-treemap');
  if(restyle&&el&&window.Plotly&&S)Plotly.restyle(el,{'marker.colors':[S.colors],text:[S.text],'marker.cmin':S.cmin,'marker.cmax':S.cmax,'marker.cmid':S.cmid},[0]);
  const rows=DATA.filter(r=>isNum(r[factor])).sort((a,b)=>(a[factor]-b[factor])*(dir==='asc'?1:-1));
  const lo=S?S.cmin:0,hi=S?S.cmax:100,mid=S?S.cmid:50,pos=v=>(Math.max(lo,Math.min(hi,v))-lo)/(hi-lo)*100,m=pos(mid);
  document.getElementById('rlist').innerHTML=rows.map((r,i)=>{const v=r[factor],p=pos(v),c=colorAt(norm(Math.max(lo,Math.min(hi,v)),S||{cmin:0,cmax:100,cmid:50}));
    const l=Math.min(p,m),w=Math.max(Math.abs(p-m),1.2);
    return `<div class="rk" title="${r.name} · ${r.sector}"><span class="n">${i+1}</span><b>${r.ticker}</b><span class="nm">${r.name}</span><span class="v">${F.f(v)}</span>`+
      `<span class="bt"><i style="left:${factor==='range_pos'?0:l}%;width:${factor==='range_pos'?Math.max(p,1.2):w}%;background:${c}"></i>${factor==='range_pos'?'':`<b style="left:${m}%"></b>`}</span></div>`}).join('');
  const miss=DATA.length-rows.length;
  document.getElementById('rnote').textContent=rows.length+' stocks'+(miss?` · ${miss} without data not shown`:'')+(factor==='range_pos'?'':' · grey tick = zero');
  document.getElementById('rlist').scrollTop=0;
}
update(true);
// the map is drawn before the ranking panel exists, so re-fit it to its column
const tmEl=document.getElementById('fig-treemap');
if(tmEl&&window.Plotly){Plotly.Plots.resize(tmEl);window.addEventListener('load',()=>Plotly.Plots.resize(tmEl))}
"""

TABLE_JS = """
const DATA = __DATA__;
const COLS = [
 ['ticker','Ticker','l',v=>`<b>${v}</b>`],['name','Company','l nm',v=>v],['sector','Sector','l',v=>v],
 ['price','Price','',v=>fx(v,2,'$')],['low_52w','52w low','',v=>fx(v,2,'$')],
 ['range_pos','52-week range','',(v,r)=>isNum(v)?`<span class="rb" title="${v.toFixed(0)}% of range"><i style="left:${v}%"></i></span>`:'–'],
 ['high_52w','52w high','',v=>fx(v,2,'$')],['range_pos','Pos.','',v=>fx(v,0,'','%')],
 ['pct_from_high','From high','',v=>sg(v)],['ret_1y','1-yr','',v=>sg(v)],
 ['market_cap','Mkt cap','',v=>isNum(v)?(v>=1e9?'$'+(v/1e9).toFixed(1)+'b':'$'+(v/1e6).toFixed(0)+'m'):'–'],
 ['pe_trailing','P/E','',v=>fx(v,1)],['pe_forward','Fwd P/E','',v=>fx(v,1)],
 ['eps_trailing','EPS','',v=>fx(v,2)],['div_yield','Div yld','',v=>fx(v,1,'','%')]];
function isNum(v){return v!==null&&v!==undefined&&!Number.isNaN(v)}
function fx(v,d,p='',s=''){return isNum(v)?p+v.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d})+s:'–'}
function sg(v){return isNum(v)?`<span class="${v<0?'neg':'pos'}">${v>0?'+':''}${v.toFixed(1)}%</span>`:'–'}
let sortKey='market_cap',sortCol=10,asc=false;
const thead=document.querySelector('#t thead'),tbody=document.querySelector('#t tbody');
thead.innerHTML='<tr>'+COLS.map((c,i)=>`<th class="${c[2].split(' ')[0]}" data-i="${i}">${c[1]}</th>`).join('')+'</tr>';
const q=document.getElementById('q'),sec=document.getElementById('sec');
[...new Set(DATA.map(r=>r.sector))].sort().forEach(s=>sec.add(new Option(s,s)));
function render(){
  const term=q.value.trim().toLowerCase(),sv=sec.value;
  let rows=DATA.filter(r=>(!sv||r.sector===sv)&&(!term||(r.ticker+' '+r.name+' '+r.industry).toLowerCase().includes(term)));
  rows.sort((a,b)=>{const x=a[sortKey],y=b[sortKey];
    if(!isNum(x)&&typeof x!=='string')return 1; if(!isNum(y)&&typeof y!=='string')return -1;
    return (typeof x==='string'?x.localeCompare(y):x-y)*(asc?1:-1)});
  tbody.innerHTML=rows.map(r=>'<tr>'+COLS.map(c=>`<td class="${c[2]}">${c[3](r[c[0]],r)}</td>`).join('')+'</tr>').join('');
  thead.querySelectorAll('th').forEach((th,i)=>{th.classList.toggle('s',i===sortCol);th.classList.toggle('asc',i===sortCol&&asc)});
  document.getElementById('cnt').textContent=rows.length+' of '+DATA.length+' stocks';
}
thead.addEventListener('click',e=>{const th=e.target.closest('th');if(!th)return;const i=+th.dataset.i;
  if(i===sortCol)asc=!asc;else{sortCol=i;sortKey=COLS[i][0];asc=typeof DATA[0][sortKey]==='string'}render()});
q.addEventListener('input',render);sec.addEventListener('change',render);render();
"""


def build_html(df: pd.DataFrame, path: str, source_note: str, cdn: bool = False) -> None:
    cfg = {"displaylogo": False, "responsive": True,
           "toImageButtonOptions": {"format": "png", "scale": 2}}
    builders = {
        "treemap": chart_treemap, "range": chart_range_swarm, "pe": chart_pe_swarm,
        "vm": chart_value_momentum, "outlook": chart_pe_outlook, "conc": chart_concentration,
    }
    div = {}
    for k, fn in builders.items():  # one bad chart (e.g. too little data) shouldn't sink the page
        try:
            div[k] = fn(df).to_html(full_html=False, include_plotlyjs=False, config=cfg, div_id=f"fig-{k}")
        except Exception as e:
            print(f"  warning: chart '{k}' skipped ({e.__class__.__name__}: {e})")
            div[k] = '<p class="d">Not enough data to draw this chart today.</p>'
    if cdn:  # small file for hosting: load plotly.js from a CDN
        plotly_tag = (f'<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@'
                      f'{get_plotlyjs_version()}/plotly.min.js"></script>')
    else:        # fully offline single file
        plotly_tag = f"<script>{get_plotlyjs()}</script>"

    cap = df["market_cap"].sum()
    kpis = [
        (f"{len(df)}", "stocks with data"),
        (f"A${cap / 1e12:.2f}tn" if cap >= 1e12 else f"A${cap / 1e9:,.0f}b", "combined market cap"),
        (f"{df.loc[df['pe_trailing'] > 0, 'pe_trailing'].median():.1f}x", "median trailing P/E"),
        (f"{(df['range_pos'] >= 90).sum()}", "within 10% of 52-wk high"),
        (f"{(df['range_pos'] <= 10).sum()}", "within 10% of 52-wk low"),
        (f"{(df['ret_1y'] > 0).sum()} / {df['ret_1y'].notna().sum()}", "up over the past year"),
    ]
    cols = ["ticker", "name", "sector", "industry", "price", "low_52w", "high_52w", "range_pos",
            "pct_from_high", "ret_1y", "earnings_yield", "market_cap", "pe_trailing", "pe_forward", "eps_trailing",
            "div_yield"]
    records = json.loads(df[cols].replace({np.nan: None}).to_json(orient="records"))

    def card(title, desc, key, extra=""):
        return f'<div class="card"><h2>{title}</h2><p class="d">{desc}</p>{div[key]}{extra}</div>'

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ASX 200 Dashboard</title>
<style>{CSS}</style>{plotly_tag}</head><body><div class="wrap">
<h1>ASX 200 market dashboard</h1>
<p class="sub">{source_note} · Hover any mark for full details · Camera icon on each chart saves a PNG</p>
<div class="kpis">{''.join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in kpis)}</div>
<div class="card"><h2>Market map</h2>
<p class="d">Every company as a tile sized by market cap and grouped by sector. Pick a factor to colour the map and
rank every stock by it. Click a sector to zoom in; click the bar above the map to zoom out.</p>
<div class="seg" id="factors" role="tablist"></div>
<p class="fdesc" id="fdesc"></p>
<div class="mapwrap"><div class="map">{div["treemap"]}</div>
<aside class="rank"><div class="rhead"><span id="rtitle"></span>
<div class="seg sm" id="rdir"><button data-d="asc"></button><button data-d="desc"></button></div></div>
<div class="rlist" id="rlist"></div><div class="rnote" id="rnote"></div></aside></div></div>
{card("Where every stock sits in its 52-week range", "Each dot is one company, placed between its 52-week low (0%) "
      "and high (100%). Sectors are sorted by their median (black tick): strongest at the top, weakest at the bottom.",
      "range")}
{card("Valuation spread by sector", "Trailing P/E for every profitable company on a log scale; black ticks mark each "
      "sector's median. Loss-making companies have no meaningful P/E: they're counted in each row label and listed "
      "below.", "pe", loss_makers_html(df))}
<div class="grid2">
{card("Value vs momentum", "Earnings yield (the inverse of P/E, so loss-makers still show) against how far each stock "
      "trades below its 52-week high. Bubble size = market cap; dotted lines are index medians.", "vm")}
{card("Earnings outlook", "Trailing vs forward P/E. Stocks below the diagonal are expected to earn more next year; "
      "above it, less. Bubble size = market cap.", "outlook")}
</div>
{card("How concentrated is the index?", "Cumulative share of total market cap as you add companies from largest "
      "to smallest.", "conc")}
<div class="card"><h2>All stocks</h2><p class="d">Click a column header to sort. The bar shows where today's price sits in the 52-week range.</p>
<div class="controls"><input id="q" placeholder="Search ticker, company or industry…"><select id="sec"><option value="">All sectors</option></select>
<span id="cnt" style="align-self:center;color:var(--ink2);font-size:13px"></span></div>
<div class="tbl"><table id="t"><thead></thead><tbody></tbody></table></div></div>
<p class="foot">Data: Yahoo Finance via yfinance. Figures can be delayed or incomplete; some dual-listed companies report
EPS in USD, so compare their P/E with care. Not financial advice.</p>
</div><script>{TABLE_JS.replace("__DATA__", json.dumps(records))}\n{FACTOR_JS.replace("__TM__", json.dumps(TREEMAP_STATES)).replace("__DIV__", json.dumps(DIVERGING))}</script></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# --------------------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="ASX 200 data fetch + dashboard")
    ap.add_argument("--cache", action="store_true", help="skip the download; use the latest saved CSV")
    ap.add_argument("--tickers", help="CSV with a 'ticker' column to use instead of the ASX 200 list")
    ap.add_argument("--workers", type=int, default=6, help="parallel requests for fundamentals (default 6)")
    ap.add_argument("--out", default="output", help="output folder (default ./output)")
    ap.add_argument("--no-open", action="store_true", help="don't open the dashboard in a browser")
    ap.add_argument("--cdn", action="store_true",
                    help="load the charting library from the web (≈200 KB file instead of ≈5 MB; for hosting)")
    a = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if a.cache:
        df = load_latest(a.out)
        note = "Cached data"
    else:
        tickers = get_constituents(a.tickers)
        df = build_dataset(tickers, a.workers)
        if df.empty:
            sys.exit("No data came back from Yahoo Finance - check your connection or try again later.")
        csv = save_dataset(df, a.out)
        print(f"Saved {len(df)} rows -> {csv}")
        missing = df["market_cap"].isna().sum()
        if missing:
            print(f"Note: {missing} stocks came back without fundamentals (likely rate-limited). "
                  f"Re-run later, or try --workers 3.")
        note = f"Data as of {date.today():%d %b %Y}"

    out_html = os.path.join(a.out, "asx200_dashboard.html")
    build_html(df, out_html, note, cdn=a.cdn)
    print(f"Dashboard -> {os.path.abspath(out_html)}")
    if not a.no_open:
        webbrowser.open("file://" + os.path.abspath(out_html))


if __name__ == "__main__":
    main()
