# ASX 200 market dashboard

**Live dashboard: https://gazza-2105.github.io/asx200-dashboard/**

An interactive dashboard covering every S&P/ASX 200 company. Data comes from Yahoo Finance
via [yfinance](https://github.com/ranaroussi/yfinance) and refreshes automatically every
weekday evening after the ASX closes.

## What it shows

| Chart | What it answers |
|---|---|
| Market map | Treemap of all 200 companies sized by market cap, grouped by sector; colour switches between 52-week position, 1-year return and earnings yield |
| 52-week range | Beeswarm showing where every stock sits between its 52-week low and high, by sector |
| Valuation spread | Trailing P/E for every profitable company, by sector, with medians |
| Value vs momentum | Earnings yield vs distance from 52-week high, in four quadrants |
| Earnings outlook | Trailing vs forward P/E: who is expected to grow earnings |
| Concentration | How much of the index the top 10 / 20 / 50 companies make up |
| Full table | Sortable, searchable table with inline 52-week range bars |

Metrics collected per stock: price, 52-week high/low, market cap, trailing and forward P/E,
EPS, dividend yield, 1-year return, sector and industry. The raw data is published alongside
the dashboard as [`asx200_data.csv`](https://gazza-2105.github.io/asx200-dashboard/asx200_data.csv).

## How it works

1. Pulls the current ASX 200 constituent list (Wikipedia, with a built-in fallback).
2. Bulk-downloads a year of daily prices, then fetches fundamentals per stock in parallel
   with retry and back-off to cope with Yahoo's rate limits.
3. Derives 52-week range position, drawdown from high, earnings yield and dividend yield.
4. Builds the charts with Plotly and writes a single self-contained HTML page.
5. A GitHub Actions workflow runs this on a schedule and deploys the page to GitHub Pages.

## Run it yourself

```bash
pip install -r requirements.txt
python asx200_dashboard.py            # fetch data + open dashboard
python asx200_dashboard.py --cache    # rebuild charts from the last download
```

*Data from Yahoo Finance; may be delayed or incomplete. Not financial advice.*
