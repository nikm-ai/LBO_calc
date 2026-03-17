import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy_financial as npf

# ── Pure-numpy Sobol low-discrepancy sequence (no scipy dependency) ────────
_SOBOL_DIRS = [
    None,
    [1],
    [1, 1],
    [1, 3, 7],
    [1, 1, 5, 13],
]

def _sobol_sequence(n: int, d: int, seed: int = 0) -> np.ndarray:
    BITS = 32
    rng  = np.random.default_rng(seed)
    V = np.zeros((d, BITS), dtype=np.uint32)
    for i in range(BITS):
        V[0, i] = np.uint32(1 << (BITS - 1 - i))
    poly_s  = [1, 1, 2, 3, 3]
    poly_a  = [0, 1, 1, 2, 1]
    m_init  = [
        [],
        [1],
        [1, 3],
        [1, 3, 1],
        [1, 1, 1, 3],
    ]
    for dim in range(1, min(d, 5)):
        s = poly_s[dim]
        a = poly_a[dim]
        m = list(m_init[dim])
        for i in range(s, BITS):
            v = m[i - s] ^ (m[i - s] >> np.uint32(s))
            for k in range(1, s):
                if (a >> np.uint32(s - 1 - k)) & 1:
                    v ^= m[i - k]
            m.append(int(v))
        for i in range(BITS):
            V[dim, i] = np.uint32(int(m[i]) << (BITS - 1 - i))
    X   = np.zeros(d, dtype=np.uint32)
    pts = np.empty((n, d), dtype=np.float64)
    scale = np.float64(2 ** BITS)
    for i in range(n):
        pts[i] = X.astype(np.float64) / scale
        c = int(int(~np.uint32(i) & np.uint32(i + 1)).bit_length()) - 1
        if c < BITS:
            for dim in range(d):
                X[dim] ^= V[dim, c]
    masks = rng.integers(0, 2**31, size=d, dtype=np.int64).astype(np.uint32)
    raw   = (pts * scale).astype(np.uint64)
    for dim in range(d):
        raw[:, dim] ^= masks[dim]
    pts = raw.astype(np.float64) / scale
    return np.clip(pts, 1e-10, 1 - 1e-10)

st.set_page_config(
    page_title="Leveraged Buyout Analysis: A Structured Returns Framework",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

  .block-container { padding-top: 3.5rem; padding-bottom: 4rem; max-width: 1200px; }
  [data-testid="stSidebar"] { display: none; }
  [data-testid="collapsedControl"] { display: none; }

  .paper-title {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 31px; font-weight: 500; line-height: 1.25;
    color: var(--text-color); margin-bottom: 0.4rem; letter-spacing: -0.01em;
  }
  .paper-byline {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px; color: var(--text-color); opacity: 0.5;
    margin-bottom: 1.5rem; letter-spacing: 0.01em;
  }
  .abstract-box {
    border-top: 1px solid rgba(128,128,128,0.25);
    border-bottom: 1px solid rgba(128,128,128,0.25);
    padding: 1.2rem 0; margin-bottom: 2rem;
  }
  .abstract-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.4; color: var(--text-color); margin-bottom: 7px;
  }
  .abstract-text {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15.5px; line-height: 1.8; color: var(--text-color); max-width: 900px;
  }
  .sec-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px; font-weight: 600; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.55;
    margin: 2.5rem 0 0.85rem; padding-bottom: 6px;
    border-bottom: 1px solid rgba(128,128,128,0.18);
  }
  .kpi-card {
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 3px; padding: 1rem 1.1rem;
    background: rgba(128,128,128,0.03);
  }
  .kpi-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 9px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.45; color: var(--text-color); margin-bottom: 6px;
  }
  .kpi-value {
    font-family: 'DM Sans', sans-serif;
    font-size: 26px; font-weight: 500; line-height: 1.1; color: var(--text-color);
  }
  .kpi-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px; margin-top: 4px; opacity: 0.55; color: var(--text-color);
  }
  .pos  { color: #2e7d4f !important; opacity: 1 !important; font-weight: 600; }
  .neg  { color: #b94040 !important; opacity: 1 !important; font-weight: 600; }
  .neut { color: #1a4f82 !important; opacity: 1 !important; font-weight: 600; }
  .warn { color: #c47a00 !important; opacity: 1 !important; font-weight: 600; }

  .fig-caption {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 13.5px; line-height: 1.75; color: var(--text-color);
    opacity: 0.8; margin-top: 0.1rem; margin-bottom: 1.5rem; font-style: italic;
  }
  .fig-caption b { font-style: normal; font-weight: 600; opacity: 1; color: var(--text-color); }

  .explainer-head {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 19px; font-weight: 500; color: var(--text-color);
    margin: 0.5rem 0 0.4rem; letter-spacing: -0.005em;
  }
  .explainer-body {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15px; line-height: 1.8; color: var(--text-color); opacity: 0.85;
    margin-bottom: 0.75rem;
  }

  /* QMC risk panel */
  .risk-banner {
    background: rgba(128,128,128,0.04);
    border: 1px solid rgba(128,128,128,0.12);
    border-radius: 3px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
  }
  .risk-banner-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.45; color: var(--text-color); margin-bottom: 0.6rem;
  }
  .risk-stat {
    display: inline-block; margin-right: 2rem;
  }
  .risk-stat-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 9px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; opacity: 0.4; color: var(--text-color);
  }
  .risk-stat-val {
    font-family: 'DM Sans', sans-serif;
    font-size: 18px; font-weight: 500; color: var(--text-color); display: block;
  }

  .scenario-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.05em;
    padding: 2px 8px; border-radius: 2px;
    background: rgba(26,79,130,0.12); color: #1a4f82;
    margin-right: 6px; vertical-align: middle;
  }

  .paper-footer {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 12px; color: var(--text-color); opacity: 0.35;
    margin-top: 4rem; padding-top: 1rem;
    border-top: 1px solid rgba(128,128,128,0.15); line-height: 1.7;
  }

  label, .stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 0.02em; opacity: 0.7;
  }

  .appendix-term {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px; font-weight: 600; letter-spacing: 0.03em;
    color: var(--text-color); margin-bottom: 2px;
  }
  .appendix-def {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 14.5px; line-height: 1.75; color: var(--text-color);
    opacity: 0.82; margin-bottom: 1rem;
  }
  .appendix-group-head {
    font-family: 'DM Sans', sans-serif;
    font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.35;
    margin: 1.8rem 0 0.8rem; padding-bottom: 4px;
    border-bottom: 1px solid rgba(128,128,128,0.12);
  }
  .appendix-note {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 13.5px; line-height: 1.8; color: var(--text-color);
    opacity: 0.7; font-style: italic; margin-top: 1.5rem;
  }

  /* Hide Streamlit manage-app button */
  [data-testid="manage-app-button"] { display: none !important; }
  .stAppDeployButton { display: none !important; }

  /* Restored original param-table styles */
  .param-table {
    width: 100%; border-collapse: collapse;
    font-family: 'DM Sans', sans-serif; font-size: 12px;
    margin-bottom: 0.5rem;
  }
  .param-table th {
    font-size: 9px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.4; color: var(--text-color);
    border-bottom: 1px solid rgba(128,128,128,0.2);
    padding: 6px 10px 6px 0; text-align: left;
  }
  .param-table td {
    padding: 5px 10px 5px 0;
    border-bottom: 1px solid rgba(128,128,128,0.07);
    color: var(--text-color); font-size: 12px; vertical-align: top;
  }
  .param-name  { font-weight: 500; }
  .param-def   { opacity: 0.6; font-size: 11px; font-family: 'EB Garamond', Georgia, serif; font-style: italic; }
  .param-value { font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)

# ── CHANGE 1: Dark chart theme ─────────────────────────────────────────────
# All charts now use a dark background that matches the Streamlit dark UI.
CHART_BG  = "#1c1c1e"          # near-black, consistent with Streamlit dark sidebar
CHART_TEXT = "#d4d0c8"          # warm off-white for axis labels
GRID_COLOR = "#2e2e32"          # subtle dark grid lines
LINE_COLOR = "#3a3a3e"          # axis line color
FONT_CH    = dict(size=12, color=CHART_TEXT, family="DM Sans, Arial, sans-serif")
LEGEND     = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                  font=dict(size=11, color=CHART_TEXT), bgcolor="rgba(0,0,0,0)")
BASE       = dict(plot_bgcolor=CHART_BG, paper_bgcolor=CHART_BG, font=FONT_CH,
                  margin=dict(l=8, r=8, t=20, b=8), legend=LEGEND)

BLUE   = "#4a8fd4"; LBLUE  = "#3d7ab5"; LLBLUE = "#1e3a5f"
GREEN  = "#4caf7a"; LGREEN = "#6ab06a"; LLGREEN= "#1a3d2a"
RED    = "#d45a5a"; LRED   = "#c47a7a"
AMBER  = "#d4a040"
GRAY   = "#888888"

SCEN_COLORS = ["#3d7ab5", "#4caf7a", "#d45a5a"]

def ax(title, grid=True):
    return dict(
        title=dict(text=title, font=dict(size=12, color=CHART_TEXT)),
        tickfont=dict(size=11, color=CHART_TEXT),
        gridcolor=GRID_COLOR if grid else "rgba(0,0,0,0)",
        linecolor=LINE_COLOR, linewidth=1, showline=True,
        showgrid=grid, zeroline=False, ticks="outside", ticklen=3,
    )

def fmt_m(v):
    if abs(v) >= 1000: return f"${v/1000:.2f}B"
    return f"${v:.1f}M"

def fmt_pct(v): return f"{v*100:.1f}%"

def irr_calc(cfs):
    try:    return npf.irr(cfs)
    except: return np.nan

def kde_curve(data, n_pts=300, bw_factor=1.0):
    data  = np.asarray(data)
    n     = len(data)
    std   = np.std(data)
    if std == 0 or n < 2:
        return np.array([]), np.array([])
    bw = bw_factor * 1.06 * std * n ** (-0.2)
    lo, hi = data.min() - 3 * bw, data.max() + 3 * bw
    x  = np.linspace(lo, hi, n_pts)
    z  = (x[:, None] - data[None, :]) / bw
    density = np.exp(-0.5 * z**2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    return x, density

# ══════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="paper-title">Leveraged Buyout Analysis: A Structured Returns Framework</div>
<div class="paper-byline">
  Interactive model for LBO transaction structuring, return attribution, sensitivity analysis, and Quasi-Monte Carlo risk simulation
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# EXPLAINER
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">What is a leveraged buyout?</div>', unsafe_allow_html=True)
st.markdown("""
<div class="abstract-box">
  <div class="abstract-label">Overview</div>
  <div class="abstract-text">
    A leveraged buyout (LBO) is an acquisition financed primarily with debt, leaving the buyer,
    typically a private equity sponsor, to contribute a relatively small equity check. The
    acquired company's own cash flows service the debt over the holding period. Upon exit, the
    sponsor receives the residual equity value after repaying outstanding debt. Leverage
    amplifies equity returns when the underlying business return exceeds the cost of debt,
    and destroys value when it does not.
  </div>
</div>
""", unsafe_allow_html=True)

col_e1, col_e2, col_e3 = st.columns(3)
with col_e1:
    st.markdown('<div class="explainer-head">The mechanics of leverage</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explainer-body">
      In a conventional acquisition, a buyer pays the full purchase price in cash or stock.
      In an LBO, the buyer finances 50 to 70% of the purchase price with debt secured against
      the target company's assets and cash flows, contributing only the remainder as equity.
      Because the debt claim is fixed, any improvement in enterprise value flows
      disproportionately to equity holders. A business purchased for $600M with $360M of debt
      and $240M of equity that exits at $700M returns $430M to equity holders: a 79% gain on
      a 17% increase in enterprise value.
    </div>""", unsafe_allow_html=True)
with col_e2:
    st.markdown('<div class="explainer-head">Sources of equity return</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explainer-body">
      Equity returns in an LBO come from three places. First, <em>EBITDA growth</em>: if the
      business earns more at exit, the enterprise value rises proportionally given a constant
      exit multiple. Second, <em>multiple expansion</em>: if the market prices the business
      at a higher earnings multiple at exit, enterprise value rises even without profit growth.
      This is the most volatile and least controllable return driver. Third, <em>debt paydown</em>:
      as free cash flow retires principal, a larger share of enterprise value accrues to equity
      holders at exit.
    </div>""", unsafe_allow_html=True)
with col_e3:
    st.markdown('<div class="explainer-head">Key metrics and thresholds</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explainer-body">
      PE investors evaluate transactions primarily on <em>IRR</em> and <em>MOIC</em>. An IRR
      above 20% is generally considered strong; below 15% is marginal for most institutional
      sponsors. A MOIC above 3.0x is excellent; below 2.0x is typically below hurdle.
      Leverage quality is assessed through <em>net leverage</em> (debt/EBITDA) and
      <em>interest coverage</em> (EBITDA/interest). Entry leverage above 7x EBITDA is
      considered aggressive, and coverage below 2x raises refinancing risk.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PARAMETERS — original slider/number_input widgets + styled HTML glossary
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Model parameters</div>', unsafe_allow_html=True)
st.markdown("""<div class="explainer-body" style="margin-bottom:1rem;">
  Adjust the parameters below to model different transaction structures. All figures in USD millions.
  The model recomputes all outputs, projections, and sensitivity tables in real time.
</div>""", unsafe_allow_html=True)

r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
with r1c1: entry_ev_ebitda = st.number_input("Entry EV / EBITDA (x)",  value=12.0, step=0.5,  min_value=4.0,  max_value=30.0)
with r1c2: ebitda_entry    = st.number_input("Entry EBITDA ($M)",       value=50.0, step=5.0,  min_value=1.0)
with r1c3: revenue_entry   = st.number_input("Entry revenue ($M)",      value=200.0, step=10.0, min_value=1.0)
with r1c4: debt_pct        = st.slider("Debt / EV (%)",                  30, 80, 60, step=5)
with r1c5: interest_rate   = st.number_input("Interest rate (%)",        value=7.0,  step=0.25, min_value=1.0,  max_value=20.0)

r2c1, r2c2, r2c3, r2c4, r2c5, r2c6 = st.columns(6)
with r2c1: rev_growth       = st.slider("Revenue growth (%/yr)",          0, 30, 8,   step=1)
with r2c2: margin_expansion = st.slider("Margin expansion (bps/yr)",    -100, 200, 50, step=10)
with r2c3: capex_pct        = st.slider("CapEx (% of revenue)",           1, 15, 4,   step=1)
with r2c4: amort_pct        = st.slider("Amortization (% initial debt)",  0, 20, 5,   step=1)
with r2c5: exit_ev_ebitda   = st.number_input("Exit EV / EBITDA (x)",   value=11.0, step=0.5,  min_value=3.0,  max_value=30.0)
with r2c6: hold_years       = st.slider("Holding period (years)",         3, 10, 5)

nwc_pct  = 10
tax_rate = 25
da_pct   = 3

# ── Parameter glossary (styled HTML table, two-column layout) ──────────────
st.markdown('<div class="sec-header">Parameter definitions</div>', unsafe_allow_html=True)

params_left = [
    ("Entry EV / EBITDA", f"{entry_ev_ebitda:.1f}x",
     "The purchase price expressed as a multiple of the target's trailing twelve-month EBITDA. Typical LBO entry multiples range from 7x to 14x depending on sector and market conditions."),
    ("Entry EBITDA", fmt_m(ebitda_entry),
     "Earnings before interest, taxes, depreciation, and amortization at the time of acquisition. Serves as the primary valuation anchor and debt sizing metric."),
    ("Entry revenue", fmt_m(revenue_entry),
     "Total revenue at acquisition, used to compute the initial EBITDA margin and project forward revenue under the growth assumption."),
    ("Debt / EV", f"{debt_pct}%",
     "The proportion of the total enterprise value financed with debt at entry. Higher leverage amplifies equity returns when the business return exceeds the cost of debt, but increases default risk."),
    ("Interest rate", f"{interest_rate:.2f}%",
     "The blended annual interest rate on all debt tranches. Represents the hurdle rate below which the underlying business must generate returns for leverage to be accretive."),
    ("Amortization", f"{amort_pct}%/yr",
     "Annual mandatory principal repayment as a percentage of the initial debt balance. Reduces outstanding debt and interest expense over the holding period."),
]
params_right = [
    ("Revenue growth", f"{rev_growth}%/yr",
     "Projected annual revenue growth rate, applied uniformly over the holding period. Drives EBITDA growth alongside margin expansion."),
    ("Margin expansion", f"{margin_expansion} bps/yr",
     "Annual improvement in EBITDA margin, in basis points. Reflects operational improvements, pricing power, or cost reduction initiatives."),
    ("CapEx", f"{capex_pct}% of rev.",
     "Capital expenditure as a percentage of revenue, deducted from EBITDA to compute free cash flow."),
    ("Exit EV / EBITDA", f"{exit_ev_ebitda:.1f}x",
     "The multiple at which the business is sold. Multiple compression (exit < entry) destroys value; expansion amplifies it."),
    ("Holding period", f"{hold_years} yrs",
     "The number of years between acquisition and exit. Longer holds allow more time for operational improvement and debt paydown."),
    ("Tax rate", f"{tax_rate}%",
     "Effective corporate income tax rate. Determines the tax shield value of interest deductions and the after-tax cash flow available for debt service."),
]

gl1, gl2 = st.columns(2)
for col, params in [(gl1, params_left), (gl2, params_right)]:
    with col:
        rows = ""
        for name, val, defn in params:
            rows += f"""<tr>
              <td class="param-name">{name}</td>
              <td class="param-value">{val}</td>
              <td class="param-def">{defn}</td>
            </tr>"""
        st.markdown(f"""<table class="param-table">
          <thead><tr>
            <th style="width:18%;">Parameter</th>
            <th style="width:10%;">Value</th>
            <th>Definition</th>
          </tr></thead><tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# MODEL ENGINE (unchanged)
# ══════════════════════════════════════════════════════════════════════════
def run_model(entry_ev_ebitda, ebitda_entry, revenue_entry, debt_pct,
              interest_rate, rev_growth, margin_expansion, capex_pct,
              amort_pct, exit_ev_ebitda, hold_years,
              nwc_pct=10, tax_rate=25, da_pct=3):
    purchase_price = entry_ev_ebitda * ebitda_entry
    debt_entry     = purchase_price * (debt_pct / 100)
    equity_entry   = purchase_price - debt_entry
    entry_margin   = ebitda_entry / revenue_entry

    schedule = []
    debt_bal = debt_entry
    for yr in range(1, hold_years + 1):
        rev          = revenue_entry * ((1 + rev_growth / 100) ** yr)
        margin       = min(entry_margin + (margin_expansion / 10000) * yr, 0.60)
        ebitda       = rev * margin
        da           = rev * (da_pct / 100)
        ebit         = ebitda - da
        interest_exp = debt_bal * (interest_rate / 100)
        ebt          = ebit - interest_exp
        tax          = max(ebt, 0) * (tax_rate / 100)
        net_income   = ebt - tax
        capex        = rev * (capex_pct / 100)
        prev_rev     = revenue_entry * ((1 + rev_growth / 100) ** (yr - 1))
        delta_nwc    = (rev - prev_rev) * (nwc_pct / 100)
        fcf          = ebitda - capex - delta_nwc - tax - interest_exp
        amort        = min(debt_entry * (amort_pct / 100), debt_bal)
        debt_bal     = max(debt_bal - amort, 0)
        schedule.append(dict(
            Year=yr, Revenue=rev, EBITDA=ebitda, EBITDA_Margin=margin,
            DA=da, EBIT=ebit, Interest=interest_exp, EBT=ebt,
            Tax=tax, Net_Income=net_income, CapEx=capex,
            Delta_NWC=delta_nwc, FCF=fcf, Debt_Balance=debt_bal, Amortization=amort,
        ))

    df           = pd.DataFrame(schedule)
    exit_ebitda  = df.iloc[-1]["EBITDA"]
    exit_ev      = exit_ev_ebitda * exit_ebitda
    exit_debt    = df.iloc[-1]["Debt_Balance"]
    exit_equity  = exit_ev - exit_debt
    moic         = exit_equity / equity_entry
    equity_cfs   = [-equity_entry] + [0] * (hold_years - 1) + [exit_equity]
    irr_levered  = irr_calc(equity_cfs)
    ulev_cfs     = [-purchase_price] + list(df["FCF"].values[:-1]) + [df["FCF"].values[-1] + exit_ev]
    irr_unlevered= irr_calc(ulev_cfs)
    total_paydown= debt_entry - exit_debt
    ebitda_cagr  = (exit_ebitda / ebitda_entry) ** (1 / hold_years) - 1

    return dict(
        df=df, purchase_price=purchase_price, debt_entry=debt_entry,
        equity_entry=equity_entry, entry_margin=entry_margin,
        exit_ebitda=exit_ebitda, exit_ev=exit_ev, exit_debt=exit_debt,
        exit_equity=exit_equity, moic=moic, irr_levered=irr_levered,
        irr_unlevered=irr_unlevered, total_paydown=total_paydown,
        ebitda_cagr=ebitda_cagr,
        ev_from_growth=(exit_ebitda - ebitda_entry) * exit_ev_ebitda,
        ev_from_multiple=ebitda_entry * (exit_ev_ebitda - entry_ev_ebitda),
        total_gain=exit_equity - equity_entry,
    )

m = run_model(entry_ev_ebitda, ebitda_entry, revenue_entry, debt_pct,
              interest_rate, rev_growth, margin_expansion, capex_pct,
              amort_pct, exit_ev_ebitda, hold_years,
              nwc_pct=nwc_pct, tax_rate=tax_rate, da_pct=da_pct)
df             = m["df"]
purchase_price = m["purchase_price"]
debt_entry     = m["debt_entry"]
equity_entry   = m["equity_entry"]
entry_margin   = m["entry_margin"]
exit_ebitda    = m["exit_ebitda"]
exit_ev        = m["exit_ev"]
exit_debt      = m["exit_debt"]
exit_equity    = m["exit_equity"]
moic           = m["moic"]
irr_levered    = m["irr_levered"]
irr_unlevered  = m["irr_unlevered"]
total_paydown  = m["total_paydown"]
ebitda_cagr    = m["ebitda_cagr"]
ev_from_growth  = m["ev_from_growth"]
ev_from_multiple= m["ev_from_multiple"]
total_gain     = m["total_gain"]
rev_cagr       = (df.iloc[-1]["Revenue"] / revenue_entry) ** (1 / hold_years) - 1

irr_str   = f"{irr_levered*100:.1f}%"   if not np.isnan(irr_levered)   else "n/a"
irr_u_str = f"{irr_unlevered*100:.1f}%" if not np.isnan(irr_unlevered) else "n/a"

pct_g = ev_from_growth   / total_gain * 100 if total_gain != 0 else 0
pct_m = ev_from_multiple / total_gain * 100 if total_gain != 0 else 0
pct_p = total_paydown    / total_gain * 100 if total_gain != 0 else 0

# ══════════════════════════════════════════════════════════════════════════
# TRANSACTION SUMMARY
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="abstract-box" style="margin-top:2rem;">
  <div class="abstract-label">Transaction summary</div>
  <div class="abstract-text">
    At {entry_ev_ebitda:.1f}x entry on {fmt_m(ebitda_entry)} of EBITDA, the purchase price is {fmt_m(purchase_price)},
    financed with {fmt_m(debt_entry)} of debt ({debt_pct:.0f}% of EV) at {interest_rate:.2f}% and {fmt_m(equity_entry)} of sponsor equity.
    Over a {hold_years}-year hold, EBITDA is projected to grow at a {fmt_pct(ebitda_cagr)} CAGR
    to {fmt_m(exit_ebitda)}. At {exit_ev_ebitda:.1f}x exit, enterprise value reaches {fmt_m(exit_ev)};
    after repaying {fmt_m(exit_debt)} of remaining debt, the sponsor takes out {fmt_m(exit_equity)}.
    The levered IRR is <b>{irr_str}</b> on a <b>{moic:.2f}x</b> MOIC.
    The underlying business earns <b>{irr_u_str}</b> unlevered, putting the leverage contribution
    at <b>{(irr_levered - irr_unlevered)*100:.1f} percentage points</b>.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI SUMMARY
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">1. Transaction structure and returns summary</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    ("Entry enterprise value",  fmt_m(purchase_price),  f"{entry_ev_ebitda:.1f}x EBITDA · {fmt_m(debt_entry)} debt + {fmt_m(equity_entry)} equity", "neut"),
    ("Sponsor equity invested", fmt_m(equity_entry),    f"{100-debt_pct:.0f}% of EV · entry leverage {debt_entry/ebitda_entry:.1f}x EBITDA", "neut"),
    ("Exit equity proceeds",    fmt_m(exit_equity),     f"{fmt_m(exit_ev)} EV less {fmt_m(exit_debt)} remaining debt", "pos" if exit_equity > equity_entry else "neg"),
    ("Levered IRR",             irr_str,                f"MOIC: {moic:.2f}x over {hold_years} years",
      "pos" if not np.isnan(irr_levered) and irr_levered > 0.20 else "neut" if not np.isnan(irr_levered) and irr_levered > 0.12 else "neg"),
    ("Unlevered IRR",           irr_u_str,              f"Underlying business return · leverage adds {(irr_levered-irr_unlevered)*100:.1f} pp", "neut"),
]
for col, (label, value, sub, cls) in zip([k1,k2,k3,k4,k5], kpis):
    with col:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {cls}">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="fig-caption" style="margin-top:0.75rem;">
  <b>Table 1.</b> Summary transaction metrics for the base case.
  Entry enterprise value of {fmt_m(purchase_price)} is financed with {fmt_m(debt_entry)} of debt
  ({debt_pct:.0f}% leverage) at {interest_rate:.2f}% interest and {fmt_m(equity_entry)} of equity.
  The levered IRR of {irr_str} reflects the equity return inclusive of financial leverage;
  the unlevered IRR of {irr_u_str} reflects the underlying business return without leverage benefit.
  The {(irr_levered-irr_unlevered)*100:.1f} pp spread is the leverage effect, which is
  {'positive here because the business return exceeds' if irr_unlevered > interest_rate/100 else 'negative here because the cost of debt exceeds the business return at'}
  {interest_rate:.2f}%.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — CAPITAL STRUCTURE + WATERFALL
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">2. Capital structure and equity value creation</div>', unsafe_allow_html=True)

col_donut, col_wf = st.columns(2)

with col_donut:
    fig_cap = go.Figure(go.Pie(
        labels=["Debt", "Sponsor equity"],
        values=[debt_entry, equity_entry],
        hole=0.56,
        marker=dict(colors=[LRED, LBLUE], line=dict(color=CHART_BG, width=2.5)),
        textfont=dict(size=12, color=CHART_TEXT),
        textinfo="label+percent",
        hovertemplate="%{label}: $%{value:.1f}M<extra></extra>",
    ))
    fig_cap.update_layout(
        **{**BASE, "margin": dict(l=8, r=8, t=30, b=8)},
        height=300, showlegend=False,
        annotations=[dict(
            text=f"<b>{fmt_m(purchase_price)}</b><br><span style='font-size:10px'>EV</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=CHART_TEXT, family="DM Sans, Arial"),
        )],
    )
    st.plotly_chart(fig_cap, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 1.</b> Entry capital structure. Debt of {fmt_m(debt_entry)} ({debt_pct:.0f}% of EV)
      and sponsor equity of {fmt_m(equity_entry)} ({100-debt_pct:.0f}% of EV).
      Entry net leverage is {debt_entry/ebitda_entry:.1f}x EBITDA.
    </div>""", unsafe_allow_html=True)

with col_wf:
    fig_wf = go.Figure(go.Waterfall(
        x=["Entry equity", "EBITDA growth", "Multiple change", "Debt paydown", "Exit equity"],
        y=[equity_entry, ev_from_growth, ev_from_multiple, total_paydown, exit_equity],
        measure=["absolute", "relative", "relative", "relative", "absolute"],
        connector=dict(line=dict(color=LINE_COLOR, width=1)),
        increasing=dict(marker=dict(color=LGREEN)),
        decreasing=dict(marker=dict(color=LRED)),
        totals=dict(marker=dict(color=LBLUE)),
        text=[fmt_m(v) for v in [equity_entry, ev_from_growth, ev_from_multiple, total_paydown, exit_equity]],
        textposition="outside",
        textfont=dict(size=11, color=CHART_TEXT),
    ))
    fig_wf.update_layout(
        **BASE, height=300,
        yaxis=dict(**ax("Equity value ($M)"), tickprefix="$"),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color=CHART_TEXT),
                   linecolor=LINE_COLOR, linewidth=1, showline=True),
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 2.</b> Equity value creation waterfall from entry to exit.
      EBITDA growth contributes {fmt_m(ev_from_growth)} ({pct_g:.0f}% of total gain),
      multiple {'expansion' if ev_from_multiple >= 0 else 'compression'} {fmt_m(ev_from_multiple)} ({pct_m:.0f}%),
      and debt paydown {fmt_m(total_paydown)} ({pct_p:.0f}%).
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — LEVERAGE AND COVERAGE
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">3. Leverage and coverage metrics over the holding period</div>', unsafe_allow_html=True)

years_range  = list(range(1, hold_years + 1))
lev_turns    = (df["Debt_Balance"] / df["EBITDA"]).values
cov_ratio    = (df["EBITDA"] / df["Interest"]).values
entry_lev    = debt_entry / ebitda_entry
exit_lev     = exit_debt / exit_ebitda if exit_ebitda > 0 else 0
entry_cov    = ebitda_entry / (debt_entry * interest_rate / 100)
exit_cov     = exit_ebitda / (exit_debt * interest_rate / 100) if exit_debt > 0 else 99.0

fig_lev = make_subplots(specs=[[{"secondary_y": True}]])
fig_lev.add_trace(go.Bar(
    x=years_range, y=df["Debt_Balance"].values,
    name="Debt balance ($M)", marker_color=RED, opacity=0.6,
), secondary_y=False)
fig_lev.add_trace(go.Scatter(
    x=years_range, y=lev_turns,
    name="Net leverage (Debt / EBITDA, x)", mode="lines+markers",
    line=dict(color=BLUE, width=2.2), marker=dict(size=6),
), secondary_y=True)
fig_lev.add_trace(go.Scatter(
    x=years_range, y=cov_ratio,
    name="Interest coverage (EBITDA / Interest, x)", mode="lines+markers",
    line=dict(color=GREEN, width=2.2, dash="dash"), marker=dict(size=6, symbol="diamond"),
), secondary_y=True)
fig_lev.update_layout(
    **BASE, height=340,
    xaxis=dict(**ax("Year"), dtick=1),
    yaxis=dict(**ax("Debt balance ($M)"), tickprefix="$"),
    yaxis2=dict(
        title=dict(text="Multiple (x)", font=dict(size=12, color=CHART_TEXT)),
        tickfont=dict(size=11, color=CHART_TEXT),
        showgrid=False, zeroline=False, overlaying="y", side="right",
        linecolor=LINE_COLOR, linewidth=1, showline=True,
    ),
)
st.plotly_chart(fig_lev, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 3.</b> Debt balance (bars, left axis), net leverage in turns of EBITDA (solid line, right axis),
  and interest coverage ratio (dashed line, right axis) over the holding period.
  Entry leverage of {entry_lev:.1f}x declines to {exit_lev:.1f}x at exit through {fmt_m(total_paydown)} of
  scheduled amortization and EBITDA growth. Coverage improves from {entry_cov:.1f}x at entry
  to {exit_cov:.1f}x at exit.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — INCOME STATEMENT
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">4. Projected income statement</div>', unsafe_allow_html=True)

pl_data = {"Metric": [
    "Revenue ($M)", "EBITDA ($M)", "EBITDA Margin (%)",
    "D&A ($M)", "EBIT ($M)", "Interest expense ($M)",
    "EBT ($M)", "Tax ($M)", "Net income ($M)",
]}
for _, row in df.iterrows():
    yr = f"Year {int(row['Year'])}"
    pl_data[yr] = [
        f"${row['Revenue']:.1f}", f"${row['EBITDA']:.1f}",
        f"{row['EBITDA_Margin']*100:.1f}%", f"${row['DA']:.1f}",
        f"${row['EBIT']:.1f}", f"(${row['Interest']:.1f})",
        f"${row['EBT']:.1f}", f"(${row['Tax']:.1f})",
        f"${row['Net_Income']:.1f}",
    ]
st.dataframe(pd.DataFrame(pl_data), use_container_width=True, hide_index=True)
st.markdown(f"""<div class="fig-caption">
  <b>Table 2.</b> Projected income statement over the {hold_years}-year holding period.
  Revenue grows at {fmt_pct(rev_growth/100)}/year; EBITDA margin expands {margin_expansion:.0f} bps/year
  from an entry margin of {entry_margin*100:.1f}%. D&A is set at {da_pct}% of revenue.
  Interest expense declines each year as debt amortizes at {amort_pct:.0f}% of the initial balance annually.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — REVENUE / EBITDA + FCF
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">5. Revenue, EBITDA, and free cash flow projections</div>', unsafe_allow_html=True)

ch1, ch2 = st.columns(2)
with ch1:
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(x=years_range, y=df["Revenue"].values,
        name="Revenue ($M)", marker_color=LLBLUE, opacity=0.8))
    fig_rev.add_trace(go.Bar(x=years_range, y=df["EBITDA"].values,
        name="EBITDA ($M)", marker_color=LBLUE))
    fig_rev.add_trace(go.Scatter(
        x=years_range, y=(df["EBITDA_Margin"] * 100).values,
        name="EBITDA margin (%)", mode="lines+markers",
        line=dict(color=GREEN, width=2.2), marker=dict(size=6), yaxis="y2",
    ))
    fig_rev.update_layout(
        **BASE, height=320, barmode="overlay",
        xaxis=dict(**ax("Year"), dtick=1),
        yaxis=dict(**ax("Value ($M)"), tickprefix="$"),
        yaxis2=dict(
            title=dict(text="EBITDA margin (%)", font=dict(size=12, color=CHART_TEXT)),
            tickfont=dict(size=11, color=CHART_TEXT), ticksuffix="%",
            showgrid=False, zeroline=False, overlaying="y", side="right",
            linecolor=LINE_COLOR, linewidth=1, showline=True,
        ),
    )
    st.plotly_chart(fig_rev, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 4.</b> Revenue (light bars) and EBITDA (dark bars) with EBITDA margin (right axis).
      Revenue CAGR: {fmt_pct(rev_cagr)}. EBITDA grows from {fmt_m(ebitda_entry)} to
      {fmt_m(exit_ebitda)} ({fmt_pct(ebitda_cagr)} CAGR).
    </div>""", unsafe_allow_html=True)

with ch2:
    fig_fcf = go.Figure(go.Bar(
        x=years_range, y=df["FCF"].values,
        marker_color=[GREEN if v >= 0 else RED for v in df["FCF"].values], opacity=0.8,
        text=[fmt_m(v) for v in df["FCF"].values],
        textposition="outside", textfont=dict(size=11, color=CHART_TEXT),
    ))
    fig_fcf.update_layout(
        **BASE, height=320, showlegend=False,
        yaxis={**ax("Levered FCF ($M)"), "zeroline": True, "zerolinecolor": LINE_COLOR},
        xaxis=dict(**ax("Year"), dtick=1),
    )
    st.plotly_chart(fig_fcf, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 5.</b> Levered free cash flow (post-interest, post-tax, after CapEx and NWC changes)
      by year. Cumulative FCF over the holding period: {fmt_m(df['FCF'].sum())}.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — VALUE CREATION BRIDGE
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">6. Return attribution: sources of equity value creation</div>', unsafe_allow_html=True)

b1, b2, b3, b4 = st.columns(4)
for col, (label, value, sub, cls) in zip([b1,b2,b3,b4], [
    ("Total equity gain",  fmt_m(total_gain),        f"{moic:.2f}x MOIC on {fmt_m(equity_entry)} invested",
     "pos" if total_gain > 0 else "neg"),
    ("From EBITDA growth", fmt_m(ev_from_growth),    f"{pct_g:.0f}% of total gain · {fmt_pct(ebitda_cagr)} EBITDA CAGR",
     "pos" if ev_from_growth > 0 else "neg"),
    ("From multiple",      fmt_m(ev_from_multiple),  f"{pct_m:.0f}% of total gain · {'expansion' if ev_from_multiple >= 0 else 'compression'}",
     "pos" if ev_from_multiple > 0 else "neg"),
    ("From debt paydown",  fmt_m(total_paydown),     f"{pct_p:.0f}% of total gain · {fmt_m(debt_entry)} → {fmt_m(exit_debt)}", "pos"),
]):
    with col:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {cls}">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")
br1, br2 = st.columns(2)

with br1:
    valid = [(v, l, c) for v, l, c in [
        (ev_from_growth,   "EBITDA growth",  LGREEN),
        (ev_from_multiple, "Multiple change", LBLUE if ev_from_multiple >= 0 else LRED),
        (total_paydown,    "Debt paydown",    LLBLUE),
    ] if v > 0]
    if valid:
        va, vl, vc = zip(*valid)
        fig_pie = go.Figure(go.Pie(
            labels=vl, values=va, hole=0.52,
            marker=dict(colors=list(vc), line=dict(color=CHART_BG, width=2)),
            textfont=dict(size=12, color=CHART_TEXT),
            textinfo="label+percent",
            hovertemplate="%{label}: $%{value:.1f}M<extra></extra>",
        ))
        fig_pie.update_layout(
            **{**BASE, "margin": dict(l=8, r=8, t=30, b=8)},
            height=300, showlegend=False,
            annotations=[dict(
                text=f"<b>{moic:.2f}x</b><br>MOIC",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=CHART_TEXT, family="DM Sans, Arial"),
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 6.</b> Decomposition of total equity value creation by source (positive contributors only).
      {'EBITDA growth is the dominant driver.' if pct_g > 50 else 'Multiple change is the dominant driver.' if abs(pct_m) > pct_g else 'Debt paydown is the dominant driver.'}
    </div>""", unsafe_allow_html=True)

with br2:
    dp_range = list(range(20, 81, 5))
    irr_lev_curve = []
    for dp in dp_range:
        de  = purchase_price * (dp / 100)
        ee  = purchase_price - de
        ed  = exit_debt * (de / debt_entry) if debt_entry > 0 else exit_debt
        eeq = exit_ev - ed
        v   = irr_calc([-ee] + [0]*(hold_years-1) + [eeq])
        irr_lev_curve.append(v * 100 if not np.isnan(v) else None)

    fig_lev2 = go.Figure()
    fig_lev2.add_trace(go.Scatter(
        x=dp_range, y=irr_lev_curve, name="Levered IRR",
        mode="lines+markers", line=dict(color=BLUE, width=2.5), marker=dict(size=5),
    ))
    fig_lev2.add_hline(y=irr_unlevered*100 if not np.isnan(irr_unlevered) else 0,
                       line_dash="dash", line_color=GRAY, line_width=1.5,
                       annotation_text=f"Unlevered IRR ({irr_u_str})",
                       annotation_font=dict(size=10, color=GRAY),
                       annotation_position="top left")
    fig_lev2.add_hline(y=interest_rate, line_dash="dot", line_color=RED, line_width=1.5,
                       annotation_text=f"Cost of debt ({interest_rate:.1f}%)",
                       annotation_font=dict(size=10, color=RED),
                       annotation_position="bottom right")
    fig_lev2.add_vline(x=debt_pct, line_dash="dot", line_color=GRAY,
                       annotation_text=f"Base ({debt_pct:.0f}%)",
                       annotation_font=dict(size=10, color=GRAY),
                       annotation_position="top right")
    fig_lev2.update_layout(
        **BASE, height=300,
        yaxis=dict(**ax("IRR (%)"), ticksuffix="%"),
        xaxis=dict(**ax("Debt / EV at entry (%)"), ticksuffix="%"),
        showlegend=False,
    )
    st.plotly_chart(fig_lev2, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 7.</b> Levered IRR as a function of entry leverage. The unlevered IRR ({irr_u_str})
      and cost of debt ({interest_rate:.1f}%) are reference lines. Leverage is accretive where the
      curve lies above the unlevered return; it is destructive below the cost of debt line.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — SENSITIVITY TABLES
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">7. Sensitivity analysis: exit multiple vs. entry leverage</div>', unsafe_allow_html=True)
st.markdown(f"""<div class="abstract-text" style="margin-bottom:1.25rem;font-size:14px;">
  The tables below present levered IRR and MOIC across a range of exit multiples (columns)
  and entry leverage levels (rows), holding all other parameters at base case values.
  Green shading indicates IRR above 20% or MOIC above 3.0x; red shading indicates IRR below 12%
  or MOIC below 2.0x.
</div>""", unsafe_allow_html=True)

exit_mult_range = [round(exit_ev_ebitda + d, 1) for d in [-3.0, -1.5, 0, +1.5, +3.0]]
debt_pct_range  = [max(30, min(80, debt_pct + d)) for d in [-20, -10, 0, +10, +20]]

rows_irr, rows_moic = [], []
for dp in debt_pct_range:
    r_i, r_m = [], []
    for em in exit_mult_range:
        de  = purchase_price * (dp / 100)
        ee  = purchase_price - de
        ed  = exit_debt * (de / debt_entry) if debt_entry > 0 else exit_debt
        eeq = em * exit_ebitda - ed
        vi  = irr_calc([-ee] + [0]*(hold_years-1) + [eeq])
        vm  = eeq / ee if ee > 0 else np.nan
        r_i.append(f"{vi*100:.1f}%" if not np.isnan(vi) else "n/a")
        r_m.append(f"{vm:.2f}x"     if not np.isnan(vm) else "n/a")
    rows_irr.append(r_i); rows_moic.append(r_m)

sens_irr_df  = pd.DataFrame(rows_irr,
    index=[f"{d:.0f}% debt" for d in debt_pct_range],
    columns=[f"{m:.1f}x" for m in exit_mult_range])
sens_moic_df = pd.DataFrame(rows_moic,
    index=[f"{d:.0f}% debt" for d in debt_pct_range],
    columns=[f"{m:.1f}x" for m in exit_mult_range])

def style_irr(v):
    try:
        n = float(v.replace("%",""))
        if n >= 20: return "background-color:rgba(46,125,79,0.25);color:#6ab06a;font-weight:600;"
        if n < 12:  return "background-color:rgba(185,64,64,0.25);color:#d45a5a;font-weight:600;"
    except: pass
    return ""

def style_moic(v):
    try:
        n = float(v.replace("x",""))
        if n >= 3.0: return "background-color:rgba(46,125,79,0.25);color:#6ab06a;font-weight:600;"
        if n < 2.0:  return "background-color:rgba(185,64,64,0.25);color:#d45a5a;font-weight:600;"
    except: pass
    return ""

st1, st2 = st.columns(2)
with st1:
    st.markdown("**Levered IRR: exit multiple (columns) vs. entry debt % (rows)**")
    st.dataframe(sens_irr_df.style.applymap(style_irr), use_container_width=True)
with st2:
    st.markdown("**MOIC: exit multiple (columns) vs. entry debt % (rows)**")
    st.dataframe(sens_moic_df.style.applymap(style_moic), use_container_width=True)

st.markdown(f"""<div class="fig-caption">
  <b>Table 3 and Table 4.</b> Sensitivity of levered IRR and MOIC to exit multiple and entry leverage.
  Green: IRR ≥ 20% / MOIC ≥ 3.0x. Red: IRR < 12% / MOIC < 2.0x.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — OPERATING SENSITIVITY HEATMAP
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">8. IRR sensitivity: revenue growth vs. margin expansion</div>', unsafe_allow_html=True)

growth_range = list(range(max(0, int(rev_growth) - 8), int(rev_growth) + 10, 2))
margin_range = list(range(max(-100, int(margin_expansion) - 100), int(margin_expansion) + 150, 50))

heat_z = []
for mg in margin_range:
    row_vals = []
    for gr in growth_range:
        er  = revenue_entry * ((1 + gr/100) ** hold_years)
        em  = min(entry_margin + (mg/10000)*hold_years, 0.60)
        eeb = er * em
        eev = exit_ev_ebitda * eeb
        eeq = eev - exit_debt
        v   = irr_calc([-equity_entry] + [0]*(hold_years-1) + [eeq])
        row_vals.append(round(v*100, 1) if not np.isnan(v) else None)
    heat_z.append(row_vals)

fig_heat = go.Figure(go.Heatmap(
    z=heat_z,
    x=[f"{g}%" for g in growth_range],
    y=[f"{m} bps" for m in margin_range],
    colorscale=[[0, "#7a1a1a"], [0.45, "#2a2a2e"], [1, "#1a5c30"]],
    zmid=irr_levered * 100 if not np.isnan(irr_levered) else 15,
    text=[[f"{v:.1f}%" if v is not None else "n/a" for v in row] for row in heat_z],
    texttemplate="%{text}", textfont=dict(size=10, color=CHART_TEXT),
    hovertemplate="Growth: %{x}<br>Margin exp.: %{y}<br>IRR: %{text}<extra></extra>",
    showscale=True,
    colorbar=dict(
        title=dict(text="IRR (%)", font=dict(size=11, color=CHART_TEXT)),
        tickfont=dict(size=10, color=CHART_TEXT), thickness=12, len=0.8,
    ),
))
if int(rev_growth) in growth_range and int(margin_expansion) in margin_range:
    bx = growth_range.index(int(rev_growth))
    by = margin_range.index(int(margin_expansion))
    fig_heat.add_shape(type="rect",
        x0=bx-0.5, x1=bx+0.5, y0=by-0.5, y1=by+0.5,
        line=dict(color=CHART_TEXT, width=2))
fig_heat.update_layout(
    **{**BASE, "margin": dict(l=8, r=60, t=20, b=8)},
    height=380,
    xaxis=dict(title=dict(text="Annual revenue growth rate", font=dict(size=12, color=CHART_TEXT)),
               tickfont=dict(size=11, color=CHART_TEXT), showgrid=False,
               linecolor=LINE_COLOR, linewidth=1, showline=False),
    yaxis=dict(title=dict(text="Annual EBITDA margin expansion (bps)", font=dict(size=12, color=CHART_TEXT)),
               tickfont=dict(size=11, color=CHART_TEXT), showgrid=False,
               linecolor=LINE_COLOR, linewidth=1, showline=False),
)
st.plotly_chart(fig_heat, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 8.</b> Levered IRR heatmap across revenue growth rates (columns) and annual EBITDA margin
  expansion (rows), with exit multiple and leverage held at base case values.
  The base case ({rev_growth:.0f}% growth, {margin_expansion:.0f} bps/year) is outlined.
  Revenue growth is typically the more powerful lever, reflecting the multiplicative effect of
  EBITDA margin on a larger revenue base.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 9 — QUASI-MONTE CARLO RISK SIMULATION
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">9. Quasi-Monte Carlo risk simulation</div>', unsafe_allow_html=True)

st.markdown("""
<div class="abstract-box">
  <div class="abstract-label">Methodology</div>
  <div class="abstract-text">
    This section uses <em>Quasi-Monte Carlo (QMC) simulation</em> with low-discrepancy Sobol
    sequences, a statistically sounder alternative to standard pseudo-random Monte Carlo.
    Sobol sampling covers the parameter space far more uniformly than random draws; 512 samples
    achieve convergence that standard Monte Carlo typically needs 5,000 or more samples to reach.
    Each simulation draws jointly from distributions over five uncertainty drivers: revenue growth,
    margin expansion, exit multiple, interest rate, and entry leverage. The full IRR and MOIC are
    computed for each path, producing an empirical return distribution with good tail coverage
    and clean downside quantification.
  </div>
  <div class="abstract-text" style="margin-top:0.75rem; font-size:14px; opacity:0.75;">
    <b>Limitations.</b> The model assumes uniform annual amortization, a flat revenue growth
    rate, and an exit multiple that does not vary with holding period. These simplifications
    depart from actual deal mechanics, which involve tiered debt tranches, seasonal cash flows,
    and multiple compression over longer holds. Results should be read as structurally
    illustrative, not as forecasts.
  </div>
</div>
""", unsafe_allow_html=True)

qmc_c1, qmc_c2, qmc_c3 = st.columns(3)
with qmc_c1:
    qmc_n = st.select_slider("QMC sample size (Sobol)", options=[128, 256, 512, 1024], value=512)
    rev_vol   = st.slider("Revenue growth uncertainty (±pp)", 1, 15, 6)
    margin_vol= st.slider("Margin expansion uncertainty (±bps)", 10, 150, 60)
with qmc_c2:
    mult_vol  = st.slider("Exit multiple uncertainty (±turns)", 0.5, 4.0, 2.0, step=0.25)
    rate_vol  = st.slider("Interest rate uncertainty (±pp)", 0.25, 3.0, 1.5, step=0.25)
with qmc_c3:
    lev_vol   = st.slider("Leverage uncertainty (±pp of EV)", 2, 20, 10)
    corr_coef = st.slider("Growth–Multiple correlation", -0.8, 0.8, 0.3, step=0.1,
                          help="Positive: high-growth deals command higher exit multiples.")

@st.cache_data(show_spinner=False)
def run_qmc(n, rev_growth, margin_expansion, exit_ev_ebitda, interest_rate, debt_pct,
            rev_vol, margin_vol, mult_vol, rate_vol, lev_vol, corr_coef,
            entry_ev_ebitda, ebitda_entry, revenue_entry, amort_pct,
            hold_years, capex_pct, nwc_pct, tax_rate, da_pct):
    m_power = int(np.ceil(np.log2(max(n, 2))))
    u = _sobol_sequence(2 ** m_power, d=5, seed=42)[:n]

    def ndtr(x):
        t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (
               1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        p = 1.0 - (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2) * poly
        return np.where(x >= 0, p, 1.0 - p)

    def ndtri(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        a = np.array([2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637])
        b = np.array([-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833])
        c = np.array([0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
                      0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
                      0.0000321767881768, 0.0000002888167364, 0.0000003960315187])
        y = p - 0.5
        r = np.empty_like(p)
        mid = np.abs(y) < 0.42
        ym = y[mid]; r2 = ym * ym
        r[mid] = ym * (((a[3]*r2+a[2])*r2+a[1])*r2+a[0]) / \
                      ((((b[3]*r2+b[2])*r2+b[1])*r2+b[0])*r2+1.0)
        tail = ~mid; pt = p[tail]
        pt = np.where(y[tail] > 0, 1.0 - pt, pt)
        s = np.log(-np.log(pt))
        tv = c[0]+s*(c[1]+s*(c[2]+s*(c[3]+s*(c[4]+s*(c[5]+s*(c[6]+s*(c[7]+s*c[8])))))))
        r[tail] = np.where(y[tail] > 0, tv, -tv)
        return r

    C = np.eye(5)
    C[0, 2] = C[2, 0] = corr_coef
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        L = np.eye(5)

    z      = ndtri(np.clip(u, 1e-6, 1 - 1e-6))
    z_corr = (L @ z.T).T
    u_corr = ndtr(z_corr)

    gr_samples = np.clip(rev_growth + (u_corr[:, 0] - 0.5) * 2 * rev_vol, 0, 40)
    mg_samples = margin_expansion + (u_corr[:, 1] - 0.5) * 2 * margin_vol
    em_samples = np.clip(exit_ev_ebitda + (u_corr[:, 2] - 0.5) * 2 * mult_vol, 3.0, 25.0)
    ir_samples = np.clip(interest_rate + (u_corr[:, 3] - 0.5) * 2 * rate_vol, 1.0, 20.0)
    dp_samples = np.clip(debt_pct + (u_corr[:, 4] - 0.5) * 2 * lev_vol, 10, 85)

    purchase_price = entry_ev_ebitda * ebitda_entry
    irrs, moics = [], []

    for gr, mg, em, ir, dp in zip(gr_samples, mg_samples, em_samples, ir_samples, dp_samples):
        debt_e = purchase_price * (dp / 100)
        eq_e   = purchase_price - debt_e
        entry_m= ebitda_entry / revenue_entry
        debt_b = debt_e
        for yr in range(1, hold_years + 1):
            rev   = revenue_entry * ((1 + gr/100) ** yr)
            margin= min(entry_m + (mg/10000) * yr, 0.60)
            ebitda_y = rev * margin
            da_y  = rev * (da_pct / 100)
            ebit  = ebitda_y - da_y
            i_exp = debt_b * (ir / 100)
            ebt   = ebit - i_exp
            tax_  = max(ebt, 0) * (tax_rate / 100)
            capex_y= rev * (capex_pct / 100)
            prev_r = revenue_entry * ((1 + gr/100) ** (yr-1))
            d_nwc  = (rev - prev_r) * (nwc_pct / 100)
            amort_  = min(debt_e * (amort_pct / 100), debt_b)
            debt_b = max(debt_b - amort_, 0)

        exit_ebitda_s = revenue_entry * ((1 + gr/100)**hold_years) * min(
            entry_m + (mg/10000)*hold_years, 0.60)
        exit_ev_s  = em * exit_ebitda_s
        exit_eq_s  = exit_ev_s - debt_b
        moic_s     = exit_eq_s / eq_e if eq_e > 0 else np.nan
        irr_s      = irr_calc([-eq_e] + [0]*(hold_years-1) + [exit_eq_s])

        irrs.append(irr_s * 100 if not np.isnan(irr_s) else np.nan)
        moics.append(moic_s)

    return np.array(irrs), np.array(moics), gr_samples, em_samples

with st.spinner(f"Running {qmc_n}-path Sobol simulation…"):
    qmc_irrs, qmc_moics, qmc_gr, qmc_em = run_qmc(
        qmc_n, rev_growth, margin_expansion, exit_ev_ebitda, interest_rate, debt_pct,
        rev_vol, margin_vol, mult_vol, rate_vol, lev_vol, corr_coef,
        entry_ev_ebitda, ebitda_entry, revenue_entry, amort_pct,
        hold_years, capex_pct, nwc_pct, tax_rate, da_pct,
    )

valid_irrs  = qmc_irrs[~np.isnan(qmc_irrs)]
valid_moics = qmc_moics[~np.isnan(qmc_moics)]

pct_above_hurdle = np.mean(valid_irrs >= 20) * 100
pct_below_floor  = np.mean(valid_irrs < 12) * 100
var_5pct         = np.percentile(valid_irrs, 5)
median_irr       = np.median(valid_irrs)
irr_std          = np.std(valid_irrs)
sharpe_irr       = (median_irr - interest_rate) / irr_std if irr_std > 0 else np.nan
median_moic      = np.median(valid_moics)
moic_p10         = np.percentile(valid_moics, 10)

sharpe_cls = "pos" if sharpe_irr >= 1.5 else "warn" if sharpe_irr >= 0.8 else "neg"
st.markdown(f"""
<div class="risk-banner">
  <div class="risk-banner-title">QMC simulation summary: {qmc_n} Sobol paths</div>
  <span class="risk-stat">
    <span class="risk-stat-label">Median IRR</span>
    <span class="risk-stat-val {'pos' if median_irr >= 20 else 'warn' if median_irr >= 12 else 'neg'}">{median_irr:.1f}%</span>
  </span>
  <span class="risk-stat">
    <span class="risk-stat-label">5th Pctile IRR (VaR)</span>
    <span class="risk-stat-val {'pos' if var_5pct >= 12 else 'warn' if var_5pct >= 0 else 'neg'}">{var_5pct:.1f}%</span>
  </span>
  <span class="risk-stat">
    <span class="risk-stat-label">P(IRR ≥ 20%)</span>
    <span class="risk-stat-val {'pos' if pct_above_hurdle >= 60 else 'warn' if pct_above_hurdle >= 35 else 'neg'}">{pct_above_hurdle:.0f}%</span>
  </span>
  <span class="risk-stat">
    <span class="risk-stat-label">P(IRR &lt; 12%)</span>
    <span class="risk-stat-val {'neg' if pct_below_floor >= 25 else 'warn' if pct_below_floor >= 10 else 'pos'}">{pct_below_floor:.0f}%</span>
  </span>
  <span class="risk-stat">
    <span class="risk-stat-label">IRR Sharpe (vs. cost of debt)</span>
    <span class="risk-stat-val {sharpe_cls}">{sharpe_irr:.2f}x</span>
  </span>
  <span class="risk-stat">
    <span class="risk-stat-label">Median MOIC</span>
    <span class="risk-stat-val {'pos' if median_moic >= 2.5 else 'warn' if median_moic >= 1.5 else 'neg'}">{median_moic:.2f}x</span>
  </span>
  <span class="risk-stat">
    <span class="risk-stat-label">P10 MOIC (downside)</span>
    <span class="risk-stat-val {'neg' if moic_p10 < 1.5 else 'warn'}">{moic_p10:.2f}x</span>
  </span>
</div>
""", unsafe_allow_html=True)

qc1, qc2 = st.columns(2)

with qc1:
    fig_irr_dist = go.Figure()
    fig_irr_dist.add_trace(go.Histogram(
        x=valid_irrs, nbinsx=35,
        marker_color=LBLUE, opacity=0.55,
        name="Simulated IRR",
        hovertemplate="IRR: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))
    kde_x, kde_dens = kde_curve(valid_irrs, bw_factor=0.9)
    if len(kde_x):
        bin_w   = (valid_irrs.max() - valid_irrs.min()) / 35
        kde_cnt = kde_dens * len(valid_irrs) * bin_w
        fig_irr_dist.add_trace(go.Scatter(
            x=kde_x, y=kde_cnt, mode="lines", name="KDE",
            line=dict(color=CHART_TEXT, width=2.5), hoverinfo="skip",
        ))
    fig_irr_dist.add_vrect(
        x0=valid_irrs.min(), x1=var_5pct,
        fillcolor=RED, opacity=0.15, line_width=0,
        annotation_text="5th pctile", annotation_position="top left",
        annotation_font=dict(size=9, color=RED),
    )
    fig_irr_dist.add_vline(x=irr_levered*100, line_dash="dash", line_color=BLUE, line_width=2,
                            annotation_text=f"Base ({irr_str})",
                            annotation_font=dict(size=10, color=BLUE),
                            annotation_position="top right")
    fig_irr_dist.add_vline(x=median_irr, line_dash="dot", line_color=GREEN, line_width=1.5,
                            annotation_text=f"Median ({median_irr:.1f}%)",
                            annotation_font=dict(size=10, color=GREEN),
                            annotation_position="top left")
    fig_irr_dist.add_vline(x=20, line_dash="dot", line_color=GRAY, line_width=1,
                            annotation_text="20% hurdle",
                            annotation_font=dict(size=9, color=GRAY),
                            annotation_position="bottom right")
    fig_irr_dist.update_layout(
        **BASE, height=320, showlegend=False, barmode="overlay",
        xaxis=dict(**ax("Levered IRR (%)"), ticksuffix="%"),
        yaxis=dict(**ax("Count")),
    )
    st.plotly_chart(fig_irr_dist, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 9.</b> Distribution of levered IRR across {qmc_n} Sobol QMC paths, with Gaussian
      KDE overlay. The red-shaded region marks the 5th percentile tail (VaR at {var_5pct:.1f}%).
      {pct_above_hurdle:.0f}% of simulated paths clear the 20% institutional hurdle.
    </div>""", unsafe_allow_html=True)

with qc2:
    fig_moic_dist = go.Figure()
    fig_moic_dist.add_trace(go.Histogram(
        x=valid_moics, nbinsx=35,
        marker_color=LGREEN, opacity=0.55,
        name="Simulated MOIC",
        hovertemplate="MOIC: %{x:.2f}x<br>Count: %{y}<extra></extra>",
    ))
    kde_mx, kde_md = kde_curve(valid_moics, bw_factor=0.9)
    if len(kde_mx):
        bin_wm   = (valid_moics.max() - valid_moics.min()) / 35
        kde_mcnt = kde_md * len(valid_moics) * bin_wm
        fig_moic_dist.add_trace(go.Scatter(
            x=kde_mx, y=kde_mcnt, mode="lines", name="KDE",
            line=dict(color=CHART_TEXT, width=2.5), hoverinfo="skip",
        ))
    fig_moic_dist.add_vrect(
        x0=valid_moics.min(), x1=moic_p10,
        fillcolor=RED, opacity=0.15, line_width=0,
        annotation_text="P10 downside", annotation_position="top left",
        annotation_font=dict(size=9, color=RED),
    )
    fig_moic_dist.add_vline(x=moic, line_dash="dash", line_color=BLUE, line_width=2,
                             annotation_text=f"Base ({moic:.2f}x)",
                             annotation_font=dict(size=10, color=BLUE),
                             annotation_position="top right")
    fig_moic_dist.add_vline(x=3.0, line_dash="dot", line_color=GRAY, line_width=1,
                             annotation_text="3.0x excellent",
                             annotation_font=dict(size=9, color=GRAY),
                             annotation_position="bottom right")
    fig_moic_dist.update_layout(
        **BASE, height=320, showlegend=False, barmode="overlay",
        xaxis=dict(**ax("MOIC (x)"), ticksuffix="x"),
        yaxis=dict(**ax("Count")),
    )
    st.plotly_chart(fig_moic_dist, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 10.</b> Distribution of MOIC across {qmc_n} Sobol paths, with Gaussian KDE overlay.
      P10 downside MOIC: {moic_p10:.2f}x; median MOIC: {median_moic:.2f}x.
      {np.mean(valid_moics >= 3.0)*100:.0f}% of paths achieve the 3.0x excellent threshold.
    </div>""", unsafe_allow_html=True)

fig_scatter = make_subplots(
    rows=2, cols=2,
    column_widths=[0.82, 0.18],
    row_heights=[0.18, 0.82],
    shared_xaxes=True, shared_yaxes=True,
    horizontal_spacing=0.01, vertical_spacing=0.01,
)
fig_scatter.add_trace(go.Scatter(
    x=qmc_gr, y=qmc_em, mode="markers",
    marker=dict(
        color=qmc_irrs,
        colorscale=[[0, "#7a1a1a"], [0.45, "#2e2e38"], [1, "#1a5c30"]],
        size=8, opacity=0.85,
        colorbar=dict(
            title=dict(text="IRR (%)", font=dict(size=11, color=CHART_TEXT)),
            tickfont=dict(size=10, color=CHART_TEXT), thickness=10,
            x=0.81, len=0.82, y=0.09, yanchor="bottom",
        ),
        cmin=np.percentile(valid_irrs, 2),
        cmax=np.percentile(valid_irrs, 98),
        line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
    ),
    hovertemplate="Growth: %{x:.1f}%<br>Exit multiple: %{y:.1f}x<br>IRR: %{marker.color:.1f}%<extra></extra>",
), row=2, col=1)
fig_scatter.add_trace(go.Scatter(
    x=[rev_growth], y=[exit_ev_ebitda], mode="markers",
    marker=dict(symbol="star", size=16, color=BLUE, line=dict(color=CHART_TEXT, width=1.5)),
    name="Base case", hovertemplate="Base case<extra></extra>",
), row=2, col=1)
fig_scatter.add_trace(go.Box(
    x=qmc_gr, name="", marker_color=LBLUE, line_color=LBLUE, fillcolor=LLBLUE,
    boxpoints=False, notched=False, hoverinfo="skip", showlegend=False,
), row=1, col=1)
fig_scatter.add_trace(go.Box(
    y=qmc_em, name="", marker_color=LGREEN, line_color=LGREEN, fillcolor=LLGREEN,
    boxpoints=False, notched=False, hoverinfo="skip", showlegend=False, orientation="v",
), row=2, col=2)
fig_scatter.update_layout(
    **BASE, height=420, showlegend=False,
    xaxis2=dict(**ax("Revenue growth (%)", grid=True), ticksuffix="%", domain=[0, 0.81]),
    yaxis3=dict(**ax("Exit multiple (x)", grid=True), ticksuffix="x"),
    xaxis=dict(title=dict(text="Growth distribution", font=dict(size=10, color=GRAY)),
               showticklabels=False, showgrid=False, zeroline=False,
               linecolor=LINE_COLOR, linewidth=1),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, linecolor=LINE_COLOR, linewidth=1),
    yaxis4=dict(title=dict(text="Multiple dist.", font=dict(size=10, color=GRAY)),
                showticklabels=False, showgrid=False, zeroline=False, linecolor=LINE_COLOR, linewidth=1),
    xaxis4=dict(showticklabels=False, showgrid=False, zeroline=False, linecolor=LINE_COLOR, linewidth=1),
)
st.plotly_chart(fig_scatter, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 11.</b> Joint distribution of {qmc_n} Sobol paths across simulated revenue growth and exit
  multiple, colored by realized levered IRR. Marginal box plots show the univariate spread.
  The Cholesky-induced correlation (ρ = {corr_coef:.1f}) between growth and exit multiple
  is visible in the diagonal tilt of the point cloud. Base case marked with a star.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 10 — SCENARIO COMPARISON
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">10. Scenario comparison</div>', unsafe_allow_html=True)

st.markdown("""<div class="explainer-body" style="margin-bottom:1rem;">
  Define up to two additional scenarios to compare against the current base case.
  Entry EBITDA, revenue, CapEx, amortization, and hold period are inherited from the base case;
  the six inputs below are the ones that typically differ across deal structures.
</div>""", unsafe_allow_html=True)

sc_col1, sc_col2 = st.columns(2)
scenarios = [("Base case", entry_ev_ebitda, ebitda_entry, revenue_entry, debt_pct,
               interest_rate, rev_growth, margin_expansion, capex_pct, amort_pct,
               exit_ev_ebitda, hold_years)]

_sc_defaults = {
    2: dict(entry=min(entry_ev_ebitda + 2.0, 20.0), debt=min(debt_pct + 10, 80),
            grow=max(rev_growth - 4, 0), marg=max(margin_expansion - 50, -100),
            exit=max(exit_ev_ebitda - 2.0, 4.0), rate=min(interest_rate + 1.5, 15.0),
            label="Scenario B — Downside"),
    3: dict(entry=max(entry_ev_ebitda - 1.5, 5.0), debt=max(debt_pct - 5, 30),
            grow=min(rev_growth + 4, 25), marg=min(margin_expansion + 75, 200),
            exit=min(exit_ev_ebitda + 2.0, 22.0), rate=max(interest_rate - 0.5, 2.0),
            label="Scenario C — Upside"),
}

def scenario_inputs(label, idx):
    d = _sc_defaults[idx]
    st.markdown(f"**{label}**")
    c1, c2, c3 = st.columns(3)
    with c1:
        s_entry = st.number_input("Entry EV/EBITDA", value=float(d["entry"]),
                                  step=0.5, min_value=4.0, max_value=30.0, key=f"s{idx}_entry")
        s_debt  = st.slider("Debt/EV (%)", 30, 80, int(d["debt"]), step=5, key=f"s{idx}_debt")
    with c2:
        s_grow  = st.slider("Rev growth (%/yr)", 0, 30, int(d["grow"]), step=1, key=f"s{idx}_grow")
        s_marg  = st.slider("Margin expansion (bps)", -100, 200, int(d["marg"]), step=10, key=f"s{idx}_marg")
    with c3:
        s_exit  = st.number_input("Exit EV/EBITDA", value=float(d["exit"]),
                                  step=0.5, min_value=3.0, max_value=30.0, key=f"s{idx}_exit")
        s_rate  = st.number_input("Interest rate (%)", value=float(d["rate"]),
                                  step=0.25, min_value=1.0, max_value=20.0, key=f"s{idx}_rate")
    return (label, s_entry, ebitda_entry, revenue_entry, s_debt, s_rate,
            s_grow, s_marg, capex_pct, amort_pct, s_exit, hold_years)

with sc_col1:
    sc2 = scenario_inputs(_sc_defaults[2]["label"], 2)
with sc_col2:
    sc3 = scenario_inputs(_sc_defaults[3]["label"], 3)

all_scenarios = [scenarios[0], sc2, sc3]
sc_results = []
for sc in all_scenarios:
    label = sc[0]
    args  = sc[1:]
    r = run_model(*args, nwc_pct=nwc_pct, tax_rate=tax_rate, da_pct=da_pct)
    sc_results.append({
        "Scenario": label,
        "Entry EV/EBITDA": f"{args[0]:.1f}x",
        "Debt/EV": f"{args[3]:.0f}%",
        "Rev Growth": f"{args[5]:.0f}%",
        "Margin Exp.": f"{args[6]:.0f} bps",
        "Exit EV/EBITDA": f"{args[9]:.1f}x",
        "Entry Equity": fmt_m(r["equity_entry"]),
        "Exit Equity": fmt_m(r["exit_equity"]),
        "MOIC": f"{r['moic']:.2f}x",
        "Levered IRR": f"{r['irr_levered']*100:.1f}%" if not np.isnan(r["irr_levered"]) else "n/a",
        "Unlevered IRR": f"{r['irr_unlevered']*100:.1f}%" if not np.isnan(r["irr_unlevered"]) else "n/a",
        "EBITDA CAGR": f"{r['ebitda_cagr']*100:.1f}%",
    })

sc_df = pd.DataFrame(sc_results).set_index("Scenario")
st.dataframe(sc_df, use_container_width=True)

sc_names_short = ["Base", "Downside", "Upside"]
sc_irrs   = [float(r["Levered IRR"].replace("%","")) if r["Levered IRR"] != "n/a" else 0 for r in sc_results]
sc_moics  = [float(r["MOIC"].replace("x","")) for r in sc_results]

fig_sc = make_subplots(rows=1, cols=2,
    subplot_titles=["Levered IRR by Scenario", "MOIC by Scenario"])
for i, (nm, irr_v, moic_v) in enumerate(zip(sc_names_short, sc_irrs, sc_moics)):
    c = SCEN_COLORS[i]
    fig_sc.add_trace(go.Bar(x=[nm], y=[irr_v], marker_color=c, opacity=0.8,
                             text=[f"{irr_v:.1f}%"], textposition="outside",
                             textfont=dict(size=12, color=CHART_TEXT), showlegend=False), row=1, col=1)
    fig_sc.add_trace(go.Bar(x=[nm], y=[moic_v], marker_color=c, opacity=0.8,
                             text=[f"{moic_v:.2f}x"], textposition="outside",
                             textfont=dict(size=12, color=CHART_TEXT), showlegend=False), row=1, col=2)

fig_sc.add_hline(y=20, line_dash="dot", line_color=GRAY, line_width=1,
                  annotation_text="20% hurdle", annotation_font=dict(size=9, color=GRAY), row=1, col=1)
fig_sc.add_hline(y=3.0, line_dash="dot", line_color=GRAY, line_width=1,
                  annotation_text="3.0x excellent", annotation_font=dict(size=9, color=GRAY), row=1, col=2)
fig_sc.update_layout(
    **BASE, height=340,
    yaxis=dict(**ax("IRR (%)"), ticksuffix="%"),
    yaxis2=dict(**ax("MOIC (x)"), ticksuffix="x"),
    bargap=0.35,
)
st.plotly_chart(fig_sc, use_container_width=True)
st.markdown("""<div class="fig-caption">
  <b>Figure 12.</b> Side-by-side comparison of levered IRR and MOIC across the three scenarios.
  Dashed reference lines mark the 20% IRR institutional hurdle and 3.0x MOIC excellent threshold.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# APPENDIX
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Appendix: terminology, metrics, and methodology</div>', unsafe_allow_html=True)

st.markdown("""
<div class="abstract-box">
  <div class="abstract-label">Who this appendix is for</div>
  <div class="abstract-text">
    This appendix is written for readers who are not financial professionals or data scientists.
    It explains every term used in the model in plain language, describes how the key metrics
    are calculated, and walks through the statistical methods used in the risk simulation section.
    No prior knowledge of finance or statistics is assumed.
  </div>
</div>
""", unsafe_allow_html=True)

app1, app2 = st.columns(2)

with app1:
    st.markdown("""
<div class="appendix-group-head">Deal structure and valuation</div>

<div class="appendix-term">Enterprise Value (EV)</div>
<div class="appendix-def">The total price paid to acquire a business, including both the debt
borrowed and the equity contributed by the buyer. Think of it as what you would pay to own the
company outright with no debt attached. In the model, EV equals the entry EV/EBITDA multiple
times the company's EBITDA at acquisition.</div>

<div class="appendix-term">EBITDA</div>
<div class="appendix-def">Earnings Before Interest, Taxes, Depreciation, and Amortization.
It is the most common measure of a business's underlying operating profit, stripping out
financing costs and accounting adjustments to give a cleaner view of cash generation.</div>

<div class="appendix-term">EV/EBITDA multiple</div>
<div class="appendix-def">The ratio of enterprise value to EBITDA. Higher multiples reflect
faster-growing businesses or competitive auction processes. If the exit multiple is lower than
the entry multiple, that is multiple compression, which hurts returns.</div>

<div class="appendix-term">Leveraged Buyout (LBO)</div>
<div class="appendix-def">An acquisition where a large portion of the purchase price is
financed with borrowed money. The borrowed money is secured against the company's assets and
repaid from future cash flows. Gains in company value are amplified on the smaller equity check.</div>

<div class="appendix-group-head">Debt and cash flow</div>

<div class="appendix-term">Net leverage (Debt/EBITDA)</div>
<div class="appendix-def">Total debt divided by annual EBITDA. How many years of operating
earnings to repay all debt. Seven times is aggressive; below four times is comfortable.</div>

<div class="appendix-term">Interest coverage (EBITDA/Interest)</div>
<div class="appendix-def">Annual EBITDA divided by annual interest expense. Below 1.5x, lenders
get nervous; above 3x is generally considered safe.</div>

<div class="appendix-term">Free Cash Flow (FCF)</div>
<div class="appendix-def">Cash generated after paying for operations, CapEx, interest, and taxes.
In an LBO, FCF services and eventually retires the debt.</div>
""", unsafe_allow_html=True)

with app2:
    st.markdown("""
<div class="appendix-group-head">Return metrics</div>

<div class="appendix-term">IRR (Internal Rate of Return)</div>
<div class="appendix-def">The annualized return on an investment, accounting for timing of cash flows.
PE firms generally target IRRs above 20% and consider anything below 15% marginal.</div>

<div class="appendix-term">MOIC (Multiple of Invested Capital)</div>
<div class="appendix-def">Total cash returned divided by total cash invested, regardless of time.
A MOIC of 3.0x is considered excellent; below 2.0x is generally below hurdle.</div>

<div class="appendix-term">Levered IRR vs. Unlevered IRR</div>
<div class="appendix-def">Levered IRR is calculated on the equity investment after debt costs.
Unlevered IRR is the return as if the business were purchased entirely with equity. The difference
is the leverage effect — positive when the business return exceeds the cost of debt.</div>

<div class="appendix-group-head">Risk simulation methodology</div>

<div class="appendix-term">Quasi-Monte Carlo (QMC) and Sobol sequences</div>
<div class="appendix-def">A more efficient version of Monte Carlo. Sobol sequences spread samples
evenly across the parameter space. 512 Sobol samples produce answers nearly as accurate as 5,000+
standard random samples.</div>

<div class="appendix-term">Correlated parameters (Cholesky method)</div>
<div class="appendix-def">Generates random samples that respect a specified correlation structure.
Here, the correlation between revenue growth and exit multiple is adjustable; a positive value means
high-growth paths also tend to exit at higher multiples.</div>

<div class="appendix-term">Value at Risk (VaR) and percentile metrics</div>
<div class="appendix-def">The 5th percentile IRR is the downside floor: in 95 out of 100 scenarios,
IRR exceeds this number. P10 MOIC means 90% of simulated paths land above it.</div>

<div class="appendix-note">
  All model outputs are for educational and illustrative purposes only and do not
  constitute investment advice.
</div>
""", unsafe_allow_html=True)

st.markdown(f"""<div class="paper-footer">
  Model parameters: Entry EV/EBITDA = {entry_ev_ebitda:.1f}x; Entry EBITDA = {fmt_m(ebitda_entry)};
  Entry revenue = {fmt_m(revenue_entry)}; Debt/EV = {debt_pct:.0f}%; Interest rate = {interest_rate:.2f}%;
  Principal amortization = {amort_pct:.0f}%/year; Revenue growth = {rev_growth:.0f}%/year;
  EBITDA margin expansion = {margin_expansion:.0f} bps/year; CapEx = {capex_pct:.0f}% of revenue;
  NWC = {nwc_pct}% of incremental revenue; D&A = {da_pct}% of revenue; Tax rate = {tax_rate:.0f}%;
  Exit EV/EBITDA = {exit_ev_ebitda:.1f}x; Holding period = {hold_years} years.
  QMC simulation uses scrambled Sobol low-discrepancy sequences (pure numpy) with Cholesky-induced
  correlation between revenue growth and exit multiple. All figures in USD millions.
  This model is for educational and illustrative purposes only and does not constitute investment advice.
</div>""", unsafe_allow_html=True)
