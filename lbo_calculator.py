import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy_financial as npf

st.set_page_config(
    page_title="Leveraged Buyout Analysis: A Structured Returns Framework",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@400;500;600&display=swap');

  .block-container { padding-top: 4rem; padding-bottom: 4rem; max-width: 1160px; }
  [data-testid="stSidebar"] { display: none; }
  [data-testid="collapsedControl"] { display: none; }

  .paper-title {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 30px; font-weight: 500; line-height: 1.3;
    color: var(--text-color); margin-bottom: 0.4rem; letter-spacing: -0.01em;
  }
  .paper-byline {
    font-family: 'Inter', sans-serif;
    font-size: 13px; color: var(--text-color); opacity: 0.5;
    margin-bottom: 1.5rem; letter-spacing: 0.01em;
  }
  .abstract-box {
    border-top: 1px solid rgba(128,128,128,0.25);
    border-bottom: 1px solid rgba(128,128,128,0.25);
    padding: 1.2rem 0; margin-bottom: 2rem;
  }
  .abstract-label {
    font-family: 'Inter', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.4; color: var(--text-color); margin-bottom: 7px;
  }
  .abstract-text {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15.5px; line-height: 1.8; color: var(--text-color); max-width: 900px;
  }
  .sec-header {
    font-family: 'Inter', sans-serif;
    font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.4;
    margin: 2.5rem 0 0.75rem; padding-bottom: 5px;
    border-bottom: 1px solid rgba(128,128,128,0.15);
  }
  .kpi-card {
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 3px; padding: 1rem 1.1rem;
    background: rgba(128,128,128,0.03);
  }
  .kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 9px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.45; color: var(--text-color); margin-bottom: 6px;
  }
  .kpi-value {
    font-family: 'Inter', sans-serif;
    font-size: 26px; font-weight: 500; line-height: 1.1; color: var(--text-color);
  }
  .kpi-sub {
    font-family: 'Inter', sans-serif;
    font-size: 12px; margin-top: 4px; opacity: 0.55; color: var(--text-color);
  }
  .pos  { color: #2e7d4f !important; opacity: 1 !important; font-weight: 600; }
  .neg  { color: #b94040 !important; opacity: 1 !important; font-weight: 600; }
  .neut { color: #1a4f82 !important; opacity: 1 !important; font-weight: 600; }

  .fig-caption {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 13.5px; line-height: 1.75; color: var(--text-color);
    opacity: 0.8; margin-top: 0.1rem; margin-bottom: 1.5rem; font-style: italic;
    word-spacing: 0.02em; white-space: normal; word-break: normal; overflow-wrap: normal;
  }
  .fig-caption b { font-style: normal; font-weight: 600; opacity: 1; color: var(--text-color); }

  /* Explainer prose */
  .explainer-head {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 19px; font-weight: 500; color: var(--text-color);
    margin: 0.5rem 0 0.4rem; letter-spacing: -0.005em;
  }
  .explainer-body {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15px; line-height: 1.8; color: var(--text-color); opacity: 0.85;
    word-spacing: 0.02em; margin-bottom: 0.75rem;
  }
  .explainer-term {
    font-family: 'Inter', sans-serif;
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.55;
    margin-bottom: 4px; margin-top: 1rem;
  }

  /* Parameter table */
  .param-table {
    width: 100%; border-collapse: collapse;
    font-family: 'Inter', sans-serif; font-size: 12px;
    margin-bottom: 0.5rem;
  }
  .param-table th {
    font-size: 9px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.4; color: var(--text-color);
    border-bottom: 1px solid rgba(128,128,128,0.2);
    padding: 6px 10px 6px 0; text-align: left;
  }
  .param-table td {
    padding: 5px 10px 5px 0;
    border-bottom: 1px solid rgba(128,128,128,0.07);
    color: var(--text-color); font-size: 12px; vertical-align: top;
  }
  .param-name { font-weight: 500; }
  .param-def  { opacity: 0.6; font-size: 11px; font-family: 'EB Garamond', Georgia, serif; font-style: italic; }
  .param-value { font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; padding-right: 0; white-space: nowrap; }

  .paper-footer {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 12px; color: var(--text-color); opacity: 0.35;
    margin-top: 4rem; padding-top: 1rem;
    border-top: 1px solid rgba(128,128,128,0.15); line-height: 1.7;
  }

  /* Streamlit widget labels */
  label, .stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 0.02em; opacity: 0.7;
  }
  [data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)

# ── Chart helpers ──────────────────────────────────────────────────────────
FONT   = dict(size=12, color="#1a1a1a", family="Inter, Arial, sans-serif")
LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
              font=dict(size=11, color="#1a1a1a"), bgcolor="rgba(0,0,0,0)")
BASE   = dict(plot_bgcolor="white", paper_bgcolor="white", font=FONT,
              margin=dict(l=8, r=8, t=20, b=8), legend=LEGEND)

BLUE   = "#1a4f82"; LBLUE  = "#3d7ab5"; LLBLUE = "#b8cfe0"
GREEN  = "#2e7d4f"; LGREEN = "#6ab06a"
RED    = "#b94040"; LRED   = "#c47a7a"
GRAY   = "#888888"

def ax(title, grid=True):
    return dict(
        title=dict(text=title, font=dict(size=12, color="#333333")),
        tickfont=dict(size=11, color="#444444"),
        gridcolor="#f0f0f0" if grid else "rgba(0,0,0,0)",
        linecolor="#dddddd", linewidth=1, showline=True,
        showgrid=grid, zeroline=False, ticks="outside", ticklen=3,
    )

def fmt_m(v):
    if abs(v) >= 1000: return f"${v/1000:.2f}B"
    return f"${v:.1f}M"

def fmt_pct(v): return f"{v*100:.1f}%"

def irr_calc(cfs):
    try: return npf.irr(cfs)
    except Exception: return np.nan

# ══════════════════════════════════════════════════════════════════════════
# TITLE BLOCK
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="paper-title">Leveraged Buyout Analysis: A Structured Returns Framework</div>
<div class="paper-byline">
  Interactive model for LBO transaction structuring, return attribution, and sensitivity analysis
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# LBO EXPLAINER
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">What is a leveraged buyout?</div>', unsafe_allow_html=True)

st.markdown("""
<div class="abstract-box">
  <div class="abstract-label">Overview</div>
  <div class="abstract-text">
    A leveraged buyout (LBO) is an acquisition in which the purchase price is financed primarily
    with debt, leaving the acquirer — typically a private equity sponsor — to contribute a
    relatively small equity check. The acquired company's own cash flows service the debt over
    the holding period. Upon exit, the sponsor receives the residual equity value after repaying
    outstanding debt. The use of leverage amplifies equity returns when the underlying business
    return exceeds the cost of debt, and destroys value when it does not.
  </div>
</div>
""", unsafe_allow_html=True)

col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    st.markdown('<div class="explainer-head">The mechanics of leverage</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explainer-body">
      In a conventional acquisition, a buyer pays the full purchase price in cash or stock.
      In an LBO, the buyer finances 50–70% of the purchase price with debt — secured against
      the target company's assets and cash flows — and contributes only the remainder as equity.
      This structure means that a given improvement in enterprise value translates into a
      proportionally larger improvement in equity value, because the debt claim is fixed.
      A business purchased for $600M with $360M of debt and $240M of equity that exits at
      $700M returns $430M to equity holders — a 79% gain on a 17% increase in enterprise value.
    </div>""", unsafe_allow_html=True)

with col_e2:
    st.markdown('<div class="explainer-head">Sources of equity return</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explainer-body">
      Equity returns in an LBO come from three sources. First, <em>EBITDA growth</em>: if the
      business earns more at exit than at entry, the enterprise value rises proportionally
      (assuming a constant exit multiple). Second, <em>multiple expansion</em>: if the market
      assigns a higher earnings multiple at exit than at entry, the enterprise value rises even
      without profit growth — this is the most volatile and least controllable return driver.
      Third, <em>debt paydown</em>: as free cash flow retires debt principal over the holding
      period, a larger share of a given enterprise value accrues to equity holders at exit.
    </div>""", unsafe_allow_html=True)

with col_e3:
    st.markdown('<div class="explainer-head">Key metrics and thresholds</div>', unsafe_allow_html=True)
    st.markdown("""<div class="explainer-body">
      Private equity investors evaluate LBO transactions primarily on <em>internal rate of
      return (IRR)</em> and <em>multiple of invested capital (MOIC)</em>. An IRR above 20%
      is generally considered a strong return; below 15% is marginal for most institutional
      sponsors. A MOIC above 3.0x is excellent; below 2.0x is typically below hurdle.
      Leverage quality is assessed through <em>net leverage</em> (debt/EBITDA) and
      <em>interest coverage</em> (EBITDA/interest). Entry leverage above 7x EBITDA is
      considered aggressive; coverage below 2x raises refinancing risk.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# PARAMETER INPUTS — inline grid, no sidebar
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Model parameters</div>', unsafe_allow_html=True)

st.markdown("""<div class="explainer-body" style="margin-bottom:1rem;">
  Adjust the parameters below to model different transaction structures. All figures are in
  USD millions unless stated. The model recomputes all outputs, projections, and sensitivity
  tables in real time.
</div>""", unsafe_allow_html=True)

# Row 1: Entry parameters
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
with r1c1: entry_ev_ebitda  = st.number_input("Entry EV / EBITDA (x)",  value=12.0, step=0.5,  min_value=4.0,  max_value=30.0)
with r1c2: ebitda_entry     = st.number_input("Entry EBITDA ($M)",       value=50.0, step=5.0,  min_value=1.0)
with r1c3: revenue_entry    = st.number_input("Entry revenue ($M)",      value=200.0, step=10.0, min_value=1.0)
with r1c4: debt_pct         = st.slider("Debt / EV (%)",                  30, 80, 60, step=5)
with r1c5: interest_rate    = st.number_input("Interest rate (%)",        value=7.0,  step=0.25, min_value=1.0,  max_value=20.0)

# Row 2: Operating + exit parameters
r2c1, r2c2, r2c3, r2c4, r2c5, r2c6 = st.columns(6)
with r2c1: rev_growth        = st.slider("Revenue growth (%/yr)",          0, 30, 8,   step=1)
with r2c2: margin_expansion  = st.slider("Margin expansion (bps/yr)",    -100, 200, 50, step=10)
with r2c3: capex_pct         = st.slider("CapEx (% of revenue)",           1, 15, 4,   step=1)
with r2c4: amort_pct         = st.slider("Amortization (% initial debt)",  0, 20, 5,   step=1)
with r2c5: exit_ev_ebitda    = st.number_input("Exit EV / EBITDA (x)",   value=11.0, step=0.5,  min_value=3.0,  max_value=30.0)
with r2c6: hold_years        = st.slider("Holding period (years)",         3, 10, 5)

# Hidden but needed
nwc_pct  = 10
tax_rate = 25
da_pct   = 3

# ── Parameter glossary table ───────────────────────────────────────────────
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
     "Annual improvement in EBITDA margin, in basis points (hundredths of a percent). Reflects operational improvements, pricing power, or cost reduction initiatives."),
    ("CapEx", f"{capex_pct}% of rev.",
     "Capital expenditure as a percentage of revenue, deducted from EBITDA to compute free cash flow. Maintenance CapEx is required to sustain the business; growth CapEx funds expansion."),
    ("Exit EV / EBITDA", f"{exit_ev_ebitda:.1f}x",
     "The multiple at which the business is sold. Multiple compression (exit < entry) destroys value; expansion (exit > entry) amplifies it. Typically set conservatively equal to or below entry."),
    ("Holding period", f"{hold_years} yrs",
     "The number of years between acquisition and exit. Longer holds allow more time for operational improvement and debt paydown, but also extend the period of illiquidity."),
    ("Tax rate", f"{tax_rate}%",
     "Effective corporate income tax rate applied to pre-tax income. Determines the tax shield value of interest deductions and the after-tax cash flow available for debt service."),
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
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# MODEL ENGINE
# ══════════════════════════════════════════════════════════════════════════
purchase_price  = entry_ev_ebitda * ebitda_entry
debt_entry      = purchase_price * (debt_pct / 100)
equity_entry    = purchase_price - debt_entry
entry_margin    = ebitda_entry / revenue_entry

years_range = list(range(1, hold_years + 1))
schedule    = []
debt_bal    = debt_entry

for yr in years_range:
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

df             = pd.DataFrame(schedule)
exit_ebitda    = df.iloc[-1]["EBITDA"]
exit_ev        = exit_ev_ebitda * exit_ebitda
exit_debt      = df.iloc[-1]["Debt_Balance"]
exit_equity    = exit_ev - exit_debt
moic           = exit_equity / equity_entry
equity_cfs     = [-equity_entry] + [0] * (hold_years - 1) + [exit_equity]
irr_levered    = irr_calc(equity_cfs)
ulevered_cfs   = [-purchase_price] + list(df["FCF"].values[:-1]) + [df["FCF"].values[-1] + exit_ev]
irr_unlevered  = irr_calc(ulevered_cfs)
total_paydown  = debt_entry - exit_debt
ebitda_cagr    = (exit_ebitda / ebitda_entry) ** (1 / hold_years) - 1
rev_cagr       = (df.iloc[-1]["Revenue"] / revenue_entry) ** (1 / hold_years) - 1

ev_from_growth   = (exit_ebitda - ebitda_entry) * exit_ev_ebitda
ev_from_multiple = ebitda_entry * (exit_ev_ebitda - entry_ev_ebitda)
total_gain       = exit_equity - equity_entry

irr_str    = f"{irr_levered*100:.1f}%"   if not np.isnan(irr_levered)   else "n/a"
irr_u_str  = f"{irr_unlevered*100:.1f}%" if not np.isnan(irr_unlevered) else "n/a"

# ══════════════════════════════════════════════════════════════════════════
# TRANSACTION SUMMARY
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="abstract-box" style="margin-top:2rem;">
  <div class="abstract-label">Transaction summary</div>
  <div class="abstract-text">
    This model structures an acquisition of a business with {fmt_m(ebitda_entry)} of EBITDA at
    an entry multiple of {entry_ev_ebitda:.1f}x, implying a total enterprise value of {fmt_m(purchase_price)}.
    The transaction is financed with {fmt_m(debt_entry)} of debt ({debt_pct}% of EV) at {interest_rate:.2f}%
    and {fmt_m(equity_entry)} of sponsor equity ({100-debt_pct}% of EV).
    Over a {hold_years}-year holding period, EBITDA is projected to grow at a {fmt_pct(ebitda_cagr)} CAGR
    to {fmt_m(exit_ebitda)}, with exit at {exit_ev_ebitda:.1f}x implying an exit enterprise value of {fmt_m(exit_ev)}.
    After repaying {fmt_m(exit_debt)} of remaining debt, the sponsor realizes {fmt_m(exit_equity)} in proceeds.
    The levered IRR is <b>{irr_str}</b> with a MOIC of <b>{moic:.2f}x</b>.
    The unlevered IRR on the underlying business is <b>{irr_u_str}</b>, implying a leverage effect of
    <b>{(irr_levered - irr_unlevered)*100:.1f} percentage points</b>.
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
    ("Sponsor equity invested", fmt_m(equity_entry),    f"{100-debt_pct}% of EV · entry leverage {debt_entry/ebitda_entry:.1f}x EBITDA", "neut"),
    ("Exit equity proceeds",    fmt_m(exit_equity),     f"{fmt_m(exit_ev)} EV less {fmt_m(exit_debt)} remaining debt", "pos" if exit_equity > equity_entry else "neg"),
    ("Levered IRR",             irr_str,                f"MOIC: {moic:.2f}x over {hold_years} years", "pos" if not np.isnan(irr_levered) and irr_levered > 0.20 else "neut" if not np.isnan(irr_levered) and irr_levered > 0.12 else "neg"),
    ("Unlevered IRR",           irr_u_str,              f"Underlying business return · leverage adds {(irr_levered-irr_unlevered)*100:.1f} pp", "neut"),
]
for col, (label, value, sub, cls) in zip([k1,k2,k3,k4,k5], kpis):
    with col:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value <{cls}>">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="fig-caption" style="margin-top:0.75rem;">
  <b>Table 1.</b> Summary transaction metrics for the base case.
  Entry enterprise value of {fmt_m(purchase_price)} is financed with {fmt_m(debt_entry)} of debt
  ({debt_pct}% leverage) at {interest_rate:.2f}% interest and {fmt_m(equity_entry)} of equity.
  The levered IRR of {irr_str} reflects the equity return inclusive of financial leverage;
  the unlevered IRR of {irr_u_str} reflects the underlying business return without leverage benefit.
  The {(irr_levered-irr_unlevered)*100:.1f} percentage-point spread is the leverage effect,
  which is {'positive' if irr_levered > irr_unlevered else 'negative'} here because the
  {'business return exceeds' if irr_unlevered > interest_rate/100 else 'cost of debt exceeds the business return at'}
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
        marker=dict(colors=[LRED, LBLUE], line=dict(color="white", width=2.5)),
        textfont=dict(size=12, color="#1a1a1a"),
        textinfo="label+percent",
        hovertemplate="%{label}: $%{value:.1f}M<extra></extra>",
    ))
    fig_cap.update_layout(
        **{**BASE, "margin": dict(l=8, r=8, t=30, b=8)},
        height=300, showlegend=False,
        annotations=[dict(
            text=f"<b>{fmt_m(purchase_price)}</b><br><span style='font-size:10px'>EV</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#1a1a1a", family="Inter, Arial"),
        )],
    )
    st.plotly_chart(fig_cap, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 1.</b> Entry capital structure. Debt of {fmt_m(debt_entry)} ({debt_pct}% of EV)
      and sponsor equity of {fmt_m(equity_entry)} ({100-debt_pct}% of EV).
      Entry net leverage is {debt_entry/ebitda_entry:.1f}x EBITDA.
    </div>""", unsafe_allow_html=True)

with col_wf:
    fig_wf = go.Figure(go.Waterfall(
        x=["Entry equity", "EBITDA growth", "Multiple change", "Debt paydown", "Exit equity"],
        y=[equity_entry, ev_from_growth, ev_from_multiple, total_paydown, exit_equity],
        measure=["absolute", "relative", "relative", "relative", "absolute"],
        connector=dict(line=dict(color="#dddddd", width=1)),
        increasing=dict(marker=dict(color=LGREEN)),
        decreasing=dict(marker=dict(color=LRED)),
        totals=dict(marker=dict(color=LBLUE)),
        text=[fmt_m(v) for v in [equity_entry, ev_from_growth, ev_from_multiple, total_paydown, exit_equity]],
        textposition="outside",
        textfont=dict(size=11, color="#333333"),
    ))
    fig_wf.update_layout(
        **BASE, height=300,
        yaxis=dict(**ax("Equity value ($M)"), tickprefix="$"),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#444444"),
                   linecolor="#dddddd", linewidth=1, showline=True),
        showlegend=False,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    pct_g = ev_from_growth / total_gain * 100    if total_gain != 0 else 0
    pct_m = ev_from_multiple / total_gain * 100  if total_gain != 0 else 0
    pct_p = total_paydown / total_gain * 100      if total_gain != 0 else 0
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 2.</b> Equity value creation waterfall from entry to exit.
      EBITDA growth contributes {fmt_m(ev_from_growth)} ({pct_g:.0f}% of total gain),
      multiple {'expansion' if ev_from_multiple >= 0 else 'compression'} {fmt_m(ev_from_multiple)}
      ({pct_m:.0f}%), and debt paydown {fmt_m(total_paydown)} ({pct_p:.0f}%).
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — LEVERAGE AND COVERAGE
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">3. Leverage and coverage metrics over the holding period</div>', unsafe_allow_html=True)

lev_turns    = (df["Debt_Balance"] / df["EBITDA"]).values
cov_ratio    = (df["EBITDA"] / df["Interest"]).values
entry_lev    = debt_entry / ebitda_entry
exit_lev     = exit_debt / exit_ebitda if exit_ebitda > 0 else 0
entry_cov    = ebitda_entry / (debt_entry * interest_rate / 100)
exit_cov     = exit_ebitda / (exit_debt * interest_rate / 100) if exit_debt > 0 else 99.0

fig_lev = make_subplots(specs=[[{"secondary_y": True}]])
fig_lev.add_trace(go.Bar(
    x=years_range, y=df["Debt_Balance"].values,
    name="Debt balance ($M)", marker_color=LRED, opacity=0.7,
), secondary_y=False)
fig_lev.add_trace(go.Scatter(
    x=years_range, y=lev_turns,
    name="Net leverage (Debt / EBITDA, x)",
    mode="lines+markers", line=dict(color=BLUE, width=2.2),
    marker=dict(size=6),
), secondary_y=True)
fig_lev.add_trace(go.Scatter(
    x=years_range, y=cov_ratio,
    name="Interest coverage (EBITDA / Interest, x)",
    mode="lines+markers", line=dict(color=GREEN, width=2.2, dash="dash"),
    marker=dict(size=6, symbol="diamond"),
), secondary_y=True)

fig_lev.update_layout(
    **BASE, height=340,
    xaxis=dict(**ax("Year"), dtick=1),
    yaxis=dict(**ax("Debt balance ($M)"), tickprefix="$"),
    yaxis2=dict(
        title=dict(text="Multiple (x)", font=dict(size=12, color="#333333")),
        tickfont=dict(size=11, color="#444444"),
        showgrid=False, zeroline=False, overlaying="y", side="right",
        linecolor="#dddddd", linewidth=1, showline=True,
    ),
)
st.plotly_chart(fig_lev, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 3.</b> Debt balance (bars, left axis), net leverage in turns of EBITDA (solid line,
  right axis), and interest coverage ratio (dashed line, right axis) over the holding period.
  Entry leverage of {entry_lev:.1f}x declines to {exit_lev:.1f}x at exit through {fmt_m(total_paydown)} of
  scheduled amortization and EBITDA growth. Coverage improves from {entry_cov:.1f}x at entry
  to {exit_cov:.1f}x at exit, reflecting both debt reduction and earnings expansion.
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
        f"${row['Revenue']:.1f}",
        f"${row['EBITDA']:.1f}",
        f"{row['EBITDA_Margin']*100:.1f}%",
        f"${row['DA']:.1f}",
        f"${row['EBIT']:.1f}",
        f"(${row['Interest']:.1f})",
        f"${row['EBT']:.1f}",
        f"(${row['Tax']:.1f})",
        f"${row['Net_Income']:.1f}",
    ]
st.dataframe(pd.DataFrame(pl_data), use_container_width=True, hide_index=True)
st.markdown(f"""<div class="fig-caption">
  <b>Table 2.</b> Projected income statement over the {hold_years}-year holding period.
  Revenue grows at {fmt_pct(rev_growth/100)}/year; EBITDA margin expands {margin_expansion} bps/year
  from an entry margin of {entry_margin*100:.1f}%. D&A is set at {da_pct}% of revenue.
  Interest expense declines each year as debt amortizes at {amort_pct}% of the initial balance annually.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — REVENUE / EBITDA + FCF
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">5. Revenue, EBITDA, and free cash flow projections</div>', unsafe_allow_html=True)

ch1, ch2 = st.columns(2)

with ch1:
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(
        x=years_range, y=df["Revenue"].values,
        name="Revenue ($M)", marker_color=LLBLUE, opacity=0.8,
    ))
    fig_rev.add_trace(go.Bar(
        x=years_range, y=df["EBITDA"].values,
        name="EBITDA ($M)", marker_color=LBLUE,
    ))
    fig_rev.add_trace(go.Scatter(
        x=years_range, y=(df["EBITDA_Margin"] * 100).values,
        name="EBITDA margin (%)",
        mode="lines+markers", line=dict(color=GREEN, width=2.2),
        marker=dict(size=6), yaxis="y2",
    ))
    fig_rev.update_layout(
        **BASE, height=320, barmode="overlay",
        xaxis=dict(**ax("Year"), dtick=1),
        yaxis=dict(**ax("Value ($M)"), tickprefix="$"),
        yaxis2=dict(
            title=dict(text="EBITDA margin (%)", font=dict(size=12, color="#333333")),
            tickfont=dict(size=11, color="#444444"), ticksuffix="%",
            showgrid=False, zeroline=False, overlaying="y", side="right",
            linecolor="#dddddd", linewidth=1, showline=True,
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
        marker_color=[GREEN if v >= 0 else RED for v in df["FCF"].values],
        opacity=0.8,
        text=[fmt_m(v) for v in df["FCF"].values],
        textposition="outside",
        textfont=dict(size=11, color="#333333"),
    ))
    fig_fcf.update_layout(
        **BASE, height=320, showlegend=False,
        yaxis=dict(**ax("Levered FCF ($M)"), zeroline=True, zerolinecolor="#dddddd"),
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
    ("Total equity gain",  fmt_m(total_gain),        f"{moic:.2f}x MOIC on {fmt_m(equity_entry)} invested", "pos" if total_gain > 0 else "neg"),
    ("From EBITDA growth", fmt_m(ev_from_growth),    f"{pct_g:.0f}% of total gain · {fmt_pct(ebitda_cagr)} EBITDA CAGR", "pos" if ev_from_growth > 0 else "neg"),
    ("From multiple",      fmt_m(ev_from_multiple),  f"{pct_m:.0f}% of total gain · {'expansion' if ev_from_multiple >= 0 else 'compression'}", "pos" if ev_from_multiple > 0 else "neg"),
    ("From debt paydown",  fmt_m(total_paydown),     f"{pct_p:.0f}% of total gain · {fmt_m(debt_entry)} → {fmt_m(exit_debt)}", "pos"),
]):
    with col:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value <{cls}>">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")

br1, br2 = st.columns(2)

with br1:
    # Decomposition donut
    valid = [(v, l, c) for v, l, c in [
        (ev_from_growth,   "EBITDA growth",    LGREEN),
        (ev_from_multiple, "Multiple change",  LBLUE if ev_from_multiple >= 0 else LRED),
        (total_paydown,    "Debt paydown",     LLBLUE),
    ] if v > 0]
    if valid:
        va, vl, vc = zip(*valid)
        fig_pie = go.Figure(go.Pie(
            labels=vl, values=va, hole=0.52,
            marker=dict(colors=list(vc), line=dict(color="white", width=2)),
            textfont=dict(size=12, color="#1a1a1a"),
            textinfo="label+percent",
            hovertemplate="%{label}: $%{value:.1f}M<extra></extra>",
        ))
        fig_pie.update_layout(
            **{**BASE, "margin": dict(l=8, r=8, t=30, b=8)},
            height=300, showlegend=False,
            annotations=[dict(
                text=f"<b>{moic:.2f}x</b><br>MOIC",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#1a1a1a", family="Inter, Arial"),
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown(f"""<div class="fig-caption">
      <b>Figure 6.</b> Decomposition of total equity value creation by source (positive contributors only).
      {'EBITDA growth is the dominant driver.' if pct_g > 50 else 'Multiple change is the dominant driver.' if abs(pct_m) > pct_g else 'Debt paydown is the dominant driver.'}
    </div>""", unsafe_allow_html=True)

with br2:
    # Levered vs unlevered IRR curve
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
        x=dp_range, y=irr_lev_curve,
        name="Levered IRR", mode="lines+markers",
        line=dict(color=BLUE, width=2.5), marker=dict(size=5),
    ))
    fig_lev2.add_hline(y=irr_unlevered*100 if not np.isnan(irr_unlevered) else 0,
                       line_dash="dash", line_color=GRAY, line_width=1.5,
                       annotation_text=f"Unlevered IRR ({irr_u_str})",
                       annotation_font=dict(size=10, color=GRAY),
                       annotation_position="top left")
    fig_lev2.add_hline(y=interest_rate, line_dash="dot", line_color=LRED, line_width=1.5,
                       annotation_text=f"Cost of debt ({interest_rate:.1f}%)",
                       annotation_font=dict(size=10, color=LRED),
                       annotation_position="bottom right")
    fig_lev2.add_vline(x=debt_pct, line_dash="dot", line_color="#aaaaaa",
                       annotation_text=f"Base ({debt_pct}%)",
                       annotation_font=dict(size=10, color="#666666"),
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
  The center cell reflects the base case ({exit_ev_ebitda:.1f}x exit, {debt_pct}% debt).
  Green shading indicates IRR above 20% or MOIC above 3.0x; red shading indicates IRR below 12%
  or MOIC below 2.0x.
</div>""", unsafe_allow_html=True)

exit_mult_range = [round(exit_ev_ebitda + d, 1) for d in [-3.0, -1.5, 0, +1.5, +3.0]]
debt_pct_range  = [max(30, debt_pct + d) for d in [-20, -10, 0, +10, +20]]
debt_pct_range  = [min(80, v) for v in debt_pct_range]

def sens_irr_val(em, dp):
    de  = purchase_price * (dp / 100)
    ee  = purchase_price - de
    ed  = exit_debt * (de / debt_entry) if debt_entry > 0 else exit_debt
    eeq = em * exit_ebitda - ed
    v   = irr_calc([-ee] + [0]*(hold_years-1) + [eeq])
    return v

def sens_moic_val(em, dp):
    de  = purchase_price * (dp / 100)
    ee  = purchase_price - de
    ed  = exit_debt * (de / debt_entry) if debt_entry > 0 else exit_debt
    eeq = em * exit_ebitda - ed
    return eeq / ee if ee > 0 else np.nan

rows_irr, rows_moic = [], []
for dp in debt_pct_range:
    r_i, r_m = [], []
    for em in exit_mult_range:
        vi = sens_irr_val(em, dp)
        vm = sens_moic_val(em, dp)
        r_i.append(f"{vi*100:.1f}%" if not np.isnan(vi) else "n/a")
        r_m.append(f"{vm:.2f}x"     if not np.isnan(vm) else "n/a")
    rows_irr.append(r_i)
    rows_moic.append(r_m)

sens_irr_df  = pd.DataFrame(rows_irr,
    index=[f"{d}% debt" for d in debt_pct_range],
    columns=[f"{m:.1f}x" for m in exit_mult_range])
sens_moic_df = pd.DataFrame(rows_moic,
    index=[f"{d}% debt" for d in debt_pct_range],
    columns=[f"{m:.1f}x" for m in exit_mult_range])

def style_irr(v):
    try:
        n = float(v.replace("%",""))
        if n >= 20: return "background-color:rgba(46,125,79,0.18);color:#1a4f82;font-weight:600;"
        if n < 12:  return "background-color:rgba(185,64,64,0.15);color:#7a1a1a;font-weight:600;"
    except: pass
    return ""

def style_moic(v):
    try:
        n = float(v.replace("x",""))
        if n >= 3.0: return "background-color:rgba(46,125,79,0.18);color:#1a4f82;font-weight:600;"
        if n < 2.0:  return "background-color:rgba(185,64,64,0.15);color:#7a1a1a;font-weight:600;"
    except: pass
    return ""

st1, st2 = st.columns(2)
with st1:
    st.markdown("**Levered IRR — exit multiple (columns) vs. entry debt % (rows)**")
    st.dataframe(sens_irr_df.style.applymap(style_irr), use_container_width=True)
with st2:
    st.markdown("**MOIC — exit multiple (columns) vs. entry debt % (rows)**")
    st.dataframe(sens_moic_df.style.applymap(style_moic), use_container_width=True)

st.markdown(f"""<div class="fig-caption">
  <b>Table 3 and Table 4.</b> Sensitivity of levered IRR and MOIC to exit multiple and entry leverage.
  Green: IRR ≥ 20% / MOIC ≥ 3.0x. Red: IRR < 12% / MOIC < 2.0x.
  At the base exit multiple of {exit_ev_ebitda:.1f}x, the transaction generates above-hurdle returns
  across most leverage levels. The analysis illustrates that {'multiple compression' if ev_from_multiple < 0 else 'multiple expansion'} is the
  {'primary risk to returns' if ev_from_multiple < 0 else 'primary upside driver'} in the base case.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8 — OPERATING SENSITIVITY HEATMAP
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">8. IRR sensitivity: revenue growth vs. margin expansion</div>', unsafe_allow_html=True)

growth_range = list(range(max(0, rev_growth - 8), rev_growth + 10, 2))
margin_range = list(range(max(-100, margin_expansion - 100), margin_expansion + 150, 50))

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
    colorscale=[[0, "#c47a7a"], [0.45, "#f7f7f7"], [1, "#6ab06a"]],
    zmid=irr_levered * 100 if not np.isnan(irr_levered) else 15,
    text=[[f"{v:.1f}%" if v is not None else "n/a" for v in row] for row in heat_z],
    texttemplate="%{text}",
    textfont=dict(size=10, color="#1a1a1a"),
    hovertemplate="Growth: %{x}<br>Margin exp.: %{y}<br>IRR: %{text}<extra></extra>",
    showscale=True,
    colorbar=dict(
        title=dict(text="IRR (%)", font=dict(size=11, color="#333333")),
        tickfont=dict(size=10, color="#444444"),
        thickness=12, len=0.8,
    ),
))

# Mark base case cell
if rev_growth in growth_range and margin_expansion in margin_range:
    bx = growth_range.index(rev_growth)
    by = margin_range.index(margin_expansion)
    fig_heat.add_shape(type="rect",
        x0=bx-0.5, x1=bx+0.5, y0=by-0.5, y1=by+0.5,
        line=dict(color="#1a1a1a", width=2))

fig_heat.update_layout(
    **{**BASE, "margin": dict(l=8, r=60, t=20, b=8)},
    height=380,
    xaxis=dict(
        title=dict(text="Annual revenue growth rate", font=dict(size=12, color="#333333")),
        tickfont=dict(size=11, color="#444444"), showgrid=False,
        linecolor="#dddddd", linewidth=1, showline=False,
    ),
    yaxis=dict(
        title=dict(text="Annual EBITDA margin expansion (bps)", font=dict(size=12, color="#333333")),
        tickfont=dict(size=11, color="#444444"), showgrid=False,
        linecolor="#dddddd", linewidth=1, showline=False,
    ),
)
st.plotly_chart(fig_heat, use_container_width=True)
st.markdown(f"""<div class="fig-caption">
  <b>Figure 8.</b> Levered IRR heatmap across revenue growth rates (columns) and annual EBITDA margin
  expansion (rows), with exit multiple and leverage held at base case values.
  The base case ({rev_growth}% growth, {margin_expansion} bps/year) is outlined in black.
  Green shading indicates higher returns; red indicates lower. Revenue growth is typically the
  more powerful lever, reflecting the multiplicative effect of EBITDA margin on a larger revenue base.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""<div class="paper-footer">
  Model parameters: Entry EV/EBITDA = {entry_ev_ebitda:.1f}x; Entry EBITDA = {fmt_m(ebitda_entry)};
  Entry revenue = {fmt_m(revenue_entry)}; Debt/EV = {debt_pct}%; Interest rate = {interest_rate:.2f}%;
  Principal amortization = {amort_pct}%/year of initial debt; Revenue growth = {rev_growth}%/year;
  EBITDA margin expansion = {margin_expansion} bps/year; CapEx = {capex_pct}% of revenue;
  NWC = {nwc_pct}% of incremental revenue; D&A = {da_pct}% of revenue; Tax rate = {tax_rate}%;
  Exit EV/EBITDA = {exit_ev_ebitda:.1f}x; Holding period = {hold_years} years.
  IRR computed using Newton-Raphson iteration (numpy-financial v1.0).
  All figures in USD millions. Sensitivity analysis holds exit debt constant at base case.
  This model is for educational and illustrative purposes only and does not constitute
  investment advice.
</div>""", unsafe_allow_html=True)
