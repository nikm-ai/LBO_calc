import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy_financial as npf

st.set_page_config(
    page_title="Leveraged Buyout Analysis: A Structured Returns Framework",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@400;500;600&display=swap');

  .block-container { padding-top: 4rem; padding-bottom: 3rem; max-width: 1200px; }

  .paper-title {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 28px; font-weight: 500; line-height: 1.3;
    color: var(--text-color); margin-bottom: 0.4rem; letter-spacing: -0.01em;
  }
  .paper-byline {
    font-family: 'Inter', sans-serif;
    font-size: 13px; color: var(--text-color); opacity: 0.55;
    margin-bottom: 1.5rem; letter-spacing: 0.01em;
  }
  .abstract-box {
    border-top: 1px solid rgba(128,128,128,0.25);
    border-bottom: 1px solid rgba(128,128,128,0.25);
    padding: 1.1rem 0; margin-bottom: 2rem;
  }
  .abstract-label {
    font-family: 'Inter', sans-serif;
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; opacity: 0.45; color: var(--text-color); margin-bottom: 6px;
  }
  .abstract-text {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 15px; line-height: 1.75; color: var(--text-color); max-width: 860px;
  }
  .sec-header {
    font-family: 'Inter', sans-serif;
    font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.4;
    margin: 2.25rem 0 0.75rem; padding-bottom: 5px;
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
    font-size: 28px; font-weight: 500; line-height: 1.1; color: var(--text-color);
  }
  .kpi-sub {
    font-family: 'Inter', sans-serif;
    font-size: 12px; margin-top: 4px; opacity: 0.55; color: var(--text-color);
  }
  .pos  { color: #2e7d4f !important; opacity: 1 !important; font-weight: 500; }
  .neg  { color: #b94040 !important; opacity: 1 !important; font-weight: 500; }
  .neut { color: #1a4f82 !important; opacity: 1 !important; font-weight: 500; }
  .fig-caption {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 13.5px; line-height: 1.75; color: var(--text-color);
    opacity: 0.8; margin-top: 0.1rem; margin-bottom: 1.25rem; font-style: italic;
    word-spacing: 0.02em; letter-spacing: normal; white-space: normal;
    word-break: normal; overflow-wrap: normal;
  }
  .fig-caption b { font-style: normal; font-weight: 600; opacity: 1; color: var(--text-color); }
  .note-head {
    font-family: 'Inter', sans-serif;
    font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; color: var(--text-color); opacity: 0.6;
    margin-bottom: 8px; word-spacing: normal; white-space: normal;
  }
  .note-body {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 14px; line-height: 1.75; color: var(--text-color); opacity: 0.85;
    word-spacing: 0.02em; letter-spacing: normal; white-space: normal;
    word-break: normal; overflow-wrap: normal;
  }
  .paper-footer {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 12.5px; color: var(--text-color); opacity: 0.4;
    margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid rgba(128,128,128,0.15); line-height: 1.6;
  }
  .sensitivity-cell-high { background: rgba(46,125,79,0.15) !important; }
  .sensitivity-cell-low  { background: rgba(185,64,64,0.12) !important; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(128,128,128,0.02);
    border-right: 1px solid rgba(128,128,128,0.12);
  }
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stNumberInput label {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important; font-weight: 500 !important;
    letter-spacing: 0.02em; color: var(--text-color) !important; opacity: 0.7;
  }
  [data-testid="stSidebar"] .stCaption {
    font-family: 'EB Garamond', Georgia, serif;
    font-size: 12px; opacity: 0.45; color: var(--text-color); line-height: 1.6;
  }
  [data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif; font-size: 13px;
    font-weight: 500; letter-spacing: 0.02em;
  }
</style>
""", unsafe_allow_html=True)

# ── Chart helpers ──────────────────────────────────────────────────────────
FONT   = dict(size=12, color="#1a1a1a", family="Inter, Arial, sans-serif")
LEGEND = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
              font=dict(size=11, color="#1a1a1a"), bgcolor="rgba(0,0,0,0)")
BASE   = dict(plot_bgcolor="white", paper_bgcolor="white", font=FONT,
              margin=dict(l=8, r=8, t=20, b=8), legend=LEGEND)

BLUE   = "#1a4f82"
LBLUE  = "#3d7ab5"
LLBLUE = "#b8cfe0"
GREEN  = "#2e7d4f"
LGREEN = "#6ab06a"
RED    = "#b94040"
LRED   = "#c47a7a"
AMBER  = "#a06020"
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
    """Format as $XM or $X.XXM"""
    if abs(v) >= 1000:
        return f"${v/1000:.2f}B"
    return f"${v:.1f}M"

def fmt_pct(v):
    return f"{v*100:.1f}%"

def irr_calc(cashflows):
    """Compute IRR from a list of cashflows (first element is negative investment)."""
    try:
        return npf.irr(cashflows)
    except Exception:
        return np.nan

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — DEAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p style="font-family:Inter,sans-serif;font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;opacity:0.45;margin-bottom:0.5rem;">Entry parameters</p>', unsafe_allow_html=True)

    entry_ev_ebitda = st.number_input("Entry EV / EBITDA (x)", value=12.0, step=0.5, min_value=4.0, max_value=30.0)
    ebitda_entry    = st.number_input("Entry EBITDA ($M)", value=50.0, step=5.0, min_value=1.0)
    debt_pct        = st.slider("Debt / EV at entry (%)", 30, 80, 60, step=5)
    interest_rate   = st.number_input("Interest rate (%)", value=7.0, step=0.25, min_value=1.0, max_value=20.0)
    amort_pct       = st.slider("Annual principal amortization (% of initial debt)", 0, 20, 5, step=1)

    st.markdown('<hr style="border:none;border-top:1px solid rgba(128,128,128,0.18);margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Inter,sans-serif;font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;opacity:0.45;margin-bottom:0.5rem;">Operating assumptions</p>', unsafe_allow_html=True)

    revenue_entry   = st.number_input("Entry revenue ($M)", value=200.0, step=10.0, min_value=1.0)
    rev_growth      = st.slider("Annual revenue growth (%)", 0, 30, 8, step=1)
    ebitda_margin_entry = ebitda_entry / revenue_entry * 100
    margin_expansion = st.slider("Annual EBITDA margin expansion (bps)", -100, 200, 50, step=10)
    capex_pct       = st.slider("CapEx as % of revenue", 1, 15, 4, step=1)
    nwc_pct         = st.slider("Change in NWC as % of revenue growth", 0, 30, 10, step=1)

    st.markdown('<hr style="border:none;border-top:1px solid rgba(128,128,128,0.18);margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Inter,sans-serif;font-size:9px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;opacity:0.45;margin-bottom:0.5rem;">Exit parameters</p>', unsafe_allow_html=True)

    hold_years      = st.slider("Holding period (years)", 3, 10, 5)
    exit_ev_ebitda  = st.number_input("Exit EV / EBITDA (x)", value=11.0, step=0.5, min_value=3.0, max_value=30.0)
    tax_rate        = st.slider("Effective tax rate (%)", 15, 35, 25, step=1)
    da_pct          = st.slider("D&A as % of revenue", 1, 10, 3, step=1)

    st.markdown('<hr style="border:none;border-top:1px solid rgba(128,128,128,0.18);margin:1rem 0;">', unsafe_allow_html=True)
    st.caption("All figures in USD millions unless stated. IRR computed using Newton-Raphson on unlevered and levered free cash flow streams.")

# ══════════════════════════════════════════════════════════════════════════
# MODEL ENGINE
# ══════════════════════════════════════════════════════════════════════════
purchase_price = entry_ev_ebitda * ebitda_entry
debt_entry     = purchase_price * (debt_pct / 100)
equity_entry   = purchase_price - debt_entry
entry_margin   = ebitda_entry / revenue_entry

years_range = list(range(1, hold_years + 1))

# Build annual P&L and cash flow schedule
schedule = []
debt_bal = debt_entry

for yr in years_range:
    rev          = revenue_entry * ((1 + rev_growth / 100) ** yr)
    margin       = entry_margin + (margin_expansion / 10000) * yr
    margin       = min(margin, 0.60)
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

    schedule.append({
        "Year": yr,
        "Revenue": rev,
        "EBITDA": ebitda,
        "EBITDA Margin": margin,
        "D&A": da,
        "EBIT": ebit,
        "Interest": interest_exp,
        "EBT": ebt,
        "Tax": tax,
        "Net Income": net_income,
        "CapEx": capex,
        "Delta NWC": delta_nwc,
        "FCF": fcf,
        "Debt Balance": debt_bal,
        "Amortization": amort,
    })

df = pd.DataFrame(schedule)

# Exit
exit_ebitda   = df.iloc[-1]["EBITDA"]
exit_ev       = exit_ev_ebitda * exit_ebitda
exit_debt     = df.iloc[-1]["Debt Balance"]
exit_equity   = exit_ev - exit_debt
moic          = exit_equity / equity_entry
equity_cfs    = [-equity_entry] + [0] * (hold_years - 1) + [exit_equity]
irr_levered   = irr_calc(equity_cfs)
total_debt_paydown = debt_entry - exit_debt

# Unlevered IRR (entry/exit EV, no debt)
ulevered_cfs  = [-purchase_price] + list(df["FCF"].values[:-1]) + [df["FCF"].values[-1] + exit_ev]
irr_unlevered = irr_calc(ulevered_cfs)

# EBITDA CAGR
ebitda_cagr   = (exit_ebitda / ebitda_entry) ** (1 / hold_years) - 1
rev_cagr      = (df.iloc[-1]["Revenue"] / revenue_entry) ** (1 / hold_years) - 1

# Value creation bridge (sources of return)
# 1. EBITDA growth contribution
ev_from_ebitda_growth = (exit_ebitda - ebitda_entry) * exit_ev_ebitda
# 2. Multiple expansion/compression
ev_from_multiple = ebitda_entry * (exit_ev_ebitda - entry_ev_ebitda)
# 3. Debt paydown
ev_from_paydown = total_debt_paydown

# ══════════════════════════════════════════════════════════════════════════
# SENSITIVITY TABLES
# ══════════════════════════════════════════════════════════════════════════
def compute_irr_for(exit_mult, entry_debt_pct, _hold=hold_years, _eq_entry=equity_entry,
                    _exit_ebitda=exit_ebitda, _exit_debt=exit_debt, _cfs=equity_cfs):
    """Compute levered IRR for a given exit multiple and entry debt %."""
    ep = purchase_price
    de = ep * (entry_debt_pct / 100)
    ee = ep - de
    # Approximate exit debt as proportional paydown
    ed = exit_debt * (de / debt_entry) if debt_entry > 0 else exit_debt
    ex_eq = exit_mult * exit_ebitda - ed
    cfs = [-ee] + [0] * (_hold - 1) + [ex_eq]
    return irr_calc(cfs)

def compute_moic_for(exit_mult, entry_debt_pct):
    ep = purchase_price
    de = ep * (entry_debt_pct / 100)
    ee = ep - de
    ed = exit_debt * (de / debt_entry) if debt_entry > 0 else exit_debt
    ex_eq = exit_mult * exit_ebitda - ed
    return ex_eq / ee if ee > 0 else np.nan

exit_mult_range  = [round(exit_ev_ebitda - 3, 1), round(exit_ev_ebitda - 1.5, 1),
                    round(exit_ev_ebitda, 1),
                    round(exit_ev_ebitda + 1.5, 1), round(exit_ev_ebitda + 3, 1)]
debt_pct_range   = [max(30, debt_pct - 20), max(30, debt_pct - 10),
                    debt_pct,
                    min(80, debt_pct + 10), min(80, debt_pct + 20)]

sens_irr  = pd.DataFrame(index=[f"{d}% debt" for d in debt_pct_range],
                          columns=[f"{m}x exit" for m in exit_mult_range])
sens_moic = pd.DataFrame(index=[f"{d}% debt" for d in debt_pct_range],
                          columns=[f"{m}x exit" for m in exit_mult_range])

for dp in debt_pct_range:
    for em in exit_mult_range:
        v_irr  = compute_irr_for(em, dp)
        v_moic = compute_moic_for(em, dp)
        sens_irr.loc[f"{dp}% debt",  f"{em}x exit"] = f"{v_irr*100:.1f}%" if not np.isnan(v_irr) else "n/a"
        sens_moic.loc[f"{dp}% debt", f"{em}x exit"] = f"{v_moic:.2f}x"    if not np.isnan(v_moic) else "n/a"

# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="paper-title">Leveraged Buyout Analysis: A Structured Returns Framework</div>
<div class="paper-byline">
  Interactive model for LBO transaction structuring, return attribution, and sensitivity analysis
</div>
""", unsafe_allow_html=True)

irr_str   = f"{irr_levered*100:.1f}%" if not np.isnan(irr_levered) else "n/a"
irr_u_str = f"{irr_unlevered*100:.1f}%" if not np.isnan(irr_unlevered) else "n/a"

st.markdown(f"""
<div class="abstract-box">
  <div class="abstract-label">Transaction summary</div>
  <div class="abstract-text">
    This model structures a leveraged buyout of a business with {fmt_m(ebitda_entry)} of EBITDA
    at an entry multiple of {entry_ev_ebitda:.1f}x, implying a total enterprise value of
    {fmt_m(purchase_price)}. The transaction is financed with {fmt_m(debt_entry)} of debt
    ({debt_pct}% of EV) and {fmt_m(equity_entry)} of sponsor equity.
    Over a {hold_years}-year holding period, EBITDA is projected to grow at a {fmt_pct(ebitda_cagr)}
    CAGR to {fmt_m(exit_ebitda)}, with exit at {exit_ev_ebitda:.1f}x implying an exit enterprise value
    of {fmt_m(exit_ev)}. After repaying {fmt_m(exit_debt)} of remaining debt, the sponsor
    realizes {fmt_m(exit_equity)} in proceeds. The levered IRR is <b>{irr_str}</b>
    with a MOIC of <b>{moic:.2f}x</b>. The unlevered IRR on the underlying business is
    <b>{irr_u_str}</b>.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tab_returns, tab_schedule, tab_bridge, tab_sensitivity = st.tabs([
    "Returns Analysis",
    "Financial Projections",
    "Value Creation Bridge",
    "Sensitivity Analysis",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: RETURNS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tab_returns:

    st.markdown('<div class="sec-header">1. Transaction structure and returns summary</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Entry enterprise value", fmt_m(purchase_price), f"{entry_ev_ebitda:.1f}x entry EBITDA", "neut"),
        ("Sponsor equity invested", fmt_m(equity_entry), f"{100-debt_pct}% of EV", "neut"),
        ("Exit equity proceeds", fmt_m(exit_equity), f"{fmt_m(exit_ev)} EV less {fmt_m(exit_debt)} debt", "pos" if exit_equity > equity_entry else "neg"),
        ("Levered IRR", irr_str, f"MOIC: {moic:.2f}x over {hold_years} years", "pos" if irr_levered > 0.20 else "neut" if irr_levered > 0.12 else "neg"),
        ("Unlevered IRR", irr_u_str, "Business return ex-leverage", "neut"),
    ]
    for col, (label, value, sub, cls) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value <{cls}>">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="fig-caption" style="margin-top:0.75rem;">
      <b>Table 1.</b> Summary transaction metrics.
      Entry enterprise value of {fmt_m(purchase_price)} is financed with {fmt_m(debt_entry)} of debt
      ({debt_pct}% leverage) at {interest_rate:.2f}% interest and {fmt_m(equity_entry)} of equity.
      The levered IRR of {irr_str} reflects the equity return inclusive of the leverage effect;
      the unlevered IRR of {irr_u_str} reflects the underlying business return on invested capital
      without the benefit of financial leverage. The {(irr_levered - irr_unlevered)*100:.1f} percentage-point
      spread between levered and unlevered returns represents the value of the leverage effect,
      which is positive when the business return exceeds the cost of debt ({interest_rate:.2f}%).
    </div>""", unsafe_allow_html=True)

    # ── Capital structure donut + Equity bridge waterfall ──────────────────
    st.markdown('<div class="sec-header">2. Capital structure and equity value creation</div>', unsafe_allow_html=True)

    col_donut, col_waterfall = st.columns([2, 3])

    with col_donut:
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Pie(
            labels=["Debt", "Sponsor equity"],
            values=[debt_entry, equity_entry],
            hole=0.55,
            marker=dict(colors=[LRED, LBLUE], line=dict(color="white", width=2)),
            textfont=dict(size=12, color="#1a1a1a"),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}M<extra></extra>",
        ))
        fig_cap.update_layout(
            **{**BASE, "margin": dict(l=8, r=8, t=30, b=8)},
            height=280,
            showlegend=False,
            annotations=[dict(
                text=f"<b>{fmt_m(purchase_price)}</b><br>EV",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#1a1a1a", family="Inter, Arial"),
            )],
        )
        st.plotly_chart(fig_cap, use_container_width=True)

        st.markdown(f"""<div class="fig-caption">
          <b>Figure 1.</b> Entry capital structure.
          Debt of {fmt_m(debt_entry)} ({debt_pct}% of EV) and sponsor equity
          of {fmt_m(equity_entry)} ({100-debt_pct}% of EV).
        </div>""", unsafe_allow_html=True)

    with col_waterfall:
        # Waterfall: equity entry -> EBITDA growth -> multiple -> debt paydown -> exit equity
        wf_x = ["Entry equity", "EBITDA growth", "Multiple change", "Debt paydown", "Exit equity"]
        wf_y = [equity_entry, ev_from_ebitda_growth, ev_from_multiple, total_debt_paydown, exit_equity]
        wf_base = [0, equity_entry, equity_entry + ev_from_ebitda_growth,
                   equity_entry + ev_from_ebitda_growth + ev_from_multiple, 0]
        wf_colors = [LBLUE,
                     GREEN if ev_from_ebitda_growth >= 0 else RED,
                     GREEN if ev_from_multiple >= 0 else RED,
                     GREEN,
                     BLUE]
        wf_type = ["absolute", "relative", "relative", "relative", "absolute"]

        fig_wf = go.Figure(go.Waterfall(
            x=wf_x,
            y=wf_y,
            measure=wf_type,
            base=0,
            connector=dict(line=dict(color="#dddddd", width=1)),
            increasing=dict(marker=dict(color=GREEN)),
            decreasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=BLUE)),
            text=[fmt_m(v) for v in wf_y],
            textposition="outside",
            textfont=dict(size=11, color="#333333"),
        ))
        fig_wf.update_layout(
            **BASE, height=280,
            yaxis=dict(**ax("Equity value ($M)"), tickprefix="$", ticksuffix="M"),
            xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#444444"),
                       linecolor="#dddddd", linewidth=1, showline=True),
            showlegend=False,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown(f"""<div class="fig-caption">
          <b>Figure 2.</b> Equity value creation waterfall from entry to exit.
          EBITDA growth contributes {fmt_m(ev_from_ebitda_growth)} ({fmt_pct(ev_from_ebitda_growth/exit_equity) if exit_equity > 0 else 'n/a'} of exit equity),
          multiple {'expansion' if ev_from_multiple >= 0 else 'compression'} contributes {fmt_m(ev_from_multiple)}
          ({'+' if ev_from_multiple >= 0 else ''}{fmt_m(ev_from_multiple)}), and debt paydown contributes {fmt_m(total_debt_paydown)}.
        </div>""", unsafe_allow_html=True)

    # ── Debt schedule ──────────────────────────────────────────────────────
    st.markdown('<div class="sec-header">3. Leverage and coverage metrics over the holding period</div>', unsafe_allow_html=True)

    fig_lev = make_subplots(specs=[[{"secondary_y": True}]])
    fig_lev.add_trace(go.Bar(
        x=years_range, y=df["Debt Balance"].values,
        name="Debt balance ($M)",
        marker_color=LRED, opacity=0.75,
    ), secondary_y=False)
    fig_lev.add_trace(go.Scatter(
        x=years_range, y=(df["Debt Balance"] / df["EBITDA"]).values,
        name="Net leverage (Debt / EBITDA)",
        mode="lines+markers",
        line=dict(color=BLUE, width=2),
        marker=dict(size=6),
    ), secondary_y=True)
    fig_lev.add_trace(go.Scatter(
        x=years_range, y=(df["EBITDA"] / df["Interest"]).values,
        name="Interest coverage (EBITDA / Interest)",
        mode="lines+markers",
        line=dict(color=GREEN, width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
    ), secondary_y=True)

    fig_lev.update_layout(
        **BASE, height=320,
        yaxis=dict(**ax("Debt balance ($M)"), tickprefix="$"),
        yaxis2=dict(**ax("Multiple (x)", grid=False), overlaying="y", side="right",
                    title=dict(text="Multiple (x)", font=dict(size=12, color="#333333"))),
        xaxis=dict(**ax("Year"), dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color="#1a1a1a")),
    )
    st.plotly_chart(fig_lev, use_container_width=True)

    entry_leverage = debt_entry / ebitda_entry
    exit_leverage  = exit_debt / exit_ebitda if exit_ebitda > 0 else 0
    entry_coverage = ebitda_entry / (debt_entry * interest_rate / 100)
    exit_coverage  = exit_ebitda / (exit_debt * interest_rate / 100) if exit_debt > 0 else float("inf")

    st.markdown(f"""<div class="fig-caption">
      <b>Figure 3.</b> Debt balance (bars, left axis), net leverage in turns of EBITDA (solid line, right axis),
      and interest coverage ratio (dashed line, right axis) over the holding period.
      Entry leverage of {entry_leverage:.1f}x declines to {exit_leverage:.1f}x at exit through a combination of
      EBITDA growth and {fmt_m(total_debt_paydown)} of scheduled principal amortization.
      Interest coverage improves from {entry_coverage:.1f}x at entry to
      {exit_coverage:.1f}x at exit, reflecting both debt reduction and EBITDA expansion.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2: FINANCIAL PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════
with tab_schedule:

    st.markdown('<div class="sec-header">4. Projected income statement</div>', unsafe_allow_html=True)

    # P&L table
    pl_rows = ["Revenue", "EBITDA", "EBITDA Margin", "D&A", "EBIT", "Interest", "EBT", "Tax", "Net Income"]
    pl_labels = ["Revenue ($M)", "EBITDA ($M)", "EBITDA Margin (%)",
                 "D&A ($M)", "EBIT ($M)", "Interest expense ($M)",
                 "EBT ($M)", "Tax ($M)", "Net income ($M)"]
    pl_data = {"Metric": pl_labels}

    for _, row in df.iterrows():
        yr_col = f"Year {int(row['Year'])}"
        pl_data[yr_col] = [
            f"${row['Revenue']:.1f}",
            f"${row['EBITDA']:.1f}",
            f"{row['EBITDA Margin']*100:.1f}%",
            f"${row['D&A']:.1f}",
            f"${row['EBIT']:.1f}",
            f"(${row['Interest']:.1f})",
            f"${row['EBT']:.1f}",
            f"(${row['Tax']:.1f})",
            f"${row['Net Income']:.1f}",
        ]

    pl_df = pd.DataFrame(pl_data)
    st.dataframe(pl_df, use_container_width=True, hide_index=True)

    st.markdown(f"""<div class="fig-caption">
      <b>Table 2.</b> Projected income statement over the {hold_years}-year holding period.
      Revenue grows at a {fmt_pct(rev_growth/100)} annual rate; EBITDA margin expands by
      {margin_expansion} basis points per year from an entry margin of {ebitda_margin_entry:.1f}%.
      Interest expense reflects the outstanding debt balance at the beginning of each period.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-header">5. Free cash flow and debt schedule</div>', unsafe_allow_html=True)

    # Revenue and EBITDA chart
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
        x=years_range, y=(df["EBITDA Margin"] * 100).values,
        name="EBITDA margin (%)", mode="lines+markers",
        line=dict(color=GREEN, width=2),
        marker=dict(size=6),
        yaxis="y2",
    ))
    fig_rev.update_layout(
        **BASE, height=320, barmode="overlay",
        yaxis=dict(**ax("Value ($M)"), tickprefix="$"),
        yaxis2=dict(**ax("Margin (%)", grid=False), overlaying="y", side="right",
                    ticksuffix="%",
                    title=dict(text="EBITDA margin (%)", font=dict(size=12, color="#333333"))),
        xaxis=dict(**ax("Year"), dtick=1),
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown(f"""<div class="fig-caption">
      <b>Figure 4.</b> Revenue (light bars) and EBITDA (dark bars) projections over the holding period,
      with EBITDA margin on the right axis.
      Revenue CAGR of {fmt_pct(rev_cagr)} and margin expansion of {margin_expansion} bps/year
      drive EBITDA from {fmt_m(ebitda_entry)} at entry to {fmt_m(exit_ebitda)} at exit,
      a {fmt_pct(ebitda_cagr)} CAGR.
    </div>""", unsafe_allow_html=True)

    # FCF chart
    fig_fcf = go.Figure()
    fig_fcf.add_trace(go.Bar(
        x=years_range, y=df["FCF"].values,
        name="Free cash flow",
        marker_color=[GREEN if v >= 0 else RED for v in df["FCF"].values],
        opacity=0.8,
        text=[fmt_m(v) for v in df["FCF"].values],
        textposition="outside",
        textfont=dict(size=11, color="#333333"),
    ))
    fig_fcf.update_layout(
        **BASE, height=280, showlegend=False,
        yaxis=dict(**ax("FCF ($M)"), zeroline=True, zerolinecolor="#dddddd"),
        xaxis=dict(**ax("Year"), dtick=1),
    )
    st.plotly_chart(fig_fcf, use_container_width=True)

    st.markdown(f"""<div class="fig-caption">
      <b>Figure 5.</b> Levered free cash flow (post-interest, post-tax, post-CapEx and NWC) by year.
      FCF reflects cash available after servicing debt obligations and funding the business.
      Cumulative FCF over the holding period is {fmt_m(df['FCF'].sum())}.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: VALUE CREATION BRIDGE
# ══════════════════════════════════════════════════════════════════════════
with tab_bridge:

    st.markdown('<div class="sec-header">6. Return attribution: sources of equity value creation</div>', unsafe_allow_html=True)

    total_gain    = exit_equity - equity_entry
    pct_ebitda    = ev_from_ebitda_growth / total_gain * 100 if total_gain != 0 else 0
    pct_multiple  = ev_from_multiple / total_gain * 100      if total_gain != 0 else 0
    pct_paydown   = total_debt_paydown / total_gain * 100    if total_gain != 0 else 0

    b1, b2, b3, b4 = st.columns(4)
    bridge_kpis = [
        ("Total equity gain", fmt_m(total_gain), f"{moic:.2f}x MOIC on {fmt_m(equity_entry)}", "pos" if total_gain > 0 else "neg"),
        ("EBITDA growth", fmt_m(ev_from_ebitda_growth), f"{pct_ebitda:.0f}% of total gain", "pos" if ev_from_ebitda_growth > 0 else "neg"),
        ("Multiple change", fmt_m(ev_from_multiple), f"{pct_multiple:.0f}% of total gain", "pos" if ev_from_multiple > 0 else "neg"),
        ("Debt paydown", fmt_m(total_debt_paydown), f"{pct_paydown:.0f}% of total gain", "pos"),
    ]
    for col, (label, value, sub, cls) in zip([b1,b2,b3,b4], bridge_kpis):
        with col:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value <{cls}>">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="fig-caption" style="margin-top:0.75rem;">
      <b>Table 3.</b> Attribution of total equity gain of {fmt_m(total_gain)} to three sources.
      EBITDA growth accounts for {pct_ebitda:.0f}% of the gain, multiple change for {pct_multiple:.0f}%,
      and debt paydown for {pct_paydown:.0f}%.
    </div>""", unsafe_allow_html=True)

    # Attribution donut
    st.markdown('<div class="sec-header">7. Return decomposition and leverage effect</div>', unsafe_allow_html=True)

    col_pie, col_lever = st.columns(2)

    with col_pie:
        attrs    = [ev_from_ebitda_growth, max(ev_from_multiple, 0), total_debt_paydown]
        labels   = ["EBITDA growth", "Multiple expansion", "Debt paydown"]
        colors_p = [GREEN, LBLUE, LRED]
        # Filter out zero/negative for the donut
        pos_mask = [v > 0 for v in attrs]
        attrs_f  = [v for v, m in zip(attrs, labels) if v > 0]
        labels_f = [l for l, m in zip(labels, [True, ev_from_multiple > 0, True]) if m]

        valid_attrs  = [(v, l, c) for v, l, c in zip(attrs, labels, colors_p) if v > 0]
        if valid_attrs:
            va, vl, vc = zip(*valid_attrs)
            fig_pie = go.Figure(go.Pie(
                labels=vl, values=va, hole=0.5,
                marker=dict(colors=list(vc), line=dict(color="white", width=2)),
                textfont=dict(size=12, color="#1a1a1a"),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.1f}M<extra></extra>",
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
          <b>Figure 6.</b> Decomposition of total equity value creation by source.
          {'EBITDA growth is the primary driver, consistent with an operationally intensive value creation thesis.' if pct_ebitda > 50 else 'Multiple expansion is the primary driver, reflecting a re-rating of the business at exit.' if pct_multiple > pct_ebitda else 'Debt paydown is the primary driver, consistent with a financial engineering-led return profile.'}
        </div>""", unsafe_allow_html=True)

    with col_lever:
        # Levered vs unlevered IRR across debt levels
        dp_range_cont = list(range(20, 81, 5))
        irr_lev_range = []
        irr_ulev_line = [irr_unlevered * 100] * len(dp_range_cont)

        for dp in dp_range_cont:
            v = compute_irr_for(exit_ev_ebitda, dp)
            irr_lev_range.append(v * 100 if not np.isnan(v) else None)

        fig_lev2 = go.Figure()
        fig_lev2.add_trace(go.Scatter(
            x=dp_range_cont, y=irr_lev_range,
            name="Levered IRR",
            mode="lines+markers",
            line=dict(color=BLUE, width=2.5),
            marker=dict(size=5),
        ))
        fig_lev2.add_trace(go.Scatter(
            x=dp_range_cont, y=irr_ulev_line,
            name="Unlevered IRR",
            mode="lines",
            line=dict(color=GRAY, width=1.5, dash="dash"),
        ))
        fig_lev2.add_trace(go.Scatter(
            x=dp_range_cont, y=[interest_rate] * len(dp_range_cont),
            name="Cost of debt",
            mode="lines",
            line=dict(color=LRED, width=1.5, dash="dot"),
        ))
        fig_lev2.add_vline(x=debt_pct, line_dash="dot", line_color="#aaaaaa",
                           annotation_text=f"Current ({debt_pct}%)",
                           annotation_font=dict(size=10, color="#666666"),
                           annotation_position="top right")
        fig_lev2.update_layout(
            **BASE, height=300,
            yaxis=dict(**ax("IRR (%)"), ticksuffix="%"),
            xaxis=dict(**ax("Debt / EV at entry (%)"), ticksuffix="%"),
        )
        st.plotly_chart(fig_lev2, use_container_width=True)

        st.markdown(f"""<div class="fig-caption">
          <b>Figure 7.</b> Levered IRR as a function of entry leverage, holding all other parameters constant.
          The unlevered IRR ({irr_u_str}) and cost of debt ({interest_rate:.1f}%) are shown as reference lines.
          When the unlevered return exceeds the cost of debt, additional leverage amplifies the equity return;
          the leverage effect reverses when the cost of debt exceeds the underlying business return.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: SENSITIVITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tab_sensitivity:

    st.markdown('<div class="sec-header">8. IRR sensitivity: exit multiple vs. entry leverage</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="abstract-text" style="margin-bottom:1rem;font-size:14px;">
      The tables below present levered IRR and MOIC across a range of exit multiples (columns)
      and entry leverage levels (rows), holding all other model parameters constant.
      The center cell reflects the base case ({exit_ev_ebitda:.1f}x exit, {debt_pct}% debt).
      Green shading indicates IRR above 20%; red shading indicates IRR below 12%.
    </div>""", unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)

    with sc1:
        st.markdown("**Levered IRR sensitivity**")

        def style_irr(val):
            try:
                v = float(val.replace("%",""))
                if v >= 20: return "background-color: rgba(46,125,79,0.18); color: #1a4f82; font-weight:600;"
                if v < 12:  return "background-color: rgba(185,64,64,0.15); color: #7a1a1a; font-weight:600;"
            except Exception:
                pass
            return ""

        st.dataframe(
            sens_irr.style.applymap(style_irr),
            use_container_width=True,
        )

    with sc2:
        st.markdown("**MOIC sensitivity**")

        def style_moic(val):
            try:
                v = float(val.replace("x",""))
                if v >= 3.0: return "background-color: rgba(46,125,79,0.18); color: #1a4f82; font-weight:600;"
                if v < 2.0:  return "background-color: rgba(185,64,64,0.15); color: #7a1a1a; font-weight:600;"
            except Exception:
                pass
            return ""

        st.dataframe(
            sens_moic.style.applymap(style_moic),
            use_container_width=True,
        )

    st.markdown(f"""<div class="fig-caption">
      <b>Table 4 and Table 5.</b> Sensitivity of levered IRR and MOIC to exit multiple and entry leverage.
      Green cells: IRR >= 20% / MOIC >= 3.0x (institutional return threshold).
      Red cells: IRR < 12% / MOIC < 2.0x (below typical hurdle rate).
      The sensitivity illustrates that at the base case exit multiple of {exit_ev_ebitda:.1f}x,
      the transaction generates acceptable returns across most leverage levels,
      but is sensitive to multiple compression below {exit_mult_range[1]:.1f}x.
    </div>""", unsafe_allow_html=True)

    # IRR vs revenue growth and margin expansion heatmap
    st.markdown('<div class="sec-header">9. IRR sensitivity: operating performance</div>', unsafe_allow_html=True)

    growth_range  = list(range(max(0, rev_growth - 8), rev_growth + 10, 2))
    margin_range  = list(range(max(-100, margin_expansion - 100), margin_expansion + 150, 50))

    heat_z = []
    for mg in margin_range:
        row_vals = []
        for gr in growth_range:
            # Recompute exit EBITDA with different growth/margin
            exit_rev_s   = revenue_entry * ((1 + gr / 100) ** hold_years)
            exit_margin_s = entry_margin + (mg / 10000) * hold_years
            exit_ebitda_s = exit_rev_s * max(exit_margin_s, 0.01)
            exit_ev_s     = exit_ev_ebitda * exit_ebitda_s
            exit_eq_s     = exit_ev_s - exit_debt
            cfs_s         = [-equity_entry] + [0] * (hold_years - 1) + [exit_eq_s]
            v             = irr_calc(cfs_s)
            row_vals.append(round(v * 100, 1) if not np.isnan(v) else None)
        heat_z.append(row_vals)

    fig_heat = go.Figure(go.Heatmap(
        z=heat_z,
        x=[f"{g}%" for g in growth_range],
        y=[f"{m} bps" for m in margin_range],
        colorscale=[[0, "#c47a7a"], [0.4, "#f5f5f5"], [1, "#6ab06a"]],
        zmid=irr_levered * 100,
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
    # Mark base case
    base_x_idx = growth_range.index(rev_growth) if rev_growth in growth_range else len(growth_range)//2
    base_y_idx = margin_range.index(margin_expansion) if margin_expansion in margin_range else len(margin_range)//2
    fig_heat.add_shape(
        type="rect",
        x0=base_x_idx - 0.5, x1=base_x_idx + 0.5,
        y0=base_y_idx - 0.5, y1=base_y_idx + 0.5,
        line=dict(color="#1a1a1a", width=2),
    )
    fig_heat.update_layout(
        **{**BASE, "margin": dict(l=8, r=60, t=20, b=8)},
        height=360,
        xaxis=dict(**ax("Revenue growth rate"), title=dict(text="Annual revenue growth rate",
                   font=dict(size=12, color="#333333")), showline=False),
        yaxis=dict(**ax("EBITDA margin expansion"), title=dict(text="Annual margin expansion (bps)",
                   font=dict(size=12, color="#333333")), showline=False),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(f"""<div class="fig-caption">
      <b>Figure 8.</b> Levered IRR heatmap across revenue growth rates (x-axis) and annual EBITDA margin
      expansion in basis points (y-axis). The base case ({rev_growth}% growth, {margin_expansion} bps/year
      margin expansion) is outlined in black. Green shading indicates higher IRR; red shading indicates
      lower IRR. The analysis illustrates that returns are most sensitive to revenue growth,
      with margin expansion providing a secondary but meaningful contribution.
    </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown(f"""<div class="paper-footer">
      Model parameters: Entry EV/EBITDA = {entry_ev_ebitda:.1f}x; Entry EBITDA = {fmt_m(ebitda_entry)};
      Entry revenue = {fmt_m(revenue_entry)}; Debt/EV = {debt_pct}%; Interest rate = {interest_rate:.2f}%;
      Principal amortization = {amort_pct}%/year; Revenue growth = {rev_growth}%/year;
      Margin expansion = {margin_expansion} bps/year; CapEx = {capex_pct}% of revenue;
      NWC drag = {nwc_pct}% of revenue growth; D&A = {da_pct}% of revenue;
      Tax rate = {tax_rate}%; Exit EV/EBITDA = {exit_ev_ebitda:.1f}x; Holding period = {hold_years} years.
      IRR computed using Newton-Raphson iteration (numpy-financial). All figures in USD millions.
      Sensitivity analysis holds exit multiple and exit debt constant at base case values.
    </div>""", unsafe_allow_html=True)
