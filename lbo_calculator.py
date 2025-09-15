import streamlit as st

st.header("LBO Calculator")

purchase_price = st.number_input("Purchase Price ($M)", value=100)
debt_percent = st.slider("Debt %", min_value=0, max_value=100, value=60)
interest_rate = st.number_input("Interest Rate (%)", value=5.0)
exit_multiple = st.number_input("Exit Multiple", value=10.0)
years = st.slider("Holding Period (Years)", 1, 10, 5)

# Simple logic
debt = purchase_price * (debt_percent / 100)
equity = purchase_price - debt
exit_value = exit_multiple * (purchase_price / exit_multiple)
return_equity = exit_value - debt
irr = (return_equity / equity) ** (1 / years) - 1

st.write(f"Initial Equity: ${equity:.2f}M")
st.write(f"Exit Value: ${exit_value:.2f}M")
st.write(f"IRR: {irr * 100:.2f}%")
