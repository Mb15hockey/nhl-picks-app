import streamlit as st

st.set_page_config(page_title="NHL Picks", layout="centered")

st.title("🏒 NHL Picks App")

st.write("This app is running successfully on Railway.")

odds_text = st.text_area(
    "Paste NHL moneyline odds here:",
    placeholder="Example:\nRangers -130\nBruins +115"
)

if st.button("Run Picks"):
    st.success("Model ran successfully (placeholder).")
    st.write("Odds you entered:")
    st.code(odds_text)
