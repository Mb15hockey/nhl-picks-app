import streamlit as st

st.set_page_config(page_title="NHL Picks", layout="centered")

st.title("🏒 NHL Picks App")
st.write("App is live on Railway ✅")

odds_text = st.text_area(
    "Paste odds here (one line per bet):",
    placeholder="Rangers -130\nBruins +115"
)

if st.button("Run Picks"):
    st.success("Model ran (placeholder).")
    st.code(odds_text)
