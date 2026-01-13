# ===============================
# NHL VALUE BETTING APP
# ===============================

import os
import csv
import requests
import streamlit as st
from datetime import datetime, timezone

# -------------------------------
# BASIC HELPERS
# -------------------------------
def implied_prob_from_odds(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def payout_from_odds(stake: float, odds: int) -> float:
    if odds > 0:
        return stake + (stake * odds / 100)
    return stake + (stake * 100 / abs(odds))

def profit_per_1_stake(odds: int) -> float:
    if odds > 0:
        return odds / 100
    return 100 / abs(odds)

def ev_per_1_stake(p_true: float, odds: int) -> float:
    return p_true * profit_per_1_stake(odds) - (1 - p_true)

def devig_two_way(p1, p2):
    s = p1 + p2
    return (p1 / s, p2 / s) if s > 0 else (0.5, 0.5)

def confidence_label(ev):
    if ev >= 0.05:
        return "STRONG ✅"
    if ev >= 0.03:
        return "MEDIUM 👍"
    if ev >= 0.015:
        return "SMALL ⚠️"
    return "TINY"

# -------------------------------
# ODDS FETCH
# -------------------------------
API_KEY = os.getenv("ODDS_API_KEY", "").strip()

def fetch_odds():
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# -------------------------------
# CANDIDATE BUILDER
# -------------------------------
def pick_candidates(games):
    candidates = []

    for g in games:
        home = g["home_team"]
        away = g["away_team"]
        game_id = g["id"]

        books = g.get("bookmakers", [])

        ml_probs = []
        ml_best = {home: None, away: None}

        for b in books:
            for m in b.get("markets", []):
                if m["key"] == "h2h":
                    o1, o2 = m["outcomes"]
                    p1, p2 = o1["price"], o2["price"]
                    ip1 = implied_prob_from_odds(p1)
                    ip2 = implied_prob_from_odds(p2)
                    t1, t2 = devig_two_way(ip1, ip2)

                    if o1["name"] == home:
                        ml_probs.append((t1, t2))
                    else:
                        ml_probs.append((t2, t1))

                    for o in [o1, o2]:
                        team = o["name"]
                        if ml_best[team] is None or o["price"] > ml_best[team]["odds"]:
                            ml_best[team] = {"odds": o["price"], "book": b["title"]}

        if ml_probs:
            p_home = sum(x[0] for x in ml_probs) / len(ml_probs)
            p_away = sum(x[1] for x in ml_probs) / len(ml_probs)

            for team, p_true in [(home, p_home), (away, p_away)]:
                odds = ml_best[team]["odds"]
                ev = ev_per_1_stake(p_true, odds)
                candidates.append({
                    "game_id": game_id,
                    "pick": f"{team} ML",
                    "market": "ML",
                    "odds": odds,
                    "book": ml_best[team]["book"],
                    "ev": ev
                })

    return candidates

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="NHL Value Bets", page_icon="🏒")
st.title("🏒 NHL Value Betting Model")

st.caption("Finds mispriced odds before games start. Bets only when the math says yes.")

# --- Intuitive Controls ---
pickiness = st.select_slider(
    "How picky should the model be?",
    options=["🔓 Loose", "⚖️ Balanced", "🔒 Strict"],
    value="⚖️ Balanced"
)

EV_MAP = {
    "🔓 Loose": 0.015,
    "⚖️ Balanced": 0.025,
    "🔒 Strict": 0.035
}
min_ev = EV_MAP[pickiness]

max_picks = st.slider("How many top bets to show?", 1, 10, 5)
daily_bankroll = st.number_input("Daily bankroll ($)", 10, 2000, 100, 10)

if not API_KEY:
    st.error("ODDS_API_KEY not set in Railway variables.")
    st.stop()

# -------------------------------
# RUN
# -------------------------------
if st.button("Run Model"):
    games = fetch_odds()
    cands = pick_candidates(games)

    cands.sort(key=lambda x: x["ev"], reverse=True)
    filtered = [c for c in cands if c["ev"] >= min_ev]

    if not filtered:
        best = max(cands, key=lambda x: x["ev"])
        st.warning(
            f"PASS — Best edge found was {best['ev']:.3f} per $1, below your {min_ev:.3f} threshold."
        )
    else:
        st.subheader("Ranked Bets (Best → Worst)")

        picks = filtered[:max_picks]
        stake = daily_bankroll / len(picks)

        for i, p in enumerate(picks, start=1):
            with st.container(border=True):
                st.markdown(
                    f"### {i}. {p['pick']} ({p['market']}) {p['odds']} @ {p['book']} — **{confidence_label(p['ev'])}**"
                )
                st.markdown(f"**Stake:** ${stake:.0f}")
                st.markdown(f"**Expected Value:** +${p['ev'] * stake:.2f}")
