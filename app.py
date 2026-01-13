import os
import requests
import streamlit as st

st.set_page_config(page_title="NHL Picks", page_icon="🏒", layout="centered")

API_KEY = os.getenv("ODDS_API_KEY", "").strip()

def american_to_implied_prob(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)

def devig_two_way(p1: float, p2: float):
    s = p1 + p2
    if s <= 0:
        return 0.5, 0.5
    return p1 / s, p2 / s

def profit_per_1_stake(odds: int) -> float:
    if odds > 0:
        return odds / 100.0
    return 100.0 / (-odds)

def ev_per_1_stake(p_true: float, odds: int) -> float:
    profit = profit_per_1_stake(odds)
    return p_true * profit - (1 - p_true)

def fetch_odds():
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def pick_candidates(games_json):
    candidates = []

    for g in games_json:
        home = g.get("home_team")
        away = g.get("away_team")
        books = g.get("bookmakers", []) or []

        # ----- MONEYLINE -----
        ml_book_probs = []  # list of (p_home_true, p_away_true)
        ml_best = {home: {"odds": None, "book": None}, away: {"odds": None, "book": None}}

        # ----- PUCK LINE (SPREADS) -----
        # Key by line (e.g., "home_-1.5__away_+1.5") because books may offer multiple.
        pl_book_probs = {}  # line_id -> list of (p_outcomeA_true, p_outcomeB_true, outcomeA_key, outcomeB_key)
        pl_best = {}        # line_id -> { outcome_key: {"odds": int, "book": str, "point": float} }

        for b in books:
            book_name = b.get("title") or b.get("key") or "book"
            for m in (b.get("markets") or []):
                mkey = m.get("key")
                outcomes = m.get("outcomes") or []

                # Moneyline
                if mkey == "h2h" and len(outcomes) == 2:
                    o1, o2 = outcomes[0], outcomes[1]
                    n1, n2 = o1.get("name"), o2.get("name")
                    p1, p2 = o1.get("price"), o2.get("price")
                    if not isinstance(p1, int) or not isinstance(p2, int):
                        continue

                    ip1 = american_to_implied_prob(p1)
                    ip2 = american_to_implied_prob(p2)
                    t1, t2 = devig_two_way(ip1, ip2)

                    # map to home/away
                    if n1 == home and n2 == away:
                        ml_book_probs.append((t1, t2))
                    elif n1 == away and n2 == home:
                        ml_book_probs.append((t2, t1))
                    else:
                        continue

                    # track best odds
                    for nm, od in [(n1, p1), (n2, p2)]:
                        if nm in ml_best:
                            cur = ml_best[nm]["odds"]
                            if cur is None or od > cur:
                                ml_best[nm] = {"odds": od, "book": book_name}

                # Puck line (spreads)
                if mkey == "spreads" and len(outcomes) == 2:
                    o1, o2 = outcomes[0], outcomes[1]
                    n1, n2 = o1.get("name"), o2.get("name")
                    p1, p2 = o1.get("price"), o2.get("price")
                    pt1, pt2 = o1.get("point"), o2.get("point")
                    if not isinstance(p1, int) or not isinstance(p2, int):
                        continue
                    if not isinstance(pt1, (int, float)) or not isinstance(pt2, (int, float)):
                        continue

                    # Build stable outcome keys like "Rangers -1.5"
                    k1 = f"{n1} {pt1:+g}"
                    k2 = f"{n2} {pt2:+g}"
                    line_id = f"{k1}__vs__{k2}"

                    ip1 = american_to_implied_prob(p1)
                    ip2 = american_to_implied_prob(p2)
                    t1, t2 = devig_two_way(ip1, ip2)

                    pl_book_probs.setdefault(line_id, []).append((t1, t2, k1, k2))
                    pl_best.setdefault(line_id, {})
                    for k, od, pt in [(k1, p1, pt1), (k2, p2, pt2)]:
                        cur = pl_best[line_id].get(k, {}).get("odds")
                        if cur is None or od > cur:
                            pl_best[line_id][k] = {"odds": od, "book": book_name, "point": pt}

        # Moneyline consensus true probs (average de-vig across books)
        if ml_book_probs:
            p_home = sum(x[0] for x in ml_book_probs) / len(ml_book_probs)
            p_away = sum(x[1] for x in ml_book_probs) / len(ml_book_probs)

            for team, p_true in [(home, p_home), (away, p_away)]:
                best = ml_best.get(team)
                if best and best["odds"] is not None:
                    odds = best["odds"]
                    ev = ev_per_1_stake(p_true, odds)
                    candidates.append({
                        "market": "ML",
                        "pick": f"{team} ML",
                        "odds": odds,
                        "book": best["book"],
                        "ev": ev
                    })

        # Puck line consensus per line_id (average de-vig across books)
        for line_id, rows in pl_book_probs.items():
            # Each row is (t1, t2, k1, k2) but k1/k2 same within line_id
            p1 = sum(r[0] for r in rows) / len(rows)
            p2 = sum(r[1] for r in rows) / len(rows)
            k1 = rows[0][2]
            k2 = rows[0][3]

            for k, p_true in [(k1, p1), (k2, p2)]:
                best = pl_best[line_id].get(k)
                if best and best["odds"] is not None:
                    odds = best["odds"]
                    ev = ev_per_1_stake(p_true, odds)
                    candidates.append({
                        "market": "PL",
                        "pick": k,  # already "Team ±1.5"
                        "odds": odds,
                        "book": best["book"],
                        "ev": ev
                    })

    return candidates

def stake_split(total: int, n: int):
    if n <= 0:
        return []
    if n == 1:
        return [total]
    if n == 2:
        return [total // 2, total - (total // 2)]
    # n >= 3 -> 3 bets max for now
    a = total // 3
    return [a, a, total - 2 * a]

# ---------- UI ----------
st.title("🏒 NHL Picks")
st.caption("Pregame only • Picks only • Moneyline + Puck line • Top 1–3 by edge")

daily_bankroll = st.number_input("Daily bankroll ($)", min_value=10, max_value=1000, value=100, step=10)
min_ev = st.slider("Minimum EV (edge) per $1 stake", min_value=0.00, max_value=0.10, value=0.03, step=0.005)
max_picks = st.selectbox("Max picks", options=[1, 2, 3], index=2)

if not API_KEY:
    st.error("ODDS_API_KEY is not set in Railway Variables.")
    st.stop()

if st.button("Run Picks"):
    try:
        games = fetch_odds()
        cands = pick_candidates(games)

        # Filter by EV threshold
        cands = [c for c in cands if c["ev"] >= float(min_ev)]
        cands.sort(key=lambda x: x["ev"], reverse=True)

        # Picks only: output up to max_picks
        picks = cands[: int(max_picks)]

        if not picks:
            st.write("PASS")
        else:
            stakes = stake_split(int(daily_bankroll), len(picks))
            # PICKS ONLY output
            for i, (p, stake) in enumerate(zip(picks, stakes), start=1):
                st.write(f"{i}) {p['pick']} ({p['market']}) {p['odds']} @ {p['book']} — ${stake}")

    except requests.HTTPError as e:
        st.error(f"Odds API error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
