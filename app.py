import os
import requests
import streamlit as st

# -------------------------------
# BASIC HELPERS
# -------------------------------
def implied_prob_from_odds(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def profit_per_1_stake(odds: int) -> float:
    # profit (not return) per $1 stake
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)

def ev_per_1_stake(p_true: float, odds: int) -> float:
    # EV per $1 stake
    return p_true * profit_per_1_stake(odds) - (1.0 - p_true)

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
        "markets": "h2h,spreads,totals",   # ✅ added totals
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# -------------------------------
# CANDIDATE BUILDER (ML + PL + TOTALS)
# -------------------------------
def pick_candidates(games):
    candidates = []

    for g in games:
        home = g.get("home_team")
        away = g.get("away_team")
        game_id = g.get("id")
        books = g.get("bookmakers", []) or []

        if not home or not away or not game_id:
            continue

        # -------------------
        # MONEYLINE (h2h)
        # -------------------
        ml_probs = []  # list of (p_home_true, p_away_true)
        ml_best = {home: None, away: None}

        # -------------------
        # PUCK LINE (spreads)
        # -------------------
        # keyed by line_id = "Team -1.5__vs__Team +1.5"
        pl_probs = {}  # line_id -> list of (p1_true, p2_true, k1, k2)
        pl_best = {}   # line_id -> { key -> {odds, book} }

        # -------------------
        # TOTALS (totals)
        # -------------------
        # keyed by total_point (like 5.5, 6.0)
        tot_probs = {}  # total_point -> list of (p_over_true, p_under_true, over_key, under_key)
        tot_best = {}   # total_point -> { key -> {odds, book} }

        for b in books:
            book_name = b.get("title") or b.get("key") or "book"

            for m in (b.get("markets") or []):
                mkey = m.get("key")
                outcomes = m.get("outcomes") or []

                # ---------- MONEYLINE ----------
                if mkey == "h2h" and len(outcomes) == 2:
                    o1, o2 = outcomes[0], outcomes[1]
                    n1, n2 = o1.get("name"), o2.get("name")
                    p1, p2 = o1.get("price"), o2.get("price")
                    if not isinstance(p1, int) or not isinstance(p2, int):
                        continue

                    ip1 = implied_prob_from_odds(p1)
                    ip2 = implied_prob_from_odds(p2)
                    t1, t2 = devig_two_way(ip1, ip2)

                    # map to home/away order
                    if n1 == home and n2 == away:
                        ml_probs.append((t1, t2))
                    elif n1 == away and n2 == home:
                        ml_probs.append((t2, t1))
                    else:
                        continue

                    # best prices
                    for o in outcomes:
                        nm = o.get("name")
                        od = o.get("price")
                        if nm in ml_best and isinstance(od, int):
                            if ml_best[nm] is None or od > ml_best[nm]["odds"]:
                                ml_best[nm] = {"odds": od, "book": book_name}

                # ---------- PUCK LINE (SPREADS) ----------
                if mkey == "spreads" and len(outcomes) == 2:
                    o1, o2 = outcomes[0], outcomes[1]
                    n1, n2 = o1.get("name"), o2.get("name")
                    p1, p2 = o1.get("price"), o2.get("price")
                    pt1, pt2 = o1.get("point"), o2.get("point")

                    if not isinstance(p1, int) or not isinstance(p2, int):
                        continue
                    if not isinstance(pt1, (int, float)) or not isinstance(pt2, (int, float)):
                        continue

                    k1 = f"{n1} {pt1:+g}"
                    k2 = f"{n2} {pt2:+g}"
                    line_id = f"{k1}__vs__{k2}"

                    ip1 = implied_prob_from_odds(p1)
                    ip2 = implied_prob_from_odds(p2)
                    t1, t2 = devig_two_way(ip1, ip2)

                    pl_probs.setdefault(line_id, []).append((t1, t2, k1, k2))
                    pl_best.setdefault(line_id, {})

                    for k, od in [(k1, p1), (k2, p2)]:
                        cur = pl_best[line_id].get(k)
                        if cur is None or od > cur["odds"]:
                            pl_best[line_id][k] = {"odds": od, "book": book_name}

                # ---------- TOTALS ----------
                if mkey == "totals" and len(outcomes) == 2:
                    o1, o2 = outcomes[0], outcomes[1]

                    # Odds API usually uses names "Over"/"Under" with same point
                    n1, n2 = o1.get("name"), o2.get("name")
                    p1, p2 = o1.get("price"), o2.get("price")
                    pt1, pt2 = o1.get("point"), o2.get("point")

                    if not isinstance(p1, int) or not isinstance(p2, int):
                        continue
                    if not isinstance(pt1, (int, float)) or not isinstance(pt2, (int, float)):
                        continue

                    # total points number should match; if not, skip
                    if float(pt1) != float(pt2):
                        continue

                    total_point = float(pt1)
                    over_key = f"Over {total_point:g}"
                    under_key = f"Under {total_point:g}"

                    ip1 = implied_prob_from_odds(p1)
                    ip2 = implied_prob_from_odds(p2)
                    t1, t2 = devig_two_way(ip1, ip2)

                    # map to Over/Under order
                    if str(n1).lower() == "over" and str(n2).lower() == "under":
                        p_over, p_under = t1, t2
                        over_odds, under_odds = p1, p2
                        over_book, under_book = book_name, book_name
                    elif str(n1).lower() == "under" and str(n2).lower() == "over":
                        p_over, p_under = t2, t1
                        over_odds, under_odds = p2, p1
                        over_book, under_book = book_name, book_name
                    else:
                        # unexpected naming, skip
                        continue

                    tot_probs.setdefault(total_point, []).append((p_over, p_under, over_key, under_key))
                    tot_best.setdefault(total_point, {})

                    # best prices for over/under at this total
                    cur_over = tot_best[total_point].get(over_key)
                    if cur_over is None or over_odds > cur_over["odds"]:
                        tot_best[total_point][over_key] = {"odds": over_odds, "book": over_book}

                    cur_under = tot_best[total_point].get(under_key)
                    if cur_under is None or under_odds > cur_under["odds"]:
                        tot_best[total_point][under_key] = {"odds": under_odds, "book": under_book}

        # ---- build ML candidates from consensus ----
        if ml_probs and ml_best[home] and ml_best[away]:
            p_home = sum(x[0] for x in ml_probs) / len(ml_probs)
            p_away = sum(x[1] for x in ml_probs) / len(ml_probs)

            for team, p_true in [(home, p_home), (away, p_away)]:
                best = ml_best[team]
                odds = int(best["odds"])
                ev = ev_per_1_stake(float(p_true), odds)
                candidates.append({
                    "game_id": game_id,
                    "game": f"{away} @ {home}",
                    "pick": f"{team} ML",
                    "market": "ML",
                    "odds": odds,
                    "book": best["book"],
                    "ev": float(ev),
                })

        # ---- build PL candidates from consensus ----
        for line_id, rows in pl_probs.items():
            p1 = sum(r[0] for r in rows) / len(rows)
            p2 = sum(r[1] for r in rows) / len(rows)
            k1 = rows[0][2]
            k2 = rows[0][3]

            for k, p_true in [(k1, p1), (k2, p2)]:
                best = pl_best[line_id].get(k)
                if not best:
                    continue
                odds = int(best["odds"])
                ev = ev_per_1_stake(float(p_true), odds)
                candidates.append({
                    "game_id": game_id,
                    "game": f"{away} @ {home}",
                    "pick": k,
                    "market": "PL",
                    "odds": odds,
                    "book": best["book"],
                    "ev": float(ev),
                })

        # ---- build TOTAL candidates from consensus ----
        for total_point, rows in tot_probs.items():
            p_over = sum(r[0] for r in rows) / len(rows)
            p_under = sum(r[1] for r in rows) / len(rows)
            over_key = rows[0][2]
            under_key = rows[0][3]

            for k, p_true in [(over_key, p_over), (under_key, p_under)]:
                best = tot_best[total_point].get(k)
                if not best:
                    continue
                odds = int(best["odds"])
                ev = ev_per_1_stake(float(p_true), odds)
                candidates.append({
                    "game_id": game_id,
                    "game": f"{away} @ {home}",
                    "pick": k,
                    "market": "TOTAL",
                    "odds": odds,
                    "book": best["book"],
                    "ev": float(ev),
                })

    return candidates

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="NHL Value Bets", page_icon="🏒")
st.title("🏒 NHL Value Betting Model")
st.caption("Now supports: Moneyline • Puck Line • Totals (Over/Under). Ranked best → worst.")

pickiness = st.select_slider(
    "How picky should the model be?",
    options=["🔓 Loose", "⚖️ Balanced", "🔒 Strict"],
    value="⚖️ Balanced",
)

EV_MAP = {"🔓 Loose": 0.015, "⚖️ Balanced": 0.025, "🔒 Strict": 0.035}
min_ev = EV_MAP[pickiness]

max_picks = st.slider("How many top bets to show?", 1, 15, 8)
daily_bankroll = st.number_input("Daily bankroll ($)", 10, 2000, 100, 10)

if not API_KEY:
    st.error("ODDS_API_KEY not set in Railway variables.")
    st.stop()

if st.button("Run Model"):
    games = fetch_odds()
    cands = pick_candidates(games)

    cands.sort(key=lambda x: x["ev"], reverse=True)
    filtered = [c for c in cands if c["ev"] >= min_ev]

    if not cands:
        st.warning("PASS — no odds returned.")
        st.stop()

    if not filtered:
        best = cands[0]
        st.warning(f"PASS — Best edge found was {best['ev']:.3f} per $1, below your {min_ev:.3f} threshold.")
        st.write("Closest opportunities:")
        for i, c in enumerate(cands[:5], start=1):
            st.write(f"{i}) {c['game']} — {c['pick']} ({c['market']}) {c['odds']} @ {c['book']} — EV {c['ev']:.3f}")
        st.stop()

    picks = filtered[:max_picks]
    stake = daily_bankroll / len(picks)

    st.subheader("Ranked Bets (Best → Worst)")
    for i, p in enumerate(picks, start=1):
        with st.container(border=True):
            st.markdown(f"### {i}. {p['game']} — {p['pick']} ({p['market']}) {p['odds']} @ {p['book']} — **{confidence_label(p['ev'])}**")
            st.markdown(f"**Stake (even split):** ${stake:.0f}")
            st.markdown(f"**Expected Value (this bet):** +${p['ev'] * stake:.2f}")
