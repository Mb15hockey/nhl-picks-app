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

def odds_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))

def stake_split_flat(total: int, n: int):
    if n <= 0:
        return []
    base = total // n
    stakes = [base] * n
    rem = total - base * n
    for i in range(rem):
        stakes[i] += 1
    return stakes

def stake_split_kelly_scaled(total: int, picks: list, kelly_fraction: float):
    """
    Fractional Kelly sizing using p_true (model) and odds.
    kelly_fraction:
      0.25 = quarter Kelly (conservative)
      0.50 = half Kelly (balanced)
    Caps single bet at 60% of bankroll as a safety guard.
    """
    bankroll = float(total)
    cap_per_pick = 0.60

    # compute raw Kelly fractions
    raw_fs = []
    for p in picks:
        p_true = float(p["p_true"])
        odds = int(p["odds"])
        dec = odds_to_decimal(odds)
        b = dec - 1.0
        q = 1.0 - p_true
        if b <= 0:
            f_star = 0.0
        else:
            f_star = (b * p_true - q) / b  # full Kelly
            f_star = max(0.0, f_star)

        f_star *= float(kelly_fraction)      # quarter/half Kelly
        f_star = min(f_star, cap_per_pick)   # cap
        raw_fs.append(f_star)

    s = sum(raw_fs)
    if s <= 0:
        return stake_split_flat(total, len(picks))

    # normalize to bankroll
    raw_stakes = [bankroll * (f / s) for f in raw_fs]
    stakes = [int(round(x)) for x in raw_stakes]

    # correct rounding overflow
    while sum(stakes) > total and any(x > 0 for x in stakes):
        i = stakes.index(max(stakes))
        stakes[i] -= 1

    # ensure at least $1 if possible
    for i in range(len(stakes)):
        if stakes[i] == 0 and total >= len(stakes):
            stakes[i] = 1
    while sum(stakes) > total:
        i = stakes.index(max(stakes))
        stakes[i] -= 1

    return stakes

# -------------------------------
# ODDS FETCH
# -------------------------------
API_KEY = os.getenv("ODDS_API_KEY", "").strip()

def fetch_odds():
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",  # ✅ ML + PL + TOTALS
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
        home = g.get("home_team")
        away = g.get("away_team")
        game_id = g.get("id")
        books = g.get("bookmakers", []) or []

        if not home or not away or not game_id:
            continue

        # --- ML ---
        ml_probs = []  # list of (p_home_true, p_away_true)
        ml_best = {home: None, away: None}

        # --- PL ---
        pl_probs = {}  # line_id -> list of (p1_true, p2_true, k1, k2)
        pl_best = {}   # line_id -> { k -> {odds, book} }

        # --- TOTALS ---
        tot_probs = {}  # total_point -> list of (p_over_true, p_under_true, over_key, under_key)
        tot_best = {}   # total_point -> { key -> {odds, book} }

        for b in books:
            book_name = b.get("title") or b.get("key") or "book"
            for m in (b.get("markets") or []):
                mkey = m.get("key")
                outcomes = m.get("outcomes") or []
                if len(outcomes) != 2:
                    continue

                o1, o2 = outcomes[0], outcomes[1]

                # ---------- MONEYLINE ----------
                if mkey == "h2h":
                    n1, n2 = o1.get("name"), o2.get("name")
                    p1, p2 = o1.get("price"), o2.get("price")
                    if not isinstance(p1, int) or not isinstance(p2, int):
                        continue

                    ip1 = implied_prob_from_odds(p1)
                    ip2 = implied_prob_from_odds(p2)
                    t1, t2 = devig_two_way(ip1, ip2)

                    if n1 == home and n2 == away:
                        ml_probs.append((t1, t2))
                    elif n1 == away and n2 == home:
                        ml_probs.append((t2, t1))
                    else:
                        continue

                    for o in (o1, o2):
                        nm, od = o.get("name"), o.get("price")
                        if nm in ml_best and isinstance(od, int):
                            if ml_best[nm] is None or od > ml_best[nm]["odds"]:
                                ml_best[nm] = {"odds": od, "book": book_name}

                # ---------- PUCK LINE ----------
                if mkey == "spreads":
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
                if mkey == "totals":
                    n1, n2 = str(o1.get("name", "")).lower(), str(o2.get("name", "")).lower()
                    p1, p2 = o1.get("price"), o2.get("price")
                    pt1, pt2 = o1.get("point"), o2.get("point")
                    if not isinstance(p1, int) or not isinstance(p2, int):
                        continue
                    if not isinstance(pt1, (int, float)) or not isinstance(pt2, (int, float)):
                        continue
                    if float(pt1) != float(pt2):
                        continue

                    total_point = float(pt1)
                    over_key = f"Over {total_point:g}"
                    under_key = f"Under {total_point:g}"

                    ip1 = implied_prob_from_odds(p1)
                    ip2 = implied_prob_from_odds(p2)
                    t1, t2 = devig_two_way(ip1, ip2)

                    # map to over/under
                    if n1 == "over" and n2 == "under":
                        p_over, p_under = t1, t2
                        over_odds, under_odds = p1, p2
                    elif n1 == "under" and n2 == "over":
                        p_over, p_under = t2, t1
                        over_odds, under_odds = p2, p1
                    else:
                        continue

                    tot_probs.setdefault(total_point, []).append((p_over, p_under, over_key, under_key))
                    tot_best.setdefault(total_point, {})

                    cur_over = tot_best[total_point].get(over_key)
                    if cur_over is None or over_odds > cur_over["odds"]:
                        tot_best[total_point][over_key] = {"odds": over_odds, "book": book_name}

                    cur_under = tot_best[total_point].get(under_key)
                    if cur_under is None or under_odds > cur_under["odds"]:
                        tot_best[total_point][under_key] = {"odds": under_odds, "book": book_name}

        # ---- ML candidates ----
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
                    "p_true": float(p_true),
                    "ev": float(ev),
                })

        # ---- PL candidates ----
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
                    "p_true": float(p_true),
                    "ev": float(ev),
                })

        # ---- TOTAL candidates ----
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
                    "p_true": float(p_true),
                    "ev": float(ev),
                })

    return candidates

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="NHL Picks", page_icon="🏒", layout="centered")

st.title("🏒 NHL Picks")
st.caption("Pregame only • Ranked best → worst • ML + Puck Line + Totals")

if not API_KEY:
    st.error("ODDS_API_KEY not set in Railway variables.")
    st.stop()

# ✅ Better than a slider for 3 options
pickiness = st.radio(
    "Pickiness (minimum edge required)",
    ["Loose (more bets)", "Balanced", "Strict (fewer bets)"],
    index=1,
    horizontal=True,
)

EV_MAP = {
    "Loose (more bets)": 0.015,
    "Balanced": 0.025,
    "Strict (fewer bets)": 0.035
}
min_ev = EV_MAP[pickiness]

st.caption(f"Current threshold: **{min_ev:.3f} EV per $1** (≈ **${min_ev*100:.2f} per $100** long-run)")

show_top_n = st.slider("Show top N picks (ranked)", 1, 25, 10)
stake_top_n = st.slider("How many to actually bet today?", 1, 10, 3)

daily_bankroll = st.number_input("Daily bankroll ($)", min_value=10, max_value=5000, value=100, step=10)

stake_method = st.radio(
    "Stake method",
    ["Flat (even split)", "Quarter Kelly (conservative)", "Half Kelly (balanced)"],
    index=0,
    horizontal=True,
)

if st.button("Run Model"):
    games = fetch_odds()
    cands = pick_candidates(games)
    if not cands:
        st.warning("PASS — no odds returned.")
        st.stop()

    cands.sort(key=lambda x: x["ev"], reverse=True)
    best_any = float(cands[0]["ev"])

    filtered = [c for c in cands if float(c["ev"]) >= float(min_ev)]
    filtered.sort(key=lambda x: x["ev"], reverse=True)

    if not filtered:
        st.warning(
            f"PASS — Best edge found was {best_any:.3f} EV per $1 "
            f"(below your {min_ev:.3f} threshold)."
        )
        st.subheader("Closest opportunities (below threshold)")
        for i, c in enumerate(cands[:5], start=1):
            st.write(f"{i}) {c['game']} — {c['pick']} ({c['market']}) {c['odds']} @ {c['book']} — EV {c['ev']:.3f}")
        st.stop()

    ranked = filtered[: int(show_top_n)]
    to_stake = ranked[: int(min(stake_top_n, len(ranked)))]

    # Stakes for the staked picks only
    if stake_method.startswith("Flat"):
        stakes = stake_split_flat(int(daily_bankroll), len(to_stake))
    else:
        k_frac = 0.25 if "Quarter" in stake_method else 0.50
        stakes = stake_split_kelly_scaled(int(daily_bankroll), to_stake, kelly_fraction=k_frac)

    st.subheader(f"Ranked Picks (Top {len(ranked)} shown)")
    st.caption("Stake is only assigned to the top picks you chose to bet today.")

    for i, c in enumerate(ranked, start=1):
        ev1 = float(c["ev"])
        ev100 = ev1 * 100.0
        label = confidence_label(ev1)

        stake_text = ""
        if c in to_stake:
            s = stakes[to_stake.index(c)]
            stake_text = f" • **Stake: ${int(s)}**"

        with st.container(border=True):
            st.markdown(
                f"### {i}. {c['game']} — {c['pick']} ({c['market']}) {c['odds']} @ {c['book']} "
                f"— **{label}**{stake_text}"
            )
            st.caption(f"Edge: ${ev100:.2f} per $100 (EV per $1: {ev1:.3f})")
