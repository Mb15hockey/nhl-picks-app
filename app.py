import os
import json
import sqlite3
from datetime import datetime, timezone
import requests
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
API_KEY = os.getenv("ODDS_API_KEY", "").strip()
DB_PATH = os.getenv("DB_PATH", "app.db")
SPORT_KEY = "icehockey_nhl"

# Tune over time. Defaults to 1.0 if not listed.
BOOK_WEIGHTS = {
    "Pinnacle": 2.5,
    "Circa Sports": 2.0,
    "BetRivers": 1.2,
    "DraftKings": 1.0,
    "FanDuel": 1.0,
    "BetMGM": 1.0,
    "Caesars": 1.0,
}

# =========================================================
# BASIC HELPERS
# =========================================================
def w(book_name: str) -> float:
    return float(BOOK_WEIGHTS.get(book_name, 1.0))

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

def fair_american_from_p(p: float) -> int:
    # convert true probability to fair American odds (no vig)
    p = min(max(p, 1e-6), 1 - 1e-6)
    dec = 1.0 / p
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100))
    return int(round(-100.0 / (dec - 1.0)))

# =========================================================
# DB LAYER
# =========================================================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS odds_snapshots (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts_utc TEXT NOT NULL,
      sport TEXT NOT NULL,
      payload_json TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS bets (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      placed_ts_utc TEXT NOT NULL,
      game_id TEXT NOT NULL,
      game TEXT NOT NULL,
      market TEXT NOT NULL,
      pick TEXT NOT NULL,
      book TEXT NOT NULL,
      odds_placed INTEGER NOT NULL,
      p_true REAL NOT NULL,
      ev REAL NOT NULL,
      stake INTEGER NOT NULL,
      close_odds INTEGER,
      clv_cents REAL,
      status TEXT DEFAULT 'OPEN',     -- OPEN/WIN/LOSS/PUSH/VOID
      settled_ts_utc TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_snapshot(games, sport=SPORT_KEY):
    conn = db()
    conn.execute(
        "INSERT INTO odds_snapshots (ts_utc, sport, payload_json) VALUES (?, ?, ?)",
        (now_utc_iso(), sport, json.dumps(games)),
    )
    conn.commit()
    conn.close()

def save_bets(to_stake, stakes):
    conn = db()
    for c, s in zip(to_stake, stakes):
        conn.execute("""
        INSERT INTO bets (
          placed_ts_utc, game_id, game, market, pick, book,
          odds_placed, p_true, ev, stake
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_utc_iso(),
            c["game_id"], c["game"], c["market"], c["pick"], c["book"],
            int(c["odds"]), float(c["p_true"]), float(c["ev"]), int(s)
        ))
    conn.commit()
    conn.close()

def clv_cents(odds_bet: int, odds_close: int) -> float:
    p_bet = implied_prob_from_odds(odds_bet)
    p_close = implied_prob_from_odds(odds_close)
    return (p_close - p_bet) * 100.0  # + is good: you beat the close

def update_bet_status(bet_id: int, new_status: str):
    conn = db()
    settled = None
    if new_status in ("WIN", "LOSS", "PUSH", "VOID"):
        settled = now_utc_iso()
    conn.execute(
        "UPDATE bets SET status=?, settled_ts_utc=? WHERE id=?",
        (new_status, settled, bet_id),
    )
    conn.commit()
    conn.close()

# =========================================================
# ODDS FETCH
# =========================================================
def fetch_odds():
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",  # ML + PL + TOTALS
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

# =========================================================
# CANDIDATE BUILDER (WITH BOOK WEIGHTING)
# =========================================================
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
        ml_probs = []  # list of (p_home_true, p_away_true, weight)
        ml_best = {home: None, away: None}

        # --- PL ---
        pl_probs = {}  # line_id -> list of (p1_true, p2_true, k1, k2, weight)
        pl_best = {}   # line_id -> { k -> {odds, book} }

        # --- TOTALS ---
        tot_probs = {}  # total_point -> list of (p_over_true, p_under_true, over_key, under_key, weight)
        tot_best = {}   # total_point -> { key -> {odds, book} }

        for b in books:
            book_name = b.get("title") or b.get("key") or "book"
            bw = w(book_name)

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
                        ml_probs.append((t1, t2, bw))
                    elif n1 == away and n2 == home:
                        ml_probs.append((t2, t1, bw))
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

                    pl_probs.setdefault(line_id, []).append((t1, t2, k1, k2, bw))
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

                    if n1 == "over" and n2 == "under":
                        p_over, p_under = t1, t2
                        over_odds, under_odds = p1, p2
                    elif n1 == "under" and n2 == "over":
                        p_over, p_under = t2, t1
                        over_odds, under_odds = p2, p1
                    else:
                        continue

                    tot_probs.setdefault(total_point, []).append((p_over, p_under, over_key, under_key, bw))
                    tot_best.setdefault(total_point, {})

                    cur_over = tot_best[total_point].get(over_key)
                    if cur_over is None or over_odds > cur_over["odds"]:
                        tot_best[total_point][over_key] = {"odds": over_odds, "book": book_name}

                    cur_under = tot_best[total_point].get(under_key)
                    if cur_under is None or under_odds > cur_under["odds"]:
                        tot_best[total_point][under_key] = {"odds": under_odds, "book": book_name}

        # ---- ML candidates ----
        if ml_probs and ml_best[home] and ml_best[away]:
            wt_sum = sum(wt for (_, _, wt) in ml_probs) or 1.0
            p_home = sum(ph * wt for (ph, _, wt) in ml_probs) / wt_sum
            p_away = sum(pa * wt for (_, pa, wt) in ml_probs) / wt_sum

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
                    "n_books": len(ml_probs),
                })

        # ---- PL candidates ----
        for line_id, rows in pl_probs.items():
            wt_sum = sum(r[4] for r in rows) or 1.0
            p1 = sum(r[0] * r[4] for r in rows) / wt_sum
            p2 = sum(r[1] * r[4] for r in rows) / wt_sum
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
                    "n_books": len(rows),
                })

        # ---- TOTAL candidates ----
        for total_point, rows in tot_probs.items():
            wt_sum = sum(r[4] for r in rows) or 1.0
            p_over = sum(r[0] * r[4] for r in rows) / wt_sum
            p_under = sum(r[1] * r[4] for r in rows) / wt_sum
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
                    "n_books": len(rows),
                })

    return candidates

# =========================================================
# CLV UPDATER
# =========================================================
def find_best_odds_for_bet(cands, bet_row):
    for c in cands:
        if (
            c["game_id"] == bet_row["game_id"]
            and c["market"] == bet_row["market"]
            and c["pick"] == bet_row["pick"]
        ):
            return int(c["odds"])
    return None

def update_closing_lines_for_open_bets():
    games = fetch_odds()
    save_snapshot(games)
    cands = pick_candidates(games)

    conn = db()
    open_bets = conn.execute("SELECT * FROM bets WHERE status='OPEN'").fetchall()

    updated = 0
    for b in open_bets:
        close = find_best_odds_for_bet(cands, b)
        if close is None:
            continue
        cents = clv_cents(int(b["odds_placed"]), int(close))
        conn.execute("""
          UPDATE bets
          SET close_odds=?, clv_cents=?
          WHERE id=?
        """, (int(close), float(cents), int(b["id"])))
        updated += 1

    conn.commit()
    conn.close()
    return updated

# =========================================================
# INIT DB BEFORE UI
# =========================================================
init_db()

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="NHL Picks", page_icon="🏒", layout="centered")
st.title("🏒 NHL Picks")
st.caption("Pregame only • Ranked best → worst • ML + Puck Line + Totals • Tracker + CLV")

if not API_KEY:
    st.error("ODDS_API_KEY not set in Railway variables.")
    st.stop()

tab_picks, tab_tracker = st.tabs(["Picks", "Tracker"])

# -----------------------
# PICKS TAB
# -----------------------
with tab_picks:
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

    show_top_n = st.slider("Show up to N opportunities (ranked)", 1, 25, 10)
    st.caption("These are all qualifying +EV bets you can review (not all will be staked).")

    stake_top_n = st.slider("Number of bets to stake today", 1, 10, 3)
    st.caption("Stakes are assigned only to the top bets you choose to stake.")

    daily_bankroll = st.number_input("Daily bankroll ($)", min_value=10, max_value=5000, value=100, step=10)

    stake_method = st.radio(
        "Stake method",
        ["Flat (even split)", "Quarter Kelly (conservative)", "Half Kelly (balanced)"],
        index=0,
        horizontal=True,
    )

    st.caption("Confidence reflects **edge size (EV)**, not the chance of winning any single bet.")
    st.caption("Legend: STRONG ≥ 0.050 • MEDIUM ≥ 0.030 • SMALL ≥ 0.015 • TINY < 0.015 (EV per $1)")

    if st.button("Run Model"):
        games = fetch_odds()
        save_snapshot(games)

        cands = pick_candidates(games)
        if not cands:
            st.warning("PASS — no odds returned.")
            st.stop()

        # Sort all candidates by EV
        cands.sort(key=lambda x: x["ev"], reverse=True)
        best_any = float(cands[0]["ev"])

        # Filter by threshold
        filtered = [c for c in cands if float(c["ev"]) >= float(min_ev)]
        filtered.sort(key=lambda x: x["ev"], reverse=True)

        if not filtered:
            st.warning(
                f"PASS — Best edge found was {best_any:.3f} EV per $1 "
                f"(below your {min_ev:.3f} threshold)."
            )
            st.subheader("Closest opportunities (below threshold)")
            for i, c in enumerate(cands[:5], start=1):
                st.write(
                    f"{i}) {c['game']} — {c['pick']} ({c['market']}) "
                    f"{c['odds']} @ {c['book']} — EV {c['ev']:.3f}"
                )
            st.stop()

        # "Show up to N opportunities"
        ranked = filtered[: int(show_top_n)]
        to_stake = ranked[: int(min(stake_top_n, len(ranked)))]

        # Stakes for the staked picks only
        if stake_method.startswith("Flat"):
            stakes = stake_split_flat(int(daily_bankroll), len(to_stake))
        else:
            k_frac = 0.25 if "Quarter" in stake_method else 0.50
            stakes = stake_split_kelly_scaled(int(daily_bankroll), to_stake, kelly_fraction=k_frac)

        # Clear, non-confusing headers
        qualifying_count = len(filtered)
        shown_count = len(ranked)

        st.subheader(f"Ranked Opportunities — {qualifying_count} passed your EV filter")
        if qualifying_count < show_top_n:
            st.caption(
                f"You asked to show up to **{show_top_n}**, but only **{qualifying_count}** met the minimum edge today."
            )
        else:
            st.caption(f"Showing the top **{shown_count}** opportunities (ranked best → worst).")

        st.caption("Only the top bets you selected in **Number of bets to stake today** will show a stake amount.")

        total_staked = sum(int(s) for s in stakes) if stakes else 0
        unused = int(daily_bankroll) - total_staked
        st.caption(f"Bankroll: **${int(daily_bankroll)}** • Staked: **${total_staked}** • Unused: **${unused}**")

        for i, c in enumerate(ranked, start=1):
            ev1 = float(c["ev"])
            ev100 = ev1 * 100.0
            label = confidence_label(ev1)

            is_staked = (c in to_stake)
            tag = "✅ STAKED" if is_staked else "👀 WATCH"

            stake_text = ""
            if is_staked:
                s = stakes[to_stake.index(c)]
                stake_text = f" • **Stake: ${int(s)}**"

            with st.container(border=True):
                st.markdown(
                    f"### {i}. {tag} — {c['game']} — {c['pick']} ({c['market']}) "
                    f"{c['odds']} @ {c['book']} — **{label}**{stake_text}"
                )
                st.caption(
                    f"Edge: ${ev100:.2f} per $100 (EV per $1: {ev1:.3f}) • Books used: {c.get('n_books','—')}"
                )

                with st.expander("Why this bet?"):
                    fair = fair_american_from_p(float(c["p_true"]))
                    st.write(f"Fair (no-vig) line: **{fair}**")
                    st.write(f"Best available: **{c['odds']} @ {c['book']}**")
                    st.write(f"Model p_true: **{float(c['p_true']):.3f}**")
                    st.write(f"EV per $100: **${float(c['ev']) * 100:.2f}**")

        st.divider()
        st.subheader("Action")
        if st.button("Save these bets to Tracker"):
            if not to_stake:
                st.warning("No staked bets to save (increase ‘Number of bets to stake today’ or loosen your filter).")
            else:
                save_bets(to_stake, stakes)
                st.success("Saved! Go to the Tracker tab to view CLV and results.")

# -----------------------
# TRACKER TAB
# -----------------------
with tab_tracker:
    st.subheader("Bet Tracker (CLV + Results)")
    st.caption("Update closing lines near game time to measure CLV. Mark results manually (MVP).")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Update closing lines (CLV) for OPEN bets"):
            n = update_closing_lines_for_open_bets()
            st.success(f"Updated closing lines for {n} open bets.")
    with col2:
        st.write("")

    conn = db()
    rows = conn.execute("""
      SELECT id, placed_ts_utc, game, market, pick, book,
             odds_placed, stake, close_odds, clv_cents, status
      FROM bets
      ORDER BY id DESC
      LIMIT 200
    """).fetchall()
    conn.close()

    if not rows:
        st.info("No bets saved yet. Run the model, then save bets to start tracking.")
    else:
        clv_vals = [r["clv_cents"] for r in rows if r["clv_cents"] is not None]
        avg_clv = sum(clv_vals) / len(clv_vals) if clv_vals else None

        st.metric("Tracked bets", len(rows))
        if avg_clv is not None:
            st.metric("Avg CLV (cents)", f"{avg_clv:.2f}")

        st.divider()

        for r in rows:
            with st.container(border=True):
                st.markdown(f"**{r['game']}** — {r['pick']} ({r['market']})")
                st.caption(f"Placed: {r['placed_ts_utc']} • Book: {r['book']} • Stake: ${r['stake']}")

                if r["clv_cents"] is not None:
                    st.write(
                        f"Odds placed: **{r['odds_placed']}** | "
                        f"Close: **{r['close_odds'] if r['close_odds'] is not None else '—'}** | "
                        f"CLV: **{r['clv_cents']:.2f}** cents"
                    )
                else:
                    st.write(
                        f"Odds placed: **{r['odds_placed']}** | Close: **—** | CLV: **—**"
                    )

                status = st.selectbox(
                    "Set status",
                    ["OPEN", "WIN", "LOSS", "PUSH", "VOID"],
                    index=["OPEN", "WIN", "LOSS", "PUSH", "VOID"].index(r["status"]),
                    key=f"status_{r['id']}"
                )
                if st.button("Save status", key=f"save_status_{r['id']}"):
                    update_bet_status(int(r["id"]), status)
                    st.success("Status updated.")
