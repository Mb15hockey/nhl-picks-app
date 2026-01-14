import os
import json
import sqlite3
from datetime import datetime, timezone, date
import requests
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
API_KEY = os.getenv("ODDS_API_KEY", "").strip()
DB_PATH = os.getenv("DB_PATH", "app.db")

# Popular pro leagues
LEAGUES = {
    "NHL": "icehockey_nhl",
    "AHL": "icehockey_ahl",
    "SHL (Sweden)": "icehockey_sweden_hockey_league",
    "HockeyAllsvenskan (Sweden)": "icehockey_sweden_allsvenskan",
    "Liiga (Finland)": "icehockey_liiga",
    # Optional later:
    # "Mestis (Finland)": "icehockey_mestis",
}

PRO_MIX_5 = ["NHL", "AHL", "SHL (Sweden)", "HockeyAllsvenskan (Sweden)", "Liiga (Finland)"]

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

def is_bovada(book: str) -> bool:
    if not book:
        return False
    b = str(book).strip().lower()
    return b == "bovada" or "bovada" in b

def implied_prob_from_odds(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def profit_per_1_stake(odds: int) -> float:
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)

def ev_per_1_stake(p_true: float, odds: int) -> float:
    return p_true * profit_per_1_stake(odds) - (1.0 - p_true)

def devig_two_way(p1, p2):
    s = p1 + p2
    return (p1 / s, p2 / s) if s > 0 else (0.5, 0.5)

def confidence_label(ev: float) -> str:
    if ev >= 0.05:
        return "STRONG"
    if ev >= 0.03:
        return "MEDIUM"
    if ev >= 0.015:
        return "SMALL"
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

        f_star *= float(kelly_fraction)
        f_star = min(f_star, cap_per_pick)
        raw_fs.append(f_star)

    s = sum(raw_fs)
    if s <= 0:
        return stake_split_flat(total, len(picks))

    raw_stakes = [bankroll * (f / s) for f in raw_fs]
    stakes = [int(round(x)) for x in raw_stakes]

    while sum(stakes) > total and any(x > 0 for x in stakes):
        i = stakes.index(max(stakes))
        stakes[i] -= 1

    for i in range(len(stakes)):
        if stakes[i] == 0 and total >= len(stakes):
            stakes[i] = 1
    while sum(stakes) > total:
        i = stakes.index(max(stakes))
        stakes[i] -= 1

    return stakes

def fair_american_from_p(p: float) -> int:
    p = min(max(p, 1e-6), 1 - 1e-6)
    dec = 1.0 / p
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100))
    return int(round(-100.0 / (dec - 1.0)))

def slate_quality_and_plan(qualified_sorted, daily_bankroll: int):
    """
    Uses TOP 8 qualified bets to score slate. EV is per $1.
    Returns: (quality_str, ev_sum_topk, suggested_bets_tonight, suggested_bet_sizing, bankroll_to_use)
    """
    n = len(qualified_sorted)
    topk = min(8, n)
    ev_sum = sum(float(x["ev"]) for x in qualified_sorted[:topk]) if topk > 0 else 0.0

    thin = (n < 4) or (ev_sum < 0.18)
    rich = (n >= 9) or (ev_sum >= 0.35)

    if thin:
        quality = "THIN"
        suggested_bets_tonight = min(3, n) if n > 0 else 0
        suggested_bet_sizing = "Quarter Kelly (conservative)"
        bankroll_use_ratio = 0.80
    elif rich:
        quality = "RICH"
        suggested_bets_tonight = min(10, n) if n > 0 else 0
        suggested_bet_sizing = "Flat (even split)"
        bankroll_use_ratio = 1.00
    else:
        quality = "NORMAL"
        suggested_bets_tonight = min(8, n) if n > 0 else 0
        suggested_bet_sizing = "Flat (even split)"
        bankroll_use_ratio = 1.00

    bankroll_to_use = int(round(daily_bankroll * bankroll_use_ratio))
    bankroll_to_use = max(0, bankroll_to_use)

    return quality, ev_sum, suggested_bets_tonight, suggested_bet_sizing, bankroll_to_use

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
      status TEXT DEFAULT 'OPEN',
      settled_ts_utc TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_snapshot(games, sport: str):
    conn = db()
    conn.execute(
        "INSERT INTO odds_snapshots (ts_utc, sport, payload_json) VALUES (?, ?, ?)",
        (now_utc_iso(), sport, json.dumps(games)),
    )
    conn.commit()
    conn.close()

def save_bets(to_bet, bets):
    conn = db()
    for c, s in zip(to_bet, bets):
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
    return (p_close - p_bet) * 100.0

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
# ODDS FETCH (PER LEAGUE)
# =========================================================
def fetch_odds(sport_key: str):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    # Invisible quota logging (Railway logs only)
    rem = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    last = r.headers.get("x-requests-last")
    print(f"[OddsAPI] sport={sport_key} remaining={rem} used={used} last_cost={last}")

    return r.json()

# =========================================================
# CANDIDATE BUILDER (WITH BOOK WEIGHTING)
# =========================================================
def pick_candidates(games, league_label: str):
    candidates = []

    for g in games:
        home = g.get("home_team")
        away = g.get("away_team")
        game_id = g.get("id")
        books = g.get("bookmakers", []) or []

        if not home or not away or not game_id:
            continue

        ml_probs = []
        ml_best = {home: None, away: None}

        pl_probs = {}
        pl_best = {}

        tot_probs = {}
        tot_best = {}

        for b in books:
            book_name = b.get("title") or b.get("key") or "book"
            bw = w(book_name)

            for m in (b.get("markets") or []):
                mkey = m.get("key")
                outcomes = m.get("outcomes") or []
                if len(outcomes) != 2:
                    continue

                o1, o2 = outcomes[0], outcomes[1]

                # MONEYLINE
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

                # SPREADS / PUCK LINE
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

                    for k, od in ((k1, p1), (k2, p2)):
                        cur = pl_best[line_id].get(k)
                        if cur is None or od > cur["odds"]:
                            pl_best[line_id][k] = {"odds": od, "book": book_name}

                # TOTALS
                if mkey == "totals":
                    n1 = str(o1.get("name", "")).lower()
                    n2 = str(o2.get("name", "")).lower()
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

        # ML candidates
        if ml_probs and ml_best[home] and ml_best[away]:
            wt_sum = sum(wt for (_, _, wt) in ml_probs) or 1.0
            p_home = sum(ph * wt for (ph, _, wt) in ml_probs) / wt_sum
            p_away = sum(pa * wt for (_, pa, wt) in ml_probs) / wt_sum

            for team, p_true in ((home, p_home), (away, p_away)):
                best = ml_best[team]
                odds = int(best["odds"])
                ev = ev_per_1_stake(float(p_true), odds)
                candidates.append({
                    "league": league_label,
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

        # PL candidates
        for line_id, rows in pl_probs.items():
            wt_sum = sum(r[4] for r in rows) or 1.0
            p1 = sum(r[0] * r[4] for r in rows) / wt_sum
            p2 = sum(r[1] * r[4] for r in rows) / wt_sum
            k1 = rows[0][2]
            k2 = rows[0][3]
            for k, p_true in ((k1, p1), (k2, p2)):
                best = pl_best[line_id].get(k)
                if not best:
                    continue
                odds = int(best["odds"])
                ev = ev_per_1_stake(float(p_true), odds)
                candidates.append({
                    "league": league_label,
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

        # TOTAL candidates
        for total_point, rows in tot_probs.items():
            wt_sum = sum(r[4] for r in rows) or 1.0
            p_over = sum(r[0] * r[4] for r in rows) / wt_sum
            p_under = sum(r[1] * r[4] for r in rows) / wt_sum
            over_key = rows[0][2]
            under_key = rows[0][3]
            for k, p_true in ((over_key, p_over), (under_key, p_under)):
                best = tot_best[total_point].get(k)
                if not best:
                    continue
                odds = int(best["odds"])
                ev = ev_per_1_stake(float(p_true), odds)
                candidates.append({
                    "league": league_label,
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
# CLV UPDATER (NHL ONLY FOR NOW)
# =========================================================
def find_best_odds_for_bet(cands, bet_row):
    for c in cands:
        if c["game_id"] == bet_row["game_id"] and c["market"] == bet_row["market"] and c["pick"] == bet_row["pick"]:
            return int(c["odds"])
    return None

def update_closing_lines_for_open_bets():
    games = fetch_odds(LEAGUES["NHL"])
    save_snapshot(games, sport=LEAGUES["NHL"])
    cands = pick_candidates(games, league_label="NHL")

    conn = db()
    open_bets = conn.execute("SELECT * FROM bets WHERE status='OPEN'").fetchall()

    updated = 0
    for b in open_bets:
        close = find_best_odds_for_bet(cands, b)
        if close is None:
            continue
        cents = clv_cents(int(b["odds_placed"]), int(close))
        conn.execute(
            "UPDATE bets SET close_odds=?, clv_cents=? WHERE id=?",
            (int(close), float(cents), int(b["id"])),
        )
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
st.set_page_config(page_title="Myles’s Professional Hockey Picks", page_icon="🏒", layout="centered")

st.markdown(
    "<style>"
    ".badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;"
    "border:1px solid rgba(255,255,255,0.15);}"
    ".badge-thin{background:rgba(255,193,7,0.15);}"
    ".badge-normal{background:rgba(13,202,240,0.15);}"
    ".badge-rich{background:rgba(25,135,84,0.15);}"
    ".badge-edge{background:rgba(25,135,84,0.20);}"
    ".badge-strong{background:rgba(25,135,84,0.18);}"
    ".badge-med{background:rgba(255,193,7,0.15);}"
    ".badge-small{background:rgba(108,117,125,0.12);}"
    ".muted{opacity:0.85;}"
    "</style>",
    unsafe_allow_html=True,
)

st.markdown("# 🏒 Myles’s Professional Hockey Picks")
st.markdown("*Pregame only · Ranked best → worst · Moneyline · Puck Line · Totals*")
st.caption(f"Updated {date.today().strftime('%B %d, %Y')}")
st.divider()

if not API_KEY:
    st.error("ODDS_API_KEY not set in Railway variables.")
    st.stop()

tab_picks, tab_tracker = st.tabs(["Picks", "Tracker"])

st.session_state.setdefault("last_to_bet", [])
st.session_state.setdefault("last_bets", [])
st.session_state.setdefault("last_ranked", [])
st.session_state.setdefault("last_plan", None)

# -----------------------
# PICKS TAB
# -----------------------
with tab_picks:
    colL, colR = st.columns(2)

    with colL:
        edge_filter = st.radio(
            "How picky should we be?",
            ["Loose (more bets)", "Balanced", "Strict (fewer bets)"],
            index=1,
            horizontal=True,
        )

        coverage = st.radio(
            "Which leagues should we include?",
            ["NHL only", "Pro Mix (NHL + 4 leagues)"],
            index=1,
            horizontal=True,
        )

        only_bov = st.toggle("Only show Bovada lines", value=False)
        st.caption("All qualifying bets will be shown (ranked). Bet amounts are only shown for the bets you place tonight.")

    with colR:
        auto_mode = st.toggle("Auto mode (recommended)", value=True)

        bets_tonight = st.slider(
            "How many bets do you want to place tonight?",
            1, 10, 6
        )

        daily_bankroll = st.number_input(
            "Tonight’s bankroll ($)",
            min_value=10, max_value=5000, value=100, step=10
        )

        bet_sizing_manual = st.radio(
            "Bet sizing",
            ["Flat (even split)", "Quarter Kelly (conservative)", "Half Kelly (balanced)"],
            index=0,
            horizontal=True,
            disabled=auto_mode,
        )

        bankroll_use_manual = st.selectbox(
            "Max bankroll to use",
            ["Auto", "100%", "80%", "60%"],
            index=0,
            disabled=auto_mode,
        )

    # League selection via toggle (cap at 5 total)
    if coverage == "NHL only":
        selected_leagues = ["NHL"]
    else:
        selected_leagues = [x for x in PRO_MIX_5 if x in LEAGUES]
        if "NHL" not in selected_leagues:
            selected_leagues = ["NHL"]

    EV_MAP = {"Loose (more bets)": 0.015, "Balanced": 0.025, "Strict (fewer bets)": 0.035}
    min_ev = float(EV_MAP[edge_filter])

    with st.expander("What do these numbers mean?", expanded=False):
        st.caption("**Model win%** = our estimated chance the bet wins.")
        st.caption("**Implied win%** = what the odds imply (before vig).")
        st.caption("**Edge (pp)** = Model win% − Implied win% (percentage points).")
        st.caption("**EV%** = expected return over many bets (not guaranteed).")
        st.markdown("- 🟢 **STRONG** ≥ 5.0% EV\n- 🟡 **MEDIUM** ≥ 3.0% EV\n- ⚪ **SMALL** ≥ 1.5% EV\n- 💤 **TINY** < 1.5% EV")

    books_txt = "Bovada only" if only_bov else "All"
    leagues_txt = ", ".join(selected_leagues)
    st.caption(
        f"Minimum edge: **{min_ev*100:.1f}% EV** (≈ **${min_ev*100:.2f} per $100**) • "
        f"Books: **{books_txt}** • Leagues: **{leagues_txt}**"
    )

    def _apply_filters_and_sort(rows, market_filter: str, league_filter: str, sort_mode: str):
        out = rows

        if market_filter == "Moneyline":
            out = [c for c in out if c.get("market") == "ML"]
        elif market_filter == "Puck Line":
            out = [c for c in out if c.get("market") == "PL"]
        elif market_filter == "Totals":
            out = [c for c in out if c.get("market") == "TOTAL"]

        if league_filter != "All":
            out = [c for c in out if c.get("league") == league_filter]

        def edge_pp(c):
            p_model = float(c["p_true"])
            p_impl = implied_prob_from_odds(int(c["odds"]))
            return (p_model - p_impl) * 100.0

        if sort_mode == "Best EV":
            out.sort(key=lambda x: float(x["ev"]), reverse=True)
        elif sort_mode == "Biggest edge (pp)":
            out.sort(key=edge_pp, reverse=True)
        elif sort_mode == "Highest model win%":
            out.sort(key=lambda x: float(x["p_true"]), reverse=True)
        else:
            out.sort(key=lambda x: float(x["ev"]), reverse=True)

        return out

    def _render_cards(ranked_rows, to_bet_rows, bet_amounts):
        for i, c in enumerate(ranked_rows, start=1):
            ev1 = float(c["ev"])
            ev_pct = ev1 * 100.0

            p_model = float(c["p_true"])
            p_implied = implied_prob_from_odds(int(c["odds"]))
            edge_pp = (p_model - p_implied) * 100.0

            label = confidence_label(ev1)

            is_bet = c in to_bet_rows
            tag = "✅ BET TONIGHT" if is_bet else "👀 QUALIFIES"

            conf_class = {"STRONG": "badge-strong", "MEDIUM": "badge-med", "SMALL": "badge-small"}.get(label, "badge-small")
            conf_badge = f'<span class="badge {conf_class}">{label}</span>'

            bet_html = ""
            if is_bet:
                amt = bet_amounts[to_bet_rows.index(c)]
                bet_html = f' • <b>Bet: ${int(amt)}</b>'

            league_tag = f"[{c.get('league','—')}]"

            header_html = (
                f"<div style='font-size:18px;font-weight:800;'>"
                f"{i}. {tag} — {league_tag} {c['game']} — {c['pick']} ({c['market']}) "
                f"{c['odds']} @ {c['book']} {conf_badge}"
                f"<span class='muted'>{bet_html}</span>"
                f"</div>"
            )

            sub_html = (
                f"<div class='muted' style='margin-top:6px;'>"
                f"EV: <b>+{ev_pct:.2f}%</b> (≈ <b>${ev_pct:.2f} per $100</b>) • "
                f"Edge: <b>{edge_pp:+.1f}pp</b> • "
                f"Model win%: <b>{p_model*100:.1f}%</b> • "
                f"Implied win%: <b>{p_implied*100:.1f}%</b> • "
                f"Books used: {c.get('n_books','—')}"
                f"</div>"
            )

            with st.container(border=True):
                st.markdown(header_html + sub_html, unsafe_allow_html=True)

                with st.expander("Why this bet?"):
                    fair = fair_american_from_p(p_model)
                    st.write(f"Fair odds (model): **{fair}**")
                    st.write(f"Best available: **{c['odds']} @ {c['book']}**")
                    st.write(f"Model win%: **{p_model*100:.1f}%**")
                    st.write(f"Book implied win%: **{p_implied*100:.1f}%**")
                    st.write(f"Win-prob edge: **{edge_pp:+.2f} percentage points**")
                    st.write(f"EV (ROI): **+{ev_pct:.2f}%** (≈ **+${ev_pct:.2f} per $100**)")

    if st.button("Run Model"):
        # Pull all leagues requested
        all_cands = []
        for lg in selected_leagues:
            sport_key = LEAGUES[lg]
            games = fetch_odds(sport_key)
            save_snapshot(games, sport=sport_key)
            all_cands.extend(pick_candidates(games, league_label=lg))

        cands = all_cands
        if not cands:
            st.warning("PASS — no odds returned.")
            st.stop()

        if only_bov:
            cands = [c for c in cands if is_bovada(c.get("book", ""))]
            if not cands:
                st.warning("No Bovada lines found for this coverage. Turn off Bovada-only or change coverage.")
                st.stop()

        # Sort all candidates by EV first (baseline)
        cands.sort(key=lambda x: float(x["ev"]), reverse=True)
        best_any = float(cands[0]["ev"])

        qualified = [c for c in cands if float(c["ev"]) >= float(min_ev)]
        qualified.sort(key=lambda x: float(x["ev"]), reverse=True)

        if not qualified:
            st.warning(
                f"PASS — Best edge found was {best_any*100:.2f}% EV (below your {min_ev*100:.2f}% threshold)."
            )
            st.subheader("Closest opportunities (below threshold)")
            for i, c in enumerate(cands[:8], start=1):
                st.write(
                    f"{i}) [{c.get('league','—')}] {c['game']} — {c['pick']} ({c['market']}) "
                    f"{c['odds']} @ {c['book']} — EV {float(c['ev'])*100:.2f}%"
                )
            st.stop()

        # --- Filters like a betting app ---
        st.divider()
        st.subheader("Filters")

        filt1, filt2, filt3 = st.columns([1.3, 1.2, 1.5])
        with filt1:
            market_filter = st.radio(
                "Market",
                ["All", "Moneyline", "Puck Line", "Totals"],
                index=0,
                horizontal=True,
            )
        with filt2:
            league_filter = st.selectbox(
                "League",
                ["All"] + selected_leagues,
                index=0,
            )
        with filt3:
            sort_mode = st.selectbox(
                "Sort",
                ["Best EV", "Biggest edge (pp)", "Highest model win%"],
                index=0,
            )

        view = _apply_filters_and_sort(qualified, market_filter, league_filter, sort_mode)

        # Slate plan & auto sizing should apply to the VIEW (what you're actually choosing bets from)
        quality, ev_sum_topk, suggested_bets_tonight, suggested_bet_sizing, bankroll_to_use_auto = slate_quality_and_plan(
            qualified_sorted=view,
            daily_bankroll=int(daily_bankroll),
        )

        if auto_mode:
            bet_sizing = suggested_bet_sizing
            bankroll_to_use = bankroll_to_use_auto
            bets_effective = min(int(suggested_bets_tonight), int(bets_tonight)) if suggested_bets_tonight else 0
        else:
            bet_sizing = bet_sizing_manual
            if bankroll_use_manual == "100%":
                bankroll_to_use = int(daily_bankroll)
            elif bankroll_use_manual == "80%":
                bankroll_to_use = int(round(daily_bankroll * 0.80))
            elif bankroll_use_manual == "60%":
                bankroll_to_use = int(round(daily_bankroll * 0.60))
            else:
                bankroll_to_use = int(daily_bankroll)
            bets_effective = min(int(bets_tonight), len(view))

        to_bet = view[: int(bets_effective)]

        if bet_sizing.startswith("Flat"):
            bet_amounts = stake_split_flat(int(bankroll_to_use), len(to_bet))
        else:
            k_frac = 0.25 if "Quarter" in bet_sizing else 0.50
            bet_amounts = stake_split_kelly_scaled(int(bankroll_to_use), to_bet, kelly_fraction=k_frac)

        st.session_state.last_ranked = view
        st.session_state.last_to_bet = to_bet
        st.session_state.last_bets = bet_amounts
        st.session_state.last_plan = {
            "quality": quality,
            "ev_sum_topk": ev_sum_topk,
            "bet_sizing": bet_sizing,
            "bankroll_to_use": bankroll_to_use,
            "min_ev": min_ev,
            "only_bov": only_bov,
            "auto_mode": auto_mode,
            "leagues": selected_leagues,
            "market_filter": market_filter,
            "league_filter": league_filter,
            "sort_mode": sort_mode,
        }

        quality_class = {"THIN": "badge-thin", "NORMAL": "badge-normal", "RICH": "badge-rich"}.get(quality, "badge-normal")
        badges = (
            f'<div class="muted">'
            f'<span class="badge {quality_class}">SLATE: {quality}</span>&nbsp;'
            f'<span class="badge">Qualifying (view): {len(view)}</span>&nbsp;'
            f'<span class="badge">Betting tonight: {len(to_bet)}</span>&nbsp;'
            f'<span class="badge">Bankroll used: ${int(bankroll_to_use)}</span>'
            f'</div>'
        )
        st.markdown(badges, unsafe_allow_html=True)

        st.caption(f"Total edge (top 8 in view): **{ev_sum_topk*100:.2f}% EV** (bigger = more opportunity).")
        if only_bov:
            st.caption("Bovada-only is ON — some nights will be thinner.")
        st.divider()

        st.subheader(f"Ranked Opportunities — {len(view)} qualify in your current view")
        st.caption(
            f"Bet amounts are shown for the **{len(to_bet)} best opportunities** in this view. "
            f"The rest still qualify but aren’t bet tonight."
        )

        total_bet = sum(int(s) for s in bet_amounts) if bet_amounts else 0
        unused = int(bankroll_to_use) - total_bet
        st.caption(f"Bankroll used: **${int(bankroll_to_use)}** • Bet: **${total_bet}** • Unused: **${unused}**")

        _render_cards(view, to_bet, bet_amounts)

        st.divider()
        st.subheader("Action")
        if st.button("Save these bets to Tracker"):
            if not st.session_state.last_to_bet:
                st.warning("No bets to save (either slate is thin or you chose too few bets tonight).")
            else:
                save_bets(st.session_state.last_to_bet, st.session_state.last_bets)
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
                st.caption(f"Placed: {r['placed_ts_utc']} • Book: {r['book']} • Bet: ${r['stake']}")

                if r["clv_cents"] is not None:
                    st.write(
                        f"Odds placed: **{r['odds_placed']}** | "
                        f"Close: **{r['close_odds'] if r['close_odds'] is not None else '—'}** | "
                        f"CLV: **{r['clv_cents']:.2f}** cents"
                    )
                else:
                    st.write("Odds placed: **{0}** | Close: **—** | CLV: **—**".format(r["odds_placed"]))

                status = st.selectbox(
                    "Set status",
                    ["OPEN", "WIN", "LOSS", "PUSH", "VOID"],
                    index=["OPEN", "WIN", "LOSS", "PUSH", "VOID"].index(r["status"]),
                    key=f"status_{r['id']}",
                )
                if st.button("Save status", key=f"save_status_{r['id']}"):
                    update_bet_status(int(r["id"]), status)
                    st.success("Status updated.")
