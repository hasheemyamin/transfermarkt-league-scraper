import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ================================================================
# Configuration
# ================================================================

BASE_URL = "https://www.transfermarkt.com"
API_BASE_URL = "https://tmapi-alpha.transfermarkt.technology"

LEAGUE_URLS = {
    "premier_league": "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1",
    "serie_a": "https://www.transfermarkt.com/serie-a/startseite/wettbewerb/IT1",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    )
}

TIMEOUT = 10
REQUEST_DELAY_SECONDS = 1.0

MAX_PLAYERS_PER_LEAGUE = 500

# ================================================================
# Data models
# ================================================================

@dataclass
class MarketValuePoint:
    """Represents a single point in the player's market value history."""
    date: str
    value: float
    currency: str


@dataclass
class PlayerData:
    player_id: str
    name: Optional[str] = None
    shirt_number: Optional[int] = None
    name_native: Optional[str] = None
    club: Optional[str] = None
    position: Optional[str] = None
    nationality: Optional[str] = None
    date_of_birth: Optional[str] = None
    age: Optional[int] = None
    height_cm: Optional[int] = None
    preferred_foot: Optional[str] = None
    contract_expires: Optional[str] = None
    current_market_value: Optional[float] = None
    current_market_value_currency: Optional[str] = None
    market_value_history: Optional[List[MarketValuePoint]] = None
    player_agent: Optional[str] = None


# ================================================================
# HTTP helpers
# ================================================================

def fetch_html(url: str) -> BeautifulSoup:
    """
    Download a page and return a BeautifulSoup object.
    Raises requests.HTTPError if status is not successful.
    """
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.content, "html.parser")


def fetch_json(url: str) -> Any:
    """
    Download JSON and return parsed Python object.
    Raises requests.HTTPError if status is not successful.
    """
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ================================================================
# League navigation (clubs -> players)
# ================================================================

def get_club_urls_for_league(league_url: str) -> List[str]:
    """
    Scrape league start page and return absolute club URLs.
    """
    soup = fetch_html(league_url)

    club_links: set[str] = set()
    for a in soup.select('table.items td.hauptlink a[href*="/startseite/verein/"]'):
        href = a.get("href")
        if not href:
            continue
        club_links.add(urljoin(BASE_URL, href))

    print(f"[INFO] Found {len(club_links)} club URLs for league: {league_url}")
    return sorted(club_links)


def get_player_urls_for_club(club_url: str) -> List[str]:
    """
    Scrape club squad page and return absolute player profile URLs.
    """
    soup = fetch_html(club_url)

    player_links: set[str] = set()
    for a in soup.select('table.items td.hauptlink a[href*="/profil/spieler/"]'):
        href = a.get("href")
        if not href:
            continue
        player_links.add(urljoin(BASE_URL, href))

    return sorted(player_links)


def collect_unique_player_urls_for_league(
    league_url: str,
    max_players: int,
) -> List[str]:
    """
    Collect unique player profile URLs across all clubs in a league,
    deduplicated by player_id, capped to max_players.
    """
    club_urls = get_club_urls_for_league(league_url)

    seen_player_ids: set[str] = set()
    player_urls: List[str] = []

    for club_idx, club_url in enumerate(club_urls, start=1):
        print(f"[INFO] ({club_idx}/{len(club_urls)}) Collecting players from club: {club_url}")
        try:
            club_player_urls = get_player_urls_for_club(club_url)
        except Exception as e:
            print(f"[WARN] Failed to fetch club squad page: {club_url} ({e})")
            continue

        for url in club_player_urls:
            pid = extract_player_id(url)
            if pid in seen_player_ids:
                continue
            seen_player_ids.add(pid)
            player_urls.append(url)

            if len(player_urls) >= max_players:
                break

        if len(player_urls) >= max_players:
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"[INFO] Collected {len(player_urls)} unique player URLs for league.")
    return player_urls


# ================================================================
# Parsing helpers
# ================================================================

def extract_player_id(player_url: str) -> str:
    match = re.search(r"(\d+)(?:/)?$", player_url)
    if not match:
        return player_url.rstrip("/").split("/")[-1]
    return match.group(1)


def get_text(element) -> Optional[str]:
    return element.get_text(strip=True) if element else None


def parse_shirt_number(soup: BeautifulSoup) -> Optional[int]:
    elem = soup.select_one("span.data-header__shirt-number")
    text = get_text(elem)
    if not text:
        return None
    text = text.replace("#", "").strip()
    try:
        return int(text)
    except ValueError:
        return None


def parse_player_name(soup: BeautifulSoup) -> Optional[str]:
    header = soup.select_one("h1.data-header__headline-wrapper")
    if not header:
        return None

    parts = []
    for text in header.stripped_strings:
        t = text.strip()
        if not t:
            continue
        if re.match(r"#\d+", t):
            continue
        parts.append(t)

    return " ".join(parts) if parts else None


def parse_native_name(soup: BeautifulSoup) -> Optional[str]:
    elem = soup.select_one("span.info-table__content.info-table__content--bold")
    return get_text(elem)


def parse_contract_expiry(soup: BeautifulSoup) -> Optional[str]:
    contract_section = soup.find(string=re.compile(r"Contract expires"))
    if not contract_section:
        return None

    parent_html = str(contract_section.find_parent())
    match = re.search(
        r"Contract expires:\s*.*?__content\">(.*?)</span>",
        parent_html,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else None


def parse_basic_info_from_table(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {
        "club": None,
        "position": None,
        "nationality": None,
        "date_of_birth": None,
        "age": None,
        "height_cm": None,
        "preferred_foot": None,
        "player_agent": None,
    }

    tables = soup.select("div.info-table")
    for table in tables:
        spans = table.select("span.info-table__content")
        for i in range(0, len(spans) - 1, 2):
            label = get_text(spans[i])
            value = get_text(spans[i + 1])
            if not label or not value:
                continue

            label_lower = label.lower()

            if "date of birth" in label_lower:
                info["date_of_birth"] = value
                age_match = re.search(r"\((\d+)\)", value)
                if age_match:
                    info["age"] = age_match.group(1)

            elif "citizenship" in label_lower or "nationality" in label_lower:
                info["nationality"] = value

            elif "height" in label_lower:
                height_match = re.search(r"([\d,\.]+)\s*m", value)
                if height_match:
                    numeric_str = height_match.group(1).replace(",", ".")
                    try:
                        meters = float(numeric_str)
                        info["height_cm"] = str(int(round(meters * 100)))
                    except ValueError:
                        pass

            elif "position" in label_lower:
                info["position"] = value

            elif "foot" in label_lower:
                info["preferred_foot"] = value

            elif "current club" in label_lower:
                info["club"] = value

            elif "player agent" in label_lower or label_lower.strip() == "agent:":
                link = spans[i + 1].select_one("a")
                info["player_agent"] = get_text(link) if link else value

    if info["club"] is None:
        club_elem = soup.select_one("a.data-header__club")
        info["club"] = get_text(club_elem) if club_elem else None

    return info


def parse_market_value_history(
    player_id: str,
) -> Tuple[Optional[float], Optional[str], Optional[List[MarketValuePoint]]]:
    api_url = f"{API_BASE_URL}/player/{player_id}/market-value-history"

    try:
        raw_json = fetch_json(api_url)
    except requests.HTTPError as e:
        print(f"[WARN] Market value history failed for player {player_id}: {e}")
        return None, None, None

    data = raw_json.get("data", {})
    history_raw = data.get("history", [])
    current_raw = data.get("current", {})

    history_points: List[MarketValuePoint] = []
    for entry in history_raw:
        mv = entry.get("marketValue", {})
        value = mv.get("value")
        currency = mv.get("currency", "EUR")
        date = mv.get("determined")
        if value is None or date is None:
            continue
        history_points.append(MarketValuePoint(date=str(date), value=float(value), currency=str(currency)))

    mv_current = current_raw.get("marketValue", {}) if current_raw else {}
    current_value = mv_current.get("value")
    current_currency = mv_current.get("currency", "EUR") if mv_current else None

    if current_value is None and history_points:
        current_value = history_points[-1].value
        current_currency = history_points[-1].currency

    return current_value, current_currency, history_points if history_points else None


# ================================================================
# Main player scrape
# ================================================================

def scrape_player(player_url: str) -> PlayerData:
    player_id = extract_player_id(player_url)
    print(f"[INFO] Scraping player {player_id} from {player_url}")

    soup = fetch_html(player_url)

    name = parse_player_name(soup)
    shirt_number = parse_shirt_number(soup)
    name_native = parse_native_name(soup)
    contract_expires = parse_contract_expiry(soup)
    basic_info = parse_basic_info_from_table(soup)

    current_mv, mv_currency, mv_history = parse_market_value_history(player_id)

    return PlayerData(
        player_id=player_id,
        name=name,
        shirt_number=shirt_number,
        name_native=name_native,
        club=basic_info.get("club"),
        position=basic_info.get("position"),
        nationality=basic_info.get("nationality"),
        date_of_birth=basic_info.get("date_of_birth"),
        age=int(basic_info["age"]) if basic_info.get("age") else None,
        height_cm=int(basic_info["height_cm"]) if basic_info.get("height_cm") else None,
        preferred_foot=basic_info.get("preferred_foot"),
        contract_expires=contract_expires,
        current_market_value=current_mv,
        current_market_value_currency=mv_currency,
        market_value_history=mv_history,
        player_agent=basic_info.get("player_agent"),
    )


# ================================================================
# Storage helpers
# ================================================================

def save_league_outputs(league_key: str, players: List[PlayerData]) -> None:
    """
    Save:
      1) flat player table -> <league_key>_players.csv
      2) market value history table -> <league_key>_market_values.csv
    """
    # Flat player table
    flat_rows = []
    for p in players:
        d = asdict(p)
        # remove nested history for the flat table
        d.pop("market_value_history", None)
        flat_rows.append(d)

    df_players = pd.DataFrame(flat_rows)
    df_players.to_csv(f"{league_key}_players.csv", index=False)

    # Market value history table (one-to-many)
    mv_rows = []
    for p in players:
        if not p.market_value_history:
            continue
        for mv in p.market_value_history:
            mv_rows.append({
                "player_id": p.player_id,
                "date": mv.date,
                "value": mv.value,
                "currency": mv.currency,
            })

    df_mv = pd.DataFrame(mv_rows)
    df_mv.to_csv(f"{league_key}_market_values.csv", index=False)

    print(f"[INFO] Saved outputs: {league_key}_players.csv and {league_key}_market_values.csv")


# ================================================================
# Script entry point
# ================================================================

def scrape_league(league_key: str, league_url: str, max_players: int) -> None:
    print(f"\n[INFO] ===== Scraping league: {league_key} =====")
    player_urls = collect_unique_player_urls_for_league(league_url, max_players=max_players)

    players: List[PlayerData] = []
    for idx, url in enumerate(player_urls, start=1):
        print(f"[INFO] [{idx}/{len(player_urls)}] {league_key} scraping: {url}")
        try:
            players.append(scrape_player(url))
        except Exception as e:
            print(f"[WARN] Failed player scrape: {url} ({e})")
        time.sleep(REQUEST_DELAY_SECONDS)

    save_league_outputs(league_key, players)


if __name__ == "__main__":
    # Scrape 500 players from each league and save to CSVs
    scrape_league("premier_league", LEAGUE_URLS["premier_league"], MAX_PLAYERS_PER_LEAGUE)
    scrape_league("serie_a", LEAGUE_URLS["serie_a"], MAX_PLAYERS_PER_LEAGUE)