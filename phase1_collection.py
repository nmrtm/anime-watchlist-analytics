"""
Fetches, processes, and aggregates anime franchise data from the Jikan API,
including fuzzy title matching, relation crawling, episode counting and batched processing.
"""

#import libraries
import requests
import pandas as pd
import time
import json
from difflib import SequenceMatcher
import re
import unicodedata
from anime_list import all_watched_anime 


#set global url
URL = "https://api.jikan.moe/v4"


# --- API/network utility ---

def get_data(url, params=None, max_retries=5, wait=1):
    """Performs a GET request with basic retry logic and error handling.

    This function sends a GET request to the specified URL with optional query
    parameters. If the server responds with a 429 (rate limit), it waits and
    retries up to `max_retries` times. Other HTTP errors or unexpected exceptions
    are logged and will cause the function to return None.

    Args:
        url (str): The endpoint URL to send the GET request to.
        params (dict, optional): Query parameters for the request.
        max_retries (int, optional): Number of retry attempts on rate limit (default 5).
        wait (int or float, optional): Seconds to wait before each attempt (default 1).

    Returns:
        requests.Response or None: The response object if successful, or None on failure.

    Side Effects:
        Prints diagnostic messages for rate limiting, errors, and failures.
    """

    retries = 0

    while retries < max_retries:
        time.sleep(wait)

        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 429:
                print(f"⚠️ Rate limited at {url}, retrying ({retries+1}/{max_retries})…")
                time.sleep(10)
                retries += 1
                continue

            resp.raise_for_status()
            return resp
        
        except requests.exceptions.HTTPError as e:
            print(f"❌ Request failed for {url} due to {e}")
            return None
        
        except Exception as e:
            print(f"❌ Unexpected error on attempt {retries+1} for {url}: {e}")
            return None
        
    print(f"❌ Max retries exceeded for {url}")
    return None


# --- Text normalisation and fuzzy matching ---

def norm(s: str) -> str:
    """Normalises a string for robust text matching and tokenisation.

    Applies Unicode normalisation, lowercases, removes apostrophes,
    replaces non-alphanumeric characters with spaces, and trims whitespace.

    Args:
        s (str): The input string to normalise.

    Returns:
        str: The cleaned, normalised string.
    """

    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"['’`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def title_variants(anime: dict) -> list[str]:
    """Returns all available title variants for a given anime entry.

    Extracts primary, English, Japanese, and any additional titles present
    in the 'titles' field.

    Args:
        anime (dict): Jikan anime record.

    Returns:
        list[str]: Unique list of title variant strings.
    """

    tv = set()
    for k in ("title", "title_english", "title_japanese"):
        if anime.get(k):
            tv.add(str(anime[k]))
    for t in (anime.get("titles") or []):
        if t.get("title"):
            tv.add(str(t["title"]))
    return list(tv)


def tokenise(s: str) -> list[str]:
    """Splits a normalised string into lowercase word tokens.

    Args:
        s (str): The string to tokenise.

    Returns:
        list[str]: List of tokens.
    """

    return norm(s).split()


def token_similarity(qtok: str, ttok: str) -> float:
    """Computes a similarity score between two tokens for fuzzy matching.

    Returns 1.0 for exact matches. If one token contains the other (e.g.,
    'summer' vs. 'summertime'), returns a boosted similarity score based on
    their overlap and position. Otherwise, falls back to the standard fuzzy
    ratio.

    Args:
        qtok (str): The query token.
        ttok (str): The title token.

    Returns:
        float: Similarity score between 0 and 1.
    """

    if not qtok or not ttok: 
        return 0.0
    if qtok == ttok:
        return 1.0
    # strong bump if one contains the other (handles summer/summertime, render/rendering)
    if qtok in ttok or ttok in qtok:
        # mild scale by ratio so 'sum' in 'summertime' isn't too strong
        base = SequenceMatcher(None, qtok, ttok).ratio()
        return max(base, 0.85 if ttok.startswith(qtok) or qtok.startswith(ttok) else 0.75)
    # fallback: fuzzy ratio
    return SequenceMatcher(None, qtok, ttok).ratio()


def tokens_cover(query_tokens: list[str], title_tokens: list[str], thresh: float) -> tuple[bool, float]:
    """Checks if every query token is strongly matched by a title token.

    For each query token, requires at least one title token with similarity
    >= `thresh`. Returns whether all are covered, and the sum of best
    similarities for each token.

    Args:
        query_tokens (list[str]): List of tokens from the query.
        title_tokens (list[str]): List of tokens from a candidate title.
        thresh (float): Minimum similarity required for a match.

    Returns:
        tuple[bool, float]: 
            - True and total score if every query token is covered.
            - False and 0.0 otherwise.
    """

    if not query_tokens or not title_tokens:
        return (False, 0.0)
    score = 0.0
    for q in query_tokens:
        best = 0.0
        for t in title_tokens:
            best = max(best, token_similarity(q, t))
            if best >= 1.0:
                break
        if best < thresh:
            return (False, 0.0)
        score += best
    return (True, score)


# --- Jikan API search and data retrieval ---

def search_main_anime(title, limit=10, strict_thresh=0.72, soft_thresh=0.62):
    """Finds the best-matching anime entry for a given title using fuzzy token-AND logic.

    Searches the Jikan API for anime candidates, then applies token-level fuzzy matching:
    - Each word in the query must match some token in a candidate title variant with similarity >= threshold.
    - Tries a strict threshold first, then a softer one if no matches are found.
    - Falls back to overall fuzzy string matching if needed.
    - Picks the candidate with the highest total per-token match score, breaking ties by popularity.

    Args:
        title (str): The anime title to search for.
        limit (int, optional): Number of search results to fetch (default is 10).
        strict_thresh (float, optional): Minimum token similarity for strict pass (default is 0.72).
        soft_thresh (float, optional): Minimum token similarity for soft pass (default is 0.62).

    Returns:
        dict or None: The best-matching anime entry (as a dict from Jikan API), or None if no suitable match is found.
    """

    url = f"{URL}/anime"
    resp = get_data(url, params={"q": title, "limit": limit})
    if not resp:
        return None

    results = resp.json().get("data", [])
    if not results:
        print(f"❌ No match for {title}")
        return None

    # Keep your original filter
    allowed_types = {"TV", "TV Special", "OVA", "ONA"}
    filtered = []
    for anime in results:
        anime_type = anime.get("type")
        ep = anime.get("episodes")
        if anime_type in allowed_types and (ep is None or ep > 1):
            filtered.append(anime)
    candidates = filtered if filtered else results

    qtokens = tokenise(title)

    def best_variant_cover(anime, thresh):
        best_ok = False
        best_score = 0.0
        for v in title_variants(anime):
            ok, score = tokens_cover(qtokens, tokenise(v), thresh)
            if ok and score > best_score:
                best_ok, best_score = True, score
        return best_ok, best_score

    # Pass 1: strict
    strict_pool = []
    for a in candidates:
        ok, score = best_variant_cover(a, strict_thresh)
        if ok:
            strict_pool.append((a, score))

    # Pass 2: soft (only if strict empty)
    pool = strict_pool
    if not pool:
        soft_pool = []
        for a in candidates:
            ok, score = best_variant_cover(a, soft_thresh)
            if ok:
                soft_pool.append((a, score))
        pool = soft_pool

    # If still nothing, fallback to your old fuzzy max on primary/english title
    if not pool:
        def best_ratio(anime):
            norms = [norm(v) for v in title_variants(anime)]
            return max((SequenceMatcher(None, norm(title), n).ratio() for n in norms), default=0.0)
        return max(candidates, key=best_ratio, default=None)

    # Tie-break within pool: higher coverage score, then better popularity (lower rank)
    def tie_break(item):
        a, coverage = item
        pop = a.get("popularity") or 10**9
        return (coverage, -min(pop, 10**9))  # maximize coverage, then popularity
    best_match, _ = max(pool, key=tie_break)
    return best_match


def get_related_ids(mal_id):
    """Fetches related anime IDs and their relation types for a given MAL ID.

    Calls the Jikan API to get all directly related anime (e.g., sequels, prequels, side stories)
    for the given MyAnimeList anime ID.

    Args:
        mal_id (int): The MyAnimeList ID for the anime.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'mal_id': The related anime's MAL ID.
            - 'relation_type': The type of relationship (e.g., 'Sequel', 'Prequel', 'Side story').
        Returns an empty list if no relations are found or on error.

    Side Effects:
        Prints error messages if parsing fails.
    """

    url = f"{URL}/anime/{mal_id}/relations"
    resp = get_data(url)

    if not resp:
        return []
    
    try:
        data = resp.json().get("data", [])
    except (ValueError, json.JSONDecodeError) as e:
        print(f"❌ Failed to parse JSON for MAL_ID {mal_id}: {e}")
        return []
    except Exception as e:
            print(f"❌ Unexpected error for MAL_ID {mal_id}: {e}")
            return []
    
    related_entries = []
    for relation in data:
        rel_type = relation.get("relation", "unknown")
        for entry in relation.get("entry", []):
            if entry.get("type") == "anime":
                related_entries.append({
                    "mal_id": entry["mal_id"],
                    "relation_type": rel_type
                })

    return related_entries


def entry_matches_root(entry, root_terms):
    """Checks if any root word appears in an anime entry's title fields.

    Used to determine whether a related anime is still part of the intended franchise
    when crawling through relations.

    Args:
        entry (dict): Anime entry with possible fields 'name', 'title', and 'title_english'.
        root_terms (list[str]): List of root words (e.g., from the original franchise title).

    Returns:
        bool: True if any root word appears in any of the title fields else False.
    """    

    titles = [
        entry.get("name", "").lower(),          # relations API usually provides this
        entry.get("title", "").lower(),
        entry.get("title_english", "").lower(),
    ]

    # Flatten root terms into words (Afro Samurai -> ["afro", "samurai"])
    root_words = []
    for term in root_terms:
        root_words.extend(term.lower().split())

    for title in titles:
        if any(word in title for word in root_words):
            return True
    return False


def get_full_franchise(mal_id, root_terms=None, seen=None, relation_map=None, depth=0, max_depth=20):
    """Recursively collects all related anime IDs for a franchise, controlling for crossover drift.

    Starts from the given MAL ID and crawls all related anime via the Jikan API, using depth-aware
    filtering:
      - At depth 0, accepts all directly related anime.
      - At depth >0, only accepts anime whose title contains any of the root words, to prevent
        drifting into unrelated crossovers.

    Args:
        mal_id (int): The starting MyAnimeList anime ID.
        root_terms (list[str], optional): Words from the original title used to filter deeper relations.
        seen (set, optional): Set of already visited MAL IDs to avoid cycles.
        relation_map (dict, optional): Dictionary accumulating {mal_id: relation_type} pairs.
        depth (int, optional): Current recursion depth (internal).
        max_depth (int, optional): Maximum recursion depth to avoid infinite loops.

    Returns:
        dict: Mapping of related MAL IDs to their relation types (does not include the seed unless added by caller).

    Side Effects:
        Prints progress for each visited MAL ID.
    """

    if seen is None:
        seen = set()
    if relation_map is None:
        relation_map = {}

    if mal_id in seen or depth > max_depth:
        return relation_map

    print(f"{'  ' * depth}🔁 Visiting MAL_ID {mal_id} (depth {depth})")
    seen.add(mal_id)

    url = f"{URL}/anime/{mal_id}/relations"
    resp = get_data(url)
    if not resp:
        return relation_map

    data = resp.json().get("data", [])
    for relation in data:
        rel_type = relation.get("relation", "unknown")
        for entry in relation.get("entry", []):
            if entry.get("type") != "anime":
                continue

            next_id = entry["mal_id"]
            if next_id in seen:
                continue

            # allow ALL direct relations (depth 0), filter only deeper hops
            if depth > 0 and root_terms and not entry_matches_root(entry, root_terms):
                continue

            relation_map[next_id] = rel_type

            get_full_franchise(
                next_id,
                root_terms=root_terms,
                seen=seen,
                relation_map=relation_map,
                depth=depth + 1,
                max_depth=max_depth
            )

    return relation_map


def get_anime_details(mal_id):
    """Fetches and flattens all details for a given anime from the Jikan API.

    Calls the Jikan API for a specific MAL ID and normalises the JSON response into
    a flat dictionary suitable for tabular analysis.

    Args:
        mal_id (int): The MyAnimeList anime ID.

    Returns:
        dict or None: Flattened anime details as a single dictionary, or None if not found.

    Side Effects:
        Prints a warning if the request fails or the data is missing.
    """
    
    url = f"{URL}/anime/{mal_id}"
    resp = get_data(url)
    if not resp:
        print(f"⚠️ No response for MAL_ID {mal_id}; returning None")
        return None

    anime = resp.json().get("data")
    if not anime:
        print(f"⚠️ No data for MAL_ID {mal_id}; returning None")
        return None

    flat_df = pd.json_normalize(anime)
    return flat_df.to_dict(orient="records")[0]


def fetch_episode_count(row):
    """Fetches the latest episode count for an anime, updating if currently airing.

    If the row has a valid episode count and is not airing, returns the existing value.
    Otherwise, pages through the Jikan API to count total episodes for the given MAL ID.

    Args:
        row (dict or pandas.Series): Anime row containing 'episodes', 'airing', and 'mal_id'.

    Returns:
        int or None: The updated episode count, or the original value if unable to fetch.

    Side Effects:
        Prints an error if repeated failures occur while fetching episode data.
    """

    default_eps = row.get("episodes")
    is_airing = row.get("airing")  # or "airing" based on your column name
    mal_id = row.get("mal_id")

    #only skip fetching if not airing AND already has valid value
    if is_airing is not True and pd.notna(default_eps):
        return default_eps

    if not mal_id:
        return default_eps

    #fetch episode data from API
    count = 0
    page = 1

    while True:
        url = f"{URL}/anime/{mal_id}/episodes"
        resp = get_data(url, params={"page": page})

        if not resp:
            print(f"❌ Skipping MAL_ID {mal_id}, page {page} due to repeated failures.")
            return default_eps

        eps = resp.json().get("data", [])
        if not eps:
            break

        count += len(eps)
        page += 1

    return count


# --- Franchise processing and batching ---

def process_franchise(title):
    """Processes a single anime franchise title into detailed, franchise-tagged records.

    Steps:
        1. Finds the best representative entry for the title using fuzzy matching.
        2. Recursively gathers all related anime IDs within the franchise.
        3. Fetches detailed info for each related entry.
        4. Assigns a consistent franchise_id (the smallest MAL ID among TV/OVA/ONA series).
        5. Tags each record with this franchise_id for grouping.

    Args:
        title (str): The anime franchise title to process.

    Returns:
        list[dict]: List of detail records for all franchise entries,
            each tagged with 'relation_type' and 'franchise_id'.
            Returns an empty list if nothing is found.

    Side Effects:
        Prints progress and warning messages; sleeps between API calls to avoid rate limiting.
    """

    time.sleep(1)
    print(f"📥 Processing: {title!r}")

    # 1) Search for the title
    main = search_main_anime(title)
    if not main:
        print(f"⚠️ Main anime not found for '{title}', skipping.")
        return []

    # 2) Collect main + related MAL_IDs
     # 2) Recursively gather all MAL IDs using title filtering
    root_terms = [w for w in re.findall(r"[a-z0-9]+", title.lower()) if len(w) > 1]
    related_map = get_full_franchise(main["mal_id"], root_terms)#, max_depth=20)
    
    # Add the original main MAL ID too
    related_map[main["mal_id"]] = "main"

    mal_ids = [{"mal_id": mid, "relation_type": rel_type} for mid, rel_type in related_map.items()]

    # mal_ids = [{"mal_id": main["mal_id"], "relation_type": "main"}]
    # related = get_related_ids(main["mal_id"])
    # mal_ids.extend(related)

    # 3) Fetch details for each ID
    details = []
    for entry in mal_ids:
        mid = entry["mal_id"]
        relation_type = entry["relation_type"]
        
        info = get_anime_details(mid)
        if info:
            info["relation_type"] = relation_type
            details.append(info)
            time.sleep(1)

    if not details:
        print(f"⚠️ Details not found for any related ID for '{title}', skipping.")
        return []

# 4) Determine franchise root: only TV‑series formats count here
    allowed_types = {"TV", "TV Special", "OVA", "ONA"}

    series_entries = []

    for d in details:
        if d.get("type") in allowed_types:
            series_entries.append(d)

    if series_entries:
        mal_ids_series = []
        for d in series_entries:
            mal_ids_series.append(d["mal_id"])
        franchise_id = min(mal_ids_series)
    else:
        mal_ids_all = []
        for d in details:
            mal_ids_all.append(d["mal_id"])
        franchise_id = min(mal_ids_all)


    # 5) Tag every record
    for d in details:
        d["franchise_id"] = franchise_id

    return details #⚠️ some crossover or multi-series specials (e.g. “X vs Y”) may appear under a single franchise. This will be handled during data cleaning.


def fetch_all_franchises(title_list):
    """Processes a list of anime franchise titles and combines all their records.

    Loops through each title, processes its franchise, and aggregates the results.

    Args:
        title_list (list[str]): List of anime franchise titles to process.

    Returns:
        list[dict]: Combined list of detailed records for all processed franchises.
    """    
    
    all_data = []
    for title in title_list:
        all_data.extend(process_franchise(title))
    return all_data


def split_into_batches(anime_list, batch_size=10):
    """Yields consecutive batches from a list.

    Args:
        anime_list (list): The input list to split into batches.
        batch_size (int, optional): The size of each batch (default is 10).

    Yields:
        list: Slices of the original list with length up to batch_size.
    """

    for i in range(0, len(anime_list), batch_size):
        yield anime_list[i: i + batch_size]


# --- Main script execution ---

if __name__ == "__main__":
    anime_franchises = all_watched_anime
    all_batches = []

    # 1) Fetch in batches
    for b, batch in enumerate(split_into_batches(anime_franchises, 5), start=1):
        print(f"\n🚀 Running Batch {b}")
        batch_data = fetch_all_franchises(batch)
        print(f"✅ Batch {b} completed")
        all_batches.extend(batch_data)
        time.sleep(20)

    #2) Build a DataFrame
    df = pd.DataFrame(all_batches)

    #3) Fill in missing Episodes    
    df["episodes"] = df.apply(fetch_episode_count, axis=1)
    df.reset_index(drop=True, inplace=True)

    #4) Save the DataFrame
    df.to_csv("anime_fran_data.csv", index=False)
    print("🎉 All data fetched, episodes filled, and saved to anime_fran_data.csv!")



