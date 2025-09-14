from typing import List, Dict, Tuple
from dotenv import load_dotenv
import psycopg
import os
import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN = True
except Exception:
    SKLEARN = False

load_dotenv(".env")
CFG = dict(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    dbname=os.getenv("DB_NAME"),
    sslmode=os.getenv("DB_SSLMODE", "require"),
)

def fetch_users_min():
    with psycopg.connect(**CFG) as conn, conn.cursor() as cur:
        cur.execute("SELECT id, email, name, interests FROM users ORDER BY created_at;")
        return cur.fetchall()

def fetch_roadmap_levels_for_users(user_ids: List[str]) -> Dict[str, Dict[str, int]]:
    if not user_ids: return {}
    with psycopg.connect(**CFG) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT user_id, title, COALESCE((data->>'level')::int, 0) AS lvl
            FROM roadmap_nodes
            WHERE user_id = ANY(%s);
        """, (user_ids,))
        out: Dict[str, Dict[str,int]] = {}
        for uid, title, lvl in cur.fetchall():
            out.setdefault(uid, {})[title] = int(lvl or 0)
        return out

def build_interest_vocab(users_rows) -> List[str]:
    vocab = set()
    for _,_,_, interests in users_rows:
        if interests:
            vocab.update(interests)
    return sorted(vocab)

def users_to_matrix(users_rows, vocab: List[str]) -> Tuple[np.ndarray, List[str]]:
    idx = {t:i for i,t in enumerate(vocab)}
    X = np.zeros((len(users_rows), len(vocab)), dtype=float)
    ids = []
    for r, (uid, email, name, interests) in enumerate(users_rows):
        ids.append(uid)
        if interests:
            for t in interests:
                if t in idx:
                    X[r, idx[t]] = 1.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    X = X / norms
    return X, ids

def find_matches_by_interests(target_user_id: str, k: int = 5) -> List[Tuple[str, float]]:
    users = fetch_users_min()
    if not users: return []
    vocab = build_interest_vocab(users)
    X, ids = users_to_matrix(users, vocab)
    if target_user_id not in ids:
        raise ValueError(f"User {target_user_id} not found.")
    t_idx = ids.index(target_user_id)
    if SKLEARN and X.shape[0] > 1:
        nn = NearestNeighbors(n_neighbors=min(k+1, X.shape[0]), metric="cosine")
        nn.fit(X)
        distances, indices = nn.kneighbors(X[t_idx:t_idx+1], return_distance=True)
        pairs = [(ids[j], 1.0 - float(dist)) for dist, j in zip(distances[0], indices[0]) if j != t_idx]
    else:
        sims = (X @ X[t_idx])
        pairs = [(ids[j], float(sims[j])) for j in range(len(ids)) if j != t_idx]
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:k]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]

def pretty_print_matches(target_user_id: str, k: int = 5):
    users = fetch_users_min()
    id2user = {uid: (email, name, interests) for uid, email, name, interests in users}
    matches = find_matches_by_interests(target_user_id, k=k)
    print(f"\nBest {k} matches for {id2user[target_user_id][1] or id2user[target_user_id][0]} ({target_user_id}):")
    for uid, score in matches:
        email, name, interests = id2user[uid]
        print(f"  - {name or email:24s}  sim={score:.3f}  interests={interests}")

if __name__ == "__main__":
    rows = fetch_users_min()
    target = None
    for uid, email, name, _ in rows:
        if email == "alice@example.com":
            target = uid; break
    if not target:
        target = rows[-1][0]
    pretty_print_matches(target_user_id=target, k=8)