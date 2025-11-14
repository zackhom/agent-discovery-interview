import json
import bm25s

try:
    import Stemmer
    STEMMER = Stemmer.Stemmer("english")
except Exception:
    STEMMER = None


# ---------- URL helper ----------

def get_primary_url(agent: dict) -> str | None:
    endpoints = agent.get("endpoints") or {}

    # 1) Prefer adaptive_resolver.url if it's a non-empty http(s) URL
    ar = endpoints.get("adaptive_resolver") or {}
    url = ar.get("url")
    if isinstance(url, str) and url.startswith("http"):
        return url

    # 2) Fall back to first static http(s) URL (skip websocket)
    static_list = endpoints.get("static") or []
    for u in static_list:
        if not isinstance(u, str):
            continue
        if u.startswith("wss://"):
            continue  # skip websocket for now
        if u.startswith("http://") or u.startswith("https://"):
            return u

    # 3) Nothing usable found
    return None


# ---------- Text canonicalization ----------

def doc_text(a: dict) -> str:
    name = a.get("name") or a.get("agent_name") or a.get("label") or a.get("id") or ""
    desc = a.get("description") or ""
    tags = []

    # collect tags/keywords from skills/capabilities/provider/jurisdiction/endpoints
    for s in a.get("skills", []) or []:
        if isinstance(s, str):
            tags.append(s)
        elif isinstance(s, dict):
            if s.get("id"):
                tags.append(str(s["id"]))
            if s.get("description"):
                tags.append(str(s["description"]))
            for k in ("inputModes", "outputModes", "supportedLanguages"):
                v = s.get(k)
                if isinstance(v, list):
                    tags += [str(x) for x in v]

    caps = a.get("capabilities") or {}
    if isinstance(caps, dict):
        mods = caps.get("modalities")
        if isinstance(mods, list):
            tags += [str(x) for x in mods]

    prov = a.get("provider") or {}
    if isinstance(prov, dict) and prov.get("name"):
        tags.append(str(prov["name"]))

    if a.get("jurisdiction"):
        tags.append(str(a["jurisdiction"]))

    # small numeric hints if present
    ev = (a.get("evaluations") or {})
    tel = (a.get("telemetry") or {}).get("metrics", {})
    numeric = " ".join(
        str(x) for x in [
            ev.get("performanceScore"),
            tel.get("latency_p95_ms"),
            tel.get("throughput_rps"),
            tel.get("availability"),
        ] if x is not None
    )

    return "\n".join([name, desc, " ".join(tags), numeric]).strip()


# ---------- Public API ----------

def load_agents(path: str = "agents.json") -> list[dict]:
    """Load agents from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        agents = json.load(f)
    if not isinstance(agents, list):
        raise ValueError("agents.json must contain a JSON array")
    return agents


def build_bm25_index(agents: list[dict]):
    """
    Build a BM25 index from the given agents.

    Returns:
        retriever: bm25s.BM25 instance
        corpus: list[str] of the text documents corresponding to agents
    """
    corpus = [doc_text(a) for a in agents]
    if not corpus:
        raise ValueError("No agents to index (empty corpus).")

    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=STEMMER)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    return retriever, corpus


def bm25_agent_urls(
    query: str,
    k: int,
    agents: list[dict],
    retriever: bm25s.BM25,
    corpus: list[str],
    verbose: bool = True,
) -> list[str]:
    """
    Run a BM25 search over agents and return a list of primary URLs
    for the top-k matching agents.

    args:
        query: search string
        k: number of top results to consider
        agents: list of agent dicts
        retriever: BM25 index object from build_bm25_index
        corpus: list of document strings (same order as agents)
        verbose: if True, print ranked agents to stdout

    returns:
        list of URL strings (only agents that have a usable primary_url)
    """
    if not corpus:
        return []

    # clamp k to valid range
    k = max(1, min(k, len(corpus)))

    q_tokens = bm25s.tokenize([query], stopwords="en", stemmer=STEMMER)
    results, scores = retriever.retrieve(q_tokens, k=k)

    urls: list[str] = []

    for i in range(results.shape[1]):
        doc_idx = int(results[0, i])
        score = float(scores[0, i])
        agent = agents[doc_idx]

        if verbose:
            name = (
                agent.get("name")
                or agent.get("agent_name")
                or agent.get("label")
                or agent.get("id")
            )
            print(f"Rank {i+1}  score={score:.3f}  id={agent.get('id')}  name={name}")

        url = get_primary_url(agent)
        if url:
            urls.append(url)

    return urls