from agent_bm25s import load_agents, build_bm25_index, bm25_agent_urls
from interview import interview_candidate

#def main():s
#    candidate_url = "http://127.0.0.1:8000/agent/telemetry"
#    task = "Help diagnose telemetry performance bottlenecks in a cloud system."

#    print(f"=== Interviewing candidate: {candidate_url} ===")
#    result = interview_candidate(candidate_url=candidate_url, task=task)
#    print(result)

def main():
    # 1) Load agents and build BM25 index
    agents = load_agents("agents.json")
    retriever, corpus = build_bm25_index(agents)

    # 2) Run BM25 to get candidate URLs
    query = "local telemetry performance optimization"
    candidate_urls = bm25_agent_urls(query, k=2, agents=agents, retriever=retriever, corpus=corpus, verbose=True,)

    # 3) Interview each candidate
    task = "Help diagnose telemetry performance bottlenecks in a cloud system."
    for url in candidate_urls:
        print("\n=== Interviewing candidate:", url, "===")
        result = interview_candidate(candidate_url=url, task=task)
        # do something with result (store, print, etc.)

if __name__ == "__main__":
    main()