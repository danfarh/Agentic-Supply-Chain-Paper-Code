import json

from langchain_core.tools import tool


@tool
def duckduckgo_web_search(query: str, max_results: int = 5) -> str:
    """
    RESEARCH AGENT:
    Perform a web search using DuckDuckGo (No API key required).

    IMPORTANT FOR EVALUATION:
    Returns a JSON string with structured retrieved_contexts containing title, URL,
    snippet, and rank. These can be archived as source snapshots for external validation.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        return json.dumps({
            "tool_name": "duckduckgo_web_search",
            "query": query,
            "error": "'duckduckgo-search' or 'ddgs' library not installed.",
            "retrieved_contexts": [],
            "display": "ERROR: 'duckduckgo-search' library not installed.",
        }, ensure_ascii=False)

    try:
        results = []
        with DDGS(timeout=20) as ddgs:
            ddg_gen = ddgs.text(query, region="wt-wt", safesearch="off", max_results=max_results)
            if ddg_gen:
                for r in ddg_gen:
                    results.append(r)

        if not results:
            return json.dumps({
                "tool_name": "duckduckgo_web_search",
                "query": query,
                "retrieved_contexts": [],
                "display": f"No results found for query: {query}",
            }, ensure_ascii=False)

        display_lines = [f"DuckDuckGo Search Results for: {query}", ""]
        retrieved_contexts = []

        for i, r in enumerate(results, start=1):
            title = r.get("title", "No Title")
            link = r.get("href", "No Link")
            body = r.get("body", "")

            display_lines.append(f"[{i}] {title}\nURL: {link}\nSnippet: {body[:300]}...\n")
            retrieved_contexts.append({
                "content": body,
                "title": title,
                "url": link,
                "rank": i,
                "search_query": query,
                "retriever": "DuckDuckGo",
            })

        return json.dumps({
            "tool_name": "duckduckgo_web_search",
            "query": query,
            "max_results": max_results,
            "retrieved_contexts": retrieved_contexts,
            "display": "\n".join(display_lines),
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "tool_name": "duckduckgo_web_search",
            "query": query,
            "error": str(e),
            "retrieved_contexts": [],
            "display": f"ERROR during DuckDuckGo search: {e}",
        }, ensure_ascii=False)