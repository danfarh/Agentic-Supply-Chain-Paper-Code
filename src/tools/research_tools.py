from langchain_core.tools import tool


@tool
def duckduckgo_web_search(query: str, max_results: int = 5) -> str:
    """
    RESEARCH AGENT:
    Perform a web search using DuckDuckGo (No API key required).
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        return "ERROR: 'duckduckgo-search' library not installed."

    try:
        results = []
        with DDGS(timeout=20) as ddgs:
            ddg_gen = ddgs.text(query, region='wt-wt', safesearch='off', max_results=max_results)
            if ddg_gen:
                for r in ddg_gen:
                    results.append(r)

        if not results:
            return f"No results found for query: {query}"

        lines = [f"DuckDuckGo Search Results for: {query}", ""]
        for i, r in enumerate(results, start=1):
            title = r.get('title', 'No Title')
            link = r.get('href', 'No Link')
            body = r.get('body', '')
            lines.append(f"[{i}] {title}\nURL: {link}\nSnippet: {body[:300]}...\n")

        return "\n".join(lines)

    except Exception as e:
        return f"ERROR during DuckDuckGo search: {e}"


@tool
def external_data_note(topic: str) -> str:
    """
    RESEARCH AGENT (fallback):
    Explain that external web / ILO / news data may require web search, and suggest
    combining the fetched numbers with the KTC dataset.

    Parameters:
    - topic: e.g. 'ILO forced labour statistics in China and Malaysia'
    """
    return (
        f"For external topic '{topic}', you can use the duckduckgo_web_search tool "
        "to retrieve recent reports or statistics. Once key numbers or qualitative "
        "findings are available, they can be compared with the KTC benchmark data, "
        "especially focusing on Remedy scores, high-risk sourcing, and regional patterns."
    )
