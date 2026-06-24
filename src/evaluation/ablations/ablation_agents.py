"""Agent builders for the MAS ablation study.

Variants implemented:
- llm_only: no tools; a direct LLM baseline.
- full_mas: all tools and all policy layers.
- mas_without_ethics: all non-ethics tools, with ethics policy removed.
- mas_without_rag: all non-RAG tools, no PDF/web retrieval tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.config.config import OPENAI_API_KEY, OPENAI_MODEL_NAME
from src.tools.analysis_tools import (
    analyze_sentiment,
    calculate_correlation,
    compare_companies,
    get_column_stats,
    perform_clustering,
    theme_medians_by_region,
)
from src.tools.data_tools import (
    filter_companies_by_score,
    get_indicator_comment,
    get_top_companies_by_metric,
    list_companies_by_region,
    list_sourcing_companies,
)
from src.tools.document_tools import company_pdf_rag
from src.tools.ethics_tools import compute_grouped_average, get_categorical_distribution
from src.tools.prediction_tools import (
    model_improvement_impact,
    project_metric_growth,
    regression_indicator_impact,
)
from src.tools.research_tools import duckduckgo_web_search


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str
    mode: str
    use_tools: bool
    include_ethics_layer: bool
    include_rag_tools: bool


VARIANTS: Dict[str, AblationVariant] = {
    "llm_only": AblationVariant(
        name="llm_only",
        description="LLM-only baseline: no structured tools, no RAG, no MAS routing.",
        mode="llm_only",
        use_tools=False,
        include_ethics_layer=False,
        include_rag_tools=False,
    ),
    "full_mas": AblationVariant(
        name="full_mas",
        description="Full MAS: structured tools, ethics layer, PDF RAG and web retrieval.",
        mode="mas",
        use_tools=True,
        include_ethics_layer=True,
        include_rag_tools=True,
    ),
    "mas_without_ethics": AblationVariant(
        name="mas_without_ethics",
        description="MAS without Ethics layer: removes ethics tools and proactive ethics-policy guidance.",
        mode="mas",
        use_tools=True,
        include_ethics_layer=False,
        include_rag_tools=True,
    ),
    "mas_without_rag": AblationVariant(
        name="mas_without_rag",
        description="MAS without RAG: removes PDF/vector retrieval and web-search retrieval tools.",
        mode="mas",
        use_tools=True,
        include_ethics_layer=True,
        include_rag_tools=False,
    ),
}


class LLMOnlyExecutor:
    """Small adapter exposing AgentExecutor-like invoke/metadata behavior."""

    def __init__(self, llm: ChatOpenAI, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory = None

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("input", ""))
        msg = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query),
        ])
        return {"output": getattr(msg, "content", str(msg)), "intermediate_steps": []}


def build_tools(*, include_ethics_layer: bool, include_rag_tools: bool) -> List[Any]:
    data_tools = [
        list_companies_by_region,
        list_sourcing_companies,
        get_top_companies_by_metric,
        get_indicator_comment,
        filter_companies_by_score,
    ]

    analysis_tools = [
        get_column_stats,
        calculate_correlation,
        compare_companies,
        perform_clustering,
        theme_medians_by_region,
        analyze_sentiment,
    ]

    prediction_tools = [
        project_metric_growth,
        model_improvement_impact,
        regression_indicator_impact,
    ]

    ethics_tools = [
        compute_grouped_average,
        get_categorical_distribution,
    ] if include_ethics_layer else []

    rag_tools = [
        company_pdf_rag,
        duckduckgo_web_search,
    ] if include_rag_tools else []

    return data_tools + analysis_tools + prediction_tools + ethics_tools + rag_tools


def base_system_instructions(*, include_ethics_layer: bool, include_rag_tools: bool, llm_only: bool = False) -> str:
    tool_section = "" if llm_only else """
You have structured tools, each representing specialised agents:

DATA tools:
- list_companies_by_region(region_substring)
- list_sourcing_companies(country_name)
- get_top_companies_by_metric(column_name, n, ascending, drop_zero)
- filter_companies_by_score(column_name, operator_str, threshold)
- get_indicator_comment(company_keyword, indicator_code)

ANALYSIS tools:
- get_column_stats(column_name)
- calculate_correlation(column_x, column_y)
- compare_companies(company_names)
- perform_clustering(features, k)
- theme_medians_by_region(region_substring)
- analyze_sentiment(text)

PREDICTION tools:
- project_metric_growth(column_name, annual_growth_rate, current_year, target_year)
- model_improvement_impact(region_name, indicator_name, target_score, target_region, coefficient_source)
- regression_indicator_impact(indicator_name, company_keyword, delta_points, coefficient_source)
"""

    ethics_section = """
ETHICS tools:
- compute_grouped_average(target_column, group_by_column)
- get_categorical_distribution(target_column, group_by_column)

ETHICS POLICY:
When discussing ethical topics, include a high-level ethical reflection. Consider how low scores in Remedy, Monitoring, or Purchasing Practices can translate into real-world risks for workers, including weak grievance mechanisms, weak oversight, or incentives that push costs and risks down the supply chain. Interpret quantitative patterns alongside worker impact.
""" if include_ethics_layer else """
ETHICS-LAYER ABLATION:
The proactive ethics layer is disabled for this run. Do not call ethics-specific tools because they are not available in this variant. Answer ethics questions using only general reasoning and any non-ethics structured evidence available from other tools.
"""

    rag_section = """
DOCUMENT / PDF tools:
- company_pdf_rag(company_name, question, k): search a pre-built Chroma vector DB of company PDFs.

RESEARCH / WEB tools:
- duckduckgo_web_search(query, max_results): retrieve external web snippets.

EXTERNAL DATA / RAG POLICY:
For external topics such as ILO forced-labour statistics, recent reports, or company PDF evidence, proactively use PDF/web retrieval tools and compare retrieved evidence with KTC benchmark data.
""" if include_rag_tools else """
RAG ABLATION:
PDF/vector retrieval and web-search retrieval are disabled for this run. Do not claim to have searched PDFs or the web. If a query requires external/PDF evidence, state that this variant cannot retrieve external/PDF context and answer only from available structured tools or general reasoning.
"""

    llm_only_section = """
LLM-ONLY BASELINE:
No tools, spreadsheet access, PDF retrieval, web retrieval, or intermediate agent layers are available. Answer directly from the model's prior knowledge and the query text. If the question requires exact dataset values, explain that this baseline cannot verify them from tools, but still provide the best possible answer with uncertainty. Do not fabricate exact numbers as if they were retrieved.
""" if llm_only else """
Behaviour guidelines:
- Decide which tools are relevant and call them with appropriate arguments.
- Prefer 1-3 tool calls per question instead of all tools.
- Never invent numeric values; use only numbers returned from tools.
- If a tool returns JSON, read its display field for the final answer. Do not expose raw JSON.
- For prediction/scenario questions, state formula, assumptions, coefficient source, and uncertainty.
- Do not predict a future rank unless current score and future score distribution/ranking assumptions are available.
- If a tool output starts with ERROR or says no data was found, explain that the requested data is unavailable and do not fabricate numbers.
"""

    routing_section = """Natural-language routing hints:
- average Total Benchmark -> get_column_stats('Total_Benchmark')
- average Remedy by region -> compute_grouped_average('Remedy', 'Region') if ethics tools are enabled
- top 5 companies -> get_top_companies_by_metric(...)
- k-means -> perform_clustering(...)
- latest ILO report -> duckduckgo_web_search(...) if RAG is enabled
- Amazon PDF/opportunities -> company_pdf_rag(...) if RAG is enabled
- UK MSA distribution -> get_categorical_distribution('UK MSA', 'Region') if ethics tools are enabled
""" if not llm_only else """
For all queries, answer directly without tools. Be explicit about uncertainty where exact dataset/PDF/web evidence is needed.
"""

    return f"""
You are evaluating a variant of a multi-agent system for the 2025 KnowTheChain ICT benchmark.
Answer in the same language as the user's question.
{tool_section}
{ethics_section}
{rag_section}
{llm_only_section}
{routing_section}
""".strip()


def build_ablation_agent(variant_name: str) -> Any:
    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant_name}'. Valid variants: {sorted(VARIANTS)}")

    variant = VARIANTS[variant_name]
    llm = ChatOpenAI(model=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)

    if not variant.use_tools:
        return LLMOnlyExecutor(
            llm=llm,
            system_prompt=base_system_instructions(
                include_ethics_layer=variant.include_ethics_layer,
                include_rag_tools=variant.include_rag_tools,
                llm_only=True,
            ),
        )

    tools = build_tools(
        include_ethics_layer=variant.include_ethics_layer,
        include_rag_tools=variant.include_rag_tools,
    )
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
        output_key="output",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", base_system_instructions(
            include_ethics_layer=variant.include_ethics_layer,
            include_rag_tools=variant.include_rag_tools,
            llm_only=False,
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )
