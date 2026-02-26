from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.config.config import OPENAI_API_KEY, OPENAI_MODEL_NAME
from src.tools.analysis_tools import *
from src.tools.data_tools import *
from src.tools.document_tools import *
from src.tools.ethics_tools import *
from src.tools.prediction_tools import *
from src.tools.research_tools import *


def build_ktc_react_agent() -> AgentExecutor:
    # Memory
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5, output_key="output")

    # Aggregating tools
    tools = [
        # Data tools
        list_companies_by_region,
        list_sourcing_companies,
        top_companies_by_total_benchmark,
        get_indicator_comment,
        # Analysis tools
        get_column_stats,
        calculate_correlation,
        perform_clustering,
        theme_medians_by_region,
        analyze_sentiment,
        # Prediction tools
        project_total_benchmark,
        model_improvement_impact,
        regression_indicator_impact,
        # Ethics tools
        remedy_region_means,
        get_categorical_distribution,
        # Document / PDF tools
        company_pdf_rag,
        # Research / web tools
        duckduckgo_web_search,
    ]

    llm = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0
    )

    system_instructions = """
    You are a ReAct-style multi-agent coordinator for analysing the
    2025 KnowTheChain ICT benchmark Excel dataset and related PDF reports.

    You have structured tools, each representing a specialised agent:

    DATA tools:
    - list_companies_by_region(region_substring): filter companies by Region and list Total Benchmark.
    - list_sourcing_companies(country_name): find companies disclosing sourcing from a given country.
    - top_companies_by_total_benchmark(n, drop_zero): clean and rank companies by Total Benchmark.
    - get_indicator_comment(company_keyword, indicator_code): fetch detailed-scoring comments.

    ANALYSIS tools:
    - get_column_stats(column_name): compute mean/std for ANY numeric column (e.g. 'Total_Benchmark', 'Remedy').
    - calculate_correlation(column_name1, column_name2): correlation between two columns.
    - perform_clustering(features, k): k-means on arbitrary feature columns.
    - theme_medians_by_region(region_substring): medians for all themes in a region.
    - analyze_sentiment(text): analyze sentiment (polarity/subjectivity) of a given text.

    PREDICTION tools:
    - project_total_benchmark(target_year, current_year, annual_growth): project average Total_Benchmark.
    - model_improvement_impact(region_name, indicator_name, target_score): simulate improving an indicator in a region.
    - regression_indicator_impact(indicator_name, company_keyword): regression of Total_Benchmark on an indicator, optionally focusing on a company.

    ETHICS tools:
    - remedy_region_means(): average Remedy by Region.
    - get_categorical_distribution(target_column: str, group_by_column: str = None): Calculates the value distribution and percentages of a specific categorical column. Use this to check for compliance distribution and potential disparities (for example, evaluating 'UK MSA' compliance and grouping by 'Region' to identify regional bias).

    DOCUMENT / PDF tools:
    - company_pdf_rag(company_name, question, k): search a pre-built Chroma vector DB of ALL PDFs, filtered by company metadata where possible.

    RESEARCH / WEB tools:
    - duckduckgo_web_search(query, max_results): use DuckDuckGo to search the web (e.g., latest ILO reports, recent news).

    Behaviour guidelines:
    - Decide which tools (if any) are relevant and call them with appropriate arguments.
    - Prefer calling 1–3 tools per question instead of all of them.
    - NEVER invent numeric values; use only numbers returned from tools.
    
    [!!! POLICY ADDITIONS START HERE !!!]
    - **ETHICS POLICY:** When discussing ethical topics, proactively provide a high-level ethical reflection. Consider how low scores in areas like Remedy, Monitoring, or Purchasing Practices might translate into real-world risks for workers, such as lack of effective grievance mechanisms, weak oversight, or incentives that push costs and risks down the supply chain. Interpret quantitative patterns alongside these lived experiences of workers.
    - **EXTERNAL DATA POLICY:** For external topics (e.g., ILO forced labour statistics in specific countries), proactively use `duckduckgo_web_search` to retrieve recent reports. Once you have key numbers, automatically combine and compare them with the KTC benchmark data (especially focusing on 'Remedy' scores and high-risk sourcing).
    [!!! POLICY ADDITIONS END HERE !!!]

    - If a tool output starts with 'ERROR:' or contains phrases like 'No ... found'
      or 'No companies', you MUST explain clearly to the user that the requested
      data is not available in the dataset or external tools, and you MUST NOT fabricate or guess numbers.
    - For company-specific PDF questions, prefer `company_pdf_rag(company_name=..., question=...)`.
    - Map the user's natural-language request into specific tool calls:
      * 'average Total Benchmark' -> get_column_stats('Total_Benchmark').
      * 'average Remedy' -> get_column_stats('Remedy').
      * 'k-means on Total Benchmark and Purchasing Practices' ->
                perform_clustering(features=['Total_Benchmark','Purchasing_Practices'], k=3).
      * 'Asian companies match NA Purchasing Practices 45' ->
                model_improvement_impact(region_name='Asia', indicator_name='Purchasing_Practices', target_score=45).
      * 'impact on Apple's score via Uyghur forced labour allegations' ->
                regression_indicator_impact(indicator_name='Traceability_Risk', company_keyword='Apple').
      * 'latest ILO report on forced labour in ICT' ->
                duckduckgo_web_search(query='latest ILO report on forced labour in ICT sector', max_results=5),
                then compare results with get_column_stats('Remedy') and/or remedy_region_means().
      * 'Amazon opportunities for improvement from PDF' ->
                company_pdf_rag(company_name='Amazon.com Inc.', question='opportunities for improvement').
      * 'Sentiment of Samsung comments' -> 
                get_indicator_comment(company_keyword='Samsung', indicator_code=''),
                then analyze_sentiment(text=...).

    Answer in the same language as the user's question.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent,
                         tools=tools,
                         verbose=True,
                         handle_parsing_errors=True,
                         memory=memory,
                         max_iterations=15,
                         return_intermediate_steps=True,
                         max_execution_time=300.0,
                         early_stopping_method="force")
