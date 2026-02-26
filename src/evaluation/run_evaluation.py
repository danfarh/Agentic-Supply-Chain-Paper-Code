import os
import time
from datetime import datetime

from src.agent_builder import build_ktc_react_agent
from src.tools.document_tools import get_vector_db

TEST_QUERIES = [
    # 1. Data Agent
    "Filter the companies in the Scoring sheet by Region = `Asia` and list their Company Names and Total Benchmark Scores.",
    "From the Non-Scored Research sheet, extract all companies that disclose sourcing from `China` and list their Company Names and Market Caps.",
    "Clean the Scoring sheet by removing any rows with missing or zero Total Benchmark Scores, then list the top 5 companies by Total Benchmark Score descending.",
    "Extract from the Detailed Scoring & Research sheet the comment for Amazon.com Inc. on indicator 1.1 (Supplier Code of Conduct).",

    # 2. Analysis Agent
    "Compute the average Total Benchmark Score across all companies in the Scoring sheet, and the standard deviation.",
    "Calculate the correlation between Market Cap and Total Benchmark Score using the Scoring sheet data.",
    "Perform k-means clustering (k=3) on companies based on Total Benchmark Score and Purchasing Practices score from Scoring sheet, and list clusters.",
    "Compute the median score for each theme (e.g., Commitment & Governance) across North American companies.",

    # 3. External data / ILO
    "Cross-verify the high-risk sourcing countries for Amazon from the PDF with current ILO statistics on forced labour prevalence in China and Malaysia.",
    "Search for the latest ILO report on forced labour in the ICT sector and compare it to the average KTC Remedy score (7/100).",
    "Validate Samsung's top rank (79.5) by searching for recent news on its supply chain practices.",
    "Fetch external data on global average market cap for ICT semiconductors and compare to KTC dataset average.",

    # 4. Prediction
    "Based on historical ranks (e.g., Amazon 2022 rank 8, 2025 rank 10), predict Amazon's 2027 rank if it improves Remedy by 10 points.",
    "Project the industry average Total Benchmark Score for 2027 if scores increase by 5% annually.",
    "Model score improvement for Asian companies if they match North America's avg Purchasing Practices (5.77).",
    "Predict the impact on Apple's score if it addresses Uyghur forced labour allegations (from data).",

    # 5. Synthesis-style
    "Synthesise key insights on top 5 companies' strengths and weaknesses from Scoring sheet.",
    "Compile a report on regional differences in Remedy scores, with ethical note on data biases.",
    "Synthesise opportunities for improvement from Amazon PDF.",
    "Compile insights on correlations between Market Cap and scores, noting ethical implications.",

    # 6. Ethics
    "Evaluate potential biases in the Scoring sheet data, using principal-agent theory.",
    "Assess if low Remedy scores (avg 7) could amplify ethical risks in supply chains.",
    "Check for regional bias in Non-Scored Research (e.g., more \"Yes\" for UK MSA in NA/Europe).",
    "Evaluate ethical risks in predicting improvements for low-scorers like BOE (0 score).",

    # 7. Text Mining / PDF
    "Extract and summarise key phrases from Amazon PDF on `Opportunities for Improvement`",
    "Perform sentiment analysis on the Detailed Scoring & Research comments for Samsung.",
    "Mine the Amazon PDF for mentions of `forced labour` and categorise themes.",
    "Analyse text from Non-Scored Research on UK MSA compliance and identify patterns.",
]

def main():
    print("🚀 Initializing Evaluation Pipeline...")
    
    # 1. Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create a timestamped filename for the results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join("logs", f"evaluation_results_{timestamp}.txt")
    
    # 2. Eagerly load databases to speed up evaluation
    print("⏳ Loading Vector DB into memory...")
    try:
        get_vector_db()
        print("✅ Vector DB loaded.")
    except Exception as e:
        print(f"⚠️ Warning: Vector DB failed to load eagerly: {e}")
        
    # (Optional) load_ktc_data() if you have an eager loader for Excel
    
    # 3. Build the Agent
    print("🤖 Building Multi-Agent System...")
    agent_executor = build_ktc_react_agent()
    
    total_queries = len(TEST_QUERIES)
    
    print(f"📄 Starting evaluation of {total_queries} queries. Results will be saved to '{log_filename}'\n")

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"=== MULTI-AGENT SYSTEM EVALUATION LOG ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Queries: {total_queries}\n")
        f.write("="*60 + "\n\n")

        for i, query in enumerate(TEST_QUERIES, start=1):
            print(f"Evaluating [{i}/{total_queries}]: {query[:60]}...")
            
            # Clear memory so previous queries don't contaminate the current one
            if agent_executor.memory:
                agent_executor.memory.clear()
            
            start_time = time.time()
            
            try:
                # Invoke the agent
                response = agent_executor.invoke({"input": query})
                answer = response.get("output", "No output returned.")
                status = "SUCCESS"
            except Exception as e:
                # Catch exceptions (like API timeouts or parsing errors) so the script doesn't crash
                answer = f"ERROR DURING EXECUTION: {str(e)}"
                status = "FAILED"
                
            elapsed_time = time.time() - start_time
            
            # Write results to file
            f.write(f"--- Query {i}/{total_queries} [{status}] ---\n")
            f.write(f"Time Taken : {elapsed_time:.2f} seconds\n")
            f.write(f"Question   : {query}\n")
            f.write(f"Answer     :\n{answer}\n")
            f.write("-" * 60 + "\n\n")
            
            # Optional: Small delay to avoid hitting OpenAI API rate limits
            time.sleep(2)

    print(f"\n✅ Evaluation complete! All results successfully saved to {log_filename}")

if __name__ == "__main__":
    main()