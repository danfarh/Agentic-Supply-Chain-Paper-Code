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
