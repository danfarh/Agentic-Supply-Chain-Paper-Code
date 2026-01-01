import asyncio

import matplotlib.pyplot as plt
import seaborn as sns
from shiny import App, ui, render, reactive

# Import backend modules
from src.agent_builder import build_ktc_react_agent
from src.data_loader import get_global_context

# Initialize Data & Agent
ctx = get_global_context()
agent_executor = build_ktc_react_agent()

# ==============================================================================
# UI Design
# ==============================================================================
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("⚙️ KTC Settings"),
        ui.hr(),

        # Dashboard Controls
        ui.h5("Dashboard Controls"),
        ui.input_select(
            "region_filter",
            "Select Region:",
            choices=["All", "Asia", "Europe", "North America"],
            selected="All"
        ),
        ui.input_slider("cluster_k", "Cluster Count (k):", min=2, max=6, value=3),

        ui.hr(),
        ui.p(ui.strong("Note:"),
             " These settings affect the Visual Dashboard tab. The Chat Agent works independently."),
        bg="#f8f9fa"
    ),

    ui.navset_card_underline(
        # Tab 1: AI Chat Interface
        ui.nav_panel(
            "💬 AI Assistant",
            ui.chat_ui("chat"),
            ui.markdown(
                """
                **Suggested Queries:**
                - *Predict Amazon's 2027 rank if it improves Remedy by 10 points.*
                - *Filter companies in Asia with Total Benchmark > 50.*
                - *What is the correlation between Market Cap and Total Score?*
                """
            )
        ),

        # Tab 2: Visual Dashboard (DSS)
        ui.nav_panel(
            "📊 Visual Dashboard",
            ui.layout_columns(
                ui.card(
                    ui.card_header("📈 Benchmark vs. Purchasing Practices (Clustering)"),
                    ui.output_plot("cluster_plot"),
                ),
                ui.card(
                    ui.card_header("🌍 Regional Score Distribution"),
                    ui.output_plot("region_plot"),
                ),
            ),
            ui.card(
                ui.card_header("📋 Raw Data Preview"),
                ui.output_data_frame("data_preview"),
                height="300px"
            )
        ),

        title="⛓️ KTC 2025 Supply Chain Decision Support System"
    )
)


# ==============================================================================
# Server Logic
# ==============================================================================
def server(input, output, session):
    # 1. Chat Logic
    chat = ui.Chat(id="chat", messages=[])

    @chat.on_user_submit
    async def _():
        user_input = chat.user_input()

        # 1. Show "Thinking..." message
        await chat.append_message(f"Thinking about: {user_input}...")

        # 2. Invoke LangChain Agent
        try:
            response = await asyncio.to_thread(
                agent_executor.invoke,
                {"input": user_input, "chat_history": []}
            )

            final_answer = response.get("output", "No response generated.")
            await chat.append_message(final_answer)

        except Exception as e:
            await chat.append_message(f"Error: {str(e)}")

    # 2. Dashboard Logic (Reactive)

    @reactive.calc
    def filtered_data():
        """Filter the global dataframe based on sidebar inputs."""
        df = ctx.scoring.copy()
        if input.region_filter() != "All":
            df = df[df["Region"].astype(str).str.contains(input.region_filter(), case=False, na=False)]
        return df

    @render.plot
    def cluster_plot():
        """Generates K-Means clustering plot on the fly."""
        df = filtered_data().dropna(subset=["Total_Benchmark", "Purchasing_Practices"])

        if len(df) < input.cluster_k():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough data for clustering", ha='center')
            return fig

        # Simple K-Means for visualization
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=input.cluster_k(), random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(df[["Total_Benchmark", "Purchasing_Practices"]])

        # Plot using Seaborn
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df,
            x="Total_Benchmark",
            y="Purchasing_Practices",
            hue="Cluster",
            palette="viridis",
            s=100,
            ax=ax
        )
        ax.set_title(f"Clustering (k={input.cluster_k()}) - {input.region_filter()} Region")
        ax.grid(True, linestyle='--', alpha=0.6)
        return fig

    @render.plot
    def region_plot():
        """Generates a boxplot of scores by theme."""
        df = filtered_data()
        cols = ["Commitment_Governance", "Traceability_Risk", "Purchasing_Practices", "Remedy"]

        # Melt for plotting
        df_melt = df.melt(value_vars=[c for c in cols if c in df.columns], var_name="Theme", value_name="Score")

        fig, ax = plt.subplots(figsize=(8, 5))

        # Plot with Seaborn (Fixes palette/hue warning)
        sns.boxplot(
            data=df_melt,
            x="Theme",
            y="Score",
            hue="Theme",
            palette="pastel",
            legend=False,
            ax=ax
        )

        # Rotate labels correctly
        ax.tick_params(axis='x', rotation=15)
        ax.set_title(f"Score Distribution by Theme - {input.region_filter()} Region")
        return fig

    @render.data_frame
    def data_preview():
        """Shows a preview table of the filtered data."""
        return render.DataGrid(
            filtered_data()[["Company", "Region", "Total_Benchmark", "Rank_2025"]].head(20),
            filters=True
        )


# Create the App object
app = App(app_ui, server)
