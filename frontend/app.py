import asyncio

import matplotlib.pyplot as plt
import seaborn as sns
from shiny import App, ui, render, reactive
from sklearn.cluster import KMeans

# Import backend modules
from src.agent_builder import build_ktc_react_agent
from src.data_loader import get_global_context

# Initialize Data & Agent
ctx = get_global_context()
agent_executor = build_ktc_react_agent()

# ==============================================================================
# UI Design & Professional Styling
# ==============================================================================
brand_css = """
    .sidebar { background-color: #1a1c23 !important; color: white !important; }
    .sidebar h3, .sidebar h5 { color: #4dabf7; font-weight: 700; }
    .sidebar hr { border-top: 1px solid #3e4451; }
    .card-header { background-color: #f8f9fa; font-weight: 600; border-bottom: 2px solid #e9ecef; }
    .nav-underline .nav-link.active { border-bottom-color: #4dabf7 !important; color: #4dabf7 !important; }
    .main-title { 
        background: linear-gradient(90deg, #1a1c23 0%, #4dabf7 100%);
        color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
"""

app_ui = ui.page_fillable(
    ui.head_content(ui.tags.style(brand_css)),

    ui.div(ui.h2("⛓️ KTC 2025 Supply Chain Decision Support System", class_="main-title")),

    ui.page_sidebar(
        ui.sidebar(
            ui.h3("⚙️ Configuration"),
            ui.hr(),
            ui.h5("Regional Policy Filter"),
            ui.input_select("region_filter", None,
                            choices=["All", "Asia", "Europe", "North America"],
                            selected="All"),
            ui.hr(),
            ui.h5("Analysis Granularity"),
            ui.input_slider("cluster_k", "Cluster Count (k):", min=2, max=6, value=3),
            ui.hr(),
            ui.value_box(
                "System Status", "Multi-Agent Active",
                showcase=ui.div("●", style="color: #2fb344; font-size: 2rem;"),
                theme="light", height="150px"
            ),
            bg="#1a1c23"
        ),

        ui.navset_card_underline(
            # Tab 1: AI Chat Interface (Policy Strategy)
            ui.nav_panel(
                "💬 AI Assistant",
                ui.layout_columns(
                    ui.card(ui.chat_ui("chat"), full_screen=True),
                    ui.card(
                        ui.card_header("💡 Agent Team Capabilities"),
                        ui.markdown(
                            """
                            Interact with our specialized **Multi-Agent Team**:

                            * **📊 Data Agent:** Filter companies by region and sourcing origin.
                            * **📈 Analysis Agent:** Compute correlations and theme medians.
                            * **🔮 Prediction Agent:** Forecast 2027 ranks and improvement impacts.
                            * **⚖️ Ethics Agent:** Analyze regional bias and UK MSA compliance.
                            * **📄 Document Agent:** Semantic search (RAG) through company PDFs.
                            * **🌐 Research Agent:** Fetch real-time ILO data and news via DuckDuckGo.
                            """
                        ),
                        ui.hr(),
                        ui.p("Quick Actions:", style="font-weight: bold;"),
                        ui.input_action_button("btn_predict", "Predict 2027 Ranks",
                                               class_="btn-outline-primary btn-sm mb-2 w-100"),
                        ui.input_action_button("btn_bias", "Check Regional Bias",
                                               class_="btn-outline-primary btn-sm mb-2 w-100"),
                        ui.input_action_button("btn_ilo", "Search Latest ILO Reports",
                                               class_="btn-outline-primary btn-sm w-100"),
                    ),
                    col_widths=(8, 4)
                )
            ),

            # Tab 2: Visual Dashboard
            ui.nav_panel(
                "📊 Analytics Dashboard",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("📈 Cluster Analysis: Benchmark vs. Practices"),
                        ui.output_plot("cluster_plot"), full_screen=True
                    ),
                    ui.card(
                        ui.card_header("🌍 Regional Score Distribution"),
                        ui.output_plot("region_plot"), full_screen=True
                    ),
                ),
                ui.card(
                    ui.card_header("📋 Dataset Explorer"),
                    ui.output_data_frame("data_preview"), height="400px"
                )
            ),
            title="KTC Analysis Suite"
        )
    )
)


def server(input, output, session):
    chat = ui.Chat(id="chat", messages=[])

    @chat.on_user_submit
    async def _():
        user_input = chat.user_input()
        await chat.append_message(f"Coordinating agents for policy query: {user_input}...")
        try:
            response = await asyncio.to_thread(agent_executor.invoke, {"input": user_input, "chat_history": []})
            await chat.append_message(response.get("output", "No response generated."))
        except Exception as e:
            await chat.append_message(f"Error: {str(e)}")

    # Reactive logic for Sidebar/Dashboard
    @reactive.calc
    def filtered_data():
        df = ctx.scoring.copy()
        if input.region_filter() != "All":
            df = df[df["Region"].astype(str).str.contains(input.region_filter(), case=False, na=False)]
        return df

    @render.plot
    def cluster_plot():
        df = filtered_data().dropna(subset=["Total_Benchmark", "Purchasing_Practices"])
        if len(df) < input.cluster_k():
            fig, ax = plt.subplots();
            ax.text(0.5, 0.5, "Insufficient data for clustering.", ha='center');
            return fig
        kmeans = KMeans(n_clusters=input.cluster_k(), random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(df[["Total_Benchmark", "Purchasing_Practices"]])
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x="Total_Benchmark", y="Purchasing_Practices", hue="Cluster", palette="viridis", s=100,
                        ax=ax)
        ax.set_title(f"Clustering Analysis (k={input.cluster_k()})")
        return fig

    @render.plot
    def region_plot():
        df = filtered_data()
        cols = ["Commitment_Governance", "Traceability_Risk", "Purchasing_Practices", "Remedy"]
        df_melt = df.melt(value_vars=[c for c in cols if c in df.columns], var_name="Theme", value_name="Score")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_melt, x="Theme", y="Score", hue="Theme", palette="pastel", legend=False, ax=ax)
        ax.tick_params(axis='x', rotation=15);
        ax.set_title(f"Distribution: {input.region_filter()} Region")
        return fig

    @render.data_frame
    def data_preview():
        return render.DataGrid(filtered_data()[["Company", "Region", "Total_Benchmark", "Rank_2025"]].head(50),
                               filters=True)


app = App(app_ui, server)
