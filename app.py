"""
Primary Streamlit Application.
Serves as the interactive graphical frontend connecting the user to the underlying
LLM parsing mechanisms and TOPSIS multi-criteria optimization engines.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os

from src.data_pipeline import MaterialsDataPipeline
from src.llm_parser import LLMInterface
from src.recommender import MaterialRecommender
from src.predictive_model import MaterialPropertyPredictor


# -------------------------------
# INITIALIZATION
# -------------------------------
load_dotenv()

st.set_page_config(
    page_title="Smart Alloy Selector",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1e3a8a;
        }
        .sub-title {
            font-size: 1.2rem;
            color: #4b5563;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


# -------------------------------
# DATA LOADING
# -------------------------------
@st.cache_data
def load_and_process_data():
    pipeline = MaterialsDataPipeline("data/materials_database.csv")
    df = pipeline.process_pipeline()

    predictor = MaterialPropertyPredictor(df)
    df = predictor.predict_missing_fatigue_strength()

    return df


try:
    df_materials = load_and_process_data()
except Exception as e:
    st.error(f"Critical Failure Loading Materials Database: {e}")
    st.stop()


# -------------------------------
# CORE ENGINE INITIALIZATION
# -------------------------------
llm_interface = LLMInterface()
recommender = MaterialRecommender(df_materials)


# -------------------------------
# HEADER
# -------------------------------
st.markdown(
    '<div class="main-title">⚙️ Smart Alloy Selector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI-Powered Multi-Objective Materials Discovery Engine</div>',
    unsafe_allow_html=True
)


# -------------------------------
# TABS
# -------------------------------
tab_search, tab_manual, tab_plots = st.tabs([
    "AI Search",
    "Manual Filter",
    "Ashby Plots"
])


# ==========================================================
# TAB 1 — AI SEARCH
# ==========================================================
with tab_search:

    st.write("### Define Engineering Operational Constraints")

    user_query = st.text_area(
        "Example: 'Designing an aerospace bracket. Requires very low density, high strength, and corrosion resistance.'",
        height=120
    )

    if st.button("Execute Autonomous Selection", type="primary"):

        if not user_query.strip():
            st.warning("Please enter an engineering query.")
            st.stop()

        if not os.getenv("GEMINI_API_KEY"):
            st.error("Gemini API key missing from environment variables.")
            st.stop()

        with st.spinner("Running AI parser and TOPSIS optimization..."):

            try:
                # -------------------------------
                # Phase 1: Parse Query
                # -------------------------------
                constraints = llm_interface.parse_query_to_constraints(user_query)

                st.write("#### 🧩 Extracted Deterministic Constraints")
                st.json(constraints.model_dump())

                # -------------------------------
                # Phase 2: Recommendation Engine
                # -------------------------------
                ranked_materials = recommender.get_recommendations(
                    constraints.model_dump()
                )

                if len(ranked_materials) == 0:
                    st.error(
                        "No materials satisfy the specified hard constraints."
                    )

                else:
                    top_material = ranked_materials.iloc[0]

                    st.success(
                        f"### 🏆 Primary Recommendation: {top_material['name']}"
                    )

                    # -------------------------------
                    # Ranked Results Table
                    # -------------------------------
                    st.write(
                        "#### Quantitative Comparison Matrix (Ranked by TOPSIS Score)"
                    )

                    display_cols = [
                        'name',
                        'category',
                        'topsis_score',
                        'density_g_cm3',
                        'yield_strength_mpa',
                        'max_service_temp_c',
                        'cost_usd_kg'
                    ]

                    st.dataframe(
                        ranked_materials[display_cols]
                        .head(5)
                        .style.background_gradient(
                            subset=['topsis_score'],
                            cmap='viridis'
                        ),
                        use_container_width=True
                    )

                    # -------------------------------
                    # Phase 3: Explanation Generation
                    # -------------------------------
                    st.write("#### 🧠 AI Materials Scientist Evaluation Rationale")

                    try:
                        explanation = llm_interface.generate_explanation(
                            user_query,
                            top_material,
                            ranked_materials.head(5).to_dict()
                        )

                        st.info(explanation)

                    except Exception:
                        st.warning(
                            "Explanation generation unavailable due to temporary API issue."
                        )

            except Exception as e:
                st.error(f"Algorithmic fault during execution phase: {e}")


# ==========================================================
# TAB 2 — MANUAL FILTER
# ==========================================================
with tab_manual:

    st.write("### Manual Parameter Override")

    col1, col2, col3 = st.columns(3)

    with col1:
        max_dens = st.slider(
            "Maximum Density (g/cm³)",
            0.0, 25.0, 25.0
        )

    with col2:
        min_yield = st.slider(
            "Minimum Yield Strength (MPa)",
            0.0, 2000.0, 0.0
        )

    with col3:
        max_cost = st.slider(
            "Maximum Cost (USD/kg)",
            0.0, 65000.0, 65000.0
        )

    filtered_manual = df_materials[
        (df_materials['density_g_cm3'] <= max_dens) &
        (df_materials['yield_strength_mpa'] >= min_yield) &
        (df_materials['cost_usd_kg'] <= max_cost)
    ]

    st.dataframe(
        filtered_manual,
        use_container_width=True
    )


# ==========================================================
# TAB 3 — ASHBY PLOTS
# ==========================================================
with tab_plots:

    st.write("### Algorithmic Performance Visualization (Ashby Analysis)")

    numeric_cols = df_materials.select_dtypes(
        include=['float64', 'int64']
    ).columns.tolist()

    col_x, col_y = st.columns(2)

    with col_x:
        x_axis = st.selectbox(
            "X-Axis Independent Variable",
            numeric_cols,
            index=0
        )

    with col_y:
        y_axis = st.selectbox(
            "Y-Axis Dependent Variable",
            numeric_cols,
            index=1
        )

    fig = px.scatter(
        df_materials,
        x=x_axis,
        y=y_axis,
        color="category",
        hover_name="name",
        title=f"{y_axis} vs {x_axis}",
        log_x=True,
        log_y=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )