import streamlit as st
import pandas as pd
import joblib
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CardioSense",
    page_icon="🫀",
    layout="wide"
)


# =========================================================
# DARK THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1117;
    color: #E8EDF2;
}

section[data-testid="stSidebar"] {
    background-color: #111923;
}

h1, h2, h3 {
    color: #F5F7FA;
}

[data-testid="stMetric"] {
    background-color: #151F2B;
    border: 1px solid #263342;
    padding: 15px;
    border-radius: 12px;
}

div.stButton > button {
    width: 100%;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "heart_prediction.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "heart_scaler.pkl"
)

COLUMNS_PATH = os.path.join(
    BASE_DIR,
    "heart_columns.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = None
scaler = None
expected_columns = None

try:

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    expected_columns = joblib.load(COLUMNS_PATH)

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )


# =========================================================
# SESSION STATE
# =========================================================

if "assessment_history" not in st.session_state:
    st.session_state.assessment_history = []

if "latest_explanation" not in st.session_state:
    st.session_state.latest_explanation = None

if "latest_input" not in st.session_state:
    st.session_state.latest_input = None

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None

if "latest_probability" not in st.session_state:
    st.session_state.latest_probability = None


# =========================================================
# HELPER FUNCTION
# =========================================================

def prepare_input(
    age,
    resting_bp,
    cholesterol,
    fasting_bs,
    max_hr,
    oldpeak,
    sex,
    chest_pain,
    resting_ecg,
    exercise_angina,
    st_slope
):

    input_data = pd.DataFrame({

        "Age": [age],

        "RestingBP": [resting_bp],

        "Cholesterol": [cholesterol],

        "FastingBS": [fasting_bs],

        "MaxHR": [max_hr],

        "Oldpeak": [oldpeak],

        "Sex": [sex],

        "ChestPainType": [chest_pain],

        "RestingECG": [resting_ecg],

        "ExerciseAngina": [exercise_angina],

        "ST_Slope": [st_slope]

    })


    encoded_data = pd.get_dummies(
        input_data,
        columns=[
            "Sex",
            "ChestPainType",
            "RestingECG",
            "ExerciseAngina",
            "ST_Slope"
        ]
    )


    encoded_data = encoded_data.reindex(
        columns=expected_columns,
        fill_value=0
    )


    numerical_columns = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak"
    ]


    encoded_data[numerical_columns] = (
        scaler.transform(
            encoded_data[numerical_columns]
        )
    )


    return encoded_data


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🫀 CardioSense")

    st.caption(
        "Explainable Cardiovascular Risk Intelligence"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Risk Assessment",
            "Risk Trajectory",
            "Explainable AI",
            "What-If Simulator",
            "Model Monitoring"
        ]
    )

    st.markdown("---")

    st.caption(
        "AI-powered cardiovascular risk analysis"
    )


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    st.title(
        "🫀 CardioSense"
    )

    st.subheader(
        "Explainable Cardiovascular Risk Intelligence"
    )

    st.write(
        "A machine-learning based dashboard for "
        "cardiovascular risk assessment, explanation, "
        "simulation and monitoring."
    )

    st.markdown("---")


    history = st.session_state.assessment_history


    if len(history) > 0:

        current_risk = (
            history[-1]["Risk Probability"]
        )

        current_risk_text = (
            f"{current_risk:.2f}%"
        )

    else:

        current_risk_text = "--"


    if len(history) > 1:

        previous_risk = (
            history[-2]["Risk Probability"]
        )

        previous_risk_text = (
            f"{previous_risk:.2f}%"
        )

        risk_change = (
            current_risk -
            previous_risk
        )

        risk_change_text = (
            f"{risk_change:+.2f}%"
        )

    else:

        previous_risk_text = "--"
        risk_change_text = "--"


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Current Risk",
            current_risk_text
        )


    with col2:

        st.metric(
            "Previous Risk",
            previous_risk_text
        )


    with col3:

        st.metric(
            "Risk Change",
            risk_change_text
        )


    with col4:

        st.metric(
            "System Status",
            "Ready" if model is not None else "Error"
        )


    st.markdown("---")


    st.subheader(
        "CardioSense Modules"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            "📈 **Risk Trajectory**\n\n"
            "Track estimated model risk across "
            "multiple assessments."
        )


    with col2:

        st.info(
            "🔍 **Explainable AI**\n\n"
            "Understand which model features "
            "contribute to the prediction."
        )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            "🧪 **What-If Simulator**\n\n"
            "Modify patient inputs and compare "
            "the original and simulated model output."
        )


    with col2:

        st.info(
            "🤖 **Model Monitoring**\n\n"
            "Monitor prediction activity, "
            "risk probabilities and input statistics."
        )


    st.markdown("---")


    st.subheader(
        "System Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Model",
            "Logistic Regression"
        )


    with col2:

        st.metric(
            "Features",
            len(expected_columns)
            if expected_columns is not None
            else "--"
        )


    with col3:

        st.metric(
            "Scaler",
            "StandardScaler"
        )


    st.markdown("---")


    st.caption(
        "CardioSense is an academic machine-learning "
        "project. Model outputs are not medical diagnoses."
    )


# =========================================================
# RISK ASSESSMENT
# =========================================================

elif page == "Risk Assessment":

    st.title(
        "🫀 Risk Assessment"
    )

    st.write(
        "Enter patient information to generate a "
        "cardiovascular risk prediction."
    )


    if (
        model is None
        or scaler is None
        or expected_columns is None
    ):

        st.error(
            "Required model files could not be loaded."
        )

        st.stop()


    st.markdown("---")


    st.subheader(
        "👤 Patient Profile"
    )


    col1, col2 = st.columns(2)


    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50,
            step=1
        )


    with col2:

        sex = st.selectbox(
            "Sex",
            ["M", "F"]
        )


    st.subheader(
        "❤️ Vitals"
    )


    col1, col2 = st.columns(2)


    with col1:

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120,
            step=1
        )


    with col2:

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=700,
            value=200,
            step=1
        )


    col1, col2 = st.columns(2)


    with col1:

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1]
        )


    with col2:

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150,
            step=1
        )


    oldpeak = st.number_input(
        "Oldpeak",
        min_value=-5.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )


    st.subheader(
        "🩺 ECG & Symptoms"
    )


    col1, col2 = st.columns(2)


    with col1:

        chest_pain = st.selectbox(
            "Chest Pain Type",
            [
                "ATA",
                "NAP",
                "TA",
                "ASY"
            ]
        )


    with col2:

        resting_ecg = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "ST",
                "LVH"
            ]
        )


    col1, col2 = st.columns(2)


    with col1:

        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            [
                "N",
                "Y"
            ]
        )


    with col2:

        st_slope = st.selectbox(
            "ST Slope",
            [
                "Up",
                "Flat",
                "Down"
            ]
        )


    st.markdown("---")


    assess = st.button(
        "🫀 Run Risk Assessment",
        use_container_width=True
    )


    if assess:

        try:

            transformed_data = prepare_input(
                age,
                resting_bp,
                cholesterol,
                fasting_bs,
                max_hr,
                oldpeak,
                sex,
                chest_pain,
                resting_ecg,
                exercise_angina,
                st_slope
            )


            prediction = model.predict(
                transformed_data
            )[0]


            if hasattr(
                model,
                "predict_proba"
            ):

                probability = (
                    model.predict_proba(
                        transformed_data
                    )[0][1]
                )

            else:

                probability = float(prediction)


            risk_percentage = (
                probability * 100
            )


            # -----------------------------------------
            # SAVE CURRENT RESULT
            # -----------------------------------------

            st.session_state.latest_prediction = (
                prediction
            )

            st.session_state.latest_probability = (
                risk_percentage
            )


            st.session_state.latest_input = {

                "Age": age,

                "RestingBP": resting_bp,

                "Cholesterol": cholesterol,

                "FastingBS": fasting_bs,

                "MaxHR": max_hr,

                "Oldpeak": oldpeak,

                "Sex": sex,

                "ChestPainType": chest_pain,

                "RestingECG": resting_ecg,

                "ExerciseAngina": exercise_angina,

                "ST_Slope": st_slope

            }


            # -----------------------------------------
            # EXPLAINABLE AI
            # -----------------------------------------

            if hasattr(
                model,
                "coef_"
            ):

                coefficients = (
                    model.coef_[0]
                )

                feature_values = (
                    transformed_data.iloc[0].values
                )

                contributions = (
                    feature_values *
                    coefficients
                )


                explanation_df = pd.DataFrame({

                    "Feature":
                        expected_columns,

                    "Value":
                        feature_values,

                    "Coefficient":
                        coefficients,

                    "Contribution":
                        contributions

                })


                explanation_df["Direction"] = (
                    explanation_df[
                        "Contribution"
                    ].apply(
                        lambda x:
                        "Higher Risk"
                        if x > 0
                        else "Lower Risk"
                    )
                )


                explanation_df[
                    "Absolute Contribution"
                ] = (
                    explanation_df[
                        "Contribution"
                    ].abs()
                )


                explanation_df = (
                    explanation_df.sort_values(
                        "Absolute Contribution",
                        ascending=False
                    )
                )


                st.session_state.latest_explanation = (
                    explanation_df
                )


            # -----------------------------------------
            # SAVE HISTORY
            # -----------------------------------------

            assessment_number = (
                len(
                    st.session_state.assessment_history
                ) + 1
            )


            st.session_state.assessment_history.append({

                "Assessment":
                    assessment_number,

                "Age":
                    age,

                "Resting BP":
                    resting_bp,

                "Cholesterol":
                    cholesterol,

                "Max HR":
                    max_hr,

                "Oldpeak":
                    oldpeak,

                "Prediction":
                    (
                        "Higher Risk"
                        if prediction == 1
                        else "Lower Risk"
                    ),

                "Risk Probability":
                    risk_percentage

            })


            # -----------------------------------------
            # DISPLAY RESULT
            # -----------------------------------------

            st.markdown("---")

            st.subheader(
                "Assessment Result"
            )


            if prediction == 1:

                st.error(
                    "⚠️ Higher Risk Prediction"
                )

            else:

                st.success(
                    "✅ Lower Risk Prediction"
                )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Predicted Class",
                    "Class 1"
                    if prediction == 1
                    else "Class 0"
                )


            with col2:

                st.metric(
                    "Model Risk Probability",
                    f"{risk_percentage:.2f}%"
                )


            if (
                st.session_state.latest_explanation
                is not None
            ):

                strongest = (
                    st.session_state
                    .latest_explanation
                    .iloc[0]
                )


                st.info(
                    f"Strongest model contribution: "
                    f"**{strongest['Feature']}** "
                    f"({strongest['Contribution']:+.4f})"
                )


            st.info(
                "This result represents the output of "
                "the trained machine-learning model and "
                "is not a medical diagnosis."
            )


        except Exception as e:

            st.error(
                f"Prediction could not be completed: {e}"
            )


# =========================================================
# RISK TRAJECTORY
# =========================================================

elif page == "Risk Trajectory":

    st.title(
        "📈 Risk Trajectory"
    )

    st.write(
        "Track how estimated model risk changes "
        "across multiple assessments."
    )


    history = st.session_state.assessment_history


    if len(history) == 0:

        st.info(
            "Run a Risk Assessment first."
        )


    else:

        df_history = pd.DataFrame(
            history
        )


        current_risk = (
            df_history[
                "Risk Probability"
            ].iloc[-1]
        )


        if len(df_history) > 1:

            previous_risk = (
                df_history[
                    "Risk Probability"
                ].iloc[-2]
            )

        else:

            previous_risk = current_risk


        risk_change = (
            current_risk -
            previous_risk
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Assessments",
                len(df_history)
            )


        with col2:

            st.metric(
                "Current Probability",
                f"{current_risk:.2f}%"
            )


        with col3:

            st.metric(
                "Change",
                f"{risk_change:+.2f}%"
            )


        st.markdown("---")


        st.subheader(
            "📈 Risk Probability Over Time"
        )


        chart_data = (
            df_history[
                [
                    "Assessment",
                    "Risk Probability"
                ]
            ]
            .set_index(
                "Assessment"
            )
        )


        st.line_chart(
            chart_data
        )


        st.markdown("---")


        st.subheader(
            "📋 Assessment History"
        )


        display_df = df_history.copy()


        display_df[
            "Risk Probability"
        ] = (
            display_df[
                "Risk Probability"
            ].round(2)
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


        st.markdown("---")


        if st.button(
            "🗑️ Clear Assessment History"
        ):

            st.session_state.assessment_history = []

            st.session_state.latest_explanation = None

            st.session_state.latest_input = None

            st.session_state.latest_prediction = None

            st.session_state.latest_probability = None

            st.rerun()


# =========================================================
# EXPLAINABLE AI
# =========================================================

elif page == "Explainable AI":

    st.title(
        "🔍 Explainable AI"
    )

    st.subheader(
        "Why did the model make this prediction?"
    )


    st.write(
        "CardioSense uses the learned coefficients of "
        "the Logistic Regression model to explain "
        "model behavior."
    )


    st.markdown("---")


    st.subheader(
        "🤖 Logistic Regression Explanation"
    )


    st.info(
        "Positive coefficients move the model toward "
        "class 1, while negative coefficients move "
        "the model toward class 0. The individual "
        "contribution depends on the transformed "
        "feature value and its coefficient."
    )


    # -----------------------------------------------------
    # GLOBAL IMPORTANCE
    # -----------------------------------------------------

    if model is not None and hasattr(
        model,
        "coef_"
    ):

        coefficients = (
            model.coef_[0]
        )


        global_df = pd.DataFrame({

            "Feature":
                expected_columns,

            "Coefficient":
                coefficients,

            "Absolute Importance":
                abs(coefficients)

        })


        global_df = (
            global_df.sort_values(
                "Absolute Importance",
                ascending=False
            )
        )


        st.subheader(
            "📊 Global Feature Importance"
        )


        chart_df = (
            global_df[
                [
                    "Feature",
                    "Coefficient"
                ]
            ]
            .set_index(
                "Feature"
            )
        )


        st.bar_chart(
            chart_df
        )


        st.markdown("---")


        coefficient_display = (
            global_df.copy()
        )


        coefficient_display[
            "Coefficient"
        ] = (
            coefficient_display[
                "Coefficient"
            ].round(4)
        )


        coefficient_display[
            "Absolute Importance"
        ] = (
            coefficient_display[
                "Absolute Importance"
            ].round(4)
        )


        st.dataframe(
            coefficient_display,
            use_container_width=True,
            hide_index=True
        )


    st.markdown("---")


    # -----------------------------------------------------
    # INDIVIDUAL EXPLANATION
    # -----------------------------------------------------

    st.subheader(
        "🧑 Individual Prediction Explanation"
    )


    explanation = (
        st.session_state.latest_explanation
    )


    if explanation is None:

        st.info(
            "Run a Risk Assessment first."
        )


    else:

        positive = explanation[
            explanation[
                "Contribution"
            ] > 0
        ].head(5)


        negative = explanation[
            explanation[
                "Contribution"
            ] < 0
        ].head(5)


        col1, col2 = st.columns(2)


        with col1:

            st.markdown(
                "### 🔴 Toward Class 1"
            )


            for _, row in positive.iterrows():

                st.write(
                    f"**{row['Feature']}** "
                    f"{row['Contribution']:+.4f}"
                )


        with col2:

            st.markdown(
                "### 🔵 Toward Class 0"
            )


            for _, row in negative.iterrows():

                st.write(
                    f"**{row['Feature']}** "
                    f"{row['Contribution']:+.4f}"
                )


        st.markdown("---")


        st.subheader(
            "📈 Feature Contributions"
        )


        contribution_chart = (
            explanation[
                [
                    "Feature",
                    "Contribution"
                ]
            ]
            .set_index(
                "Feature"
            )
        )


        st.bar_chart(
            contribution_chart
        )


        st.markdown("---")


        st.subheader(
            "📋 Detailed Explanation"
        )


        detailed_df = explanation[
            [
                "Feature",
                "Value",
                "Coefficient",
                "Contribution",
                "Direction"
            ]
        ].copy()


        detailed_df = detailed_df.round(4)


        st.dataframe(
            detailed_df,
            use_container_width=True,
            hide_index=True
        )


        st.info(
            "These explanations describe model behavior. "
            "They should not be interpreted as medical "
            "causal relationships."
        )


# =========================================================
# WHAT-IF SIMULATOR
# =========================================================

elif page == "What-If Simulator":

    st.title(
        "🧪 What-If Simulator"
    )

    st.subheader(
        "Counterfactual Model Simulation"
    )

    st.write(
        "Modify selected inputs and compare the model's "
        "original prediction with a simulated prediction."
    )


    latest_input = (
        st.session_state.latest_input
    )


    latest_probability = (
        st.session_state.latest_probability
    )


    latest_prediction = (
        st.session_state.latest_prediction
    )


    if latest_input is None:

        st.warning(
            "Please run a Risk Assessment first. "
            "The simulator uses your latest assessment "
            "as the starting point."
        )


    else:

        st.success(
            "Latest assessment loaded. "
            "You can now modify the inputs below."
        )


        st.markdown("---")


        st.subheader(
            "Current Assessment"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Current Probability",
                f"{latest_probability:.2f}%"
            )


        with col2:

            st.metric(
                "Current Prediction",
                "Class 1"
                if latest_prediction == 1
                else "Class 0"
            )


        with col3:

            st.metric(
                "Current BP",
                latest_input["RestingBP"]
            )


        with col4:

            st.metric(
                "Current Cholesterol",
                latest_input["Cholesterol"]
            )


        st.markdown("---")


        st.subheader(
            "🔬 Modify Patient Inputs"
        )


        st.caption(
            "The simulation changes the model inputs "
            "and recalculates the prediction. It does "
            "not establish causal medical effects."
        )


        col1, col2 = st.columns(2)


        with col1:

            simulated_bp = st.number_input(
                "Simulated Resting Blood Pressure",
                min_value=50,
                max_value=250,
                value=int(
                    latest_input["RestingBP"]
                ),
                step=1
            )


        with col2:

            simulated_cholesterol = st.number_input(
                "Simulated Cholesterol",
                min_value=50,
                max_value=700,
                value=int(
                    latest_input["Cholesterol"]
                ),
                step=1
            )


        col1, col2 = st.columns(2)


        with col1:

            simulated_max_hr = st.number_input(
                "Simulated Maximum Heart Rate",
                min_value=50,
                max_value=250,
                value=int(
                    latest_input["MaxHR"]
                ),
                step=1
            )


        with col2:

            simulated_oldpeak = st.number_input(
                "Simulated Oldpeak",
                min_value=-5.0,
                max_value=10.0,
                value=float(
                    latest_input["Oldpeak"]
                ),
                step=0.1
            )


        simulate = st.button(
            "🧪 Run What-If Simulation",
            use_container_width=True
        )


        if simulate:

            try:

                simulated_data = prepare_input(

                    latest_input["Age"],

                    simulated_bp,

                    simulated_cholesterol,

                    latest_input["FastingBS"],

                    simulated_max_hr,

                    simulated_oldpeak,

                    latest_input["Sex"],

                    latest_input["ChestPainType"],

                    latest_input["RestingECG"],

                    latest_input["ExerciseAngina"],

                    latest_input["ST_Slope"]

                )


                simulated_prediction = (
                    model.predict(
                        simulated_data
                    )[0]
                )


                if hasattr(
                    model,
                    "predict_proba"
                ):

                    simulated_probability = (
                        model.predict_proba(
                            simulated_data
                        )[0][1]
                        * 100
                    )

                else:

                    simulated_probability = (
                        float(
                            simulated_prediction
                        )
                        * 100
                    )


                probability_change = (
                    simulated_probability
                    - latest_probability
                )


                st.markdown("---")


                st.subheader(
                    "📊 Simulation Result"
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Original Probability",
                        f"{latest_probability:.2f}%"
                    )


                with col2:

                    st.metric(
                        "Simulated Probability",
                        f"{simulated_probability:.2f}%",
                        delta=f"{probability_change:+.2f}%"
                    )


                with col3:

                    st.metric(
                        "Simulated Prediction",
                        "Class 1"
                        if simulated_prediction == 1
                        else "Class 0"
                    )


                st.markdown("---")


                if (
                    simulated_prediction
                    != latest_prediction
                ):

                    st.warning(
                        "The model prediction changed "
                        "between the original and simulated "
                        "input."
                    )

                else:

                    st.info(
                        "The model prediction remained "
                        "the same after the simulation."
                    )


                # -----------------------------------------
                # CHANGED INPUTS
                # -----------------------------------------

                st.subheader(
                    "🔄 Changed Inputs"
                )


                comparison = pd.DataFrame({

                    "Feature": [
                        "Resting BP",
                        "Cholesterol",
                        "Maximum Heart Rate",
                        "Oldpeak"
                    ],

                    "Original": [
                        latest_input["RestingBP"],
                        latest_input["Cholesterol"],
                        latest_input["MaxHR"],
                        latest_input["Oldpeak"]
                    ],

                    "Simulated": [
                        simulated_bp,
                        simulated_cholesterol,
                        simulated_max_hr,
                        simulated_oldpeak
                    ]

                })


                comparison["Change"] = (
                    comparison["Simulated"]
                    - comparison["Original"]
                )


                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )


                st.markdown("---")


                # -----------------------------------------
                # PROBABILITY COMPARISON
                # -----------------------------------------

                st.subheader(
                    "📈 Probability Comparison"
                )


                comparison_chart = pd.DataFrame({

                    "Probability": [
                        latest_probability,
                        simulated_probability
                    ]

                }, index=[
                    "Original",
                    "Simulated"
                ])


                st.bar_chart(
                    comparison_chart
                )


                st.info(
                    "This is a counterfactual model simulation. "
                    "The change in model output should not be "
                    "interpreted as proof of a medical outcome "
                    "from changing a single variable."
                )


            except Exception as e:

                st.error(
                    f"Simulation could not be completed: {e}"
                )


# =========================================================
# MODEL MONITORING
# =========================================================

elif page == "Model Monitoring":

    st.title(
        "🤖 Model Monitoring"
    )

    st.subheader(
        "Prediction Activity & Input Monitoring"
    )

    st.write(
        "Monitor prediction activity, probability "
        "distribution and characteristics of inputs "
        "processed during the current dashboard session."
    )


    history = (
        st.session_state.assessment_history
    )


    if len(history) == 0:

        st.info(
            "No monitoring data available yet. "
            "Run some Risk Assessments first."
        )


    else:

        monitoring_df = pd.DataFrame(
            history
        )


        # =================================================
        # MAIN MONITORING METRICS
        # =================================================

        total_predictions = (
            len(monitoring_df)
        )


        higher_risk_count = (
            monitoring_df[
                monitoring_df[
                    "Prediction"
                ] == "Higher Risk"
            ].shape[0]
        )


        lower_risk_count = (
            monitoring_df[
                monitoring_df[
                    "Prediction"
                ] == "Lower Risk"
            ].shape[0]
        )


        average_probability = (
            monitoring_df[
                "Risk Probability"
            ].mean()
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Total Predictions",
                total_predictions
            )


        with col2:

            st.metric(
                "Higher Risk",
                higher_risk_count
            )


        with col3:

            st.metric(
                "Lower Risk",
                lower_risk_count
            )


        with col4:

            st.metric(
                "Average Probability",
                f"{average_probability:.2f}%"
            )


        st.markdown("---")


        # =================================================
        # PREDICTION DISTRIBUTION
        # =================================================

        st.subheader(
            "📊 Prediction Distribution"
        )


        prediction_counts = (
            monitoring_df[
                "Prediction"
            ]
            .value_counts()
        )


        st.bar_chart(
            prediction_counts
        )


        st.markdown("---")


        # =================================================
        # RISK PROBABILITY TREND
        # =================================================

        st.subheader(
            "📈 Risk Probability Trend"
        )


        risk_chart = (
            monitoring_df[
                [
                    "Assessment",
                    "Risk Probability"
                ]
            ]
            .set_index(
                "Assessment"
            )
        )


        st.line_chart(
            risk_chart
        )


        st.markdown("---")


        # =================================================
        # INPUT MONITORING
        # =================================================

        st.subheader(
            "📋 Input Monitoring"
        )


        input_columns = [
            "Age",
            "Resting BP",
            "Cholesterol",
            "Max HR",
            "Oldpeak"
        ]


        input_stats = monitoring_df[
            input_columns
        ].describe().T


        input_stats = input_stats[
            [
                "count",
                "mean",
                "std",
                "min",
                "max"
            ]
        ]


        input_stats.columns = [
            "Count",
            "Mean",
            "Std Dev",
            "Minimum",
            "Maximum"
        ]


        st.dataframe(
            input_stats.round(2),
            use_container_width=True
        )


        st.markdown("---")


        # =================================================
        # CURRENT SESSION TABLE
        # =================================================

        st.subheader(
            "🗂️ Prediction Log"
        )


        display_monitoring = (
            monitoring_df.copy()
        )


        display_monitoring[
            "Risk Probability"
        ] = (
            display_monitoring[
                "Risk Probability"
            ].round(2)
        )


        st.dataframe(
            display_monitoring,
            use_container_width=True,
            hide_index=True
        )


        st.markdown("---")


        # =================================================
        # EXPORT MONITORING DATA
        # =================================================

        st.subheader(
            "📥 Export Monitoring Data"
        )


        csv_data = (
            monitoring_df
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(
            label="Download Monitoring CSV",
            data=csv_data,
            file_name="cardiosense_monitoring.csv",
            mime="text/csv",
            use_container_width=True
        )


        st.markdown("---")


        st.caption(
            "Current monitoring statistics are based on "
            "assessments performed during the active "
            "Streamlit session. They are not a substitute "
            "for formal model validation or clinical monitoring."
        )