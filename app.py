import streamlit as st
import pickle
from openai import OpenAI
import pandas as pd

# ---------------- API ----------------
client = OpenAI(api_key="your_API_Key")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Water Quality AI",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Top navigation styling */
.stRadio > div {
    display: flex;
    justify-content: center;
    gap: 40px;
    padding: 10px;
    border-bottom: 1px solid #333;
}

/* Titles */
.title {
    font-size: 32px;
    font-weight: 600;
    margin-top: 20px;
}

.subtitle {
    color: #aaa;
    margin-bottom: 20px;
}

/* Section */
.section {
    margin-top: 30px;
}

/* Cards */
.card {
    padding: 20px;
    border-radius: 10px;
    background-color: #1c1f26;
}

/* Button */
.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 8px;
    font-size: 16px;
}

/* Alerts */
.success {
    color: #00ff99;
}
.error {
    color: #ff4d4d;
}
</style>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------
page = st.radio("", ["Home", "Prediction", "About"], horizontal=True)

# ---------------- LOAD MODEL ----------------
with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------- FUNCTIONS ----------------
def predict_potability(data):
    scaled = scaler.transform([data])
    return classifier.predict(scaled)[0]


def generate_explanation(prediction, input_data):
    try:
        status = "SAFE" if prediction == 1 else "NOT SAFE"

        prompt = f"""
        Water Quality Analysis:
        Prediction: {status}
        Parameters: {input_data}

        Explain clearly:
        - Why the water is safe or unsafe
        - Key factors
        - Suggestions
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except:
        if prediction == 1:
            return """
Water is safe. Chemical parameters are within acceptable range.
Maintain regular monitoring.
"""
        else:
            return """
Water is unsafe due to imbalance in chemical properties.
Use filtration, boiling, and purification methods.
"""

# ---------------- HOME ----------------
if page == "Home":
    st.markdown('<div class="title">Water Quality Monitoring System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered analysis platform for water safety</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("System Status", "Active")
    col2.metric("Model Type", "ML + AI")
    col3.metric("Goal", "Clean Water")

    st.markdown('<div class="section"></div>', unsafe_allow_html=True)

    st.write("""
This platform analyzes water quality using machine learning and generates intelligent explanations.

Features:
- Real-time prediction
- Dashboard visualization
- AI-based explanation
- Risk alerts
""")

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.markdown('<div class="title">Water Quality Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter parameters to analyze water safety</div>', unsafe_allow_html=True)

    # -------- INPUTS --------
    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input("pH", value=7.0)
        hardness = st.number_input("Hardness", value=150.0)
        solids = st.number_input("Solids", value=10000.0)

    with col2:
        chloramines = st.number_input("Chloramines", value=7.0)
        sulfate = st.number_input("Sulfate", value=300.0)
        conductivity = st.number_input("Conductivity", value=400.0)

    with col3:
        organic = st.number_input("Organic Carbon", value=10.0)
        trihalo = st.number_input("Trihalomethanes", value=80.0)
        turbidity = st.number_input("Turbidity", value=4.0)

    if st.button("Analyze Water Quality"):

        input_data = [ph, hardness, solids, chloramines, sulfate,
                      conductivity, organic, trihalo, turbidity]

        prediction = predict_potability(input_data)

        st.markdown("---")

        # -------- METRICS --------
        m1, m2, m3 = st.columns(3)
        m1.metric("pH", ph)
        m2.metric("Solids", solids)
        m3.metric("Turbidity", turbidity)

        # -------- STATUS --------
        if prediction == 1:
            st.success("Water is SAFE")
            st.markdown("Risk Level: LOW")
        else:
            st.error("Water is NOT SAFE")
            st.markdown("Risk Level: HIGH")

        # -------- ALERT --------
        if prediction == 0:
            st.error("Alert: Unsafe water detected. Action required.")
        else:
            st.success("Water parameters are within safe limits.")

        # -------- CHART --------
        df = pd.DataFrame({
            "Parameter": ["pH","Hardness","Solids","Chloramines","Sulfate",
                          "Conductivity","Organic","Trihalo","Turbidity"],
            "Value": input_data
        })

        st.markdown("### Parameter Visualization")
        st.bar_chart(df.set_index("Parameter"))

        # -------- SUGGESTIONS --------
        st.markdown("### Recommendations")

        if prediction == 0:
            st.warning("""
- Use filtration systems
- Boil water before consumption
- Apply purification methods
- Monitor parameters regularly
""")
        else:
            st.info("""
- Maintain current water quality
- Continue periodic testing
""")

        # -------- AI EXPLANATION --------
        with st.spinner("Generating AI explanation..."):
            explanation = generate_explanation(prediction, input_data)

        st.markdown("### AI Explanation")
        st.write(explanation)

# ---------------- ABOUT ----------------
elif page == "About":
    st.markdown('<div class="title">About System</div>', unsafe_allow_html=True)

    st.write("""
This system predicts water quality using machine learning and AI.

Features:
- Dashboard analytics
- Risk detection
- Visualization
- AI explanation system

Technologies:
- Python
- Scikit-learn
- Streamlit
- OpenAI API
""")