import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0B0F14;
}
.block-container {
    padding-top: 1rem;
}

/* Gradient Title */
h1 {
    background: linear-gradient(90deg, #4F8EF7, #00C2A8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Headings */
h2, h3 {
    color: #E5E7EB;
}

/* Divider */
hr {
    border: 1px solid #2A2F3A;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #141A21, #1B2430);
    border: 1px solid #2A2F3A;
    border-radius: 8px;
    padding: 10px;
}

/* Custom Explanation Cards */
.card {
    background-color: #141A21;
    border-left: 4px solid #4F8EF7;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 10px;
}
.card-title {
    font-weight: 600;
    color: #E5E7EB;
    margin-bottom: 5px;
}
.card-text {
    color: #AAB2C0;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


model = joblib.load("fraud_detection_random_forest.pkl")

# SIDEBAR

st.sidebar.title("Credit Card Fraud Monitoring System")

page = st.sidebar.radio(
    "Navigation",
    ["Monitoring Dashboard", "Transaction Evaluation", "Data Explorer", "Analytical Insights"]
)

uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type=["csv"])

data = None
original_data = None

# LOAD DATA

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    data = data.head(1000)  

    original_data = data.copy()

    X = data.drop("Class", axis=1, errors="ignore")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    st.write("NEW VERSION LOADED")

    data["Prediction"] = predictions
    data["Fraud Probability"] = probabilities

    def risk_label(p):
        if p < 0.3:
            return "Low"
        elif p < 0.7:
            return "Medium"
        else:
            return "High"

    data["Risk Level"] = data["Fraud Probability"].apply(risk_label)

#  DASHBOARD

if page == "Monitoring Dashboard":

    st.title("Fraud Detection Dashboard")

    if data is None:
        st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(79,142,247,0.15), rgba(0,194,168,0.10));
    border: 1px solid rgba(79,142,247,0.4);
    padding: 16px;
    border-radius: 10px;
    color: #E5E7EB;
    font-size: 14px;
    font-weight: 500;
">
Upload a dataset to begin analysis.
</div>
""", unsafe_allow_html=True)
    else:

        total = len(data)
        fraud_count = int(data["Prediction"].sum())
        fraud_rate = (fraud_count / total) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", total)
        col2.metric("Fraudulent Transactions", fraud_count)
        col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["Overview", "Analysis", "Transactions"])

      
        #  OVERVIEW
       
        with tab1:

            col1, col2 = st.columns(2)

            # DONUT CHART
            with col1:
                st.subheader("Transaction Distribution")

                fig = px.pie(
                    data,
                    names="Prediction",
                    color="Prediction",
                    color_discrete_map={0: "#00C2A8", 1: "#E05D5D"},
                    hole=0.6
                )

                fig.update_layout(
                    height=300,
                    paper_bgcolor="#0B0F14",
                    plot_bgcolor="#0B0F14",
                    font=dict(color="white")
                )

                st.plotly_chart(fig, use_container_width=True)

            # RISK BAR
            with col2:
                st.subheader("Risk Level Distribution")

                risk_counts = data["Risk Level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]

                fig = px.bar(
                    risk_counts,
                    x="Risk Level",
                    y="Count",
                    color="Risk Level",
                    color_discrete_map={
                        "Low": "#00C2A8",
                        "Medium": "#F5A623",
                        "High": "#E05D5D"
                    }
                )

                fig.update_layout(
                    height=300,
                    paper_bgcolor="#0B0F14",
                    plot_bgcolor="#0B0F14",
                    font=dict(color="white")
                )

                st.plotly_chart(fig, use_container_width=True)

            # HISTOGRAM
            st.subheader("Fraud Probability Distribution")

            fig = px.histogram(
                data,
                x="Fraud Probability",
                nbins=50,
                color_discrete_sequence=["#4F8EF7"]
            )

            fig.update_layout(
                height=300,
                paper_bgcolor="#0B0F14",
                plot_bgcolor="#0B0F14",
                font=dict(color="white")
            )

            st.plotly_chart(fig, use_container_width=True)

            # TREND
            st.subheader("Transaction Risk Trend")

            fig = px.line(
                data.head(1000),
                y="Fraud Probability"
            )

            fig.update_layout(
                height=300,
                paper_bgcolor="#0B0F14",
                plot_bgcolor="#0B0F14",
                font=dict(color="white")
            )

            st.plotly_chart(fig, use_container_width=True)

        
        # ANALYSIS
        
        with tab2:

            st.subheader("Top Feature Influence")

            X = data.drop(["Prediction", "Fraud Probability", "Risk Level"], axis=1, errors="ignore")

            importances = model.feature_importances_
            features = X.columns[:len(importances)]

            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Teal"
            )

            fig.update_layout(
                height=350,
                paper_bgcolor="#0B0F14",
                plot_bgcolor="#0B0F14",
                font=dict(color="white")
            )

            st.plotly_chart(fig, use_container_width=True)

            # BOX PLOT
            if "Amount" in data.columns:
                st.subheader("Fraud vs Normal Comparison")

                fig = px.box(
                    data,
                    x="Prediction",
                    y="Amount",
                    color="Prediction",
                    color_discrete_map={0: "#00C2A8", 1: "#E05D5D"}
                )

                fig.update_layout(
                    height=350,
                    paper_bgcolor="#0B0F14",
                    plot_bgcolor="#0B0F14",
                    font=dict(color="white")
                )

                st.plotly_chart(fig, use_container_width=True)

            
        #  TRANSACTIONS
        
        with tab3:

            st.subheader("High Risk Transactions")

            high_risk = data.sort_values(by="Fraud Probability", ascending=False).head(20)
            st.dataframe(high_risk, use_container_width=True)

            csv = high_risk.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Export Report",
                data=csv,
                file_name="fraud_report.csv",
                mime="text/csv"
            )

# SINGLE TRANSACTION

elif page == "Transaction Evaluation":

    st.title("Single Transaction Evaluation")

    if data is None:
        
        st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(79,142,247,0.15), rgba(0,194,168,0.10));
    border: 1px solid rgba(79,142,247,0.4);
    padding: 16px;
    border-radius: 10px;
    color: #E5E7EB;
    font-size: 14px;
    font-weight: 500;
">
Upload a dataset to enable input.
</div>
""", unsafe_allow_html=True)
    else:

        feature_cols = data.drop(
            ["Prediction", "Fraud Probability", "Risk Level", "Class"],
            axis=1,
            errors="ignore"
        ).columns

        input_data = {}
        cols = st.columns(2)

        for i, col in enumerate(feature_cols):
            min_val = float(data[col].min())
            max_val = float(data[col].max())

            with cols[i % 2]:
                input_data[col] = st.slider(
                    col,
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val + max_val) / 2
                )

        if st.button("Run Detection"):

            input_df = pd.DataFrame([input_data])

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            confidence = max(prob, 1-prob)
            anomaly = prob * 100

            st.markdown("---")

            if pred == 1:
                st.error(f"Fraudulent Transaction (Probability: {prob:.3f})")
            else:
                st.success(f"Legitimate Transaction (Probability: {prob:.3f})")

           
            # CARD-BASED EXPLANATION
            
            st.subheader("Key Factors Behind Prediction")

            importances = model.feature_importances_
            features = input_df.columns[:len(importances)]

            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(5)

            for _, row in importance_df.iterrows():
                val = input_df.iloc[0][row["Feature"]]

                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{row['Feature']}</div>
                    <div class="card-text">
                        Value: {val:.3f}<br>
                        Influence: High impact on prediction
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card">
                <div class="card-title">Final Decision</div>
                <div class="card-text">
                    Confidence Score: {  confidence:.2f} <br>
            Anomaly Score: {  anomaly:.1f}/100 <br>
            Risk Level: {" High" if prob>0.7 else " Medium" if prob>0.3 else " Low"} <br>
                    The prediction is based on combined deviations across multiple high-impact features,
                    indicating a strong likelihood of fraud.
                </div>
            </div>
            """, unsafe_allow_html=True)
           
# DATA VIEW & INSIGHTS

elif page == "Data Explorer":
    st.title("Dataset Overview")
    if data is None:
        st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(79,142,247,0.15), rgba(0,194,168,0.10));
    border: 1px solid rgba(79,142,247,0.4);
    padding: 16px;
    border-radius: 10px;
    color: #E5E7EB;
    font-size: 14px;
    font-weight: 500;
">
 Try uploading the dataset.
</div>
""", unsafe_allow_html=True)
    if data is not None:
        st.markdown("### Actual Dataset")
        st.dataframe(original_data)
        st.write("Rows:", original_data.shape[0])
        st.write("Columns:", original_data.shape[1])
        st.markdown("---")
        st.subheader("Processed Dataset")
        st.dataframe(data)

# INSIGHTS

elif page == "Analytical Insights":

    st.title("Dataset Insights")

    if data is None:
        st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(79,142,247,0.15), rgba(0,194,168,0.10));
    border: 1px solid rgba(79,142,247,0.4);
    padding: 16px;
    border-radius: 10px;
    color: #E5E7EB;
    font-size: 14px;
    font-weight: 500;
">
 upload the dataset to get the insights.
</div>
""", unsafe_allow_html=True)
    else:

        total = len(data)
        fraud_count = int(data["Prediction"].sum())
        fraud_rate = (fraud_count / total) * 100
        features = data.shape[1]

       
        #  STAT CARDS
       
        st.markdown("### Summary Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
    <div style="background:#141A21; padding:16px; border-radius:10px; text-align:center; border:1px solid #2A2F3A;">
        <div style="font-size:24px; font-weight:600; color:#4F8EF7;">
            {total:,}
        </div>
        <div style="color:#AAB2C0; font-size:13px;">
            Total Transactions
        </div>
    </div>
    """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
    <div style="background:#141A21; padding:16px; border-radius:10px; text-align:center; border:1px solid #2A2F3A;">
        <div style="font-size:24px; font-weight:600; color:#00C2A8;">
            {features}
        </div>
        <div style="color:#AAB2C0; font-size:13px;">
            Features
        </div>
    </div>
    """, unsafe_allow_html=True)

        with col3:
             st.markdown(f"""
    <div style="background:#141A21; padding:16px; border-radius:10px; text-align:center; border:1px solid #2A2F3A;">
        <div style="font-size:24px; font-weight:600; color:#E05D5D;">
            {fraud_count}
        </div>
        <div style="color:#AAB2C0; font-size:13px;">
            Fraud Cases
        </div>
    </div>
    """, unsafe_allow_html=True)

        with col4:
           st.markdown(f"""
    <div style="background:#141A21; padding:16px; border-radius:10px; text-align:center; border:1px solid #2A2F3A;">
        <div style="font-size:24px; font-weight:600; color:#F5A623;">
            {fraud_rate:.4f}%
        </div>
        <div style="color:#AAB2C0; font-size:13px;">
            Fraud Rate
        </div>
    </div>
    """, unsafe_allow_html=True)

        st.markdown("---")

      
        #  DYNAMIC OBSERVATIONS
      
        st.markdown("### Key Observations")

        if fraud_rate < 0.5:
            st.markdown("""
- The dataset is highly imbalanced with extremely low fraud occurrence.
- Most transactions are legitimate, making fraud detection a rare-event classification problem.
""")
        elif fraud_rate < 2:
            st.markdown("""
- The dataset shows moderate imbalance with noticeable fraud cases.
- Model performance must balance precision and recall carefully.
""")
        else:
            st.markdown("""
- The dataset contains relatively higher fraud activity.
- Risk detection becomes more prominent and easier to identify patterns.
""")

        st.markdown("""
- Fraud detection is performed using probability-based classification.
- High-risk transactions are identified through combined feature deviations.
- Feature importance analysis highlights the most influential variables in decision-making.
""")