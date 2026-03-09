import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Load model
model = joblib.load("fraud_detection_random_forest.pkl")

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown(
"<h4 style='color:#9BA4B5;'>AI-powered transaction risk monitoring system</h4>",
unsafe_allow_html=True
)

st.divider()

uploaded_file = st.file_uploader("Upload Transaction Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head(20), use_container_width=True)

    if st.button("Run Fraud Detection"):

        X = data.drop("Class", axis=1, errors="ignore")

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:,1]

        data["Prediction"] = predictions
        data["Fraud Probability"] = probabilities

        fraud_count = sum(predictions)
        normal_count = len(predictions) - fraud_count

        fraud_rate = (fraud_count / len(data)) * 100
        fraud_percent = (fraud_count / (fraud_count + normal_count)) * 100

        st.divider()

        st.subheader("Detection Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Transactions", len(data))
        col2.metric("Fraud Detected", fraud_count)
        col3.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        st.divider()

        # Pie Chart
        st.subheader("Transaction Distribution")

        fig, ax = plt.subplots(figsize=(4,4), facecolor="#0E1117")

        labels = ["Normal", "Fraud"]
        values = [normal_count, fraud_count]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,  # makes donut chart
            marker=dict(colors=["#00F5D4", "#FF4D6D"]),
            textinfo="percent+label"
        )])
 
        fig.update_layout(
            height=350,
            width=350,
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(color="white", size=14),
            margin=dict(t=20, b=20, l=20, r=20)
        )

        # col1, col2, col3 = st.columns([1,2,1])
        # with col2:
        #      st.plotly_chart(fig)

        # st.divider()

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fraud_percent,
            title={'text': "Fraud Risk Level"},
    
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF4D6D"},
                'steps': [
                     {'range': [0, 20], 'color': "#00F5D4"},
                     {'range': [20, 50], 'color': "#FFA500"},
                     {'range': [50, 100], 'color': "#FF4D6D"}
                ],
            }
        ))

        fig_gauge.update_layout(
             height=300,
             width=400,
             paper_bgcolor="#0E1117",
             font=dict(color="white")
        )

        # st.plotly_chart(fig_gauge)
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig)

        with col2:
            st.plotly_chart(fig_gauge)

        st.subheader("Top Fraud Influencing Features")

        importances = model.feature_importances_
        features = X.columns

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        })

        importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Turbo"
        )

        fig.update_layout(
             height=400,
             paper_bgcolor="#0E1117",
             plot_bgcolor="#0E1117",
             font=dict(color="white"),
             margin=dict(l=40, r=40, t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Fraud Transactions")

        fraud_df = data[data["Prediction"] == 1]

        st.dataframe(fraud_df, use_container_width=True)

        st.subheader("Transactions Risk Analysis")

        st.dataframe(data.head(500), use_container_width=True)