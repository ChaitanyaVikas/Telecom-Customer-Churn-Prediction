import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. TITLE & DESCRIPTION
st.title("📱 Telecom Customer Churn Prediction")
st.markdown("This app predicts whether a customer is likely to leave based on their contract details.")

# 2. LOAD DATA (Cache it so it doesn't reload every time)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

df = load_data()

# 3. SIDEBAR FOR USER INPUT
st.sidebar.header("User Input Features")

def user_input_features():
    tenure = st.sidebar.slider('Tenure (Months)', 1, 72, 12)
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18.0, 118.0, 70.0)
    total_charges = st.sidebar.number_input('Total Charges ($)', 0.0, 10000.0, 1500.0)
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    
    data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. PREPARE DATA FOR MODELING
# We combine user input with the original data to ensure the encoding matches perfectly
combined_df = pd.concat([input_df, df.drop(columns=['Churn'])], axis=0)

# One-Hot Encoding
combined_df_encoded = pd.get_dummies(combined_df)

# Separate input line from the rest of the data
final_input = combined_df_encoded[:1]
X = combined_df_encoded[1:]
y = df['Churn']

# 5. TRAIN MODEL (In real apps, we would load a saved model, but training here works for small data)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 6. PREDICTION
prediction = model.predict(final_input)
prediction_proba = model.predict_proba(final_input)

# ... (keep all your existing code) ...

# 7. SHOW RESULTS (Existing code)
st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ High Risk: Customer is likely to CHURN.")
else:
    st.success(f"✅ Low Risk: Customer is likely to STAY.")

# --- NEW DYNAMIC GRAPH CODE STARTS HERE ---
st.subheader("Confidence Score")
# Create a simple dataframe for the probabilities
prob_df = pd.DataFrame({
    "Outcome": ["Stay (Low Risk)", "Leave (High Risk)"],
    "Probability": prediction_proba[0]
})

# Create a bar chart that updates instantly
st.bar_chart(prob_df.set_index("Outcome"))
# --- NEW CODE ENDS ---

# 8. SHOW VISUALIZATION (Keep your existing static graph below as "Model Logic")
st.subheader("Model Logic (Global Feature Importance)")
# ... (rest of your existing graph code)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_5 = feature_importances.sort_values(ascending=False).head(5)

fig, ax = plt.subplots()
sns.barplot(x=top_5.values, y=top_5.index, palette='viridis', ax=ax)
ax.set_title("Top Factors Influencing this Prediction")
st.pyplot(fig)