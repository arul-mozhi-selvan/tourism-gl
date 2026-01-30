import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# -------------------------
# Configuration
# -------------------------
HF_MODEL_REPO = "arulmozhiselvan/superkart-model"
MODEL_FILENAME = "best_model_v1.joblib"

# -------------------------
# Download & load model
# -------------------------
model = None
try:
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=MODEL_FILENAME, repo_type="model", token=os.getenv("HF_TOKEN"))
    model = joblib.load(model_path)
    st.write(f"Loaded model from Hugging Face: {HF_MODEL_REPO}/{MODEL_FILENAME}")
except Exception as e:
    st.warning(f"Could not download model from Hugging Face ({HF_MODEL_REPO}).\nError: {e}\nFalling back to local file if present.")
    if os.path.exists(MODEL_FILENAME):
        model = joblib.load(MODEL_FILENAME)
        st.write(f"Loaded local model file: {MODEL_FILENAME}")
    else:
        st.error("Model not available. Please upload the model to HF or place it locally.")
        st.stop()

# -------------------------
# Streamlit UI
# -------------------------
st.title("SuperKart Sales Prediction App")
st.write("Predict product sales at different stores using trained ML model.")

# --- Customer details
Product_Weight = st.number_input("Product Weight", value=12.66)
Product_Sugar_Content = st.selectbox("Sugar Content", ["Low Sugar","No Sugar","Medium Sugar","High Sugar"])
Product_Allocated_Area = st.number_input("Allocated Area", value=0.027, step=0.001, format="%.3f")
Product_Type = st.text_input("Product Type", "Frozen Foods")
Product_MRP = st.number_input("Product MRP", value=117.08)
Store_Id = st.text_input("Store Id", "OUT004")
Store_Establishment_Year = st.number_input("Store Establishment Year", value=2009, step=1)
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.text_input("Store Type", "Supermarket Type2")


# Assemble input into DataFrame matching training columns (raw — pipeline should handle preprocessing)
input_df = pd.DataFrame([{
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_Type": Product_Type,
    "Product_MRP": Product_MRP,
    "Store_Id": Store_Id,
    "Store_Establishment_Year": Store_Establishment_Year,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type
}])

st.subheader("Input Preview")
st.dataframe(input_df.T, width=700)

# Prediction
if st.button("Predict Sales"):
    prediction = model.predict(input_df)
    st.subheader("Prediction Result")
    st.write(f"Predicted Product Store Sales Total: {prediction[0]:.2f}")
