import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="Nairobi House Price Predictor", layout="centered")

st.title("🏠 Nairobi House Price Predictor")

mode = st.sidebar.radio("Mode", ["Pricing App", "Insights Dashboard"])
st.caption("Quick MVP pricing interface (loads best_model.pkl and predicts price in KES).")

DATA_PATH = Path("data/clean_listings.csv")
MODEL_PATH = Path("models/best_model.pkl")

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Missing file: {DATA_PATH}. Run `python src/clean_features.py` first.")
        st.stop()
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Missing model: {MODEL_PATH}. Run `python src/train_compare_models.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# Build dropdown options from your dataset
locations = sorted([x for x in df["location"].dropna().unique()])
property_types = sorted([x for x in df["property_type"].dropna().unique()])


if mode == "Pricing App":
    st.subheader("Enter property details")

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("Location", locations)
        property_type = st.selectbox("Property type", property_types)
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=2, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2, step=1)

    with col2:
        size_sqm = st.number_input("Size (sqm)", min_value=1.0, max_value=2000.0, value=95.0, step=1.0)
        amenity_score = st.number_input("Amenity score (count)", min_value=0, max_value=100, value=3, step=1)

    # Create input row matching training features
    input_df = pd.DataFrame([{
        "location": location,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "size_sqm": size_sqm,
        "amenity_score": amenity_score
    }])

    st.divider()

    if st.button("Predict Price (KES)"):
        pred = model.predict(input_df)[0]
        st.success(f"Estimated Price: KES {pred:,.0f}")

        st.write("Input used:")
        st.dataframe(input_df, width="stretch")

    st.caption("Note: This is a demo-scale dataset. Predictions may not generalize until dataset size increases.")
    
else:
    st.subheader("📊 Market Insights Dashboard")

    df_dash = df.copy()
    df_dash = df_dash.dropna(subset=["price_kes", "location"])

    st.markdown("### Median price by location")
    loc_median = (
        df_dash.groupby("location")["price_kes"]
        .median()
        .sort_values(ascending=False)
        .head(15)
    )
    st.bar_chart(loc_median)

    if "price_per_sqm" in df_dash.columns:
        st.markdown("### Median price per sqm by location")
        ppm = (
            df_dash.dropna(subset=["price_per_sqm"])
            .groupby("location")["price_per_sqm"]
            .median()
            .sort_values(ascending=False)
            .head(15)
        )
        st.bar_chart(ppm)

    if "amenity_score" in df_dash.columns:
        st.markdown("### Amenity score vs price (median)")
        amen = (
            df_dash.groupby("amenity_score")["price_kes"]
            .median()
            .sort_index()
        )
        st.line_chart(amen)