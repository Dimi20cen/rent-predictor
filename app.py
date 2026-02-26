import streamlit as st
import pandas as pd
import numpy as np
import pickle

from src.ml_pipeline import (
    ArtifactValidationError,
    ContractValidationError,
    get_model_version,
    predict_rent,
    resolve_feature_names,
    transform_features_for_model,
    validate_model_artifacts,
)

# Set Page Config
st.set_page_config(page_title="Swiss Rent Predictor", layout="centered")

# --- 1. Load Resources ---
@st.cache_resource
def load_resources():
    validate_model_artifacts("models/model_manifest.json")

    # Load Model
    with open('models/xgb_rent_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load Zip Encoder
    with open('models/zip_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    # Load feature columns used during training
    feature_columns = resolve_feature_names(model, "models/feature_columns.json")
        
    # Load Cleaned Data (for dropdown options only)
    # We only need unique Zips, Cantons, and Tax info
    df = pd.read_pickle('data/processed/02_featured_data.pkl')
    
    model_version = get_model_version("models/model_manifest.json")
    return model, encoder, feature_columns, df, model_version

try:
    model, encoder_zip, model_feature_names, df_ref, model_version = load_resources()
except ArtifactValidationError as exc:
    st.error(f"Artifact integrity error: {exc}")
    st.stop()
except FileNotFoundError as exc:
    st.error(f"Missing required artifact: {exc}")
    st.stop()

# --- 2. Helper Functions (Recreating NB 02 Logic) ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

HUBS = {
    'Zurich_HB': (47.378177, 8.540192),
    'Geneva_Cornavin': (46.210226, 6.142456),
    'Basel_SBB': (47.547412, 7.589556),
    'Bern_HB': (46.948833, 7.439122),
    'Lausanne_Gare': (46.516777, 6.629095)
}

# --- 3. UI Layout ---
st.title("🇨🇭 Swiss Rental Price Predictor")
st.markdown("Estimate the fair market value of an apartment in Switzerland using Machine Learning (XGBoost).")
st.caption("Demo includes artifact integrity checks and displays the active model version.")

SCENARIOS = {
    "Custom": {},
    "Zurich Starter Flat": {
        "area": 58,
        "rooms": 2.0,
        "floor": 3,
        "canton": "ZH",
        "subtype": "FLAT",
        "has_lake": False,
        "is_new": False,
        "is_quiet": True,
    },
    "Geneva Family Apartment": {
        "area": 110,
        "rooms": 4.5,
        "floor": 4,
        "canton": "GE",
        "subtype": "FLAT",
        "has_lake": False,
        "is_new": True,
        "is_quiet": True,
    },
    "Lakeside Premium": {
        "area": 145,
        "rooms": 5.0,
        "floor": 6,
        "canton": "ZG",
        "subtype": "PENTHOUSE",
        "has_lake": True,
        "is_new": True,
        "is_quiet": True,
    },
}

with st.sidebar:
    st.subheader("Model Metadata")
    st.caption(f"Version: `{model_version}`")
    st.success("Artifacts verified against manifest")
    selected_scenario_name = st.selectbox("Sample Scenario", list(SCENARIOS.keys()), index=0)
    st.caption("Pick a scenario for quick demo inputs, then fine-tune fields.")

scenario = SCENARIOS[selected_scenario_name]
default_area = int(scenario.get("area", 65))
default_rooms = float(scenario.get("rooms", 2.5))
default_floor = int(scenario.get("floor", 2))
default_canton = str(scenario.get("canton", "ZH"))
default_subtype = str(scenario.get("subtype", "FLAT"))
default_has_lake = bool(scenario.get("has_lake", False))
default_is_new = bool(scenario.get("is_new", False))
default_is_quiet = bool(scenario.get("is_quiet", False))

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Living Area (m²)", min_value=10, max_value=500, value=default_area)
    rooms = st.number_input("Rooms", min_value=1.0, max_value=10.0, step=0.5, value=default_rooms)
    floor = st.number_input("Floor", min_value=-1, max_value=20, value=default_floor)

with col2:
    # Canton Selection
    cantons = sorted(df_ref['Canton'].unique())
    canton_idx = cantons.index(default_canton) if default_canton in cantons else (cantons.index('ZH') if 'ZH' in cantons else 0)
    selected_canton = st.selectbox("Canton", cantons, index=canton_idx)
    
    # Filter Zips by Canton
    canton_zips = sorted(df_ref[df_ref['Canton'] == selected_canton]['Zip'].unique())
    selected_zip = st.selectbox("Zip Code", canton_zips, index=0)
    
    # SubType
    subtypes = sorted(df_ref['SubType'].unique())
    subtype_idx = subtypes.index(default_subtype) if default_subtype in subtypes else (subtypes.index('FLAT') if 'FLAT' in subtypes else 0)
    selected_type = st.selectbox("Property Type", subtypes, index=subtype_idx)

st.markdown("### ✨ Extras")
c1, c2, c3 = st.columns(3)
with c1: has_lake = st.checkbox("Lake View", value=default_has_lake)
with c2: is_new = st.checkbox("New Building", value=default_is_new)
with c3: is_quiet = st.checkbox("Quiet Area", value=default_is_quiet)

# --- 4. Prediction Logic ---
if st.button("Predict Rent", type="primary"):
    
    # A. Get Reference Data for the selected Zip
    # We need Lat, Lon, and Tax Rate for the selected Zip
    # We take the median Lat/Lon/Tax of existing listings in that Zip
    ref_row = df_ref[df_ref['Zip'] == selected_zip].iloc[0]
    lat, lon = ref_row['Lat'], ref_row['Lon']
    tax_rate = ref_row['tax_rate']
    
    # B. Calculate Distances
    dists = {}
    for hub, coords in HUBS.items():
        dists[f'dist_to_{hub}'] = haversine_distance(lat, lon, coords[0], coords[1])
    min_dist = min(dists.values())
    
    # C. Construct Input DataFrame
    # Must match the columns XGBoost expects (except One-Hot columns which we handle manually)
    input_data = {
        'Rooms': [rooms],
        'Area_m2': [area],
        'Floor': [floor],
        'Canton': [selected_canton],
        'SubType': [selected_type],
        'Lat': [lat],
        'Lon': [lon],
        'dist_to_Zurich_HB': [dists['dist_to_Zurich_HB']],
        'dist_to_Geneva_Cornavin': [dists['dist_to_Geneva_Cornavin']],
        'dist_to_Basel_SBB': [dists['dist_to_Basel_SBB']],
        'dist_to_Bern_HB': [dists['dist_to_Bern_HB']],
        'dist_to_Lausanne_Gare': [dists['dist_to_Lausanne_Gare']],
        'dist_to_closest_hub': [min_dist],
        'tax_rate': [tax_rate],
        'is_rent_estimated': [0], # User input is "real"
        'year_built_is_missing': [1], # Assume unknown
        'is_renovated': [1 if is_new else 0],
        'Balcony': [1 if has_lake else 0], # Proxy
        'Elevator': [1],
        'Parking': [0],
        'View': [1 if has_lake else 0],
        'Fireplace': [0],
        'Child_Friendly': [0],
        'CableTV': [0],
        'New_Building': [1 if is_new else 0],
        'Minergie': [0],
        'Wheelchair': [0],
        'has_lake_view': [1 if has_lake else 0],
        'is_attic': [0],
        'is_quiet': [1 if is_quiet else 0],
        'is_sunny': [0],
        'Zip': [selected_zip] # For Encoder
    }

    try:
        X_input = pd.DataFrame(input_data)
        X_final = transform_features_for_model(X_input, encoder_zip, model_feature_names)
        prediction = predict_rent(model, X_final)[0]
    except ContractValidationError as exc:
        st.error(f"Input contract error: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    else:
        # F. Display
        st.success("Prediction complete")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Estimated Monthly Rent", f"CHF {int(prediction):,}")
        with m2:
            st.metric("Estimated Annual Rent", f"CHF {int(prediction * 12):,}")
        st.info(
            f"📍 Location: {selected_zip} (Tax Index: {tax_rate}) | "
            f"📏 Distance to Hub: {int(min_dist)} km | "
            f"🧩 Model Version: {model_version}"
        )
