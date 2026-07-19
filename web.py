import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="E-Nose VOC & TB Diagnostic System",
    page_icon="🔬",
    layout="centered"
)

# 2. AI MODEL TRAINING (Single Sensor UiO-66 Only)
@st.cache_resource
def train_voc_model():
    # Database original based on your dataset (UiO-66 values)
    raw_data = [
        ["Methyl Nicotinate", "UiO66", 0.2, 3.29993],
        ["Methyl Nicotinate", "UiO66", 1.0, 3.29967],
        ["Methyl Nicotinate", "UiO66", 5.0, 3.29834],
        ["Methyl Phenylacetate", "UiO66", 0.2, 3.29993],
        ["Methyl Phenylacetate", "UiO66", 1.0, 3.29967],
        ["Methyl Phenylacetate", "UiO66", 5.0, 3.29835],
        ["Isoprene", "UiO66", 0.2, 3.29993],
        ["Isoprene", "UiO66", 1.0, 3.29967],
        ["Isoprene", "UiO66", 5.0, 3.29834],
        ["1,2-Dimethylbenzene", "UiO66", 0.2, 3.29993],
        ["1,2-Dimethylbenzene", "UiO66", 1.0, 3.29967],
        ["1,2-Dimethylbenzene", "UiO66", 5.0, 3.29834],
        ["2-Methylpyridine", "UiO66", 0.2, 3.29993],
        ["2-Methylpyridine", "UiO66", 1.0, 3.29967],
        ["2-Methylpyridine", "UiO66", 5.0, 3.29834],
        ["Ethanol", "UiO66", 0.2, 3.29993],
        ["Ethanol", "UiO66", 1.0, 3.29967],
        ["Ethanol", "UiO66", 5.0, 3.29822]
    ]
    df_raw = pd.DataFrame(raw_data, columns=['VOC', 'MOF', 'Beta', 'Voltage'])
    
    # Simulating standard variation for robust machine learning prediction
    np.random.seed(42)
    simulated_patients = []
    for _, row in df_raw.iterrows():
        voc = row['VOC']
        base_voltage = row['Voltage']
        for i in range(100):
            v_uio = np.random.normal(base_voltage, 0.000015)
            simulated_patients.append({
                'V_UiO66': round(v_uio, 5),
                'VOC_Gas': voc
            })
    df_patients = pd.DataFrame(simulated_patients)
    
    # Single feature matrix (ZIF-8 is completely removed)
    X = df_patients[['V_UiO66']]
    y = df_patients['VOC_Gas']
    
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

ai_model = train_voc_model()
tbc_biomarkers = ['Methyl Nicotinate', 'Methyl Phenylacetate']

# 3. WEB USER INTERFACE (GUI)
st.title("🔬 VOC Analysis & TB Diagnosis - E-Nose")
st.markdown("An interactive web application utilizing a **single MOF UiO-66 sensor** to detect specific Volatile Organic Compounds (*VOC*) and determine clinical TB status.")
st.write("---")

st.subheader("📋 Input Patient Sensor Data")
col1, col2 = st.columns(2)
with col1:
    nama_pasien = st.text_input("Patient Name / ID", value="Patient #01")
with col2:
    st.text_input("Sensitivity Condition", value="Beta = 5.0 (Optimal)", disabled=True)

# Single input component for the UiO-66 voltage signal
v_uio_input = st.number_input("Sensor Voltage MOF UiO-66 (V)", min_value=3.29000, max_value=3.30000, value=3.29834, format="%.5f", step=0.00001)

st.write("")
tombol_uji = st.button("🔴 Run AI Gas Analysis", type="primary", use_container_width=True)

# 4. PREDICTION PROCESS & DISPLAY RESULTS
if tombol_uji:
    input_data = pd.DataFrame([[v_uio_input]], columns=['V_UiO66'])
    
    gas_terdeteksi = ai_model.predict(input_data)[0]
    probabilitas_semua = ai_model.predict_proba(input_data)[0]
    daftar_gas = ai_model.classes_
    
    idx_gas = np.where(daftar_gas == gas_terdeteksi)[0][0]
    persentase_gas = probabilitas_semua[idx_gas] * 100
    
    is_tbc = gas_terdeteksi in tbc_biomarkers
    status_tbc = "TB POSITIVE" if is_tbc else "TB NEGATIVE (Healthy / Other Substance)"

    st.write("---")
    st.subheader(f"📊 Analysis Results: {nama_pasien}")
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Detected VOC Type", value=gas_terdeteksi)
    with c2:
        st.metric(label="AI Confidence Level", value=f"{persentase_gas:.2f}%")
        
    st.write("### Clinical Diagnosis Conclusion:")
    if is_tbc:
        st.error(f"⚠️ **PATIENT STATUS: {status_tbc}**")
        st.markdown(f"The detected gas **{gas_terdeteksi}** is identified as an active biomarker for *Mycobacterium tuberculosis* infection in the breath sample.")
    else:
        st.success(f"✅ **PATIENT STATUS: {status_tbc}**")
        st.markdown(f"The detected gas **{gas_terdeteksi}** does not indicate any clinical signs of active Pulmonary Tuberculosis.")
        
    st.write("---")
    st.write("🔬 **Gas Spectrum Analysis Breakdown (Probabilities):**")
    df_prob = pd.DataFrame({
        'VOC Compound Type': daftar_gas,
        'Signal Similarity (%)': [f"{p*100:.2f}%" for p in probabilitas_semua]
    }).sort_values(by='Signal Similarity (%)', ascending=False)
    st.table(df_prob)
