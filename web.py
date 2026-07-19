import streamlit st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. PAGE CONFIGURATION & THEME
st.set_page_config(
    page_title="E-Nose Medical Diagnostics",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS to make the app look like a modern dashboard
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; color: #1E3A8A; }
    .reportview-container .main .block-container{ max-width: 650px; }
</style>
""", unsafe_allow_html=True)

# 2. AI MODEL TRAINING (Single Sensor UiO-66 Only)
@st.cache_resource
def train_voc_model():
    raw_data = [
        ["Methyl Nicotinate", 3.29993], ["Methyl Nicotinate", 3.29967], ["Methyl Nicotinate", 3.29834],
        ["Methyl Phenylacetate", 3.29993], ["Methyl Phenylacetate", 3.29967], ["Methyl Phenylacetate", 3.29835],
        ["Isoprene", 3.29993], ["Isoprene", 3.29967], ["Isoprene", 3.29834],
        ["1,2-Dimethylbenzene", 3.29993], ["1,2-Dimethylbenzene", 3.29967], ["1,2-Dimethylbenzene", 3.29834],
        ["2-Methylpyridine", 3.29993], ["2-Methylpyridine", 3.29967], ["2-Methylpyridine", 3.29834],
        ["Ethanol", 3.29993], ["Ethanol", 3.29967], ["Ethanol", 3.29822]
    ]
    df_raw = pd.DataFrame(raw_data, columns=['VOC', 'Voltage'])
    
    np.random.seed(42)
    simulated_patients = []
    for _, row in df_raw.iterrows():
        voc = row['VOC']
        base_voltage = row['Voltage']
        for i in range(100):
            v_uio = np.random.normal(base_voltage, 0.000015)
            simulated_patients.append({'V_UiO66': round(v_uio, 5), 'VOC_Gas': voc})
            
    df_patients = pd.DataFrame(simulated_patients)
    X = df_patients[['V_UiO66']]
    y = df_patients['VOC_Gas']
    
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

ai_model = train_voc_model()
tbc_biomarkers = ['Methyl Nicotinate', 'Methyl Phenylacetate']

# 3. HEADER SECTION
st.image("https://img.icons8.com/external-flatart-icons-flat-flatarticons/128/external-medical-biotechnology-flatart-icons-flat-flatarticons.png", width=75)
st.title("Electronic Nose Analysis System")
st.caption("Advanced Respiratory Diagnostics utilizing AI & Single MOF UiO-66 Nanomaterial Sensor")
st.write("---")

# 4. PATIENT INPUT CARD (Beta removed entirely)
with st.container():
    st.subheader("📋 Patient & Sensor Data Input")
    
    nama_pasien = st.text_input("Patient Full Name / ID", value="Patient #01", help="Enter patient identification code or full name")
    
    # Large and easy to use voltage numeric field
    v_uio_input = st.number_input(
        "Sensor Voltage MOF UiO-66 (V)", 
        min_value=3.29000, 
        max_value=3.30000, 
        value=3.29834, 
        format="%.5f", 
        step=0.00001,
        help="Input the raw voltage data read from the E-Nose hardware system."
    )

st.write("")
tombol_uji = st.button("🚀 Run Intelligent Gas Classification", type="primary")

# 5. DIAGNOSTICS & GRAPHICAL RESULTS
if tombol_uji:
    input_data = pd.DataFrame([[v_uio_input]], columns=['V_UiO66'])
    
    # AI Predictions
    gas_terdeteksi = ai_model.predict(input_data)[0]
    probabilitas_semua = ai_model.predict_proba(input_data)[0]
    daftar_gas = ai_model.classes_
    
    idx_gas = np.where(daftar_gas == gas_terdeteksi)[0][0]
    persentase_gas = probabilitas_semua[idx_gas] * 100
    
    is_tbc = gas_terdeteksi in tbc_biomarkers
    status_tbc = "TB POSITIVE" if is_tbc else "TB NEGATIVE (Healthy / Non-Biomarker)"

    st.write("---")
    st.subheader(f"📊 Diagnostic Reports Summary: {nama_pasien}")
    
    # Modern Metric Row
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="Predicted Gas Molecule", value=gas_terdeteksi)
    with col_m2:
        st.metric(label="AI Diagnostic Confidence", value=f"{persentase_gas:.2f}%")
        
    # Clinical Status Banner
    st.markdown("### Clinical Assessment:")
    if is_tbc:
        st.error(f"⚠️ **DIAGNOSIS STATUS: {status_tbc}**")
        st.info(f"💡 **Medical Note:** The identified compound (**{gas_terdeteksi}**) acts as an active organic breath biomarker for *Mycobacterium tuberculosis* bacteria.")
    else:
        st.success(f"✅ **DIAGNOSIS STATUS: {status_tbc}**")
        st.info(f"💡 **Medical Note:** The identified compound (**{gas_terdeteksi}**) does not align with typical clinical TB indicators. Patient is classified as non-infected.")
        
    # Beautiful Data Analytics Breakdown with a bar chart
    st.write("---")
    st.write("### 📈 AI Probability Spectrum Distribution")
    
    df_chart = pd.DataFrame({
        'VOC Compound': daftar_gas,
        'Similarity Probability (%)': probabilitas_semua * 100
    }).sort_values(by='Similarity Probability (%)', ascending=True)
    
    # Horizontal Bar Chart for better data visual aesthetics
    st.bar_chart(data=df_chart, x='VOC Compound', y='Similarity Probability (%)', horizontal=True, color="#2563EB")
    
    # NEW FEATURE: Structured Data Table shown directly to the user
    st.write("### 📋 Detailed Signal Analysis Report")
    
    # Create and sort dataframe based on highest probability
    df_prob = pd.DataFrame({
        'VOC Compound Type': daftar_gas,
        'Signal Match Rate': probabilitas_semua * 100
    }).sort_values(by='Signal Match Rate', ascending=False).reset_index(drop=True)
    
    # Format to look like standard table percentages
    df_prob['Signal Match Rate'] = df_prob['Signal Match Rate'].map("{:.2f}%".format)
    
    # Add a standard index counter starting from 1 instead of 0
    df_prob.index = df_prob.index + 1
    df_prob.index.name = 'No.'
    
    # Render the formal data table
    st.table(df_prob)
