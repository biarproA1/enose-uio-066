import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. DATABASE ASLI
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
    ["Ethanol", "UiO66", 1.0, 3.29964],
    ["Ethanol", "UiO66", 5.0, 3.29822]
]

df_raw = pd.DataFrame(raw_data, columns=['VOC', 'MOF', 'Beta', 'Voltage'])

# Mengubah bentuk agar MOF menjadi kolom fitur sejajar
df_base = df_raw.pivot(index=['VOC', 'Beta'], columns='MOF', values='Voltage').reset_index()

# FILTER MUTLAK: Hanya mengambil baseline di kondisi optimal (Beta == 5.0)
df_base_optimal = df_base[df_base['Beta'] == 5.0]

# Menentukan target biomarker TBC
tbc_biomarkers = ['Methyl Nicotinate', 'Methyl Phenylacetate']

# =======================================================
# 2. GENERATE POPULASI DATA PASIEN (KHUSUS BETA 5.0)
# =======================================================
np.random.seed(42)
simulated_patients = []

# 100 sampel pasien untuk setiap jenis zat pada tingkat Beta 5.0
for _, row in df_base_optimal.iterrows():
    voc = row['VOC']
    base_uio = row['UiO66']
    status_tbc = 'POSITIF TBC' if voc in tbc_biomarkers else 'NEGATIF (Sehat/Zat Lain)'

    for i in range(100):
        # Noise sensor mikro alami (Standard Deviation = 0.000015)
        v_uio = np.random.normal(base_uio, 0.000015)
        # v_zif = np.random.normal(base_zif, 0.000015) # ZIF8 tidak digunakan

        simulated_patients.append({
            'Beta': 5.0,
            'V_UiO66': round(v_uio, 5),
            # 'V_ZIF8': round(v_zif, 5), # ZIF8 tidak digunakan
            'Target_Gas': voc,
            'Status': status_tbc
        })

df_patients = pd.DataFrame(simulated_patients)

print("=== 5 SAMPLE DATABASE PASIEN (OPTIMAL BETA 5.0) ===")
print(df_patients.head().to_string(index=False))
print(f"\nTotal Database Pasien Terbuat: {len(df_patients)} data sampel.")
print("-" * 55)


# 3. MODEL AI RANDOM FOREST (DENGAN VOLTASE UiO66)
# menggunakan pembacaan voltase dari UiO66
X = df_patients[['V_UiO66']]
y = df_patients['Status']

# Split data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Inisialisasi & Fitting Model AI
ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
ml_model.fit(X_train, y_train)

# 4. HASIL EVALUASI AKHIR SISTEM
y_pred = ml_model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print("=== HASIL EVALUASI MODEL AI (V_UiO66 ONLY) ===")
print(f"Akurasi Deteksi Akhir Sistem E-Nose: {score * 100:.2f}%")
print("\n=== DETAIL DATA LAPORAN DIAGNOSIS AI === ")
print(classification_report(y_test, y_pred))  
