import random
import pandas as pd

# Definisi nilai ambang batas
threshold_uio = 2.64

dummy_patients_threshold = []

# Daftar VOC dari data asli (df_base_optimal)
available_vocs = [
    '1,2-Dimethylbenzene', '2-Methylpyridine', 'Ethanol',
    'Isoprene', 'Methyl Nicotinate', 'Methyl Phenylacetate'
]

# 50 data dummy
for i in range(50):
    # Menghasilkan nilai V_UiO66 di sekitar ambang batas
    v_uio_value = round(random.uniform(2.63, 2.65), 5)

    #status berdasarkan ambang batas
    status = 'POSITIF TBC' if v_uio_value > threshold_uio else 'NEGATIF'

    # Menetapkan VOC dan persentase secara acak
    selected_voc = random.choice(available_vocs)
    percentage = round(random.uniform(70, 100), 2) # Contoh persentase antara 70-100%

    dummy_patients_threshold.append({
        'Patient_ID': f'PASIEN_{i+1:03d}',
        'V_UiO66': v_uio_value,
        'Threshold': threshold_uio,
        'Target_Gas_Detected': selected_voc, # Nama kolom untuk VOC yang terdeteksi
        'Percentage_of_VOC': f'{percentage}%',
        'Status': status
    })

df_dummy_threshold = pd.DataFrame(dummy_patients_threshold)

print("DATA DUMMY PASIEN BERDASARKAN AMBANG BATAS V_UiO66 \n")

# Menampilkan 25 pasien pertama
print("--- Pasien 1 - 25 ---")
display(df_dummy_threshold.head(25))

# Menampilkan 25 pasien berikutnya (26 - 50)
print("\n--- Pasien 26 - 50 ---")
display(df_dummy_threshold.tail(25))
