import pandas as pd

# Baca data
df = pd.read_csv('destinasi-wisata-indonesia.csv')

# Hapus kolom yang tidak diperlukan
df = df.drop(columns=["Time_Minutes", "Column1", "_1"], errors="ignore")

# Filter hanya destinasi di Semarang dan Yogyakarta saja
selected_cities = ['Semarang', 'Yogyakarta']

df_selected = df[df['City'].str.strip().isin(selected_cities)].copy()


# Tambahkan kolom Outdoor/Indoor
def get_outdoor_indoor(row):
    indoor_categories = ['Budaya', 'Pusat Perbelanjaan', 'Tempat Ibadah', 'Museum']
    outdoor_categories = ['Taman Hiburan', 'Bahari', 'Cagar Alam', 'Taman', 'Cagar Budaya']

    if any(cat in str(row['Category']) for cat in indoor_categories):
        return 'Indoor'
    elif any(cat in str(row['Category']) for cat in outdoor_categories):
        return 'Outdoor'
    else:
        return 'Outdoor'  # Default ke Outdoor jika tidak jelas

df_selected['Outdoor/Indoor'] = df_selected.apply(get_outdoor_indoor, axis=1)

# Normalisasi kolom rating agar berbentuk decimal di bawah 10
def normalize_rating(r):
    try:
        val = float(r)
        if val > 10:
            return round(val / 10, 2)
        if val < 0 or val == 0:
            return None
        return round(val, 2)
    except Exception:
        return None

df_selected['Rating'] = df_selected['Rating'].apply(normalize_rating)

# Pastikan kolom Lat dan Long bertipe float dan valid
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

df_selected['Lat'] = df_selected['Lat'].apply(safe_float)
df_selected['Long'] = df_selected['Long'].apply(safe_float)

# Simpan ke file baru
df_selected.to_csv('destinasi-wisata-YKSM.csv', index=False)