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

# Mapping destinasi ke koordinat Google Maps (tambahkan destinasi lain sesuai kebutuhan)
coords_update = {
    "Kebun Teh Nglinggo": (-7.667222, 110.146944),
    "Pantai Patihan": (-7.997222, 110.312222),
    "Pantai Parangtritis": (-8.025833, 110.317222),
    "Kawasan Malioboro": (-7.792778, 110.365833),
    "Embung Tambakboyo": (-7.763889, 110.414444),
    "Geoforest Watu Payung Turunan": (-7.963889, 110.561944),
    "Hutan Pinus Kayon": (-7.312222, 110.519722),
    "Pantai Marina": (-6.950833, 110.423889),
    "Pantai Krakal": (-8.148889, 110.613333),
    "Pantai Sadranan": (-8.150278, 110.613611),
    "Pantai Ngandong": (-8.151389, 110.614444),
    "Pantai Sundak": (-8.149444, 110.613333),
    "Pantai Jogan": (-8.185278, 110.663333),
    "Pantai Jungwok": (-8.185833, 110.678611),
}

# Update koordinat pada DataFrame
for place, (lat, long) in coords_update.items():
    mask = df_selected['Place_Name'].str.strip().str.lower() == place.lower()
    df_selected.loc[mask, 'Lat'] = lat
    df_selected.loc[mask, 'Long'] = long

# Simpan ke file baru
df_selected.to_csv('destinasi-wisata-YKSM.csv', index=False)