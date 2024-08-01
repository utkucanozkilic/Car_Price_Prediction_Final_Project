import pandas as pd
import joblib

df = pd.read_csv('../Step 2 (Prepare Data)/car_df_for_model.csv')

pipeline = joblib.load(
    r"/Step 3 (Modelling)/pipeline_model.pkl"
    )

import streamlit as st

st.set_page_config(layout = 'wide', page_title = 'Used Car Price Prediction Project', page_icon = '🚗')
st.title("🚗 :rainbow[Used Car Price Prediction] 🚗")

# Sidebar for car features
st.sidebar.header("Car Features")

# Brand selection
brands_option = df['marka'].unique().tolist()
marka = st.sidebar.selectbox('Marka', brands_option)

# Series selection based on brand
series_options = df[df['marka'] == marka]['seri'].unique().tolist()
seri = st.sidebar.selectbox('Seri', series_options)

# Model selection based on series
model_options = df[(df['marka'] == marka) & (df['seri'] == seri)]['model'].unique().tolist()
model = st.sidebar.selectbox('Model', model_options)

# Year selection
yil = st.sidebar.number_input(
    'Yıl', min_value = 0,
    max_value = 2023, step = 1
    )

# Mileage selection
kilometre = st.sidebar.number_input('Kilometre', min_value = 0, step = 1)

# Transmission type selection
vites_tipi = st.sidebar.selectbox(
    'Vites Tipi',
    df[(df['marka'] == marka) &
       (df['seri'] == seri) &
       (df['model'] == model)]['vites_tipi'].unique().tolist()
    )

# Fuel type selection
yakit_tipi = st.sidebar.selectbox(
    'Yakıt Tipi',
    df[(df['marka'] == marka) &
       (df['seri'] == seri) &
       (df['model'] == model)]['yakit_tipi'].unique().tolist()
    )

# Body type selection
kasa_tipi = st.sidebar.selectbox(
    'Kasa Tipi',
    df[(df['marka'] == marka) &
       (df['seri'] == seri) &
       (df['model'] == model)]['kasa_tipi'].unique().tolist()
    )

# Color selection
renk = st.sidebar.selectbox(
    'Renk',
    df[(df['marka'] == marka) &
       (df['seri'] == seri) &
       (df['model'] == model)]['renk'].unique().tolist()
    )

# Paint selection
boya = st.sidebar.selectbox('Boyalı Parça Sayısı', list(range(14)))

# Change selection
degisen = st.sidebar.selectbox('Değişen Parça Sayısı', list(range(14)))

if st.sidebar.button('Tahmin Et'):
    # Kullanıcıdan alınan bilgileri bir DataFrame'e dönüştür
    input_data = pd.DataFrame(
        {
            'marka':      [marka],
            'seri':       [seri],
            'model':      [model],
            'yil':        [yil],
            'kilometre':  [kilometre],
            'vites_tipi': [vites_tipi],
            'yakit_tipi': [yakit_tipi],
            'kasa_tipi':  [kasa_tipi],
            'renk':       [renk],
            'boya':       [boya],
            'degisen':    [degisen]
            }
        )

    # Modelden tahmin edilen fiyatı al:
    predicted_price = pipeline.predict(input_data)[0]

    # Tahmin edilen fiyatı sayfanın ana kısmında göster:
    st.write("Tahmin Edilen Fiyat: {:,.2f} TL".format(predicted_price))
