import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import load

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1881)
pd.options.display.float_format = '{:.0f}'.format

# Kaydedilmiş modeli yükleme
final_model = load(r'/final_random_forest_model.joblib')
tahmin_df = pd.read_csv(r'/tahmin.csv')

tahmin_df.columns

tahmin_df['kimden'].value_counts()

marka = input("Aracın markası nedir? ")

model = input("Aracın modeli nedir? ")

seri = input("Aracın serisi nedir? ")

yil = int(input("Aracın üretim yılı nedir? "))

kilometre = int(input("Aracın kilometre bilgisi nedir? "))

vites_tipi = input("Aracın vites tipi nedir?"
                   "(Düz/Otomatik/Yarı Otomatik)")

yakit_tipi = input('Aracın yakıt tipi nedir?'
                   '(Dizel/LPG & Benzin/Benzin/Hibrit)')

motor_hacmi = int(input("Aracın motor hacmi(cc) nedir? "))

motor_gucu = int(input("Aracın motor gücü(hp) nedir? "))

yakit_deposu = int(input("Aracın yakıt deposu kapasitesi(lt) nedir? "))

kimden = input('Aracı satan kim?'
               '(Sahibinden/Galeriden/Yetkili Bayiden)')

boya = int(input("Aracın boyalı parça sayısı?"))

degisen = int(input("Aracın değişen parça sayısı?"))

tahmin_df.columns
# Kullanıcı girişlerini bir veri çerçevesine dönüştür
input_data = pd.DataFrame({
    "marka": [marka],
    'model': [model],
    'seri': [seri],
    'yil': [yil],
    'kilometre': [kilometre],
    'vites_tipi': [vites_tipi],
    'yakit_tipi': [yakit_tipi],
    'motor_hacmi': [motor_hacmi],
    'motor_gucu': [motor_gucu],
    'yakit_deposu': [yakit_deposu],
    'kimden': [kimden],
    'boya': [boya],
    'degisen': [degisen]
    })

tahmin_df = pd.concat([tahmin_df, input_data], ignore_index = True)


def outlier_threshold(dataframe, column, first_percent = 0.25, third_percent = 0.95):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return low_limit, up_limit


low_value, up_value = outlier_threshold(tahmin_df, 'kilometre')
tahmin_df['kilometre'] = tahmin_df['kilometre'].apply(lambda x: up_value if x > up_value else x)

scaler = StandardScaler()

numerical_columns = ['yil', 'kilometre', 'motor_hacmi', 'motor_gucu', 'yakit_deposu', 'boya', 'degisen']
tahmin_df[numerical_columns] = scaler.fit_transform(tahmin_df[numerical_columns])

tahmin_df = pd.get_dummies(tahmin_df, drop_first = False)
# Pipeline'dan geçir ve tahmini al
predicted_price = final_model.predict(tahmin_df.iloc[[-1]])




