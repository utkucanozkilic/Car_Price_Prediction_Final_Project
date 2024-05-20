# Prepare data:

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1881)
pd.options.display.float_format = '{:.0f}'.format

df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\Car Price Prediction\car_last.csv')
df_copy = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\Car Price Prediction\car_last.csv')

df['marka'].value_counts()
df = df[(df['marka'] == 'Renault') |
        (df['marka'] == 'Fiat') |
        (df['marka'] == 'Volkswagen') |
        (df['marka'] == 'Peugeot') |
        (df['marka'] == 'Toyota') |
        (df['marka'] == 'Hyundai') |
        (df['marka'] == 'Honda') |
        (df['marka'] == 'BMW') |
        (df['marka'] == 'Citroen') |
        (df['marka'] == 'Ford')
        ]

df.drop(['Detail_Breadcrumb', 'renk', 'ilan_no'], inplace = True, axis = 1)

# df.drop(['marka', 'model', 'seri', 'renk', 'ilan_no'], inplace = True, axis = 1)
df_y = df['Fiyat']
df.drop('Fiyat', axis = 1, inplace = True)
df = pd.get_dummies(df, drop_first = False)
df['marka'].value_counts()

df.to_csv('tahmin.csv', index = False)
# ((df['kilometre'] * df['yil']).describe()).T


# plt.figure(figsize = (10, 6))
# (df['kilometre'] * df['yil']).describe().plot(kind = 'bar', color = 'skyblue')
# plt.title('Kilometre * Yıl İstatistikleri')
# plt.ylabel('Değer')
# plt.xlabel('İstatistik')
# plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
# plt.show()


def outlier_threshold(dataframe, column, first_percent = 0.25, third_percent = 0.95):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return low_limit, up_limit


low_value, up_value = outlier_threshold(df, 'kilometre')
df['kilometre'] = df['kilometre'].apply(lambda x: up_value if x > up_value else x)
# df.drop(['kilometre', 'yil'], inplace = True, axis = 1)
scaler = StandardScaler()

numerical_columns = ['yil', 'kilometre', 'motor_hacmi', 'motor_gucu', 'yakit_deposu', 'boya', 'degisen']
# numerical_columns = ['km_yl', 'motor_hacmi', 'motor_gucu', 'yakit_deposu', 'boya', 'degisen']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

X = df
y = df_y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = RandomForestRegressor(
    max_depth = None,
    min_samples_leaf = 5,
    min_samples_split = 2,
    n_estimators = 50,
    random_state = 42,
    n_jobs = -1
    )
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
print("Eğitim Seti R^2 Skoru:", train_score)  # Eğitim Seti R^2 Skoru: 0.9584161671488526

# Test seti üzerinde modelin performansını değerlendir
test_score = model.score(X_test, y_test)
print("Test Seti R^2 Skoru:", test_score)  # Test Seti R^2 Skoru: 0.8269586726617316

# Eğitim seti tahminleri
train_preds = model.predict(X_train)

# Test seti tahminleri
test_preds = model.predict(X_test)

# Eğitim seti RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
print("Eğitim seti RMSE:", train_rmse)

# Test seti RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
print("Test seti RMSE:", test_rmse)


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    RandomForestRegressor(
        max_depth = None,
        min_samples_leaf = 2,
        min_samples_split = 5,
        n_estimators = 50,
        random_state = 42,
        n_jobs = -1
        ),
    X_train, y_train, cv = 5
    )  # cv = 5: 5 katlı çapraz doğrulama

# Çapraz Doğrulama R^2 Skorları: [0.81184632 0.91186963 0.87361725 0.9326807  0.91804628]
print("Çapraz Doğrulama R^2 Skorları:", cv_scores)

# Ortalama R^2 Skoru: 0.8896120346117289
print("Ortalama R^2 Skoru:", cv_scores.mean())


# Test seti üzerinde tahmin yap
y_pred_test = model.predict(X_test)

# Test seti üzerinde R^2 skoru
test_score = r2_score(y_test, y_pred_test)
# Test Seti R^2 Skoru: 0.922892461002403
print("Test Seti R^2 Skoru:", test_score)


from sklearn.model_selection import GridSearchCV

# Grid arama için parametrelerin oluşturulması
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# GridSearchCV oluşturma
grid_search = GridSearchCV(estimator = RandomForestRegressor(random_state = 42),
                           param_grid = param_grid,
                           scoring = 'neg_mean_squared_error',
                           cv = 5,
                           verbose = 1,
                           n_jobs = -1)

# Grid aramanın eğitilmesi
grid_search.fit(X_train, y_train)

# En iyi parametrelerin bulunması
best_params = grid_search.best_params_
print("En iyi parametreler:", best_params)

# En iyi modelin bulunması
best_model = grid_search.best_estimator_

# Eğitim seti üzerinde en iyi modelin performansının değerlendirilmesi
train_score = best_model.score(X_train, y_train)
print("Eğitim Seti R^2 Skoru:", train_score)

# Test seti üzerinde en iyi modelin performansının değerlendirilmesi
test_score = best_model.score(X_test, y_test)
print("Test Seti R^2 Skoru:", test_score)

from joblib import dump

# En iyi modeli kaydetme
dump(model, '../final_random_forest_model.joblib')


from joblib import load

# Kaydedilmiş modeli yükleme
final_model = load('../best_random_forest_model.joblib')

# Yüklü modeli kullanarak tahmin yapma
prediction = final_model.predict(X_test)

# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
#
# # Verisetini yükle
# df = pd.read_csv(r'C:\Users\Souljah_Pc\PycharmProjects\Car Price Prediction\car_last.csv')
#
#
# # Kategorik değişkenleri işle
# df = pd.get_dummies(df)
#
# # Sayısal değişkenleri ölçeklendir
# scaler = StandardScaler()
# numerical_columns = ['Fiyat', 'yil', 'kilometre', 'motor_hacmi', 'motor_gucu', 'yakit_deposu', 'boya', 'degisen']
# df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
#
# # Bağımsız ve bağımlı değişkenleri ayır
# X = df.drop('Fiyat', axis=1)
# y = df['Fiyat']
#
# # Eğitim ve test setlerini ayır
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Makine öğrenimi modelini oluştur
# model = RandomForestRegressor(n_estimators=100, random_state=42)
#
# # Modeli eğit
# model.fit(X_train, y_train)
#
# # Eğitim seti üzerinde modelin performansını değerlendir
# train_score = model.score(X_train, y_train)
# print("Eğitim Seti R^2 Skoru:", train_score)
#
# # Test seti üzerinde modelin performansını değerlendir
# test_score = model.score(X_test, y_test)
# print("Test Seti R^2 Skoru:", test_score)
#
# # Çapraz doğrulama ile modelin performansını değerlendir
# cv_scores = cross_val_score(model, X, y, cv=5)
# print("Çapraz Doğrulama Skorları:", cv_scores)
# print("Ortalama Çapraz Doğrulama Skoru:", cv_scores.mean())
#
# # Test seti üzerinde tahmin yap ve performansı değerlendir
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Ortalama Kare Hata (MSE):", mse)
#
