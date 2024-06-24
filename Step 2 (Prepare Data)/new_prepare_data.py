import pandas as pd
import re
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1881)

pd.options.display.float_format = '{:.0f}'.format


def reproduce_values(dataframe, column, short_line = False):
    if not short_line:
        mapping = {}

        na_df = dataframe[dataframe[column].isnull()]
        not_na_df = dataframe[dataframe[column].notnull()]

        for index, row in not_na_df.iterrows():
            mapping[row['Detail_Breadcrumb']] = row[column]

        for index, row in na_df.iterrows():
            value = row['Detail_Breadcrumb']
            if value in mapping:
                dataframe.at[index, column] = mapping[value]

    if short_line:
        mapping = {}

        short_line_df = dataframe[dataframe[column] == short_line]
        short_line_not_df = dataframe[dataframe[column] != short_line]

        for index, row in short_line_not_df.iterrows():
            mapping[row['Detail_Breadcrumb']] = row[column]

        for index, row in short_line_df.iterrows():
            value = row['Detail_Breadcrumb']
            if value in mapping:
                dataframe.at[index, column] = mapping[value]

    return dataframe


# 39. sütundaki verilerin farklı veri tiplerinden oluştuğu uyarısını aldık. Bu sorunu ileride çözeceğiz:
car_df = pd.read_csv('Step 1 (Get Adverts)/cars.csv')
car_df.columns

# Kullanışsız sütunları düşürme:
car_df.drop(
    ['Pagetype', 'Assignee', 'kategori', 'ilanNo', 'galeri_ID', 'photoCount', 'ilanTarihi',
     'Detail_ContactNumberCount ', 'Detail_MessageContactStatus', 'Detail_850Status', 'motor_hacmi',
     'Detail_PhoneContactStatus', 'Etiket', 'plaka_uyrugu', 'Detail_CarSegment', 'il', ''
     'motor_gucu', 'cekis', 'ort._yakit_tuketimi', 'yakit_deposu', 'takasa_uygun', 'kimden'], inplace = True, axis = 1
    )

# Anonimleştirme:
car_df['ilan_no'] = car_df['ilan_no'] + 783562

# Az miktardaki tekrarlı veriyi düşür:
car_df.drop_duplicates(keep = 'first', inplace = True)

# Elektirikli araçları ayırma:
car_df['yakit_tipi'].value_counts()
electricity_car_df = car_df[car_df['yakit_tipi'] == 'Electricity']

# Elektrikli araçları düşürme:
car_df = car_df[~(car_df['yakit_tipi'] == 'Elektrik')]

# Elektrikli araç için var olan sütunları düşür:
car_df.drop(columns = car_df.columns[-5:], inplace = True, axis = 1)

# Sıfır araçları düşürme:
car_df = car_df[~((car_df['arac_durumu'] == 'Yurtdışından İthal Sıfır') | (car_df['arac_durumu'] == 'Sıfır'))]
car_df.drop('arac_durumu', inplace = True, axis = 1)

# Tek bir null satır:
car_df.drop(index = 6617, inplace = True, axis = 0)

# Detail_Breadcrumb:
car_df.loc[:, 'Detail_Breadcrumb'] = car_df['Detail_Breadcrumb'].str.replace('Otomobil / ', '')


########################################################################################################################
################################################ MISSING VALUES ########################################################
########################################################################################################################

# kilometre, vites_tipi, yakit_tipi, kasa_tipi, renk
car_df = car_df[(car_df['kilometre'].notnull())]
car_df.isnull().sum()

car_df = car_df[(car_df['kasa_tipi'].notnull())]

car_df = car_df[(car_df['renk'].notnull())]


# kasa_tipi == '-' olan değerler (144 tane)
(car_df['kasa_tipi'] == '-').sum()
car_df = reproduce_values(car_df, 'kasa_tipi', '-')

car_df = car_df[~(car_df['kasa_tipi'] == '-')]

# renk
car_df = car_df[car_df['renk'] != '-']


########################################################################################################################
################################################### DTYPES #############################################################
########################################################################################################################

car_df.info()

# yil:
car_df['yil'] = car_df['yil'].astype('int64')

# kilometre:
car_df['kilometre'] = car_df['kilometre'].str.replace('.', '')
car_df['kilometre'] = car_df['kilometre'].str.replace(' km', '')
car_df['kilometre'] = car_df['kilometre'].str.replace('Km', '')

car_df['kilometre'] = car_df['kilometre'].astype('int64')

# boya-degisen:
car_df.info()

car_df.loc[:, 'boya'] = car_df['boya-degisen'].apply(lambda x: '0' if x == 'Tamamı orjinal' else x)
car_df.loc[:, 'boya'] = car_df['boya'].apply(lambda x: '0' if x == 'Belirtilmemiş' else x)
car_df.loc[:, 'boya'] = car_df['boya'].apply(lambda x: '13' if x == 'Tamamı boyalı' else x)
car_df.loc[:, 'boya'] = car_df['boya'].apply(
    lambda x: (re.findall(r'\d+', x.split(',')[-1]))[0]
    if ',' in x else x
    )
car_df.loc[:, 'boya'] = car_df['boya'].apply(lambda x: '0' if 'değişen' in x else x)
car_df.loc[:, 'boya'] = car_df['boya'].apply(lambda x: '0' if 'Tamamı' in x else x)
car_df.loc[:, 'boya'] = car_df['boya'].apply(
    lambda x: (re.findall(r'\d+', x.split(' ')[0]))[0]
    if 'boyalı' in x else x
    )
car_df['boya'] = car_df['boya'].astype('int64')


car_df['degisen'] = car_df['boya-degisen'].apply(lambda x: '0' if x == 'Tamamı orjinal' else x)
car_df['degisen'] = car_df['degisen'].apply(lambda x: '0' if x == 'Belirtilmemiş' else x)
car_df['degisen'] = car_df['degisen'].apply(lambda x: '13' if x == 'Tamamı değişmiş' else x)
car_df['degisen'] = car_df['degisen'].apply(
    lambda x: (re.findall(r'\d+', x.split(',')[0]))[0]
    if ',' in x else x
    )
car_df['degisen'] = car_df['degisen'].apply(lambda x: '0' if 'boyalı' in x else x)
car_df['degisen'] = car_df['degisen'].apply(
    lambda x: (re.findall(r'\d+', x.split(' ')[0]))[0]
    if 'değişen' in x else x
    )

car_df['degisen'] = car_df['degisen'].astype('int64')

# boya-degisen'i düşür:
car_df.drop('boya-degisen', axis = 1, inplace = True)

# Fiyat:
car_df['Fiyat'] = car_df['Fiyat'].astype('int64')

# Yalnızca 2000-2023 model araçlar ile çalış
car_df = car_df[car_df['yil'].isin(range(2000, 2024))]


########################################################################################################################
################################################### OUTLIERS ###########################################################
########################################################################################################################


def outlier_threshold(dataframe, column, first_percent = 0.05, third_percent = 0.95):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return low_limit, up_limit


def check_outlier(dataframe, column, first_percent = 0.25, third_percent = 0.75):
    q1 = dataframe[column].quantile(first_percent)
    q3 = dataframe[column].quantile(third_percent)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr

    return dataframe[(dataframe[column] < low_limit) | (dataframe[column] > up_limit)].any(axis = None)


def grab_outliers(dataframe, column, index = False):
    low, up = outlier_threshold(dataframe, column)

    if len(dataframe[(dataframe[column] < low) | (dataframe[column] > up)]) > 10:
        print(dataframe[(dataframe[column] < low) | (dataframe[column] > up)].head())
    else:
        print(dataframe[(dataframe[column] < low) | (dataframe[column] > up)])

    if index:
        return dataframe[(dataframe[column] < low) | (dataframe[column] > up)].index


def replacement_with_thresholds(dataframe, column):
    low_limit, up_limit = outlier_threshold(dataframe, column)
    dataframe.loc[(dataframe[column] < low_limit), column] = low_limit
    dataframe.loc[(dataframe[column] > up_limit), column] = up_limit


car_df.info()
car_df_ex = car_df.copy()
car_df = car_df_ex.copy()

# Fiyat:
plt.figure(figsize = (10, 6))
sns.boxplot(x = car_df['Fiyat'], orient = 'v')
plt.title('Fiyat Sütunu Aykırı Değer Analizi')
plt.xlabel('Fiyat')
plt.show()


# Çok fazla aykırı değer yok. Baskılama ihtiyacı yok. Aykırı görünen 5 değeri atabiliriz:
car_df['Fiyat'].sort_values(ascending = False)
car_df.drop(car_df['Fiyat'].sort_values(ascending = False).head(3).index, axis = 0, inplace = True)
car_df.drop(car_df['Fiyat'].sort_values(ascending = False).tail(2).index, axis = 0, inplace = True)

# kilometre:
plt.figure(figsize = (10, 6))
sns.boxplot(x = car_df['kilometre'])
plt.title('Kilometre Sütunu Aykırı Değer Analizi')
plt.xlabel('Kilometre')
plt.show()

# Aykırı değerleri baskıla:
replacement_with_thresholds(car_df, 'kilometre')

plt.figure(figsize = (10, 6))
sns.boxplot(x = car_df['kilometre'])
plt.title('Kilometre Sütunu Aykırı Değer Analizi')
plt.xlabel('Kilometre')
plt.show()

car_df.info()
print(car_df.sort_values(by = 'kilometre', ascending = False).head(100).to_string())
car_df = car_df[car_df['kilometre'] < 700000]

# model
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

pd.options.display.float_format = '{:.4f}'.format

car_df.columns

columns = ['Detail_Breadcrumb', 'ilan_no', 'ilan_tarihi']
car_df.drop(columns, axis = 1, inplace = True)

column_order = ['marka', 'model', 'seri', 'yil', 'kilometre', 'vites_tipi', 'yakit_tipi', 'kasa_tipi',
                'renk', 'boya', 'degisen', 'Fiyat']

car_df = car_df[column_order]

def grab_col_names(dataframe, cat_th = 10, car_th = 20, info = False):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['object', 'category', 'bool']]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64']
                   and dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtype in ['object', 'category']]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['float64', 'int64']]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    if not info:
        return cat_cols, num_cols, cat_but_car
    else:
        print("Observations: {}, Variables: {}".format(dataframe.shape[0], dataframe.shape[1]))
        print("Caterogical columns:", len(cat_cols))
        print("Numerical columns:", len(num_cols))
        print('Caterogical but cardinal columns:', len(cat_but_car))
        print('Numerical but caterogical columns:', len(num_but_cat))

        return cat_cols, num_cols, cat_but_car


def plot_grouped_importance(model, features, original_columns, top_n = None, figure_name = False):
    # Özelliklerin önemini al ve bir df olarak oluştur
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})

    # Orijinal sütunlara göre dummy değişkenleri grupla
    grouped_feature_imp = []
    for col in original_columns:
        dummy_cols = [dummy_col for dummy_col in features.columns if dummy_col.startswith(col + '_')]
        if dummy_cols:
            mean_importance = feature_imp[feature_imp['Feature'].isin(dummy_cols)]['Value'].mean()
            grouped_feature_imp.append({'Feature': col, 'Value': mean_importance})
        else:
            # Dummy olmayan özellikler için doğrudan ekle
            feature_value = feature_imp[feature_imp['Feature'] == col]['Value'].values
            if len(feature_value) > 0:
                grouped_feature_imp.append({'Feature': col, 'Value': feature_value[0]})

    # DataFrame'e dönüştür
    grouped_feature_imp = pd.DataFrame(grouped_feature_imp)

    # Verileri sıralayıp görselleştir
    if top_n:
        grouped_feature_imp = grouped_feature_imp.sort_values(by = 'Value', ascending = False).head(top_n)
    else:
        grouped_feature_imp = grouped_feature_imp.sort_values(by = 'Value', ascending = False)

    plt.figure(figsize = (20, 5))
    sns.set(font_scale = 1)
    sns.barplot(x = "Value", y = "Feature", data = grouped_feature_imp)
    plt.title('Grouped Feature Importances for')
    plt.tight_layout()
    plt.show()
    if figure_name:
        plt.savefig('grouped_importances.png')


cat_columns, num_columns, cat_but_car_columns = grab_col_names(car_df, cat_th = 14, car_th = 20, info = True)

cat_columns = cat_columns + cat_but_car_columns

# Scale:

# İşler sarpasardığında reset bloğumuz:
car_df_2 = car_df.copy()
car_df = car_df_2.copy()

# Normalizasyon öncesi hedef etiketi at (Fiyat):
num_columns.pop(-1)

scaler = MinMaxScaler()
car_df[num_columns] = scaler.fit_transform(car_df[num_columns])

# dummy_attributes:
car_df = pd.get_dummies(car_df, drop_first = True, dtype = int, dummy_na = False)

y = car_df['Fiyat']
X = car_df.drop('Fiyat', axis = 1)

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

rf_model = RandomForestRegressor(
    max_depth = None,
    min_samples_leaf = 2,
    min_samples_split = 5,
    n_estimators = 50,
    random_state = 42,
    n_jobs = None,
    verbose = 2,
    )

rf_model.fit(X_train, y_train)

# Eğitim başarı:
y_train_preds = rf_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_preds))

# Test başarı:
y_test_preds = rf_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_preds))

# Orijinal sütunları tanımla
original_columns = ['marka', 'model', 'seri', 'yil', 'kilometre', 'vites_tipi', 'yakit_tipi', 'kasa_tipi',
                    'renk', 'boya', 'degisen']

# Özelliklerin önemini gruplayarak olarak görselleştir:
plot_grouped_importance(rf_model, X_train, original_columns)

# AdaBoost
car_df = car_df_2.copy()

scaler = MinMaxScaler()
car_df[num_columns] = scaler.fit_transform(car_df[num_columns])

# dummy_attributes:
car_df = pd.get_dummies(car_df, drop_first = True, dtype = int, dummy_na = False)

y = car_df['Fiyat']
X = car_df.drop('Fiyat', axis = 1)

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

adaboost_model = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 3), n_estimators = 50, random_state = 42)
adaboost_model.fit(X_train, y_train)

y_train_pred = adaboost_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # 4351

y_test_pred = adaboost_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # 182477


#
car_df['marka_seri_model'] = car_df['marka'] + ' ' + car_df['seri'] + ' ' + car_df['model']
car_df.drop(['marka', 'seri', 'model'], inplace = True, axis = 1)

class_counts = car_df['marka_seri_model'].value_counts()
to_remove = class_counts[class_counts < 2].index
car_df = car_df[~car_df['marka_seri_model'].isin(to_remove)]

num_columns = ['yil', 'kilometre', 'boya', 'degisen']

scaler = MinMaxScaler()
car_df[num_columns] = scaler.fit_transform(car_df[num_columns])

car_df = pd.get_dummies(car_df, drop_first = False, dtype = int, dummy_na = False)

y = car_df['Fiyat']
X = car_df.drop('Fiyat', axis = 1)

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42)

model = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 15), random_state = 42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # 49326.86580001562

y_test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # 183258.55967293796
