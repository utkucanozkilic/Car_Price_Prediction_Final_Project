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

# Kullanışsız sütunları düşürme:
car_df.drop(
    ['Pagetype', 'Assignee', 'kategori', 'ilanNo', 'galeri_ID', 'photoCount', 'ilanTarihi',
     'Detail_ContactNumberCount ', 'Detail_MessageContactStatus', 'Detail_850Status',
     'Detail_PhoneContactStatus', 'Etiket', 'plaka_uyrugu', 'Detail_CarSegment'], inplace = True, axis = 1
    )

# Veri seti içinde 700'den az örneğe sahip markaları düşür (Tabakalı örneklemeden kaçınmak için):
marka_less_than_700 = car_df['marka'].value_counts()[car_df['marka'].value_counts() < 700].index.to_list()

for marka in marka_less_than_700:
    car_df = car_df[~(car_df['marka'] == marka)]

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
car_df['Detail_Breadcrumb'] = car_df['Detail_Breadcrumb'].str.replace('Otomobil / ', '')


########################################################################################################################
################################################ MISSING VALUES ########################################################
########################################################################################################################

# il:
car_df[car_df['il'].isnull()]  # 13 örnek
car_df = car_df[~(car_df['il'].isnull())]

# kilometre, vites_tipi, yakit_tipi, kasa_tipi, renk
car_df = car_df[(car_df['kilometre'].notnull())]
car_df.isnull().sum()

car_df = car_df[(car_df['kasa_tipi'].notnull())]

car_df = car_df[(car_df['renk'].notnull())]

# motor_hacmi
car_df['motor_hacmi'].isnull().sum()  # 1962 değer

# motor_hacmi boş olan hücreleri, dolu olan hücrelerden getir:
car_df = reproduce_values(car_df, 'motor_hacmi')

car_df['motor_hacmi'].isnull().sum()  # 34 örnek
car_df = car_df[car_df['motor_hacmi'].notnull()]

# motor_hacmi == '-' olan hücreleri, dolu olan hücrelerden getir:
car_df = reproduce_values(car_df, 'motor_hacmi', short_line = '-')

car_df[car_df['motor_hacmi'] == '-']  # 2 örnek
car_df = car_df[~(car_df['motor_hacmi'] == '-')]

# motor_gucu eksik ya da '-' olan hücreleri doldurmak mümkün değil:
car_df = car_df[car_df['motor_gucu'].notnull()]
car_df = car_df[car_df['motor_gucu'] != '-']

car_df.isnull().sum()

# cekis:
car_df['cekis'].value_counts()
car_df['cekis'].isnull().sum()

car_df = reproduce_values(car_df, 'cekis')
car_df[car_df['cekis'].isnull()]  # 4 örnek
car_df = car_df[car_df['cekis'].notnull()]

car_df = reproduce_values(car_df, 'cekis', short_line = '-')
car_df[car_df['cekis'] == '-']  # 8 örnek
car_df = car_df[car_df['cekis'] != '-']

# ort._yakit_tuketimi:
car_df.drop('ort._yakit_tuketimi', inplace = True, axis = 1)

car_df.isnull().sum()

# yakit_deposu:
car_df = reproduce_values(car_df, 'yakit_deposu')

car_df.isnull().sum()  # 1450 değer
car_df[car_df['yakit_deposu'].isnull()]
car_df[car_df['model'] == '320i 50th Year M Edition']

# yakit_deposu'nu düşür:
car_df.drop('yakit_deposu', inplace = True, axis = 1)

# takasa_uygun sütununda 34699 Null değer var. Düşür:
car_df.drop('takasa_uygun', inplace = True, axis = 1)

car_df.isnull().sum()

# kasa_tipi == '-' olan değerler (72 tane)
(car_df['kasa_tipi'] == '-').sum()
car_df = reproduce_values(car_df, 'kasa_tipi')  # bu değerler doldurulamıyor.

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

# motor_hacmi:
print(car_df['motor_hacmi'].to_string())

car_df['motor_hacmi'] = car_df['motor_hacmi'].str.replace(' cc', '')
car_df['motor_hacmi'] = car_df['motor_hacmi'].str.replace(' cm3 ve üzeri', '')
car_df['motor_hacmi'] = car_df['motor_hacmi'].str.replace(' cm3\\', '')
car_df['motor_hacmi'] = car_df['motor_hacmi'].str.replace(' cm3', '')

car_df['motor_hacmi'] = car_df['motor_hacmi'].apply(lambda x: x.split(' - ')[-1] if ' - ' in x else x)

car_df['motor_hacmi'] = car_df['motor_hacmi'].astype('int64')

car_df.info()

# motor_gucu:
print(car_df['motor_gucu'].to_string())

car_df['motor_gucu'] = car_df['motor_gucu'].str.replace(' hp', '')
car_df['motor_gucu'] = car_df['motor_gucu'].str.replace(' HP', '')
car_df['motor_gucu'] = car_df['motor_gucu'].apply(lambda x: x.split(' - ')[-1] if ' - ' in x else x)


car_df = car_df[~(car_df['motor_gucu'] == '601 ve üzeri')]
car_df = car_df[~(car_df['motor_gucu'] == '50\\')]

car_df['motor_gucu'] = car_df['motor_gucu'].astype('int64')

# boya-degisen:
car_df.info()

car_df = car_df[(car_df['boya-degisen'] != 'Belirtilmemiş')]
car_df.loc[:, 'boya'] = car_df['boya-degisen'].apply(lambda x: '0' if x == 'Tamamı orjinal' else x)
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

# Fiyat:
plt.figure(figsize = (10, 6))
sns.boxplot(x = car_df['Fiyat'])
plt.title('Fiyat Sütunu Aykırı Değer Analizi')
plt.xlabel('Fiyat')
plt.show()

# Çok fazla aykırı değer yok. Baskılama ihtiyacı yok. Aykırı görünen 3 değeri atabiliriz:
car_df['Fiyat'].sort_values(ascending = False)
car_df.drop(car_df['Fiyat'].sort_values(ascending = False).head(2).index, axis = 0, inplace = True)
car_df.drop(car_df['Fiyat'].sort_values(ascending = False).tail(1).index, axis = 0, inplace = True)

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

# csv olarak kaydet (by no index reset):
car_df.to_csv('Step 2 (Prepare Data)/car_last.csv', index = False, mode = 'a')