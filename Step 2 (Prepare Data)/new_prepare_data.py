import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1881)

pd.options.display.float_format = '{:.4f}'.format


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


car_df = pd.read_csv('Step 1 (Get Adverts)/cars.csv')

# Kullanışsız sütunları düşürme:
car_df.drop(
    ['Pagetype', 'Assignee', 'kategori', 'ilanNo', 'galeri_ID', 'photoCount', 'ilanTarihi',
     'Detail_ContactNumberCount ', 'Detail_MessageContactStatus', 'Detail_850Status',
     'Detail_PhoneContactStatus', 'Etiket', 'plaka_uyrugu', 'Detail_CarSegment', 'il', 'ilan_no', 'ilan_tarihi',
     'motor_hacmi', 'motor_gucu', 'cekis', 'ort._yakit_tuketimi',
     'yakit_deposu', 'takasa_uygun', 'kimden'], inplace = True, axis = 1
    )

# Az miktardaki tekrarlı veriyi düşür:
car_df.drop_duplicates(keep = 'first', inplace = True)

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
car_df.drop('Detail_Breadcrumb', inplace = True, axis = 1)

# Yalnızca 2000 ve 2023 model araçları tut:
car_df = car_df[(car_df['yil'] >= 2000) & (car_df['yil'] <= 2023)]

# Sütunları sırala:
column_order = ['marka', 'seri', 'model', 'yil', 'kilometre', 'vites_tipi', 'yakit_tipi', 'kasa_tipi',
                'renk', 'boya-degisen', 'Fiyat']

car_df = car_df[column_order]

########################################################################################################################
################################################ MISSING VALUES ########################################################
########################################################################################################################

for col in car_df.columns:
    if car_df[col].isnull().sum():
        print(col, 'has', car_df[col].isnull().sum(), 'NULL values')

car_df = car_df.dropna(axis = 0, how = 'any')

########################################################################################################################
################################################### DTYPES #############################################################
########################################################################################################################

# yil:
car_df['yil'] = car_df['yil'].astype('int64')

# kilometre:
car_df.loc[:, 'kilometre'] = car_df['kilometre'].str.replace('.', '')
car_df.loc[:, 'kilometre'] = car_df['kilometre'].str.replace(' km', '')
car_df.loc[:, 'kilometre'] = car_df['kilometre'].str.replace('Km', '')

car_df['kilometre'] = car_df['kilometre'].astype('int64')

# boya-degisen:
import re

car_df.info()
(car_df['boya-degisen'] == 'Belirtilmemiş').sum()  # 23218

# reset:
duplicate_df = car_df.copy()
car_df = duplicate_df.copy()

# boya-degisen = 'Belirtilmemiş' olan satırları hafızada tut:
car_df_boya_degisen_belirtilmemis = car_df[car_df['boya-degisen'] == 'Belirtilmemiş']


car_df.loc[:, 'boya'] = car_df['boya-degisen'].apply(lambda x: '0' if x == 'Belirtilmemiş' else x)
car_df.loc[:, 'boya'] = car_df['boya'].apply(lambda x: '0' if x == 'Tamamı orjinal' else x)
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


car_df.loc[:, 'degisen'] = car_df['boya-degisen'].apply(lambda x: '0' if x == 'Belirtilmemiş' else x)
car_df.loc[:, 'degisen'] = car_df['degisen'].apply(lambda x: '0' if x == 'Tamamı orjinal' else x)
car_df.loc[:, 'degisen'] = car_df['degisen'].apply(lambda x: '13' if x == 'Tamamı değişmiş' else x)
car_df.loc[:, 'degisen'] = car_df['degisen'].apply(
    lambda x: (re.findall(r'\d+', x.split(',')[0]))[0]
    if ',' in x else x
    )
car_df.loc[:, 'degisen'] = car_df['degisen'].apply(lambda x: '0' if 'boyalı' in x else x)
car_df.loc[:, 'degisen'] = car_df['degisen'].apply(
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
duplicate_df = car_df.copy()
car_df = duplicate_df.copy()


def outlier_threshold(dataframe, column, first_percent = 0.25, third_percent = 0.75):
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


def replacement_with_thresholds(dataframe, column, first_percent, third_percent):
    low_limit, up_limit = outlier_threshold(dataframe, column, first_percent, third_percent)
    dataframe.loc[(dataframe[column] < low_limit), column] = int(low_limit)
    dataframe.loc[(dataframe[column] > up_limit), column] = int(up_limit)


# 200 bin altı ve 20 milyon üstü araçları düşür:
car_df = car_df[(car_df['Fiyat'] >= 200000) & (car_df['Fiyat'] <= 20000000)]

# kilometre:
car_df.describe([0.01, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1]).T

# car_df['kilometre'].max()
# Out[62]: 93000000
# car_df['kilometre'].min()
# Out[63]: 0
low, up = outlier_threshold(car_df, 'kilometre', 0.5, 0.96)
car_df['kilometre'].max()
car_df['kilometre'].min()
replacement_with_thresholds(car_df, 'kilometre', 0.5, 0.96)

car_df.describe([0.01, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1]).T

# Fiyat:
low, up = outlier_threshold(car_df, 'Fiyat', 0.1, 0.9)
car_df['Fiyat'].max()
car_df['Fiyat'].min()
replacement_with_thresholds(car_df, 'Fiyat', 0.5, 0.9)

# veri setini kaydet:
car_df.to_csv('Step 2 (Prepare Data)/car_df_for_model.csv', index = False, mode = 'a')
