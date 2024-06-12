import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1881)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.0f}'.format

# df_1 = pd.read_csv('car_1.csv')
# df_2 = pd.read_csv('car_2.csv')
# df_3 = pd.read_csv('car_3.csv')
# df_4 = pd.read_csv('car_4.csv')


# car_df = pd.concat([df_1, df_2, df_3, df_4], ignore_index = True)
car_df = pd.read_csv('../Step 1 (Get Adverts)/cars.csv')

# İlk bakışta kullanışsız olan sütunları düşür:
car_df.drop(
    ['Pagetype', 'Assignee', 'ilanTarihi', 'kategori', 'galeri_ID', 'il', 'ilanNo', 'photoCount',
     'Detail_ContactNumberCount ', 'Detail_MessageContactStatus', 'Detail_850Status',
     'Detail_PhoneContactStatus', 'Etiket', 'plaka_uyrugu', 'Detail_CarSegment'], inplace = True, axis = 1
    )

# Az miktardaki tekrarlı veriyi düşür:
car_df.drop_duplicates(keep = 'first', inplace = True)

# Tek bir null satır:
car_df.drop(index = 6617, inplace = True, axis = 0)

# # Segmenti NaN olanlardan yeni bir segment_na df'i oluştur:
# segment_na = car_df[car_df['Detail_CarSegment'].isna()]
#
# # Detail_Breadcrumb: Detail_CarSegment sözlüğü(mapping) işlemleri:
# segment_not_na = car_df[~car_df['Detail_CarSegment'].isna()]
#
# mapping = {}
#
# for index, row in segment_not_na.iterrows():
#     mapping[row['Detail_Breadcrumb']] = row['Detail_CarSegment']
#
# # Sözlükteki değerlerden car_df['Detail_CarSegment'] NaN olan değerleri doldurma:
# for index, row in segment_na.iterrows():
#     breadcrumb_value = row['Detail_Breadcrumb']
#     if breadcrumb_value in mapping:
#         car_df.at[index, 'Detail_CarSegment'] = mapping[breadcrumb_value]
#
# # Geriye kalan NaN değerlerin kontrolü:
# car_df['Detail_CarSegment'].isna().sum()  # > 1085 tane daha NaN değer
#
# car_df[car_df['Detail_CarSegment'].isna()]['Detail_Breadcrumb'].value_counts()

# İlan no anonimleştirme:
car_df['ilan_no'] = car_df['ilan_no'] + 783562

# Elektrikli araçları ayırma:
car_df_elektrik = car_df[car_df['yakit_tipi'] == 'Elektrik']

car_df = car_df[car_df['yakit_tipi'] != 'Elektrik']

# Toplam 20 tane ikinci el olmayan araçları düşür:
car_df.loc[car_df['arac_durumu'].isnull(), 'arac_durumu'] = 'İkinci El'
car_df['arac_durumu'].value_counts()
car_df = car_df[car_df['arac_durumu'] == 'İkinci El']

# Kullanışsız olan sütunları düşürme:
car_df.drop(
    ['ilan_tarihi', 'kasa_tipi', 'cekis', 'arac_durumu', 'takasa_uygun', 'motor_gucu_(kw)',
     'batarya_voltaji_(v)', 'menzil', 'sarj_suresi', 'pil_kapasitesi_(kwh)'], axis = 1, inplace = True
    )


# Null analizleri:
def isnull(df, column):
    if df[column].isnull().sum() == 0:
        return False
    return True


for column in car_df.columns:
    print(column, ', status of null:', isnull(car_df, column))

# km Nulls (3 tane ve aynı zamanda yakip_tipi de NaN):
car_df = car_df[~car_df['kilometre'].isnull()]

# renk Nulls:
car_df[car_df['renk'].isnull()]
car_df['renk'].value_counts()


def fill_na_colors(df):
    df.loc[car_df['ilan_no'] == 23420534, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 23338073, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23280377, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23217866, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 21810487, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 22859038, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23534214, 'renk'] = 'Yeşil'
    df.loc[car_df['ilan_no'] == 23345660, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 23553286, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23541281, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23552190, 'renk'] = 'Mavi'
    df.loc[car_df['ilan_no'] == 23543014, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 22622295, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23514285, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23094542, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 23243250, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23547694, 'renk'] = 'Kahverengi'
    df.loc[car_df['ilan_no'] == 23322584, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 22914960, 'renk'] = 'Mavi'
    df.loc[car_df['ilan_no'] == 23373437, 'renk'] = 'Mavi'
    df.loc[car_df['ilan_no'] == 23260534, 'renk'] = 'Mavi'
    df.loc[car_df['ilan_no'] == 23506048, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23557071, 'renk'] = 'Yeşil'
    df.loc[car_df['ilan_no'] == 23467729, 'renk'] = 'Lacivert'
    df.loc[car_df['ilan_no'] == 23546632, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23534902, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 23471715, 'renk'] = 'Kırmızı'
    df.loc[car_df['ilan_no'] == 23025364, 'renk'] = 'Kırmızı'
    df.loc[car_df['ilan_no'] == 23341706, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 23159359, 'renk'] = 'Mavi'
    df.loc[car_df['ilan_no'] == 23465615, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 21625525, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23138937, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23127536, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23347461, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23506257, 'renk'] = 'Mavi'
    df.loc[car_df['ilan_no'] == 23396611, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 23228420, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 22643876, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 23196839, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 22433475, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 23541540, 'renk'] = 'Beyaz'
    df.loc[car_df['ilan_no'] == 23524116, 'renk'] = 'Siyah'
    df.loc[car_df['ilan_no'] == 22940353, 'renk'] = 'Gri'
    df.loc[car_df['ilan_no'] == 23471586, 'renk'] = 'Beyaz'


fill_na_colors(car_df)

# motor_hacmi Nulls:
car_df[car_df['motor_hacmi'].isnull()]
car_df['motor_hacmi'].isnull().sum()  # 2087
car_df[car_df['motor_hacmi'].isnull()]['Detail_Breadcrumb'].value_counts()

mapping_motor_hacmi_dict = {}

motor_hacmi_not_na_df = car_df[~car_df['motor_hacmi'].isnull()]
motor_hacmi_na_df = car_df[car_df['motor_hacmi'].isnull()]

for index, row in motor_hacmi_not_na_df.iterrows():
    mapping_motor_hacmi_dict[row['Detail_Breadcrumb']] = row['motor_hacmi']

for index, row in motor_hacmi_na_df.iterrows():
    breadcrump_value = row['Detail_Breadcrumb']
    if breadcrump_value in mapping_motor_hacmi_dict:
        car_df.at[index, 'motor_hacmi'] = mapping_motor_hacmi_dict[breadcrump_value]

# '-' olan değerlerin motor_hacmi'ni Detail_Breadcrumb'tan türetme:
for index, row in car_df.iterrows():
    if row['motor_hacmi'] == '-':
        breadcrumb = row['Detail_Breadcrumb']

        if (re.search(r'\b\d\.\d\b', breadcrumb)) is not None:
            motor_hacmi_str = re.search(r'\b\d\.\d\b', breadcrumb).group()
            motor_hacmi_float = float(motor_hacmi_str)
            motor_hacmi_cc = motor_hacmi_float * 1000
            car_df.at[index, 'motor_hacmi'] = str(int(motor_hacmi_cc)) + ' cc'

# NaN olanların motor_hacmi değerlerini Detail_Breadcrumb'tan türetme:
for index, row in car_df.iterrows():
    if pd.isnull(row['motor_hacmi']):
        breadcrumb = row['Detail_Breadcrumb']

        if (re.search(r'\b\d\.\d\b', breadcrumb)) is not None:
            motor_hacmi_str = re.search(r'\b\d\.\d\b', breadcrumb).group()
            motor_hacmi_float = float(motor_hacmi_str)
            motor_hacmi_cc = motor_hacmi_float * 1000
            car_df.at[index, 'motor_hacmi'] = str(int(motor_hacmi_cc))

# Kalan değerleri el ile girme:
print(car_df[car_df['motor_hacmi'] == '-'].to_string())
car_df.at[41633, 'motor_hacmi'] = '1600 cc'
car_df.at[42403, 'motor_hacmi'] = '1995 cc'
car_df.at[59140, 'motor_hacmi'] = '1507 cc'
car_df.at[60031, 'motor_hacmi'] = '1600 cc'
car_df.at[66354, 'motor_hacmi'] = '2497 cc'
car_df.at[73586, 'motor_hacmi'] = '1396 cc'
car_df.at[84041, 'motor_hacmi'] = '1581 cc'

# Kalanı düşürdüm:
car_df = car_df[~car_df['motor_hacmi'].isnull()]

car_df.loc[:, 'motor_hacmi'] = car_df['motor_hacmi'].str.strip(' cc')
car_df.loc[:, 'motor_hacmi'] = car_df['motor_hacmi'].str.strip(' cm3\\')
car_df.loc[:, 'motor_hacmi'] = car_df['motor_hacmi'].str.strip(' cm3 ve üzeri')
car_df.loc[:, 'motor_hacmi'] = car_df['motor_hacmi'].str.strip(' cm3')

for index, row in car_df.iterrows():
    if row['motor_hacmi'] == '1401 - 1600':
        breadcrumb = row['Detail_Breadcrumb']

        if (re.search(r'\b\d\.\d\b', breadcrumb)) is not None:
            motor_hacmi_str = re.search(r'\b\d\.\d\b', breadcrumb).group()
            motor_hacmi_float = float(motor_hacmi_str)
            motor_hacmi_cc = motor_hacmi_float * 1000
            car_df.at[index, 'motor_hacmi'] = str(int(motor_hacmi_cc))

# car_df.loc[car_df['seri'] == 'Doğan', 'motor_hacmi'] = '1581'
# car_df.loc[car_df['seri'] == 'Murat', 'motor_hacmi'] = '1297'

car_df['motor_hacmi'] = car_df['motor_hacmi'].apply(lambda x: x.split(' - ')[-1] if '-' in x else x)
car_df['motor_hacmi'] = car_df['motor_hacmi'].astype(int)

# motor_gucu Nulls:
car_df['motor_gucu'].isnull().sum()  # 2522
car_df[car_df['motor_gucu'].isnull()]

mapping_motor_gucu_dict = {}

motor_gucu_not_na_df = car_df[~car_df['motor_gucu'].isnull()]
motor_gucu_na_df = car_df[car_df['motor_gucu'].isnull()]

for index, row in motor_gucu_not_na_df.iterrows():
    mapping_motor_gucu_dict[row['Detail_Breadcrumb']] = row['motor_gucu']

for index, row in motor_gucu_na_df.iterrows():
    breadcrump_value = row['Detail_Breadcrumb']
    if breadcrump_value in mapping_motor_gucu_dict:
        car_df.at[index, 'motor_gucu'] = mapping_motor_gucu_dict[breadcrump_value]

# 39 tane null:
car_df = car_df[~car_df['motor_gucu'].isnull()]
print(car_df['motor_gucu'].value_counts().to_string())
print(car_df[car_df['motor_gucu'] == '-'].to_string())
car_df.loc[78939, 'motor_gucu'] = '115'

for index, row in car_df.iterrows():
    if row['motor_gucu'] == '-':
        breadcrumb_value = row['Detail_Breadcrumb']
        if breadcrump_value in mapping_motor_gucu_dict:
            car_df.at[index, 'motor_gucu'] = mapping_motor_gucu_dict[breadcrump_value]

car_df['motor_gucu'] = car_df['motor_gucu'].str.strip(' hp')
car_df['motor_gucu'] = car_df['motor_gucu'].str.strip(' HP ve üzeri')
car_df['motor_gucu'] = car_df['motor_gucu'].str.strip(' HP\\')
car_df['motor_gucu'] = car_df['motor_gucu'].str.strip(' HP')
car_df['motor_gucu'] = car_df['motor_gucu'].apply(lambda x: x.split(' - ')[-1] if '-' in x else x)
car_df['motor_gucu'] = car_df['motor_gucu'].astype(int)

# # 3 tane Null değer kaldı. El ile doldurma:
# car_df[car_df['motor_gucu'].isnull()]
# car_df.at[18320, 'motor_gucu'] = '65 hp'
# car_df.at[18704, 'motor_gucu'] = '143 hp'
# car_df.at[74338, 'motor_gucu'] = '110 hp'

# Şimdilik ort._yakit_tuketimi'ni düşürdüm:
car_df.drop('ort._yakit_tuketimi', axis = 1, inplace = True)
# # ort._yakit_tuketimi Nulls:
# car_df['ort._yakit_tuketimi'].isnull().sum()  # 12401
#
# mapping_ort_yakit_tuketimi_dict = {}
#
# ort_yakit_tuketimi_not_na_df = car_df[~car_df['ort._yakit_tuketimi'].isnull()]
# ort_yakit_tuketimi_na_df = car_df[car_df['ort._yakit_tuketimi'].isnull()]
#
# for index, row in ort_yakit_tuketimi_not_na_df.iterrows():
#     mapping_ort_yakit_tuketimi_dict[row['Detail_Breadcrumb']] = row['ort._yakit_tuketimi']
#
# for index, row in ort_yakit_tuketimi_na_df.iterrows():
#     breadcrump_value = row['Detail_Breadcrumb']
#     if breadcrump_value in mapping_ort_yakit_tuketimi_dict:
#         car_df.at[index, 'ort._yakit_tuketimi'] = mapping_ort_yakit_tuketimi_dict[breadcrump_value]
#

# yakit_deposu Nulls:
car_df['yakit_deposu'].isnull().sum()  # 5557
mapping_yakit_deposu_dict = {}

yakit_deposu_not_na_df = car_df[~car_df['yakit_deposu'].isnull()]
yakit_deposu_na_df = car_df[car_df['yakit_deposu'].isnull()]

for index, row in yakit_deposu_not_na_df.iterrows():
    mapping_yakit_deposu_dict[row['Detail_Breadcrumb']] = row['yakit_deposu']

for index, row in yakit_deposu_na_df.iterrows():
    breadcrump_value = row['Detail_Breadcrumb']
    if breadcrump_value in mapping_yakit_deposu_dict:
        car_df.at[index, 'yakit_deposu'] = mapping_yakit_deposu_dict[breadcrump_value]

# 1750 değer kaldı. düşürdüm:
car_df = car_df[~car_df['yakit_deposu'].isnull()]

car_df['yakit_deposu'] = car_df['yakit_deposu'].apply(lambda x: x.split(' ')[0])
car_df['yakit_deposu'] = car_df['yakit_deposu'].astype(int)

# kilometre:
car_df[~car_df['kilometre'].str.contains('km')]
car_df.at[451, 'kilometre'] = '83000'

car_df['kilometre'] = car_df['kilometre'].str.strip(' km')
car_df['kilometre'] = car_df['kilometre'].str.replace('.', '')
car_df['kilometre'] = car_df['kilometre'].astype(int)

car_df['kilometre'].describe()
car_df = car_df[car_df['kilometre'] != 0]

print(car_df.sort_values(by = 'kilometre', ascending = False).head(100).to_string())

car_df.loc[car_df['ilan_no'] == 24923583, 'kilometre'] = 127000
car_df.loc[car_df['ilan_no'] == 24558717, 'kilometre'] = 139000
car_df.loc[car_df['ilan_no'] == 24694152, 'kilometre'] = 146000
car_df.loc[car_df['ilan_no'] == 24122986, 'kilometre'] = 150000

car_df = car_df[~car_df['kilometre'] < 1900000]

# boya-degisen
print(car_df['boya-degisen'].value_counts().to_string())
print(car_df['boya'].value_counts().to_string())
print(car_df['degisen'].value_counts().to_string())

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
car_df.loc[:, 'boya'] = car_df['boya'].astype(int)


car_df['degisen'] = car_df['degisen'].apply(lambda x: '0' if x == 'Tamamı orjinal' else x)
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

car_df['degisen'] = car_df['degisen'].astype(int)

car_df.drop('boya-degisen', axis = 1, inplace = True)

car_df['Fiyat'] = car_df['Fiyat'].astype(int)
car_df['yil'] = car_df['yil'].astype(int)

# Verinin son halini kaydet:
car_df.to_csv('car_last.csv', index = False, mode = 'a')




motor_hacmi_not_short_line_df = car_df[~(car_df['motor_hacmi'] == '-')]
motor_hacmi_short_line_df = car_df[car_df['motor_hacmi'] == '-']
mapping_motor_hacmi_dict = {}

for index, row in motor_hacmi_not_short_line_df.iterrows():
    mapping_motor_hacmi_dict[row['Detail_Breadcrumb']] = row['motor_hacmi']

for index, row in motor_hacmi_short_line_df.iterrows():
    breadcrump_value = row['Detail_Breadcrumb']
    if breadcrump_value in mapping_motor_hacmi_dict:
        car_df.at[index, 'motor_hacmi'] = mapping_motor_hacmi_dict[breadcrump_value]

car_df[car_df['motor_hacmi'] == '-']  # 2 örnek
car_df = car_df[~(car_df['motor_hacmi'] == '-')]