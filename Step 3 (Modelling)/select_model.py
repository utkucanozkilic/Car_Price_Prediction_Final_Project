import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1881)
pd.options.display.float_format = '{:.0f}'.format

car_df = pd.read_csv('Step 2 (Prepare Data)/car_last.csv')

# model eğitiminde işe yaramayacak sütunları düşür:
columns = ['Detail_Breadcrumb', 'ilan_no', 'ilan_tarihi']
car_df.drop(columns, axis = 1, inplace = True)

# Sütunları sırala:
column_order = ['marka', 'model', 'seri', 'yil', 'kilometre', 'vites_tipi', 'yakit_tipi', 'kasa_tipi',
                'renk', 'motor_hacmi', 'motor_gucu', 'cekis', 'kimden', 'boya', 'degisen', 'il', 'Fiyat']

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


cat_columns, num_columns, cat_but_car_columns = grab_col_names(car_df, cat_th = 14, car_th = 20, info = True)

cat_columns = cat_columns + cat_but_car_columns

# Scale:
# Normalizasyon öncesi hedef etiketi at (Fiyat):
num_columns.pop(-1)

scaler = MinMaxScaler(feature_range = (0, 1))
car_df['kilometre'] = scaler.fit_transform(car_df[['kilometre']])