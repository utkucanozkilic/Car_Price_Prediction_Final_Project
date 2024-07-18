import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1881)
pd.options.display.float_format = '{:.4f}'.format

car_df = pd.read_csv('Step 2 (Prepare Data)/car_df_for_model.csv')

# Sütunları sırala:
column_order = ['marka', 'model', 'seri', 'yil', 'kilometre', 'vites_tipi', 'yakit_tipi', 'kasa_tipi',
                'renk', 'boya', 'degisen', 'Fiyat']

car_df = car_df[column_order]

# Özellik ve hedef değişkenlerini ayır:
X = car_df.drop('Fiyat', axis = 1)
y = car_df['Fiyat']


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


cat_columns, num_columns, cat_but_car_columns = grab_col_names(car_df, cat_th = 5, car_th = 20, info = True)

cat_columns = cat_columns + cat_but_car_columns
num_columns.remove('Fiyat')  # Hedef değişkeni at

############################################### PIPELINE ############################################################

# Sayısal sütunlar için pipeline:
num_pipeline = Pipeline(
    [
        ('scaler', MinMaxScaler())
        ]
    )

# Kategorik sütunlar için pipeline:
cat_pipeline = Pipeline(
    [
        ('onehot', OneHotEncoder(handle_unknown = 'ignore'))  # Eğitim veri setinde olmayan kategoriler göz ardı edilir.
        ]
    )

# Sayısal ve kategorik sütunları birleştir:
preprocessor = ColumnTransformer(
    [
        ('num', num_pipeline, num_columns),
        ('cat', cat_pipeline, cat_columns)
        ]
    )

# Ana pipeline:
model_pipeline = Pipeline(
    [
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(verbose = 1, random_state = 42))
        ]
    )

# Veri setini ayır:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Modeli eğit
model_pipeline.fit(X_train, y_train)

# Tahmin yap
y_pred_train = model_pipeline.predict(X_train)
y_pred_test = model_pipeline.predict(X_test)

# Model performansını değerlendir
print('Train RMSE:', mean_squared_error(y_train, y_pred_train, squared = False))
print('Test RMSE:', mean_squared_error(y_test, y_pred_test, squared = False))

# Model ve scaler değerlerini kaydet:
joblib.dump(model_pipeline, 'Step 3 (Modelling)/pipeline_model.pkl')
joblib.dump(preprocessor, 'Step 3 (Modelling)/preprocessor.pkl')
