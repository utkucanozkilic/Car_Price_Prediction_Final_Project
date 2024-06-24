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

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1881)
pd.options.display.float_format = '{:.4f}'.format

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
car_df = pd.get_dummies(car_df, drop_first = True, dtype = int, dummy_na = False)  # 3010 satır

y = car_df['Fiyat']
X = car_df.drop('Fiyat', axis = 1)

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# from sklearn.model_selection import cross_val_score
#
# cv_scores = cross_val_score(
#     RandomForestRegressor(
#         max_depth = None,
#         min_samples_leaf = 2,
#         min_samples_split = 5,
#         n_estimators = 50,
#         random_state = 42,
#         n_jobs = -1,
#         verbose = 1,
#         ),
#     X_train, y_train, cv = 5
#     )
#
# # array([0.90610499, 0.79502823, 0.90381929, 0.8553828 , 0.80652892])

rf_model = RandomForestRegressor(
    max_depth = None,
    min_samples_leaf = 2,
    min_samples_split = 5,
    n_estimators = 50,
    random_state = 42,
    n_jobs = -1,
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
                    'renk', 'motor_hacmi', 'motor_gucu', 'cekis', 'kimden', 'boya', 'degisen', 'il']

# Özelliklerin önemini gruplayarak olarak görselleştir:
plot_grouped_importance(rf_model, X_train, original_columns)


# Model başarısına etkisi düşük olan sütunları düşür:
car_df = car_df_2.copy()
car_df.drop(['kimden', 'yakit_tipi', 'kasa_tipi', 'renk', 'il'], axis = 1, inplace = True)

scaler = MinMaxScaler()
car_df[num_columns] = scaler.fit_transform(car_df[num_columns])

# dummy_attributes:
car_df = pd.get_dummies(car_df, drop_first = True, dtype = int, dummy_na = False)  # 2893 satır

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
    n_jobs = -1,
    verbose = 2,
    )

rf_model.fit(X_train, y_train)

# Eğitim başarı:
y_train_preds = rf_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_preds))  # 114.274

# Test başarı:
y_test_preds = rf_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_preds))  # 161896

# cross_val
scores = cross_validate(rf_model, X, y, cv = 5, scoring = ['r2', 'neg_root_mean_squared_error'],
                        return_train_score = True)

print('Train RMSE Scores Mean:', -(scores['train_neg_root_mean_squared_error']).mean())  # 106618
print("Test RMSE Scores Mean:", -(scores['test_neg_root_mean_squared_error'].mean()))  # 238643

print("Test RMSE Scores:", -scores['test_neg_root_mean_squared_error'])
print('Test R2 Scores:', scores['test_r2'])

print('Train R2 Scores Mean:', scores['train_r2'].mean())  # 0.96
print('Test R2 Scores Mean:', scores['test_r2'].mean())  # 0.755

# GridSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]

grid_search = GridSearchCV(RandomForestRegressor(),
                           param_grid,
                           cv = 5,
                           scoring = 'neg_root_mean_squared_error',
                           return_train_score = True,
                           verbose = 2)

grid_search.fit(X_train, y_train)

grid_search.best_params_  # {'max_features': 4, 'n_estimators': 30}
grid_search.best_score_  # 183264

# Elastic Net:
car_df = car_df_2.copy()
car_df.drop(['kimden', 'yakit_tipi', 'kasa_tipi', 'renk', 'il'], axis = 1, inplace = True)

scaler = MinMaxScaler()
car_df[num_columns] = scaler.fit_transform(car_df[num_columns])

# dummy_attributes:
car_df = pd.get_dummies(car_df, drop_first = True, dtype = int, dummy_na = False)  # 2893 satır

y = car_df['Fiyat']
X = car_df.drop('Fiyat', axis = 1)

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Gradient Boosting modelini oluşturma
gb_model = GradientBoostingRegressor(random_state = 42)

# Modeli eğitim seti üzerinde eğitme
gb_model.fit(X_train, y_train)

# Eğitim seti üzerinde tahmin yapma
y_train_pred = gb_model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

# Test seti üzerinde tahmin yapma
y_test_pred = gb_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

# RMSE skorlarını yazdırma
print("Train RMSE: ", train_rmse)  # 124932
print("Test RMSE: ", test_rmse)  # 179507

# AdaBoost
car_df = car_df_2.copy()
car_df.drop(['kimden', 'yakit_tipi', 'kasa_tipi', 'renk', 'il'], axis = 1, inplace = True)

scaler = MinMaxScaler()
car_df[num_columns] = scaler.fit_transform(car_df[num_columns])

# dummy_attributes:
car_df = pd.get_dummies(car_df, drop_first = True, dtype = int, dummy_na = False)  # 2893 satır

y = car_df['Fiyat']
X = car_df.drop('Fiyat', axis = 1)

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

adaboost_model = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators = 50, random_state = 42)
adaboost_model.fit(X_train, y_train)

y_train_pred = adaboost_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # 4351

y_test_pred = adaboost_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # 182477

# AdaBoost GridSearch:
car_df = car_df_2.copy()
car_df.drop(['kimden', 'yakit_tipi', 'kasa_tipi', 'renk', 'il'], axis = 1, inplace = True)

scaler = MinMaxScaler()
car_df[num_columns] = scaler.fit_transform(car_df[num_columns])

# dummy_attributes:
car_df = pd.get_dummies(car_df, drop_first = True, dtype = int, dummy_na = False)  # 2893 satır

y = car_df['Fiyat']
X = car_df.drop('Fiyat', axis = 1)

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

param_grid = {
    'estimator__max_depth': [10, 15, 30, None]
    }

grid_search = GridSearchCV(AdaBoostRegressor(DecisionTreeRegressor(), random_state = 42),
                           param_grid,
                           cv = 5,
                           scoring = 'neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)

best_adaboost_model = grid_search.best_estimator_

y_train_pred = best_adaboost_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # 50591.3777310589

y_test_pred = best_adaboost_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # 178491.50474238794

grid_search.best_params_  # {'estimator__max_depth': 15}

# Feature Engineering
car_df = car_df_2.copy()
car_df.drop(['renk', 'il'], axis = 1, inplace = True)

car_df['marka_seri_model'] = car_df['marka'] + ' ' + car_df['seri'] + ' ' + car_df['model']

car_df.drop(['marka', 'seri', 'model'], axis = 1, inplace = True)
columns = ['marka_seri_model'] + [col for col in car_df if col != 'marka_seri_model']
car_df = car_df[columns]

# Veri setinde yalnızca 1 kez geçen örnekleri sil (Tabakalı örnekleme yapabilmek için):
class_count = car_df['marka_seri_model'].value_counts()
to_remove = class_count[class_count < 2].index
car_df = car_df[~car_df['marka_seri_model'].isin(to_remove)]

num_columns = ['yil', 'kilometre', 'motor_hacmi', 'motor_gucu', 'boya', 'degisen']
scaler = MinMaxScaler()
car_df[num_columns] = scaler.fit_transform(car_df[num_columns])
car_df.sort_values(by = 'Fiyat')
# dummy_attributes:
car_df = pd.get_dummies(car_df, drop_first = True, dtype = int, dummy_na = False)  # 2893 satır

y = car_df['Fiyat'] - 50000
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


original_columns = ['marka', 'model', 'seri', 'yil', 'kilometre', 'vites_tipi',
                    'motor_hacmi', 'motor_gucu', 'cekis', 'boya_degisen']

# Özelliklerin önemini gruplayarak olarak görselleştir:
plot_grouped_importance(model, X_train, original_columns)
