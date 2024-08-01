import pandas as pd
import joblib


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1881)
pd.options.display.float_format = '{:.4f}'.format

# Modeli ve scaler'ı yükle:
model = joblib.load('Step 3 (Modelling)/pipeline_model.pkl')


def new_car():
    new_car = pd.DataFrame(
        {
            'marka':      [input("Araç markası (örn. Ford): ")],
            'seri':       [input("Araç serisi (örn. Focus): ")],
            'model':      [input("Araç modeli (örn. 1.6 TDCi): ")],
            'yil':        [int(input("Üretim yılı (örn. 2015): "))],
            'kilometre':  [int(input("Kilometre bilgisi (örn. 80000): "))],
            'vites_tipi': [input("Vites tipi (örn. Manuel): ")],
            'yakit_tipi': [input("Yakıt tipi (örn. Dizel): ")],
            'kasa_tipi':  [input("Kasa tipi (örn. Sedan): ")],
            'renk':       [input("Araç rengi (örn. Beyaz): ")],
            'boya':       [int(input("Boyalı parça sayısı (örn. 1): "))],
            'degisen':    [int(input("Değişen parça sayısı (örn. 0): "))]
        }
    )
    return new_car


new_car = new_car()


def get_predict(sample):
    predicted_price = model.predict(sample)
    return predicted_price


predicted_price = get_predict(new_car)
print(f"Predicted Price: {predicted_price[0]}")