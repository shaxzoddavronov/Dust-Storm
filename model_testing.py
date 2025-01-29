import numpy as np
import pandas as pd
import sklearn
import joblib

model_RF=joblib.load('model_RF.joblib')
scaler=joblib.load('scaler.joblib')

sample_dict={'Date': '2014-03-17',
 'Air Т average (°C) ': 9.0,
 'Air Т min (°C)': 1.5,
 'Air Т max (°C)': 17.2,
 'Humid f aver. (%)': 59.0,
 'Humid f min (%)': 51.0,
 'Eff Тe min (°C)': -4,
 'Eff Тe max (°C)': 14,
 'Eff Тe average (°C)': 14,
 'impulse (m/s)': 25,
 'Meteorological visibility (m)': 4000,
 'P aver. (gPa)': 1004.9,
 'P min (gPa)': 996.6,
 'P max (gPa)': 1012.9,
 'Po aver. (gPa)': 989.6,
 'Po min (gPa)': 981.3,
 'Po max (gPa)': 997.3,
 'Prec night (mm)': 0.0,
 'Prec day (mm)': 0.0,
 'Prec summary (mm)': 0.0,
 'Rain (day)': 0.0,
 'Snow (day)': 0.0,
 'fog (day)': 0.0,
 'Dust storm': 1.0, #dust storm darajasi, dust storm darajasi 0 dan katta bo'lsa dust_storm_label=1, bo'lmasa 0
 'dust_storm_label': 1, # 1-dust storm bo'lgan, 0-bo'lmagan
 'month': 3}
sample=pd.DataFrame(sample_dict,index=[0])
#cols_to_remove=~np.isin(df.columns,['Date','Dust storm','dust_storm_label'])
scaling_cols=['Air Т average (°C) ', 'Air Т min (°C)', 'Air Т max (°C)', 'Humid f aver. (%)',
            'Humid f min (%)', 'Eff Тe min (°C)', 'Eff Тe max (°C)', 'Eff Тe average (°C)', 'impulse (m/s)',
          'Meteorological visibility (m)', 'P aver. (gPa)', 'P min (gPa)', 'P max (gPa)', 'Po aver. (gPa)', 
            'Po min (gPa)', 'Po max (gPa)']
sample[scaling_cols]=scaler.transform(sample[scaling_cols])
X=sample.drop(columns=['Date','Dust storm','dust_storm_label']).values
y=sample['dust_storm_label'].values[0]

print(f"Prediction: {model_RF.predict(X)[0]}")
print(f"Actual: {y}")
