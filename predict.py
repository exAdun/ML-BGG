import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Variables to use
df = pd.read_csv('data/Test_datos_nuevos.csv') #open csvs
respuestas = pd.read_csv('data/respuestas.csv')

# Load model with pickle
with open('model/production/modelfinal', 'rb') as archivo_entrada:
    loaded_model = pickle.load(archivo_entrada) #open model


Prediccion = loaded_model.predict(df) #prediction

print("MAE test:", mean_absolute_error(respuestas, Prediccion))
print("MAPE test:", mean_absolute_percentage_error(respuestas, Prediccion))
print("MSE test:", mean_squared_error(respuestas, Prediccion))
print("RMSE test:", np.sqrt(mean_squared_error(respuestas, Prediccion)))

print("The score of the model is:", loaded_model.score(df, respuestas))

with open('data/predict.csv', 'wb') as archivo_salida:
    pickle.dump(loaded_model.predict(df), archivo_salida)