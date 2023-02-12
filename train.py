import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost
import pickle
from datetime import datetime

# Train data
train20 = pd.read_csv('data/train.csv') 
X = train20.iloc[:,1:-1] #X and y split for training
y = train20['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# My best model was XGB Regressor
xgb_final = xgboost.XGBRegressor(learning_rate = 0.1, max_depth = 5, n_estimators = 500, random_state=17) #"Final"
xgb_final.fit(X_train, y_train)

# Predict and scores
y_pred = xgb_final.predict(X_test) 
print("Train data MAE test:", mean_absolute_error(y_test, y_pred)) 
print("Train data MAPE test:", mean_absolute_percentage_error(y_test, y_pred))
print("Train data MSE test:", mean_squared_error(y_test, y_pred))
print("Train data RMSE test:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("The scoring for XGB Regressor with training data is:", xgb_final.score(X_test, y_test))
print("-"*20)

# Save model to pickle
now = datetime.now()
model_name = "model_" + now.strftime("%y%m%d%H%M%S")

with open('model/production/'+ model_name, 'wb') as archivo_salida: #save model
    pickle.dump(xgb_final, archivo_salida)