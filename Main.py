import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

anemia_data = pd.read_csv("Data/anemia.csv")
X = anemia_data.drop(columns = 'Result', axis = 1)
Y = anemia_data['Result']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=(0.2),stratify=Y,random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)

GenderValue = input("Enter the Gender as 0 or 1: ")
HemoglobValue = input("Enter Hemoglobin level: ")
MCHValue = input("Enter MCH value: ")
MCHCValue = input("Enter MCHC value: ")
MCVValue = input("Enter MCV value: ")

input_data = (float(GenderValue), float(HemoglobValue), float(MCHValue), float(MCHCValue), float(MCVValue))
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
  print('The Person does not have a heart Disease')
else:
  print('The Person has Heart Disease')
