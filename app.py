import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv("smartphones.csv")

df=df.dropna()

le = LabelEncoder()
df['processor_brand_encode'] = le.fit_transform(df.processor_brand)

x_data=df.drop(["model",'avg_rating','brand_name',"os","processor_brand","extended_memory_available"],axis=1)
y_data=df["model"]

X_train,X_test,Y_train,Y_test = train_test_split(x_data,y_data,test_size=0.2)

new_model=KNeighborsClassifier(leaf_size=38,n_neighbors=3)
new_model.fit(X_train,Y_train)

with open("model.pkl",'wb') as files:
  pickle.dump(new_model,files)