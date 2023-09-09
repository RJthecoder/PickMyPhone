import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

render_model=pickle.load(open("model.pkl", 'rb'))

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")
 
@app.route("/predict",methods=["POST","GET"])
def predict():                                                                              
    inps=[int(request.form.get('price')),int(request.form.get('5g')),int(request.form.get('cores')),float(request.form.get('speed')),int(request.form.get('capacity')),int(request.form.get('charging')),int(request.form.get('batterywatt')),int(request.form.get('ram')),int(request.form.get('internal_storage')),float(request.form.get('size')),int(request.form.get('rr')),int(request.form.get('numberofrearcameras')),int(request.form.get('mp')),int(request.form.get('fmp')),int(request.form.get('height')),int(request.form.get('width')),int(request.form.get('processer'))]
    
    df=pd.read_csv("smartphones.csv")
    df=df.dropna()

    le = LabelEncoder()
    df['processor_brand_encode'] = le.fit_transform(df.processor_brand)

    x_data=df.drop(["model",'avg_rating','brand_name',"os","processor_brand","extended_memory_available"],axis=1)
    y_data=df["model"]

    
    model = SVC(kernel='linear', C=10)
    model.fit(x_data, y_data)
    
    output=model.predict([inps])
    
    return render_template('index.html', output="We pick the best phone for you: "+output[0],showans="(predicted phone is shown at the bottom of this page)")

if __name__ =="__main__":
    app.run(debug=True)

