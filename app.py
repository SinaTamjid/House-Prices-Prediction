#importing packages
from flask import Flask,request,render_template
import joblib
import pandas as pd

#importing model
model=joblib.load(r"House-Prices-Prediction/model/model.pkl")

app=Flask(__name__)

#prediction route
@app.route('/',methods=['POST'])
def predict():
 inputs={
        'bedrooms': [float(request.form['bedrooms'])],
        'bathrooms': [float(request.form['bathrooms'])],
        'sqft_living': [float(request.form['sqft_living'])],
        'sqft_lot': [float(request.form['sqft_lot'])],
        'floors': [float(request.form['floors'])],
        'waterfront': [float(request.form.get('waterfront', 0))],  # Default to 0 if not provided
        'view': [float(request.form.get('view', 0))],  # Default to 0 if not provided
        'condition': [float(request.form['condition'])],
        'grade': [float(request.form['grade'])],
        'sqft_above': [float(request.form.get('sqft_above', 0))],  # Default to 0 if missing
        'sqft_basement': [float(request.form.get('sqft_basement', 0))],
        'yr_built': [float(request.form['yr_built'])],
        'yr_renovated': [float(request.form.get('yr_renovated', 0))],  # Default to 0 if not renovated
        'lat': [float(request.form['lat'])],
        'long': [float(request.form['long'])],
        'sqft_living15': [float(request.form['sqft_living15'])],
        'sqft_lot15': [float(request.form['sqft_lot15'])]
    
 }
 df=pd.DataFrame(data=inputs)
 prediction=model.predict(df)

 return render_template('index.html',prediction=round(prediction[0],2))


if __name__=='__main__':
 app.run(debug=True,port=5000)