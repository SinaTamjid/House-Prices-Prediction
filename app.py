from flask import Flask,request,render_template

app=Flask(__name__)

@app.route('/',methods=['POST'])
def predict():



 return 0

