from flask import Flask ,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get input values
        age = int(request.form["age"])
        gender = int(request.form["gender"])
        height = int(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = int(request.form["ap_hi"])
        ap_lo = int(request.form["ap_lo"])
        cholesterol = int(request.form["cholesterol"])
        gluc = int(request.form["gluc"])
        smoke = int(request.form["smoke"])
        alco = int(request.form["alco"])
        active = int(request.form["active"])
        
        # Convert to numpy array
        features = np.array([[age, gender, height, weight, ap_hi, ap_lo,
                              cholesterol, gluc, smoke, alco,active]])

        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "High Risk of Heart Disease"
        else:
            result = "Low Risk of Heart Disease"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")
    
app.run(debug=True)
        