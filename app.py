from flask import Flask,jsonify,render_template,request
import pickle

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")



@app.route('/predict', methods= ["post"])
def iris_pred():
     
    with open("model.pkl","rb") as model:
        ml_model = pickle.load(model)
    
    data = request.form
    SepalLengthCm = eval(data["SepalLengthCm"])
    SepalWidthCm = eval(data["SepalWidthCm"])
    PetalLengthCm = eval(data["PetalLengthCm"])
    PetalWidthCm = eval(data["PetalWidthCm"])

    result = ml_model.predict([[SepalLengthCm, SepalWidthCm,PetalLengthCm,PetalWidthCm]])
    
    if result[0] == 2:
        iris_flower = "Iris-virginica" 
    if result[0] == 0:
        iris_flower = "Iris-setosa"
    if result[0] == 1:
        iris_flower = "Iris-versicolor"  

    return iris_flower

@app.route('/welcome')
def index1():
    return "Welcome to Velocity"

if __name__ == "__main__":
    app.run(debug=True)