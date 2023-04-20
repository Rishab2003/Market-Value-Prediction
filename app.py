from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))


app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
    g1=int(request.form.get('g1'))
    g2=int(request.form.get('g2'))
    g3=int(request.form.get('g3'))
    g4=int(request.form.get('g4'))
    g5=int(request.form.get('g5'))
    g6=int(request.form.get('g6'))
    g7=int(request.form.get('g7'))

    result =model.predict(np.array([g1,g2,g3,g4,g5,g6,g7]).reshape(1,7))
    name=int(result)
    return render_template('predict.html',value =name)
    



if __name__=='__main__':
    app.run(debug=True)