import numpy as np
import pickle
from flask import Flask, render_template, url_for, request
# from sklearn.linear_model import LinearRegression

app = Flask(__name__)
lr_model_1 = pickle.load(open(
    'E:\\FS-DS-iNeuron\\02-Live-Classes\\05-ML\\01. Linear-Regression\\Task-LR-02\\lr_Model_1.pkl', 'rb'))
lr_model_2 = pickle.load(open(
    'E:\\FS-DS-iNeuron\\02-Live-Classes\\05-ML\\01. Linear-Regression\\Task-LR-02\\lr_Model_2.pkl', 'rb'))
lasso = pickle.load(open(
    'E:\\FS-DS-iNeuron\\02-Live-Classes\\05-ML\\01. Linear-Regression\\Task-LR-02\\lasso_model.pkl', 'rb'))
ridge = pickle.load(open(
    'E:\\FS-DS-iNeuron\\02-Live-Classes\\05-ML\\01. Linear-Regression\\Task-LR-02\\ridgre_model.pkl', 'rb'))
elastic = pickle.load(open(
    'E:\\FS-DS-iNeuron\\02-Live-Classes\\05-ML\\01. Linear-Regression\\Task-LR-02\\elasticnet_model.pkl', 'rb'))


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predicetion():
    '''
    Model prediction function
    '''
    given_features = [float(x) for x in request.form.values()]
    given_features = [np.array(given_features)]

    res = lr_model_1.predict(given_features)
    print('LR-1\n',res)
    output_lr_1 = round(res[0], 3)

    res = lr_model_2.predict(given_features)
    print("LR-2\n",res)
    output_lr_2 = round(res[0], 3)

    res = lasso.predict(given_features)
    print("Lasso \n",res)
    output_lasso = round(res[0], 3)

    res = ridge.predict(given_features)
    print('Ridge\n',res)
    output_ridge = round(res[0], 3)

    res = elastic.predict(given_features)
    print("Elastic\n",res)
    output_elastic = round(res[0],3)

    return render_template('index.html', prediction_text=" Linear Model-1: {}\n Linear Model-2: {}\n LASSO Model: {}\n Ridge Model: {}\n ElasticNet Model: {}".format(output_lr_1, output_lr_2,output_lasso, output_ridge,output_elastic))

# def file_upload


if __name__ == "__main__":
    app.run(debug=True)
