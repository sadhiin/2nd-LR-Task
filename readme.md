# Simple Linear Rigression Model
This is a task to build a simple Linear Rigression model and deploy on production using Flask API

## Prerequisites packages
- Numpy
- Sklearn
- Matplotlit
- Pickle
- Flask
## Project Structure
This project has `3` major segment :
- [model.ipynb](https://github.com/sadhiin/2nd-LR-Task/blob/main/model.ipynb) The Main code/notebook that contain machine learning model for selecting suitable feature, model training and prediction based on training ['data'](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv).

- [app.py](https://github.com/sadhiin/2nd-LR-Task/blob/main/app.py) - This contains Flask APIs that receives details features from user through GUI or API calls, computes the precited value based on the model and returns it.
- [templates](https://github.com/sadhiin/2nd-LR-Task/tree/main/templates) - This folder contains the HTML template to allow user to enter required details and displays the predicted result.

## Play with model
1. For this part make sure to install 2 more packages
    - Pandas
    - Statsmodels

    Execute all the the notebook cells for the traing and saving the trained model. In model section I try represent Linear Rigression model in different view of output. Presnet model in the notebook are `2 variasion of Linear Rigression model`, `Standadize data`, `Regularized Model: LASSO, Ridge, ElasticNet`.
2. After executing all cells this notebook save the trained model of all variations. Then run the app.py using below command to start Flask API
    `python app.py`
3. By default, flask will run on port 5000.
    `Navigate to URL http://localhost:5000` or `http://127.0.0.1:5000/`

For entering the feature values following [dataset page](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset).

> Made with Love💖
