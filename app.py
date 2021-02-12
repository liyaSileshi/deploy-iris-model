# Serve model as a flask application

import pickle
import numpy as np
from flask import Flask, request
print(pickle.format_version)
# model = None


# load model
model = pickle.load(open('iris_trained_model.pkl','rb'))

# def load_model():
#     print('model not loaded')
#     try:
#         global model
#         # model variable refers to the global variable
#         with open('iris_trained_model.pkl', 'rb') as f:
#             model = pickle.load(f)
#         print('model loaded')
#     except:
#         "error loading model" 
        
app = Flask(__name__)

@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        print('data',data)
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        prediction = model.predict(data)  # runs globally loaded model on the data
    return str(prediction[0])


if __name__ == '__main__':
    # load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)
