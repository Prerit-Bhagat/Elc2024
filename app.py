from flask import Flask, request, jsonify
import pickle
import numpy as np
import time
import hpelm
from hpelm import ELM # type: ignore
import joblib

app = Flask(__name__)

# Load the model
# with open('elm_model.pkl', 'rb') as file:
model = joblib.load('ekm_model.pkl')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            arr = request.form.get('arr')

            if not arr:
                return jsonify({'error': 'No array provided'}), 400

            # Clean and parse the input array
            cleaned_string = arr.strip('[]').strip()

            # old code
            # cleaned_string = cleaned_string.replace('e+', 'e')
            # arr_list = [float(num) for num in cleaned_string.split()]
            # arr_np = np.array(arr_list).reshape(1, 2048)


            # new code
            cleaned_string = ' '.join(cleaned_string.split())
            # Convert string to list of floats
            try:
                # Convert string to list of floats
                arr_list = [float(num) for num in cleaned_string.split()]
                # Reshape the list into a numpy array
                arr_np = np.array(arr_list).reshape(1, 2048)
            except ValueError:
                return jsonify({'error': 'Invalid array content'}), 400

            # Check if the array shape is correct
            if arr_np.shape != (1, 2048):
                return jsonify({'error': 'Input array shape should be (1, 2048)'}), 400

            # Debug print to check the type and content of arr_np
            # print(type(arr_np))
            # print(arr_np)

            # Format the numpy array to a string with commas
            arr_np_str = np.array2string(arr_np, separator=',').strip('[]')
            # Print formatted array string
            print('Formatted array string:', arr_np_str)

            # Perform prediction
            prediction = model.predict(arr_np).flatten().tolist()
            prediction=int(np.argmax(prediction))

            return jsonify({'result': prediction})

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    return 'Please send a POST request with the required parameters.'

if __name__ == '__main__':
    app.run(debug=True)
