
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from feature_extractor import fingerprint_features

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("saved_models/model1.h5")


def preprocess_smile(smile):
    extracted_features = np.array(fingerprint_features(smile, size=2048))
    extracted_features = extracted_features.reshape(1, -1)
    return extracted_features


@app.route('/predict', methods=["POST", "GET"])
def predict():
    """
    To get structure of model [GET]

    example:
    http://127.0.0.1:5000/predict?smile=Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C

    output:
    smile prediction:	"1"
    """
    # Retrieve query parameters related to this request.
    smile = request.args.get('smile')

    x = preprocess_smile(smile)

    # Use the model to predict the class
    pred = model.predict(x)
    # the predicted class
    label = np.argmax(pred, axis=1)[0]

    # Create and send a response to the API caller
    return jsonify({'smile prediction': str(label)})


if __name__ == '__main__':
    app.run(debug=True)
