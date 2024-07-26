from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

app = Flask(__name__)

# Define the middle layers of the CNN (Server)
def create_server_model():
    input_layer = Input(shape=(64,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(64, activation='relu')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Create the server model
server_model = create_server_model()

@app.route('/process_activations', methods=['POST'])
def process_activations():
    try:
        data = request.get_json()
        activations = np.array(data["activations"])
        server_activations = server_model.predict(activations)
        response = {
            "activations": server_activations.tolist()
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
