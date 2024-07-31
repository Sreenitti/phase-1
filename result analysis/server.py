from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

app = Flask(__name__)

# Create a global variable to store the current server model
server_model = None

# Define the server model with varying layers
def create_server_model(num_dense_layers=1):
    input_layer = Input(shape=(64,))
    x = input_layer
    for _ in range(num_dense_layers):
        x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

@app.route('/process_activations', methods=['POST'])
def process_activations():
    global server_model

    try:
        data = request.get_json()
        num_dense_layers = data.get("num_dense_layers", 1)  # Get number of dense layers from the request
        activations = np.array(data["activations"])

        # Initialize the server model with the requested number of dense layers
        server_model = create_server_model(num_dense_layers)
        
        # Predict server activations
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
