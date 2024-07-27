from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

app = Flask(__name__)

# Define the middle layers of the CNN (Server)
def create_server_model():
    input_layer = Input(shape=(64,))
    x = Dense(128, activation='relu')(input_layer)  # Output shape should be (None, 128)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Create the server model
server_model = create_server_model()
optimizer = tf.keras.optimizers.Adam()

@app.route('/process_activations', methods=['POST'])
def process_activations():
    try:
        data = request.get_json()
        activations = np.array(data["activations"], dtype=np.float32)
        # Ensure activations shape is correct
        if activations.shape[1] != 64:
            raise ValueError(f"Expected shape (None, 64), but got {activations.shape}")
        server_activations = server_model.predict(activations)
        response = {
            "activations": server_activations.tolist()
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/update_model', methods=['POST'])
def update_model():
    try:
        data = request.get_json()
        model_weights = data["model_weights"]
        
        # Convert weights from list to numpy arrays
        model_weights = [np.array(w, dtype=np.float32) for w in model_weights]

        # Print expected vs received weight shapes
        temp_model = create_server_model()
        expected_weights = temp_model.get_weights()
        print("Expected weight shapes:", [w.shape for w in expected_weights])
        print("Received weight shapes:", [w.shape for w in model_weights])

        # Ensure model and weight lengths match
        if len(model_weights) != len(expected_weights[:2]):  # We expect the first 2 weights only
            raise ValueError(f"Expected {len(expected_weights[:2])} weights but received {len(model_weights)}")
        
        temp_model.set_weights(model_weights + expected_weights[2:])  # Combine received and existing weights
        
        # Update the server model weights
        server_model.set_weights(temp_model.get_weights())
        
        response = {
            "status": "Model updated successfully"
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
