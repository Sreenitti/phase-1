from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

app = Flask(__name__)

# Define the middle layers of the model (Server)
def create_server_model():
    input_layer = Input(shape=(64,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(64, activation='relu')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Create the server model
server_model = create_server_model()

# Define an optimizer for updating the server model's weights
optimizer = tf.keras.optimizers.Adam()

# Storage for activations received from the clients
received_activations = []

@app.route('/process_activations', methods=['POST'])
def process_activations():
    global received_activations
    try:
        data = request.get_json()
        if 'activations' not in data:
            return jsonify({"error": "Missing 'activations' in request"}), 400

        activations = np.array(data["activations"])

        if activations.ndim == 1:
            activations = activations.reshape(1, -1)  # Ensure it is 2D

        if activations.shape[1] != 64:
            return jsonify({"error": "Activations shape mismatch"}), 400

        received_activations.append(activations)

        if len(received_activations) == 2:
            # Aggregate activations
            aggregated_activations = np.mean(np.vstack(received_activations), axis=0)
            aggregated_activations = aggregated_activations.reshape(1, -1)  # Ensure it is 2D
            
            # Ensure tensor is 2D
            tensor_activations = tf.convert_to_tensor(aggregated_activations, dtype=tf.float32)
            server_activations = server_model(tensor_activations)
            
            response = {
                "activations": server_activations.numpy().tolist()
            }
            received_activations.clear()  # Reset for the next round
            return jsonify(response)
        else:
            return jsonify({"status": "waiting for more activations"}), 202
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/backpropagate', methods=['POST'])
def backpropagate():
    try:
        data = request.get_json()
        if 'grads' not in data:
            return jsonify({"error": "Missing 'grads' in request"}), 400

        grads = np.array(data["grads"])

        if grads.ndim == 1:
            grads = grads.reshape(1, -1)  # Ensure it is 2D

        if grads.shape[1] != 64:
            return jsonify({"error": "Grads shape mismatch"}), 400

        grads_tensor = tf.convert_to_tensor(grads, dtype=tf.float32)

        # Ensure activations are correctly shaped
        aggregated_activations = np.mean(np.vstack(received_activations), axis=0)
        aggregated_activations = aggregated_activations.reshape(1, -1)  # Ensure it is 2D

        with tf.GradientTape() as tape:
            # Ensure activations are correctly shaped
            tensor_activations = tf.convert_to_tensor(aggregated_activations, dtype=tf.float32)
            server_outputs = server_model(tensor_activations)
            loss = tf.reduce_mean(tf.losses.mean_squared_error(server_outputs, server_outputs))

        grads_model = tape.gradient(loss, server_model.trainable_variables)
        optimizer.apply_gradients(zip(grads_model, server_model.trainable_variables))

        response = {
            "grads": grads_tensor.numpy().tolist()
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
