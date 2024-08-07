from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model

app = Flask(__name__)

# Create a global variable to store the current server model
server_model = None
optimizer = tf.keras.optimizers.Adam()  # Define a global optimizer

# Define the server model with varying layers
def create_server_model(num_dense_layers=1):
    input_layer = Input(shape=(64,))
    x = BatchNormalization()(input_layer)
    for _ in range(num_dense_layers):
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
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

@app.route('/backpropagate', methods=['POST'])
def backpropagate():
    global server_model
    if server_model is None:
        return jsonify({"error": "Server model has not been initialized"}), 400

    try:
        data = request.get_json()
        grads = np.array(data["grads"])
        activations = np.array(data["activations"])

        # Convert gradients to TensorFlow tensor
        grads_tensor = tf.convert_to_tensor(grads, dtype=tf.float32)

        # Create a TensorFlow tensor for the received activations
        activations_tensor = tf.convert_to_tensor(activations, dtype=tf.float32)

        # Use the activations to calculate gradients with respect to the server model
        with tf.GradientTape() as tape:
            tape.watch(activations_tensor)
            outputs = server_model(activations_tensor)

        # Calculate gradients of the server model with respect to the received activations
        grads_input = tape.gradient(outputs, activations_tensor, output_gradients=grads_tensor)

        # Update the server model with the gradients
        with tf.GradientTape() as tape:
            tape.watch(server_model.trainable_variables)
            server_outputs = server_model(activations_tensor)
            loss = tf.reduce_mean(tf.losses.mean_squared_error(outputs, server_outputs))

        grads_model = tape.gradient(loss, server_model.trainable_variables)
        optimizer.apply_gradients(zip(grads_model, server_model.trainable_variables))

        response = {
            "grads": grads_input.numpy().tolist()
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
