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

# Storage for activations received from clients
received_activations_client1 = None
received_activations_client2 = None

@app.route('/process_activations_client1', methods=['POST'])
def process_activations_client1():
    global received_activations_client1
    try:
        data = request.get_json()
        activations = np.array(data["activations"])
        received_activations_client1 = activations  # Store activations for client 1
        server_activations = server_model.predict(activations)
        response = {
            "activations": server_activations.tolist()
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request from client 1: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_activations_client2', methods=['POST'])
def process_activations_client2():
    global received_activations_client2
    try:
        data = request.get_json()
        activations = np.array(data["activations"])
        received_activations_client2 = activations  # Store activations for client 2
        server_activations = server_model.predict(activations)
        response = {
            "activations": server_activations.tolist()
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request from client 2: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/backpropagate', methods=['POST'])
def backpropagate():
    global received_activations_client1, received_activations_client2
    if received_activations_client1 is None or received_activations_client2 is None:
        return jsonify({"error": "Not all activations received for backpropagation"}), 400
    
    try:
        data = request.get_json()
        grads_client1 = np.array(data["grads_client1"])
        grads_client2 = np.array(data["grads_client2"])
        
        # Convert gradients to TensorFlow tensors
        grads_client1_tensor = tf.convert_to_tensor(grads_client1, dtype=tf.float32)
        grads_client2_tensor = tf.convert_to_tensor(grads_client2, dtype=tf.float32)
        
        # Calculate gradients and update server model
        with tf.GradientTape() as tape:
            tape.watch(server_model.trainable_variables)
            output_client1 = server_model(received_activations_client1)
            output_client2 = server_model(received_activations_client2)
            loss = tf.reduce_mean(tf.losses.mean_squared_error(output_client1, output_client2))
        
        grads_model = tape.gradient(loss, server_model.trainable_variables)
        optimizer.apply_gradients(zip(grads_model, server_model.trainable_variables))
        
        # Aggregate gradients from both clients
        avg_grads = (grads_client1_tensor + grads_client2_tensor) / 2
        
        response = {
            "grads": avg_grads.numpy().tolist()
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error processing backpropagation request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
