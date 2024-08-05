from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model

app = Flask(__name__)

# Define the middle layers of the model (Server)
def create_server_model():
    input_layer = Input(shape=(64,))
    x = Dense(128, activation='relu')(input_layer)
    x = BatchNormalization()(x) 
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x) 
    model = Model(inputs=input_layer, outputs=x)
    return model

# Create the server model
server_model = create_server_model()

# Define an optimizer for updating the server model's weights
optimizer = tf.keras.optimizers.Adam()

# Storage for activations received from the client
received_activations = None

@app.route('/process_activations', methods=['POST'])
def process_activations():
    global received_activations
    try:
        data = request.get_json()
        activations = np.array(data["activations"])
        received_activations = activations  # Store activations for later use
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
    global received_activations
    if received_activations is None:
        return jsonify({"error": "No activations received for backpropagation"}), 400
    
    try:
        data = request.get_json()
        grads = np.array(data["grads"])
        
        # Convert gradients to TensorFlow tensor
        grads_tensor = tf.convert_to_tensor(grads, dtype=tf.float32)

        # Create a TensorFlow tensor for the received activations
        received_activations_tensor = tf.convert_to_tensor(received_activations, dtype=tf.float32)

        # Use the activations to calculate gradients with respect to the server model
        with tf.GradientTape() as tape:
            tape.watch(received_activations_tensor)
            outputs = server_model(received_activations_tensor)
        
        # Calculate gradients of the server model with respect to the received activations
        grads_input = tape.gradient(outputs, received_activations_tensor, output_gradients=grads_tensor)
        
        # Update the server model with the gradients
        with tf.GradientTape() as tape:
            tape.watch(server_model.trainable_variables)
            server_outputs = server_model(received_activations_tensor)
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
    app.run(port=5002)
