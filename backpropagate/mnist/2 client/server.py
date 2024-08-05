from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Create the server model
def create_server_model():
    input_layer = Input(shape=(64,))
    x = Dense(128, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

server_model = create_server_model()
optimizer = tf.keras.optimizers.legacy.Adam()

# Initialize variables to store activations
received_activations = {}
clients_ready = set()
activations_sent = {}  # Store which clients have received their activations

@app.route('/process_activations', methods=['POST'])
def process_activations():
    global received_activations, clients_ready, activations_sent, aggregated_activations
    data = request.json
    client_id = data['client_id']
    activations = np.array(data['activations'])
    clients_ready.add(client_id)
    
    # Check the shape of received activations
    logging.info(f'Received activations from client {client_id} with shape: {activations.shape}')
    
    if client_id not in received_activations:
        received_activations[client_id] = activations
    
    if len(clients_ready) == 2:  # Assuming 2 clients
        # Check shapes before aggregation
        for key, value in received_activations.items():
            logging.info(f'Received activations for client {key} with shape: {value.shape}')
        
        try:
            aggregated_activations = np.mean(list(received_activations.values()), axis=0)
            
            # Notify both clients with the aggregated activations
            response = jsonify({'activations': aggregated_activations.tolist()})
            return response  # Returning response to the current client

        except Exception as e:
            logging.error(f'Error during activation aggregation: {e}')
            return jsonify({'error': str(e)})
    else:
        return jsonify({'status': 'waiting for other client'})

received_gradients = {}
aggregated_activations = []
grads_ready = set()

@app.route('/backpropagate', methods=['POST'])
def backpropagate():
    global received_activations, received_gradients, grads_ready, clients_ready
    
    data = request.json
    client_id = data['client_id']
    gradients = np.array(data['grads'])
    grads_ready.add(client_id)

    logging.info(f'Received gradients from client {client_id} with shape: {gradients.shape}')
    
    if client_id not in received_gradients:
        received_gradients[client_id] = gradients

    if len(grads_ready) < 2:
        return jsonify({"status": "waiting for all gradients"}), 200

    try:
        received_activations= {}
        clients_ready= set()
        # Aggregate gradients
        grads = []
        for client_id in sorted(received_gradients.keys()):
            client_grads = np.array(received_gradients[client_id])
            grads.append(client_grads)
        aggregated_grads = np.mean(grads, axis=0)

        # Convert gradients to tensor
        grads_tensor = tf.convert_to_tensor(aggregated_grads, dtype=tf.float32)

        # Aggregate activations from clients
        aggregated_activations_tensor = tf.convert_to_tensor(aggregated_activations)

        # Perform forward pass with aggregated activations
        with tf.GradientTape() as tape:
            tape.watch(aggregated_activations_tensor)
            server_outputs = server_model(aggregated_activations_tensor)

        # Compute gradients with respect to the aggregated activations using aggregated gradients
        grads_input = tape.gradient(server_outputs, aggregated_activations_tensor, output_gradients=grads_tensor)

        # Perform backpropagation for server model
        with tf.GradientTape() as tape:
            tape.watch(server_model.trainable_variables)
            server_outputs = server_model(aggregated_activations_tensor)
            # Use the model outputs for loss calculation
            # Assuming we have aggregated activations as the target (or some suitable loss calculation)
            loss = tf.reduce_mean(tf.losses.mean_squared_error(aggregated_activations_tensor, server_outputs))

        grads_model = tape.gradient(loss, server_model.trainable_variables)
        optimizer.apply_gradients(zip(grads_model, server_model.trainable_variables))

        response = {"grads": grads_input.numpy().tolist()}
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5002)
