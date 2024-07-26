import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import logging

app = Flask(__name__)

# Shared model and gradients tracking
server_model = None
gradients_list = []
client_gradients_received = {'client_1': False, 'client_2': False}


def create_server_model():
    input_layer = layers.Input(shape=(32,))
    server_layers = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
    output_layer = layers.Dense(3, activation='softmax')(server_layers)
    server_model = keras.Model(inputs=input_layer, outputs=output_layer)
    return server_model

server_model = create_server_model()

def deserialize_gradients(gradients_serializable):
    try:
        gradients = [np.array(grad, dtype=np.float32) for grad in gradients_serializable]
        # Log gradient shapes
        for idx, grad in enumerate(gradients):
            print(f"Deserialized gradient {idx} shape: {grad.shape}")
        return gradients
    except Exception as e:
        print(f"Error deserializing gradients: {e}")
        raise


@app.route('/update_model', methods=['POST'])
def update_model():
    global server_model, gradients_list, client_gradients_received

    try:
        data = request.get_json()
        client_id = data.get('client_id')
        gradients_serializable = data.get('gradients')

        logging.info(f"Received request from {client_id}")

        if client_id not in ['client_1', 'client_2']:
            return jsonify({'error': 'Invalid client ID'}), 400

        if gradients_serializable is None:
            return jsonify({'error': 'Invalid input data'}), 400

        gradients = deserialize_gradients(gradients_serializable)
        logging.info(f"Deserialized gradients from {client_id}: {gradients}")
 
        client_gradients_received[client_id] = True
        gradients_list.append(gradients)

        logging.info(f"Client gradients received status: {client_gradients_received}")

        if client_gradients_received['client_1'] and client_gradients_received['client_2']:
            # Aggregate gradients
            aggregated_gradients = [np.mean([grad[i] for grad in gradients_list], axis=0) for i in range(len(gradients_list[0]))]

            # Log aggregated gradients shapes
            for idx, grad in enumerate(aggregated_gradients):
                print(f"Aggregated gradient {idx} shape: {grad.shape}")

            # Log server model variable shapes before applying gradients
            for idx, var in enumerate(server_model.trainable_variables):
                print(f"Server model variable {idx} shape: {var.shape}")

            # Apply gradients
            optimizer = tf.keras.optimizers.Adam()  # Use the same optimizer as the clients
            optimizer.apply_gradients(zip(aggregated_gradients, server_model.trainable_variables))

            # Reset for next round
            client_gradients_received = {'client_1': False, 'client_2': False}
            gradients_list = []

            response = {'message': 'Model updated successfully'}
        else:
            response = {'message': 'Waiting for the other client'}

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/test_model', methods=['POST'])
def test_model():
    global server_model
    
    try:
        data = request.get_json()
        test_data = data.get('test_data')
        
        if test_data is None:
            return jsonify({'error': 'Invalid input data'}), 400
        
        test_data = np.array(test_data, dtype=np.float32)
        
        # Check if test_data is a valid shape
        if len(test_data.shape) < 2:
            raise ValueError("Test data must have at least two dimensions")
        
        predictions = server_model.predict(test_data)
        
        response = {
            'predictions': predictions.tolist()
        }
        return jsonify(response)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
