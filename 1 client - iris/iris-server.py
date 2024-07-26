from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__)

def create_server_model():
    input_layer = layers.Input(shape=(32,))  # Expecting shape (batch_size, 32) from client
    server_layers = layers.Dense(64, activation='relu')(input_layer)
    output_layer = layers.Dense(4, activation='relu')(server_layers)  # 4 classes for IRIS
    server_model = keras.Model(inputs=input_layer, outputs=output_layer)
    return server_model

server_model = create_server_model()

@app.route('/update_model', methods=['POST'])
def update_model():
    global server_model
    
    try:
        data = request.get_json()
        intermediate_outputs_serializable = data.get('intermediate_outputs')
        
        if intermediate_outputs_serializable is None:
            return jsonify({'error': 'Invalid input data'}), 400
        
        intermediate_outputs = np.array(intermediate_outputs_serializable, dtype=np.float32)

        # Dummy targets for simulation
        dummy_labels = np.zeros((intermediate_outputs.shape[0], 4), dtype=np.float32)  # 4 classes for IRIS
        server_model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Train server model with dummy labels
        server_model.fit(intermediate_outputs, dummy_labels, epochs=100, verbose=2)

        # Get processed intermediate outputs
        processed_intermediate_outputs = server_model.predict(intermediate_outputs)
        
        response = {
            'processed_intermediate_outputs': processed_intermediate_outputs.tolist()
        }
        return jsonify(response)
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
