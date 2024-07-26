import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import requests
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time

def generate_data():
    iris = load_iris()
    x_data = iris.data
    y_data = iris.target.reshape(-1, 1)
    
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_data = encoder.fit_transform(y_data)
    
    # Split the data down the middle
    split_index = len(x_data) // 2
    x_half = x_data[:split_index]
    y_half = y_data[:split_index]

    x_train, x_test, y_train, y_test= train_test_split(x_half, y_half, test_size=0.2, random_state=42)
    
    return x_train, y_train, x_test, y_test

def create_client_initial_model():
    input_layer = layers.Input(shape=(4,))
    client_layers = layers.Dense(128, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
    client_layers = layers.BatchNormalization()(client_layers)
    client_layers = layers.Activation('relu')(client_layers)
    intermediate_output_layer = layers.Dense(32, activation=None)(client_layers)
    intermediate_output_layer = layers.BatchNormalization()(intermediate_output_layer)
    intermediate_output_layer = layers.Activation('relu')(intermediate_output_layer)
    client_initial_model = keras.Model(inputs=input_layer, outputs=intermediate_output_layer)
    return client_initial_model

def create_client_final_model():
    input_layer = layers.Input(shape=(32,))  # Shape (batch_size, 32)

    # Add a Dense layer followed by BatchNormalization and Activation
    dense_layer = layers.Dense(64)(input_layer)  # Adding a Dense layer
    batch_norm = layers.BatchNormalization()(dense_layer)  # Batch Normalization
    activated = layers.Activation('relu')(batch_norm)  # Activation function
    
    # Output layer
    final_output_layer = layers.Dense(3, activation='softmax')(activated)  # 3 classes for IRIS
    
    client_final_model = keras.Model(inputs=input_layer, outputs=final_output_layer)
    return client_final_model

def send_gradients_to_server(client_id, gradients):
    server_url = 'http://127.0.0.1:5001/update_model'
    try:
        # Convert each gradient numpy array to a list
        gradients_as_lists = [grad.tolist() for grad in gradients]
        response = requests.post(server_url, json={
            'client_id': client_id,
            'gradients': gradients_as_lists
        })
        response.raise_for_status()
        server_response = response.json()
        
        if server_response['message'] == 'Waiting for the other client':
            print("Waiting for the other client. Pausing...")
            # Wait until the server indicates it's ready
            while True:
                response = requests.post(server_url, json={
                    'client_id': client_id,
                    'gradients': gradients_as_lists
                })
                response.raise_for_status()
                server_response = response.json()
                if server_response['message'] != 'Waiting for the other client':
                    break
                time.sleep(5)  # Wait before retrying
        else:
            print(server_response['message'])
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to send gradients to server: {e}")

def train(client_initial_model, client_final_model, x_train, y_train, client_id):
    # Generate intermediate outputs
    intermediate_outputs = client_initial_model.predict(x_train)
    
    # Print shapes and types for debugging
    print(f"x_train shape: {x_train.shape}, type: {type(x_train)}")
    print(f"intermediate_outputs shape: {intermediate_outputs.shape}, type: {type(intermediate_outputs)}")
    print(f"y_train shape: {y_train.shape}, type: {type(y_train)}")
    
    # Compile the final model
    client_final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    client_final_model.fit(intermediate_outputs, y_train, epochs=100, batch_size=32, verbose=2)
    
    # Obtain predictions
    predictions = client_final_model.predict(intermediate_outputs)
    print(f"predictions shape: {predictions.shape}, type: {type(predictions)}")

    # Calculate training accuracy
    y_train_pred = np.argmax(predictions, axis=1)
    y_train_true = np.argmax(y_train, axis=1)
    train_accuracy = np.mean(y_train_pred == y_train_true)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    # Compute gradients
    with tf.GradientTape() as tape:
        # Ensure intermediate_outputs are tensor
        intermediate_outputs_tf = tf.convert_to_tensor(intermediate_outputs, dtype=tf.float32)
        predictions = client_final_model(intermediate_outputs_tf, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y_train, predictions)
    
    gradients = tape.gradient(loss, client_final_model.trainable_variables)
    gradients = [grad.numpy() for grad in gradients]
    
    # Send gradients to server
    send_gradients_to_server(client_id, gradients)

def test(client_initial_model, x_test):
    intermediate_outputs = client_initial_model.predict(x_test)
    server_url = 'http://127.0.0.1:5001/test_model'
    try:
        response = requests.post(server_url, json={
            'test_data': intermediate_outputs.tolist()
        })
        response.raise_for_status()
        server_response = response.json()
        predictions = np.array(server_response['predictions'])
        return predictions
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to send test data to server: {e}")
        return None

if __name__ == "__main__":
    client_initial_model = create_client_initial_model()
    client_final_model = create_client_final_model()
    
    x_train, y_train, x_test, y_test = generate_data()
    
    # Choose client_id based on which client this is
    client_id = 'client_1'  # Change to 'client_2' for the second client
    
    # Train the model
    train(client_initial_model, client_final_model, x_train, y_train, client_id)
    
    # Test the model
    predictions = test(client_initial_model, x_test)
    if predictions is not None:
        y_test_pred = np.argmax(predictions, axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(y_test_pred == y_test_true)
        print(f"Testing Accuracy: {test_accuracy:.4f}")
