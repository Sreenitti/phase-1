import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import requests
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def generate_data():
    iris = load_iris()
    x_data = iris.data
    y_data = iris.target.reshape(-1, 1)
    
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_data = encoder.fit_transform(y_data)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test

def create_client_initial_model():
    input_layer = layers.Input(shape=(4,))
    client_layers = layers.Dense(128, activation=None)(input_layer)
    client_layers = layers.BatchNormalization()(client_layers)
    client_layers = layers.Activation('relu')(client_layers)
    intermediate_output_layer = layers.Dense(32, activation=None)(client_layers)
    intermediate_output_layer = layers.BatchNormalization()(intermediate_output_layer)
    intermediate_output_layer = layers.Activation('relu')(intermediate_output_layer)
    client_initial_model = keras.Model(inputs=input_layer, outputs=intermediate_output_layer)
    return client_initial_model


def create_client_final_model():
    input_layer = layers.Input(shape=(4,))  # Shape (batch_size, 32)
    final_output_layer = layers.Dense(3, activation='softmax')(input_layer)  # 3 classes for IRIS
    client_final_model = keras.Model(inputs=input_layer, outputs=final_output_layer)
    return client_final_model

def send_data_to_server(intermediate_outputs):
    server_url = 'http://127.0.0.1:5001/update_model'
    try:
        response = requests.post(server_url, json={
            'intermediate_outputs': intermediate_outputs.tolist()
        })
        response.raise_for_status()
        
        server_response = response.json()
        if 'processed_intermediate_outputs' in server_response:
            return np.array(server_response['processed_intermediate_outputs'])
        else:
            print("No processed outputs received from server.")
            return None
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to send data to server: {e}")
        return None

def train(client_initial_model, client_final_model, x_train, y_train):
    intermediate_outputs = client_initial_model.predict(x_train)
    processed_intermediate_outputs = send_data_to_server(intermediate_outputs)
    if processed_intermediate_outputs is not None:
        client_final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        client_final_model.fit(processed_intermediate_outputs, y_train, epochs=100, batch_size=32, verbose=2)
        
        # Calculate and print training accuracy
        train_predictions = client_final_model.predict(processed_intermediate_outputs)
        y_train_pred = np.argmax(train_predictions, axis=1)
        y_train_true = np.argmax(y_train, axis=1)
        train_accuracy = np.mean(y_train_pred == y_train_true)
        print(f"Training Accuracy: {train_accuracy:.4f}")

def test(client_initial_model, client_final_model, x_test, y_test):
    intermediate_outputs = client_initial_model.predict(x_test)
    processed_intermediate_outputs = send_data_to_server(intermediate_outputs)
    if processed_intermediate_outputs is not None:
        test_predictions = client_final_model.predict(processed_intermediate_outputs)
        
        # Calculate and print testing accuracy
        y_test_pred = np.argmax(test_predictions, axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(y_test_pred == y_test_true)
        print(f"Testing Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    client_initial_model = create_client_initial_model()
    client_final_model = create_client_final_model()
    
    x_train, y_train, x_test, y_test = generate_data()
    
    # Train the model
    train(client_initial_model, client_final_model, x_train, y_train)
    
    # Test the model
    test(client_initial_model, client_final_model, x_test, y_test)
