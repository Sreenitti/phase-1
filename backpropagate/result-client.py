import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to create a model with a specified number of dense layers
def create_model(input_shape, layers):
    input_layer = Input(shape=input_shape)
    x = input_layer
    for units in layers:
        x = Dense(units, activation='relu')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Function to perform the training and evaluation
def perform_training(client_layers, server_layers, epochs):
    # Load and preprocess data
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    # Create initial and final models
    initial_model = create_model(input_shape=(4,), layers=client_layers)
    final_model = create_model(input_shape=(64,), layers=[64, 32, 3])

    # Compile the initial model with legacy optimizer
    initial_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mean_squared_error')

    # Compile the final model
    final_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training loop
    for epoch in range(epochs):
        logging.info('Epoch %d', epoch + 1)
        
        # Forward pass through initial model
        initial_activations = initial_model(x_train, training=False)
        
        # Send activations to server
        response = requests.post("http://localhost:5001/process_activations", json={"activations": initial_activations.numpy().tolist()})
        server_activations = np.array(response.json()["activations"])

        # Train the final model using true labels
        final_model.fit(server_activations, y_train, epochs=1, verbose=0)
        
        with tf.GradientTape() as tape:
            predictions = final_model(server_activations, training=True)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_train, predictions)
        
        # Compute gradients for the final model
        gradients = tape.gradient(loss, final_model.trainable_variables)
        final_model.optimizer.apply_gradients(zip(gradients, final_model.trainable_variables))

        # Compute gradients of server activations
        with tf.GradientTape() as tape:
            tape.watch(initial_activations)
            predictions = final_model(initial_activations, training=False)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_train, predictions)
        
        grads_initial_activations = tape.gradient(loss, initial_activations)
        
        grads_initial_activations = tf.convert_to_tensor(grads_initial_activations, dtype=tf.float32)
        
        # Send gradients of server activations back to server
        response = requests.post("http://localhost:5001/backpropagate", json={"grads": grads_initial_activations.numpy().tolist()})

        grads_initial_activations = np.array(response.json()["grads"])

        # Perform backpropagation on initial model
        with tf.GradientTape() as tape:
            initial_outputs = initial_model(x_train, training=True)
        
        grads_initial_activations = tf.convert_to_tensor(grads_initial_activations, dtype=tf.float32)
        
        grads = tape.gradient(initial_outputs, initial_model.trainable_variables, output_gradients=grads_initial_activations)
        initial_model.optimizer.apply_gradients(zip(grads, initial_model.trainable_variables))

    # Evaluate final model on training set
    train_predictions = final_model.predict(server_activations)
    train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == y_train)

    # Evaluate final model on test set
    x_test_client = initial_model.predict(x_test)
    response = requests.post("http://localhost:5001/process_activations", json={"activations": x_test_client.tolist()})
    x_test_server = np.array(response.json()["activations"])
    x_test_final = final_model.predict(x_test_server)
    
    test_accuracy = np.mean(np.argmax(x_test_final, axis=1) == y_test)

    return train_accuracy, test_accuracy

# Analyze results for different configurations
def analyze_results():
    results = []
    epoch_cases = [5, 10]
    
    for epochs in epoch_cases:
        for client_layers in range(1, 3):  # Client layers from 1 to 2
            for server_layers in range(1, 3):  # Server layers from 1 to 2
                train_accuracy, test_accuracy = perform_training(
                    client_layers=[128] * client_layers, 
                    server_layers=[128] * server_layers, 
                    epochs=epochs
                )
                results.append({
                    'epochs': epochs,
                    'client_layers': client_layers,
                    'server_layers': server_layers,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy
                })
                logging.info(f"Epochs: {epochs}, Client Layers: {client_layers}, Server Layers: {server_layers}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    results = analyze_results()
    for result in results:
        print(result)
