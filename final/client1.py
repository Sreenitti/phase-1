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

# Define the initial layers of the model (Client)
def create_initial_model():
    input_layer = Input(shape=(4,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Define the final layers of the model (Client)
def create_final_model():
    input_layer = Input(shape=(64,))
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(32, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Load and preprocess data
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create initial and final models
initial_model = create_initial_model()
final_model = create_final_model()

# Compile the initial model with legacy optimizer
initial_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='mean_squared_error')

# Compile the final model
final_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Client identifier
client_id = "Client_1"  # Change to "Client_2" for the second client

# Training loop
for epoch in range(12):  # Adjust epochs as needed
    logging.info('%s - Epoch %d', client_id, epoch + 1)
    
    # Forward pass through initial model
    logging.info('%s - Performing forward pass through initial model...', client_id)
    initial_activations = initial_model(x_train, training=False)
    logging.info('%s - Initial model forward pass complete.', client_id)
    
    # Send activations to server
    logging.info('%s - Sending activations to server...', client_id)
    response = requests.post("http://localhost:5001/process_activations", json={"client_id": "Client_1", "activations": initial_activations.numpy().tolist()})
    if response.status_code == 202:
        logging.info('%s - Waiting for other client...', client_id)
        continue
    server_activations = np.array(response.json()["activations"])
    logging.info('%s - Server responded with processed activations.', client_id)

    # Train the final model using true labels
    logging.info('%s - Training the final model...', client_id)
    final_model.fit(server_activations, y_train, epochs=1, verbose=1)
    logging.info('%s - Final model training step complete.', client_id)
    
    with tf.GradientTape() as tape:
        predictions = final_model(server_activations, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_train, predictions)
    
    # Compute gradients for the final model
    gradients = tape.gradient(loss, final_model.trainable_variables)
    final_model.optimizer.apply_gradients(zip(gradients, final_model.trainable_variables))
    logging.info('%s - Final model gradients applied.', client_id)

    # Compute gradients of server activations
    logging.info('%s - Computing gradients of server activations...', client_id)
    with tf.GradientTape() as tape:
        tape.watch(initial_activations)
        predictions = final_model(initial_activations, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_train, predictions)
    
    grads_initial_activations = tape.gradient(loss, initial_activations)
    
    if grads_initial_activations is None:
        logging.error('%s - Error: Gradients for server activations are None', client_id)
    else:
        logging.info('%s - Server activations gradients computed.', client_id)
        grads_initial_activations = tf.convert_to_tensor(grads_initial_activations, dtype=tf.float32)
        
        # Send gradients of server activations back to server
        logging.info('%s - Sending gradients of server activations to server...', client_id)
        response = requests.post("http://localhost:5001/backpropagate", json={"grads": grads_initial_activations.numpy().tolist()})
        logging.info('%s - Server responded with gradients processing.', client_id)

        try:
            grads_initial_activations = np.array(response.json()["grads"])
            logging.info('%s - Received gradients from server for initial model.', client_id)
        except KeyError:
            logging.error("%s - Response does not contain 'grads' key", client_id)
            logging.error("%s - Full response: %s", client_id, response.json())
            continue
        
        # Perform backpropagation on initial model
        logging.info('%s - Performing backpropagation on initial model...', client_id)
        with tf.GradientTape() as tape:
            initial_outputs = initial_model(x_train, training=True)
        
        grads_initial_activations = tf.convert_to_tensor(grads_initial_activations, dtype=tf.float32)
        
        grads = tape.gradient(initial_outputs, initial_model.trainable_variables, output_gradients=grads_initial_activations)
        if initial_model.optimizer is not None:
            initial_model.optimizer.apply_gradients(zip(grads, initial_model.trainable_variables))
            logging.info('%s - Initial model gradients applied.', client_id)
        else:
            logging.error("%s - Error: Initial model optimizer is None", client_id)

# Evaluate final model on training set
logging.info('%s - Evaluating final model on training set...', client_id)
train_predictions = final_model.predict(server_activations)
train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == y_train)
logging.info('%s - Training Accuracy: %.4f', client_id, train_accuracy)

# Evaluate final model on test set
try:
    logging.info('%s - Evaluating final model on test set...', client_id)
    x_test_client = initial_model.predict(x_test)
    response = requests.post("http://localhost:5001/process_activations", json={"activations": x_test_client.tolist()})
    x_test_server = np.array(response.json()["activations"])
    x_test_final = final_model.predict(x_test_server)
    
    test_accuracy = np.mean(np.argmax(x_test_final, axis=1) == y_test)
    logging.info('%s - Test Accuracy: %.4f', client_id, test_accuracy)
except requests.exceptions.RequestException as e:
    logging.error("%s - Request failed: %s", client_id, e)
except (ValueError, KeyError) as e:
    logging.error("%s - Error decoding response: %s", client_id, e)
