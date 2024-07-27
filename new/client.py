import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import requests

# Define the initial layers of the CNN (Client)
def create_initial_model():
    input_layer = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

# Define the final layers of the CNN (Client)
def create_final_model():
    input_layer = Input(shape=(128,))  # Adjusted shape to match server output
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(10, activation='softmax')(x)  # Adjusted output layer for classification
    model = Model(inputs=input_layer, outputs=x)
    return model

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)

# Create initial and final models
initial_model = create_initial_model()
final_model = create_final_model()

# Compile the final model
final_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training loop
for epoch in range(1):
    # Forward pass through initial model
    initial_activations = initial_model.predict(x_train)
    
    # Send activations to server
    response = requests.post("http://localhost:5001/process_activations", json={"activations": initial_activations.tolist()})
    server_activations = np.array(response.json()["activations"], dtype=np.float32)
    
    # Convert server activations to tensor
    server_activations = tf.convert_to_tensor(server_activations, dtype=tf.float32)

    # Train the final model with server activations
    with tf.GradientTape() as tape:
        predictions = final_model(server_activations, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_train, predictions)
    
    # Compute gradients
    grads = tape.gradient(loss, final_model.trainable_variables)
    
    # Apply gradients to update model
    final_model.optimizer.apply_gradients(zip(grads, final_model.trainable_variables))
    
    # Calculate training accuracy
    train_accuracy = np.mean(np.argmax(predictions.numpy(), axis=1) == y_train)
    print(f'Training Accuracy: {train_accuracy:.4f}')
    
    # Send model weights to server
    model_weights = final_model.get_weights()[:2]  # Send the first 2 weights
    response = requests.post("http://localhost:5001/update_model", json={"model_weights": [w.tolist() for w in model_weights]})
    print(response.json())

# Evaluate final model on test set
try:
    x_test_client = initial_model.predict(x_test)
    response = requests.post("http://localhost:5001/process_activations", json={"activations": x_test_client.tolist()})
    x_test_server = np.array(response.json()["activations"], dtype=np.float32)

    x_test_server = tf.convert_to_tensor(x_test_server, dtype=tf.float32)
    x_test_final = final_model.predict(x_test_server)
    
    test_accuracy = np.mean(np.argmax(x_test_final, axis=1) == y_test)
    print(f'Test Accuracy: {test_accuracy:.4f}')
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except (ValueError, KeyError) as e:
    print(f"Error decoding response: {e}")
