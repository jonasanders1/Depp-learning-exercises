'''
Neural Network Implementation from Scratch

Task Description:
1. Implement a neural network using only numpy (no deep learning frameworks)
2. Network Architecture:
   - Input layer: 2 neurons (for 2D input data)
   - Two hidden layers (configurable number of neurons)
   - Output layer: 2 neurons with softmax activation
   
3. Implementation Requirements:
   - Implement forward and backward propagation
   - Use modular design with separate classes/functions
   - Include weight initialization methods
   - Support different activation functions (ReLU, tanh, etc.)
   - Implement gradient descent optimization
   
4. Training and Evaluation:
   - Train on the "Two Moons" dataset:
     (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
   - Dataset size: 1000 training samples, 1000 test samples
   - Optional: Test on "Two Circles" dataset:
     (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)
     
5. Experimentation Requirements:
   - Compare performance with different:
     a) Activation functions (ReLU vs tanh vs sigmoid)
     b) Weight initialization methods (random, Xavier/Glorot, He)
     c) Learning rates (try range from 0.001 to 0.1)
     d) Hidden layer sizes (e.g., [8,4], [16,8], [32,16])
   - Document findings and best performing configuration
'''

from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NeuralNetwork( object ):
   
   def __init__(self, num_inputs=2, hidden_layers=[2,2], num_outputs=2):
      self.num_inputs = num_inputs
      self.hidden_layers = hidden_layers
      self.num_outputs = num_outputs
      # self.activation = activation


      # Layer representation architecture
      layers = [num_inputs] + hidden_layers + [num_outputs]
      print(layers)

      # Initialize weights
      weights = []
      for i in range(len(layers) - 1):
         w = np.random.rand(layers[i], layers[i + 1])
         weights.append(w)
         print(f"Weight for layer {i + 1} shape: {w.shape}")
      self.weights = weights


      # Initialize activations
      activations = []
      for i in range(len(layers)):
         a = np.zeros(layers[i])
         activations.append(a)
      self.activations = activations
  
      # Initialize derivatives
      derivatives = []
      for i in range(len(layers) - 1):  # The derivatives with respect to the weights
         d = np.zeros((layers[i], layers[i + 1]))
         derivatives.append(d)
      self.derivatives = derivatives


      # Initialize biases (Column Vector)
      biases = []
      for i in range(len(layers) - 1):
         b = np.zeros((layers[i + 1], 1))  # Make sure biases are (neurons, 1)
         biases.append(b)
         print(f"Bias for layer {i + 1} shape: {b.shape}")
      self.biases = biases
            
   def _softmax(self, x):
      exp_values = np.exp(x - np.max(x))  # For numerical stability
      return exp_values / np.sum(exp_values, axis=1, keepdims=True)


   # Modify forward propagation to use softmax in the output layer
   def forward_propagate(self, X):
      activations = X
      self.activations[0] = X

      for i, w in enumerate(self.weights[:-1]):  # Apply forward propagation to all layers except the last one
         z = np.dot(activations, w) + self.biases[i].T
         activations = self._sigmoid(z)  # Sigmoid for hidden layers
         self.activations[i + 1] = activations

      # For the output layer, use softmax activation
      output = np.dot(activations, self.weights[-1]) + self.biases[-1].T
      self.activations[-1] = self._softmax(output)  # Softmax activation in output layer

      return self.activations[-1]

         

   def gradient_descent(self, learning_rate):
      
      for i in range(len(self.weights)):
         weights = self.weights[i]
         derivatives  = self.derivatives[i]
         weights += derivatives * learning_rate
      
   

   def back_propagate(self, error, verbose=False):
    for i in reversed(range(len(self.derivatives))):
        activations = self.activations[i + 1]
        
        # For the last layer, just use the error directly (no sigmoid derivative)
        if i == len(self.derivatives) - 1:  # For the output layer, use softmax derivative
            delta = error  # no need to apply _sigmoid_derivative for softmax output
        else:
            delta = error * self._sigmoid_derivative(activations)
        
        # Reshape activations and delta for matrix multiplication
        current_activations = self.activations[i]
        
        # Reshape to column vectors for proper matrix multiplication
        current_activations = current_activations.reshape(-1, 1)
        delta = delta.reshape(1, -1)
        
        # Calculate the derivatives
        self.derivatives[i] = np.dot(current_activations, delta)
        
        # Propagate error backward
        if i > 0:  # No need to calculate error for input layer
            error = np.dot(delta, self.weights[i].T)
        
        if verbose:
            print(f"Layer {i}: Derivatives:\n{self.derivatives[i]}")
    
    return error



      
   
   def _sigmoid(self, x):
      y = 1.0 / (1 + np.exp(-x))
      return y
      
   def _sigmoid_derivative(self, x):
      return x * (1.0 - x)
   
   
   def mse(self, target, output):
      return np.mean((target - output) ** 2)
      
   
   def train(self, inputs, targets, epochs, learning_rate):
      
      for epoch in range(epochs):
         
         sum_error = 0
         
         for i in range(len(inputs)):
             input = inputs[i]
             # Convert target to one-hot encoding
             target = np.zeros(self.num_outputs)
             target[int(targets[i])] = 1
             
             # Forward Propagation
             output = self.forward_propagate(input)
             
             # Calculate the error
             error = target - output
             
             # Back Propagation
             self.back_propagate(error, verbose=False)
             
             # Apply gradient descent
             self.gradient_descent(learning_rate)
             
             # Calculate and append the error
             sum_error += self.mse(target, output)
               # report error 
         
         print(f"Epoch: {epoch},\n Error: {sum_error / len(inputs)}")
      
   
   def evaluate(self, X_test, y_test):
    # Get predictions
    predictions = []
    for x in X_test:
        output = self.forward_propagate(x)
        predicted_class = np.argmax(output)
        predictions.append(predicted_class)
    
    # Calculate accuracy
    correct_predictions = sum([1 for pred, true in zip(predictions, y_test) if pred == true])
    accuracy = correct_predictions / len(y_test)
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    
    # Plot points for each class with different colors
    for class_label in [0, 1]:
        # Get indices for true labels of this class
        mask = y_test == class_label
        
        # Plot points with true labels vs predictions
        plt.scatter(
            X_test[mask, 0],  # x-coordinate (first feature)
            X_test[mask, 1],  # y-coordinate (second feature)
            c=['red' if pred != class_label else 'green' for pred in np.array(predictions)[mask]],
            marker='o',
            label=f'Class {class_label}'
        )
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Classification Results\nGreen: Correct Prediction, Red: Incorrect Prediction")
    plt.legend()
    plt.show()
    
    # Report the evaluation results
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

   
   def visualize_dataset(self, X, y):
      plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
      plt.xlabel("Feature 1")
      plt.ylabel("Feature 2")
      plt.title("make_moons dataset")
      plt.show()
   
   



if __name__ == "__main__":
 
   # Moon data into samples and labels
   X, y = make_moons(1000, random_state=42)
   # Split into train and test data (20/80 split)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 


   model = NeuralNetwork()
   model.train(inputs=X_train, targets=y_train, epochs=50, learning_rate=0.01)
   
   model.evaluate(X_test, y_test)
   