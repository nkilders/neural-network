# Neural Network

Neural Network library for Java

## Example

As an example the following code shows how to train a net to perform [XOR](https://en.wikipedia.org/wiki/XOR_gate).

```java
// Create a NeuralNetwork
NeuralNetwork net = new NeuralNetwork(2, 2, 1);

// Define training-data (inputs and targets)
double[][] inputs = new double[][] {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
};
double[][] targets = new double[][] {
        {0},
        {1},
        {1},
        {0}
};

// Train the net 100 times at a learningRate of 0.1
net.train(inputs, targets, 100, 0.1);

// Calculate the net's output for a set of input-values...
double[] output = net.propagate(input[0]);
// ... and print it to the console
System.out.println("Output: " + output[0]);
```