package de.noah.neuralnetwork;

import java.io.*;

public class NeuralNetwork implements Serializable {
    public final int[] SIZE;
    public final int LAST_LAYER;
    private final Neuron[][] NET;

    public NeuralNetwork(int... size) {
        this.SIZE = size;
        this.LAST_LAYER = SIZE.length - 1;
        this.NET = new Neuron[SIZE.length][];

        for (int layer = 0; layer < SIZE.length; layer++) {
            NET[layer] = new Neuron[SIZE[layer]];

            for (int i = 0; i < SIZE[layer]; i++) {
                NET[layer][i] = new Neuron();
            }
        }

        for (int layer = 1; layer < SIZE.length; layer++) {
            for (int neuron = 0; neuron < SIZE[layer]; neuron++) {
                Neuron out = NET[layer][neuron];

                for (int i = 0; i < SIZE[layer - 1]; i++) {
                    new Synapse(NET[layer - 1][i], out);
                }
            }
        }
    }

    public double[] propagate(final double[] input) {
        if (input.length != SIZE[0])
            throw new IllegalArgumentException("You must enter " + SIZE[0] + " input value(s)!");

        for (int inNeuron = 0; inNeuron < SIZE[0]; inNeuron++) {
            NET[0][inNeuron].output = input[inNeuron];
        }

        for (int layer = 1; layer < SIZE.length; layer++) {
            for (int neuron = 0; neuron < SIZE[layer]; neuron++) {
                final Neuron n = NET[layer][neuron];

                double d = n.bias;
                for (int synapse = 0; synapse < n.inSynapses.size(); synapse++) {
                    final Synapse s = n.inSynapses.get(synapse);

                    d += s.inNeuron.output * s.weight;
                }

                n.output = sigmoid(d);
            }
        }

        double[] output = new double[SIZE[LAST_LAYER]];
        for (int neuron = 0; neuron < SIZE[LAST_LAYER]; neuron++) {
            output[neuron] = NET[LAST_LAYER][neuron].output;
        }

        return output;
    }

    public void train(double[][] input, double[][] targets, int iterations, double learningRate) {
        for (int i = 0; i < iterations; i++) {
            final int rand = (int) (Math.random() * input.length);
            final double[] tdInput = input[rand];
            final double[] tdTarget = targets[rand];

            propagate(tdInput);

            // Last layer
            for (int neuron = 0; neuron < SIZE[LAST_LAYER]; neuron++) {
                final Neuron n = NET[LAST_LAYER][neuron];

                // Calculate error
                n.error = tdTarget[neuron] - n.output;

                // Calculate gradient
                double gradient = sigmoidDerivative(n.output) * n.error * learningRate;

                // Adjust weights
                for (int synapse = 0; synapse < n.inSynapses.size(); synapse++) {
                    final Synapse s = n.inSynapses.get(synapse);

                    s.weight += gradient * s.inNeuron.output;
                }

                // Adjust bias
                n.bias += gradient;
            }

            // Other layers
            for (int layer = LAST_LAYER - 1; layer > 0; layer--) {
                for (int neuron = 0; neuron < SIZE[layer]; neuron++) {
                    final Neuron n = NET[layer][neuron];

                    // Calculate error
                    n.error = 0.0D;
                    for (int synapse = 0; synapse < n.outSynapses.size(); synapse++) {
                        final Synapse s = n.outSynapses.get(synapse);

                        n.error += s.outNeuron.error * s.weight;
                    }

                    // Calculate gradient
                    double gradient = sigmoidDerivative(n.output) * n.error * learningRate;

                    // Adjust weights
                    for (int synapse = 0; synapse < n.inSynapses.size(); synapse++) {
                        final Synapse s = n.inSynapses.get(synapse);

                        s.weight += gradient * s.inNeuron.output;
                    }

                    // Adjust bias
                    n.bias += gradient;
                }
            }
        }
    }

    public void saveToFile(File file) throws IOException {
        ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(file));
        outputStream.writeObject(this);
        outputStream.flush();
        outputStream.close();
    }

    public static NeuralNetwork loadFromFile(File file) throws IOException, ClassNotFoundException {
        ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(file));
        NeuralNetwork network = (NeuralNetwork) inputStream.readObject();
        inputStream.close();

        return network;
    }

    private double sigmoid(double input) {
        return 1.0D / (1.0 + Math.pow(Math.E, -input));
    }

    private double sigmoidDerivative(double input) {
        return input * (1 - input);
    }

    public Neuron[][] getNet() {
        return NET;
    }
}