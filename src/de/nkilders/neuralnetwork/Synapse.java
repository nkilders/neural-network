package de.nkilders.neuralnetwork;

import java.io.Serializable;

/**
 * @author Noah Kilders
 */
public class Synapse implements Serializable {
    public Neuron inNeuron;
    public Neuron outNeuron;
    public double weight;

    public Synapse(Neuron inNeuron, Neuron outNeuron) {
        this(inNeuron, outNeuron, 0.5D - Math.random());
    }

    public Synapse(Neuron inNeuron, Neuron outNeuron, double weight) {
        inNeuron.outSynapses.add(this);
        outNeuron.inSynapses.add(this);

        this.inNeuron = inNeuron;
        this.outNeuron = outNeuron;
        this.weight = weight;
    }
}