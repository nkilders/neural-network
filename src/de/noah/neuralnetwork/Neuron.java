package de.noah.neuralnetwork;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Neuron implements Serializable {
    public double output;
    public double bias;
    public double error;

    public List<Synapse> inSynapses;
    public List<Synapse> outSynapses;

    public Neuron() {
        this.output = 0.0D;
        this.bias = 0.5D - Math.random();
        this.error = 0.0D;
        this.inSynapses = new ArrayList<>();
        this.outSynapses = new ArrayList<>();
    }
}