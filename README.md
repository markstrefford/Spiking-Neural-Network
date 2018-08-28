# Spiking Neural Network Simulator
Basic SNN propogating spikes between layers of LIF neurons

This code is designed to demo the use of a Spiking Neural Network (SNN) to propogate spikes between layers of neurons. At this stage there is no learning involved, it's purely about propogating spikes between LIF neurons.

### Dependencies:

* Python 3
* Jupyter Notebooks
* Numpy
* Matplotlib
* Random

### Findings

* The model works and it is possible to see spike trains propogate between different layers in an SNN
* Only a simple model using feedforward has been applied here
* Different spike trains are evidenced depending on the offset of the applied stimulus
* There is no real view of biological plausability here, and this code base is unlikely to offer anything in terms of a real use-case
* It has been a useful experience to understand the mechanics of a basic spiking network, and to witness it in action

### Further Development

* Explore other neuron types (Hodkins-Huxley neurons for example)
* Explore how to develop a more complex layered model with feedforward, then with feedback too
* Explore the impact of inhibitory neurons (excitory neurons are modelled above)
* Explore how to integrate this with real stimuli (for example MNIST data)
* Explore how to integrate learning into this multi-layered model


