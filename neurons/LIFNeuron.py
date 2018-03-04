"""
Create a basic LIF neuron class

Based on http://neurdon.wpengine.com/2011/01/19/neural-modeling-with-python-part-1/

- Some changes to the original model:
- This class models the state of the neuron over time
- The values have been modified to make the simulation work better (not based on any biological plausability)
- Debugging has been added

Requires Python 3.x and Numpy
"""

import numpy as np

class LIFNeuron():
    def __init__(self, neuron_label = "LIF", debug=True):
        # Simulation config (may not all be needed!!)
        self.dt = 0.125  # simulation time step
        self.t_rest = 0  # initial refractory time

        # LIF Properties
        self.Vm = np.array([0])  # Neuron potential (mV)
        self.time = np.array([0])  # Time duration for the neuron (needed?)
        self.spikes = np.array([0])  # Output (spikes) for the neuron

        self.t = 0  # Neuron time step
        self.Rm = 1  # Resistance (kOhm)
        self.Cm = 10  # Capacitance (uF)
        self.tau_m = self.Rm * self.Cm  # Time constant
        self.tau_ref = 4  # refractory period (ms)
        self.Vth = 0.75  # = 1  #spike threshold
        self.V_spike = 1  # spike delta (V)
        self.type = 'Leaky Integrate and Fire'
        self.neuron_label = neuron_label
        self.debug = debug
        if self.debug:
            print ('LIFNeuron({}): Created {} neuron starting at time {}'.format(self.neuron_label, self.type, self.t))

    def spike_generator(self, neuron_input):
        # Create local arrays for this run
        duration = len(neuron_input)
        Vm = np.zeros(duration)  # len(time)) # potential (V) trace over time
        time = np.arange(int(self.t / self.dt), int(self.t / self.dt) + duration)
        spikes = np.zeros(duration)  # len(time))

        # Seed the new array with previous value of last run
        Vm[-1] = self.Vm[-1]

        if self.debug:
            print ('LIFNeuron.spike_generator({}).initial_state(input={}, duration={}, initial Vm={}, t={}, debug={})'
                   .format(self.neuron_label, neuron_input.shape, duration, Vm[-1], self.t, self.debug))

        for i in range(duration):
            if self.debug == 'INFO':
                print ('Index {}'.format(i))

            if self.t > self.t_rest:
                Vm[i] = Vm[i - 1] + (-Vm[i - 1] + neuron_input[i - 1] * self.Rm) / self.tau_m * self.dt

                if self.debug == 'INFO':
                    print(
                    'spike_generator({}): i={}, self.t={}, Vm[i]={}, neuron_input={}, self.Rm={}, self.tau_m * self.dt = {}'
                    .format(self.neuron_label, i, self.t, Vm[i], neuron_input[i], self.Rm, self.tau_m * self.dt))

                if Vm[i] >= self.Vth:
                    spikes[i] += self.V_spike
                    self.t_rest = self.t + self.tau_ref
                    if self.debug:
                        print ('*** LIFNeuron.spike_generator({}).spike=(self.t_rest={}, self.t={}, self.tau_ref={})'
                               .format(self.neuron_label, self.t_rest, self.t, self.tau_ref))

            self.t += self.dt

        # Save state
        self.Vm = np.append(self.Vm, Vm)
        self.spikes = np.append(self.spikes, spikes)
        self.time = np.append(self.time, time)

        if self.debug:
            print ('LIFNeuron.spike_generator({}).exit_state(Vm={} at iteration i={}, time={})'
                   .format(self.neuron_label, self.Vm.shape, i, self.t))

            # return time, Vm, output
