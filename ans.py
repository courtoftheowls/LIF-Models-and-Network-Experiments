# (C) Jeff Orchard, 2025
# University of Waterloo

import numpy as np
from neuron_models import *


#================================
#
#  LIFNeuron class
#
#================================
class LIFNeuron(SpikingNeuron):

    def __init__(self, Tau_m=0.02, Tau_ref=0.002, Tau_s=0.05, label=None):
        '''
         lif = LIFNeuron(Tau_m=0.02, Tau_ref=0.002, Tau_s=0.05, label=None)
        
         Constructor for LIFNeuron class
        
         Inputs:
           Tau_m   membrane time constant, in seconds (s)
           Tau_ref refractory period (s)
           Tau_s   synaptic time constant (s)
           label   optional string
        '''
        # The base class SpikingNeuron sets:
        #   self.t
        #   self.label
        #   self.__spikes
        #   self.axon
        #   self.__Incoming_spikes_buffer
        super().__init__(label=label)

        self.tau_m = Tau_m      # membrane time constant
        self.tau_ref = Tau_ref  # refractory period
        self.tau_s = Tau_s      # synaptic time constant

        self.__v = 0.             # sub-threshold membrane potential (voltage)
        self.__s = 0.             # post-synaptic current (PSC)
        self.__dvdt = 0.          # rate of change of v
        self.__dsdt = 0.          # rate of change of s
        
        self.__ref_remaining = 0. # amount of time remaining in the refractory period
                
        # During a time step, spikes sent out from neurons are received by other neurons,
        # incrementing the receiving neuron's incoming_spikes_buffer.
        # When slope is called, that buffer is copied to self.__I so that it can be used
        # by the next call to step. That is, self.__I appears in the RHS of the
        # differential equation for s.
        self.__I = 0.             # current injected by arriving spikes
        
        # For plotting
        self.__v_history = []     # records v over time
        self.__s_history = []     # records s over time
    
        
    def slope(self):
        '''
         lif.slope()
        
         Evaluates the right-hand side of the differential equations that
         govern v and s. This function does NOT include incoming spikes.
         
         There is no output; the method simply updates the member variables
            self.__dvdt, and
            self.__dsdt
         which are used by the step method.
        '''
        self.__I = self.get_input_buffer()
        self.set_input_buffer(0.)
                
        self.__dvdt = (self.__s - self.__v)/self.tau_m  
        self.__dsdt = -self.__s/self.tau_s

        
    def step(self, dt):
        '''
         lif.step(dt)
        
         Updates the LIF neuron state by taking an Euler step in v and s.
         The length of the step is dt seconds.
         
         Input
          dt       time step (in seconds)
          
         If v reaches the threshold of 1, the neuron fires an action potential
         (spike). Linear interpolation is used to estimate the time that v=1.
         The spike time is appended to the list self.__spikes, and v
         is set to zero. After a spike, the neuron is dormant for self.tau_ref
         seconds.
        '''
        # self.__I includes all the input from spikes in the last time step
        
        # Store v (for plotting)
        self.__v_history.append(self.__v)
        self.__s_history.append(self.__s)

        self.__s =  self.__s + (self.__I/self.tau_s) + self.__dsdt * dt
        remaining_prev_time = self.__ref_remaining
        
        # Check if in refractory period
        if self.__ref_remaining > 0:
            remaining_ref_time = max(0, dt - self.__ref_remaining)
            self.__ref_remaining = max(0, self.__ref_remaining - dt)
            if remaining_ref_time > 0:
                tempv = self.__v + self.__dvdt * remaining_ref_time
            else:
                self.t += dt
                return
        else:
            tempv = self.__v + self.__dvdt * dt

        # check for spike
        if tempv > 1:
            if self.__dvdt == 0:
                spike_time = float('inf') 
            else:
                spike_time = ((1 - self.__v) / (self.__dvdt))
            self.add_spike(self.t + remaining_prev_time + spike_time)
            self.send_spike()
            self.__v = 0
            self.__ref_remaining = self.tau_ref - (dt - spike_time)
        else:
            self.__v = tempv

        # To add a spike at time self.t, use
        # self.add_spike(self.t)

        self.t += dt     # step forward in time


            
    def set_v(self, v):
        self.__v = v
    def get_v(self):
        return self.__v
    def set_s(self, s):
        self.__s = s
    def get_s(self):
        return self.__s
    def get_s_history(self):
        return self.__s_history
    def get_v_history(self):
        return self.__v_history
    
    def __repr__(self):
        '''
         lif.print(neur)
         
         Prints the current time, membrane potential, input current, and
         remaining refractory time.
        '''
        s = f'{self.t:6.4f}s: s={self.__s:5.3f}, v={self.__v:6.4f}, ref remaining={self.__ref_remaining:7.5f}s'
        return s
    
