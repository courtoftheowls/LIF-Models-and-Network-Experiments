🧠 Spiking Neural Networks: LIF Models and Network Experiments

Overview

This project implements, simulates, and analyzes leaky integrate-and-fire (LIF) spiking neuron models and networks. I developed core neuron models, built multi-neuron networks, ran simulations, and visualized the emergent firing patterns and input–output relationships.

The project explores both the implementation of numerical integration (Euler’s method) for neuron dynamics and experiments on how network topology and input currents affect spiking behavior.

🚀 Features

✅ Implemented an LIF neuron class with membrane potential, post-synaptic currents, and spike-time tracking

✅ Built a two-neuron bidirectional network with controlled input spike trains

✅ Simulated the network to produce spike raster plots

✅ Designed a larger LIF neuron layer with varied input currents and measured firing rates

✅ Analyzed the relationship between input current and output firing rates, generating insightful plots

⚙️ Tech Stack
* Python 3
* NumPy
* Matplotlib
* Custom neuron simulation modules (neuron_models.py)
* Jupyter Notebooks for experiments and visualization

📊 Key Results
* Built a dynamic LIF model that properly handles spikes, refractory periods, and synaptic integration.
* In the two-neuron network, showed reciprocal spiking behavior modulated by external inputs at precise time intervals.
* In the multi-neuron experiment, demonstrated the nonlinear relationship between input current strength and firing rate across a range of synaptic weights.
* Produced clear, labeled plots showing firing rates vs. input currents, matching theoretical expectations.
