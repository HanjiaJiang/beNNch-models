# -*- coding: utf-8 -*-
#
# hpc_benchmark.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.


"""
Random balanced network HPC benchmark
-------------------------------------

This script produces a balanced random network of `scale*11250` neurons in
which the excitatory-excitatory neurons exhibit STDP with
multiplicative depression and power-law potentiation. A mutual
equilibrium is obtained between the activity dynamics (low rate in
asynchronous irregular regime) and the synaptic weight distribution
(unimodal). The number of incoming connections per neuron is fixed
and independent of network size (indegree=11250).

This is the standard network investigated in [1]_, [2]_, [3]_.

A note on scaling
~~~~~~~~~~~~~~~

This benchmark was originally developed for very large-scale simulations on
supercomputers with more than 1 million neurons in the network and
11.250 incoming synapses per neuron. For such large networks, synaptic input
to a single neuron will be little correlated across inputs and network
activity will remain stable over long periods of time.

The original network size corresponds to a scale parameter of 100 or more.
In order to make it possible to test this benchmark script on desktop
computers, the scale parameter is set to 1 below, while the number of
11.250 incoming synapses per neuron is retained. In this limit, correlations
in input to neurons are large and will lead to increasing synaptic weights.
Over time, network dynamics will therefore become unstable and all neurons
in the network will fire in synchrony, leading to extremely slow simulation
speeds.

Therefore, the presimulation time is reduced to 50 ms below and the
simulation time to 250 ms, while we usually use 100 ms presimulation and
1000 ms simulation time.

For meaningful use of this benchmark, you should use a scale > 10 and check
that the firing rate reported at the end of the benchmark is below 10 spikes
per second.

References
~~~~~~~~

.. [1] Morrison A, Aertsen A, Diesmann M (2007). Spike-timing-dependent
       plasticity in balanced random networks. Neural Comput 19(6):1437-67
.. [2] Helias et al (2012). Supercomputers ready for use as discovery machines
       for neuroscience. Front. Neuroinform. 6:26
.. [3] Kunkel et al (2014). Spiking network simulation code for petascale
       computers. Front. Neuroinform. 8:78

"""

import os
import time
import scipy.special as sp

import nest
import nest.raster_plot
import matplotlib.pyplot as plt

M_INFO = 10
M_ERROR = 30


###############################################################################
# Parameter section
# Define all relevant parameters: changes should be made here

params = {
    'nvp': {num_vps},                  # total number of virtual processes
    'scale': {N_SCALING},              # scaling factor of the network size
                                       # total network size = scale*11250 neurons
    'simtime': {model_time_sim},       # total simulation time in ms
    'presimtime': {model_time_presim}, # simulation time until reaching equilibrium
    'dt': 0.1,                         # simulation step
    'record_spikes': {record_spikes},  # switch to record spikes of excitatory
                                       # neurons to file
    'rng_seed': {rng_seed},            # random number generator seed
    'path_name': '.',                  # path where all files will have to be written
    'log_file': 'logfile',             # naming scheme for the log files
}

###############################################################################
# Network parameters.

network_params = {
    "N_ex": 8000,  # number of excitatory neurons
    "N_in": 2000,  # number of inhibitory neurons
    "N_astro": 10000,  # number of astrocytes
    "p": 0.1,  # neuron-neuron connection probability.
    "p_syn_astro": 1.0,  # synapse-astrocyte pairing probability
    "max_astro_per_target": 10,  # max number of astrocytes per target neuron
    "astro_pool_by_index": False,  # Astrocyte pool selection by index
    "poisson_rate": 2000,  # rate of Poisson input
    "Nrec": 250,
    }

syn_params = {
    "synapse_model": "tsodyks_synapse",  # model of neuron-to-neuron and neuron-to-astrocyte connections
    "astro2post": "sic_connection",  # model of astrocyte-to-neuron connection
    "w_a2n": 1.0,  # weight of astrocyte-to-neuron connection
    "w_e": 1.0,  # weight of excitatory connection in nS
    "w_i": -4.0,  # weight of inhibitory connection in nS
    "d_e": 2.0,  # delay of excitatory connection in ms
    "d_i": 1.0,  # delay of inhibitory connection in ms
    }

###############################################################################
# Astrocyte parameters.

astrocyte_model = "ignore_and_sic"
astrocyte_params = {
    "SIC": 5.0,
    }

###############################################################################
# Neuron parameters.

neuron_model = "aeif_cond_alpha_astro"
tau_syn_ex = 2.0
tau_syn_in = 4.0

neuron_params_ex = {
    "tau_syn_ex": tau_syn_ex, # excitatory synaptic time constant in ms
    "tau_syn_in": tau_syn_in, # inhibitory synaptic time constant in ms
    }

neuron_params_in = {
    "tau_syn_ex": tau_syn_ex, # excitatory synaptic time constant in ms
    "tau_syn_in": tau_syn_in, # inhibitory synaptic time constant in ms
    }

###############################################################################
# Function for network building.

def create_astro_network(scale=1.0):
    """Create nodes for a neuron-astrocyte network."""
    print("Creating nodes ...")
    nodes_ex = nest.Create(
        neuron_model, int(network_params["N_ex"]*scale), params=neuron_params_ex)
    nodes_in = nest.Create(
        neuron_model, int(network_params["N_in"]*scale), params=neuron_params_in)
    nodes_astro = nest.Create(
        astrocyte_model, int(network_params["N_astro"]*scale), params=astrocyte_params)
    nodes_noise = nest.Create(
        "poisson_generator", params={"rate": network_params["poisson_rate"]}
        )
    return nodes_ex, nodes_in, nodes_astro, nodes_noise

def connect_astro_network(nodes_ex, nodes_in, nodes_astro, nodes_noise, scale=1.0):
    """Connect the nodes in a neuron-astrocyte network.
    The astrocytes are paired with excitatory connections only.
    """
    print("Connecting Poisson generator ...")
    assert scale >= 1.0, "scale must be >= 1.0"
    syn_params_noise = {
        "synapse_model": "static_synapse", "weight": syn_params["w_e"]
        }
    nest.Connect(
        nodes_noise, nodes_ex + nodes_in, syn_spec=syn_params_noise
        )
    print("Connecting neurons and astrocytes ...")
    conn_params_e = {
        "rule": "pairwise_bernoulli_astro",
        "astrocyte": nodes_astro,
        "p": network_params["p"]/scale,
        "p_syn_astro": network_params["p_syn_astro"]/scale,
        "max_astro_per_target": network_params["max_astro_per_target"],
        "astro_pool_by_index": network_params["astro_pool_by_index"],
        }
    syn_params_e = {
        "synapse_model": syn_params["synapse_model"],
        "weight_pre2post": syn_params["w_e"],
        "tau_psc": tau_syn_ex,
        "astro2post": syn_params["astro2post"],
        "weight_astro2post": syn_params["w_a2n"],
        "delay": syn_params["d_e"],
        }
    conn_params_i = {"rule": "pairwise_bernoulli", "p": network_params["p"]/scale}
    syn_params_i = {
        "synapse_model": syn_params["synapse_model"],
        "weight": syn_params["w_i"],
        "tau_psc": tau_syn_in,
        "delay": syn_params["d_i"],
        }
    nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_e, syn_params_e)
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_i, syn_params_i)

    return nodes_ex, nodes_in, nodes_astro, nodes_noise


def build_network():
    """Builds the network including setting of simulation and neuron
    parameters, creation of neurons and connections
    """

    tic = time.time()  # start timer on construction

    # set global kernel parameters
    nest.SetKernelStatus({'total_num_virtual_procs': params['nvp'],
                          'resolution': params['dt'],
                          'rng_seed': params['rng_seed'],
                          'overwrite_files': True})

    nest.print_time = True
    nest.overwrite_files = True

    e, i, a, n = create_astro_network(scale=params['scale'])

    nest.message(M_INFO, 'build_network',
                 'Creating excitatory spike recorder.')

    if params['record_spikes']:
        recorder_label = os.path.join(
            '.',
            'spikes')
        E_recorder = nest.Create('spike_recorder', params={
            'record_to': 'ascii',
            'label': recorder_label
        })


    BuildNodeTime = time.time() - tic
    node_memory = str(memory_thisjob())

    tic = time.time()

    connect_astro_network(e, i, a, n)

    E_neurons = e
    if params['record_spikes']:
        if params['nvp'] != 1:
            local_neurons = nest.GetLocalNodeCollection(E_neurons)
            # GetLocalNodeCollection returns a stepped composite NodeCollection, which
            # cannot be sliced. In order to allow slicing it later on, we're creating a
            # new regular NodeCollection from the plain node IDs.
            local_neurons = nest.NodeCollection(local_neurons.tolist())
        else:
            local_neurons = E_neurons

        if len(local_neurons) < network_params['Nrec']:
            network_params['Nrec'] = len(local_neurons)
            #nest.message(
            #    M_ERROR, 'build_network',
            #    """Spikes can only be recorded from local neurons, but the
            #    number of local neurons is smaller than the number of neurons
            #    spikes should be recorded from. Aborting the simulation!""")
            #exit(1)

        nest.message(M_INFO, 'build_network', 'Connecting spike recorders.')
        nest.Connect(local_neurons[:network_params['Nrec']], E_recorder,
                     'all_to_all', 'static_synapse_hpc')

    # read out time used for building
    BuildEdgeTime = time.time() - tic
    network_memory = str(memory_thisjob())

    d = {'py_time_create': BuildNodeTime,
         'py_time_connect': BuildEdgeTime,
         'node_memory': node_memory,
         'network_memory': network_memory}
    recorders = E_recorder if params['record_spikes'] else None

#    espikes = nest.Create("spike_recorder")
#    ispikes = nest.Create("spike_recorder")
#    espikes.set(label="brunel-py-ex", record_to="ascii")
#    ispikes.set(label="brunel-py-in", record_to="ascii")
#    nest.Connect(e[:100], espikes)
#    nest.Connect(i[:100], ispikes)

    return d, recorders

def run():
    """Performs a simulation, including network construction"""

    nest.ResetKernel()
    nest.set_verbosity(M_INFO)

    base_memory = str(memory_thisjob())

    build_dict, sr = build_network()

    tic = time.time()

    nest.Simulate(params['presimtime'])

    PreparationTime = time.time() - tic
    init_memory = str(memory_thisjob())

    tic = time.time()

    nest.Simulate(params['simtime'])

    SimCPUTime = time.time() - tic
    total_memory = str(memory_thisjob())

    average_rate = 0.0
    if params['record_spikes']:
        average_rate = compute_rate(sr)

    d = {'py_time_presimulate': PreparationTime,
         'py_time_simulate': SimCPUTime,
         'base_memory': base_memory,
         'init_memory': init_memory,
         'total_memory': total_memory,
         'average_rate': average_rate}
    d.update(build_dict)
    d.update(nest.GetKernelStatus())

    fn = '{fn}_{rank}.dat'.format(fn=params['log_file'], rank=nest.Rank())
    with open(fn, 'w') as f:
        for key, value in d.items():
            f.write(key + ' ' + str(value) + '\n')

    # Firing rates
#    events_ex = espks.n_events
#    events_in = ispks.n_events
#    rate_ex = events_ex / (params['presimtime'] + params['simtime']) * 1000.0 / 100
#    rate_in = events_in / (params['presimtime'] + params['simtime']) * 1000.0 / 100
#    print(f"Excitatory rate   : {rate_ex:.2f} Hz")
#    print(f"Inhibitory rate   : {rate_in:.2f} Hz")
#    nest.raster_plot.from_device(sr, hist=True)
#    plt.savefig("astrocyte_benchmark.png")

def compute_rate(sr):
    """Compute local approximation of average firing rate

    This approximation is based on the number of local nodes, number
    of local spikes and total time. Since this also considers devices,
    the actual firing rate is usually underestimated.

    """

    n_local_spikes = sr.n_events
    n_local_neurons = network_params['Nrec']
    simtime = params['simtime']
    return 1. * n_local_spikes / (n_local_neurons * simtime) * 1e3


def memory_thisjob():
    """Wrapper to obtain current memory usage"""
    nest.ll_api.sr('memory_thisjob')
    return nest.ll_api.spp()


def lambertwm1(x):
    """Wrapper for LambertWm1 function"""
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real


if __name__ == '__main__':
    run()
