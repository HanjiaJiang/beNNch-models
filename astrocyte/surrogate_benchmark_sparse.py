"""
Random balanced network with astrocyte_lr_1994 for HPC benchmark
------------------------------------------------------------------

This script creates and simulates random balanced network with the
astrocyte_lr_1994 model. This script is used for HPC benchmarks.

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
# Set network parameters.

model = "sparse"

network_params = {
    "N_ex": 8000,  # number of excitatory neurons
    "N_in": 2000,  # number of inhibitory neurons
    "N_astro": 10000,  # number of astrocytes
    "p_primary": 0.1,  # connection probability between neurons
    "p_third_if_primary": 0.5,  # probability of each created neuron-neuron connection to be paired with one astrocyte
    "pool_size": 10,  # astrocyte pool size for each target neuron
    "pool_type": "random",  # astrocyte pool will be chosen randomly for each target neuron
    "poisson_rate": 2000,  # Poisson input rate for neurons
}

syn_params = {
    "w_a2n": 0.01,  # weight of astrocyte-to-neuron connection
    "w_e": 1.0,  # weight of excitatory connection in nS
    "w_i": -4.0,  # weight of inhibitory connection in nS
    "d_e": 2.0,  # delay of excitatory connection in ms
    "d_i": 1.0 if model == "sparse" else 2.0,  # delay of inhibitory connection in ms
}

###############################################################################
# Set astrocyte parameters.

astrocyte_model = "astrocyte_surrogate"
astrocyte_params = {
    "SIC": 1.0
}

###############################################################################
# Set neuron parameters.

neuron_model = "aeif_cond_alpha_astro"
tau_syn_ex = 2.0
tau_syn_in = 4.0 if model == "sparse" else 2.0

neuron_params_ex = {
    "tau_syn_ex": tau_syn_ex,  # excitatory synaptic time constant in ms
    "tau_syn_in": tau_syn_in,  # inhibitory synaptic time constant in ms
}

neuron_params_in = {
    "tau_syn_ex": tau_syn_ex,  # excitatory synaptic time constant in ms
    "tau_syn_in": tau_syn_in,  # inhibitory synaptic time constant in ms
}

###############################################################################
# This function creates the nodes and build the network. The astrocytes only
# respond to excitatory synaptic inputs; therefore, only the excitatory
# neuron-neuron connections are paired with the astrocytes. The
# TripartiteConnect() function and the "tripartite_bernoulli_with_pool" rule
# are used to create the connectivity of the network.


def create_astro_network(scale=1.0):
    """Create nodes for a neuron-astrocyte network."""
    print("Creating nodes ...")
    assert scale >= 1.0, "scale must be >= 1.0"
    nodes_ex = nest.Create(neuron_model, int(network_params["N_ex"] * scale), params=neuron_params_ex)
    nodes_in = nest.Create(neuron_model, int(network_params["N_in"] * scale), params=neuron_params_in)
    nodes_astro = nest.Create(astrocyte_model, int(network_params["N_astro"] * scale), params=astrocyte_params)
    nodes_noise = nest.Create("poisson_generator", params={"rate": network_params["poisson_rate"]})
    return nodes_ex, nodes_in, nodes_astro, nodes_noise


def connect_astro_network(nodes_ex, nodes_in, nodes_astro, nodes_noise, scale=1.0):
    """Connect the nodes in a neuron-astrocyte network.
    The astrocytes are paired with excitatory connections only.
    """
    print("Connecting Poisson generator ...")
    assert scale >= 1.0, "scale must be >= 1.0"
    nest.Connect(nodes_noise, nodes_ex + nodes_in, syn_spec={"weight": syn_params["w_e"]})
    print("Connecting neurons and astrocytes ...")
    # excitatory connections are paired with astrocytes
    # conn_spec and syn_spec according to the "tripartite_bernoulli_with_pool" rule
    conn_params_e = {
        "rule": "tripartite_bernoulli_with_pool",
        "p_primary": network_params["p_primary"] / scale,
        "p_third_if_primary": network_params["p_third_if_primary"],
        "pool_size": network_params["pool_size"],
        "pool_type": network_params["pool_type"],
    }
    syn_params_e = {
        "primary": {
            "synapse_model": "tsodyks_synapse",
            "weight": syn_params["w_e"],
            "tau_psc": tau_syn_ex,
            "delay": syn_params["d_e"],
        },
        "third_in": {
            "synapse_model": "tsodyks_synapse",
            "weight": syn_params["w_e"],
            "tau_psc": tau_syn_ex,
            "delay": syn_params["d_e"],
        },
        "third_out": {"synapse_model": "sic_connection", "weight": syn_params["w_a2n"]},
    }
    nest.TripartiteConnect(nodes_ex, nodes_ex + nodes_in, nodes_astro, conn_spec=conn_params_e, syn_specs=syn_params_e)
    # inhibitory connections are not paired with astrocytes
    conn_params_i = {"rule": "pairwise_bernoulli", "p": network_params["p_primary"] / scale}
    syn_params_i = {
        "synapse_model": "tsodyks_synapse",
        "weight": syn_params["w_i"],
        "tau_psc": tau_syn_in,
        "delay": syn_params["d_i"],
    }
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_i, syn_params_i)


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

    nest.print_time = False
    nest.overwrite_files = True

    e, i, a, n = create_astro_network(scale=params['scale'])

    nest.message(M_INFO, 'build_network', 'Creating excitatory spike recorder.')

    if params['record_spikes']:
        recorder_label = os.path.join(
            '.',
            'spikes')
        E_recorder = nest.Create('spike_recorder', params={
            'record_to': 'ascii',
            'label': recorder_label
        })


    BuildNodeTime = time.time() - tic
#    node_memory = str(memory_thisjob())

    tic = time.time()

    connect_astro_network(e, i, a, n, scale=params['scale'])

#    E_neurons = e
#    if params['record_spikes']:
#        if params['nvp'] != 1:
#            local_neurons = nest.GetLocalNodeCollection(E_neurons)
#            # GetLocalNodeCollection returns a stepped composite NodeCollection, which
#            # cannot be sliced. In order to allow slicing it later on, we're creating a
#            # new regular NodeCollection from the plain node IDs.
#            local_neurons = nest.NodeCollection(local_neurons.tolist())
#        else:
#            local_neurons = E_neurons
#
#        if len(local_neurons) < network_params['Nrec']:
#            network_params['Nrec'] = len(local_neurons)
#            #nest.message(
#            #    M_ERROR, 'build_network',
#            #    """Spikes can only be recorded from local neurons, but the
#            #    number of local neurons is smaller than the number of neurons
#            #    spikes should be recorded from. Aborting the simulation!""")
#            #exit(1)
#
#        nest.message(M_INFO, 'build_network', 'Connecting spike recorders.')
#        nest.Connect(local_neurons[:network_params['Nrec']], E_recorder,
#                     'all_to_all', 'static_synapse_hpc')

    # read out time used for building
    BuildEdgeTime = time.time() - tic
#    network_memory = str(memory_thisjob())

    d = {'py_time_create': BuildNodeTime,
         'py_time_connect': BuildEdgeTime,
#         'node_memory': node_memory,
#         'network_memory': network_memory
    }
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

#    base_memory = str(memory_thisjob())

    build_dict, sr = build_network()

#    tic = time.time()

    nest.Simulate(params['presimtime'])

#    PreparationTime = time.time() - tic
#    init_memory = str(memory_thisjob())

#    tic = time.time()

    nest.Simulate(params['simtime'])

#    SimCPUTime = time.time() - tic
#    total_memory = str(memory_thisjob())

#    average_rate = 0.0
#    if params['record_spikes']:
#        average_rate = compute_rate(sr)

    d = {}
#    d = {'py_time_presimulate': PreparationTime,
#         'py_time_simulate': SimCPUTime,
#         'base_memory': base_memory,
#         'init_memory': init_memory,
#         'total_memory': total_memory,
#         'average_rate': average_rate
#    }
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
