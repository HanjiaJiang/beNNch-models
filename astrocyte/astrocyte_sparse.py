"""
Random balanced network with astrocyte_lr_1994 for HPC benchmark
------------------------------------------------------------------

This script creates and simulates random balanced network with the
astrocyte_lr_1994 model. This script is used for HPC benchmarks.

"""

import inspect
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

astrocyte_model = "astrocyte_lr_1994"
astrocyte_params = {
    "delta_IP3": 0.5,  # Parameter determining the increase in astrocytic IP3 concentration induced by synaptic input
    "tau_IP3": 2.0,  # Time constant of the exponential decay of astrocytic IP3
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
    if "third_factor_conn_spec" in list(inspect.signature(nest.TripartiteConnect).parameters.keys()):
        print("Using new TripartiteConnect().")
        conn_params_e = {
            "rule": "pairwise_bernoulli",
            "p": network_params["p_primary"] / scale
        }
        conn_params_astro = {
            "rule": "third_factor_bernoulli_with_pool",
            "p": network_params["p_third_if_primary"],
            "pool_size": network_params["pool_size"],
            "pool_type": network_params["pool_type"],
        }
        nest.TripartiteConnect(
            nodes_ex,
            nodes_ex + nodes_in,
            nodes_astro,
            conn_spec=conn_params_e,
            third_factor_conn_spec=conn_params_astro,
            syn_specs=syn_params_e,
        )
    else:
        print("Using old TripartiteConnect().")
        nest.TripartiteConnect(
            nodes_ex,
            nodes_ex + nodes_in,
            nodes_astro,
            conn_spec=conn_params_e,
            syn_specs=syn_params_e
        )
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

    tic = time.time()

    connect_astro_network(e, i, a, n, scale=params['scale'])

    # read out time used for building
    BuildEdgeTime = time.time() - tic

    d = {
        'py_time_create': BuildNodeTime,
        'py_time_connect': BuildEdgeTime,
    }
    recorders = E_recorder if params['record_spikes'] else None

    return d, recorders

def run(record_conn=True):
    """Performs a simulation, including network construction"""

    nest.ResetKernel()
    nest.set_verbosity(M_INFO)

    build_dict, sr = build_network()

    nest.Simulate(params['presimtime'])

    nest.Simulate(params['simtime'])

    d = {}
    d.update(build_dict)
    d.update(nest.GetKernelStatus())

    # Number of connections
    if record_conn:
        conn_data = nest.GetConnections()
        synapses = conn_data.get("synapse_model")
        connect_dict = {}
        for x in ["static_synapse", "tsodyks_synapse", "sic_connection"]:
            connect_dict[x] = synapses.count(x)
        d.update(connect_dict)

    fn = '{fn}_{rank}.dat'.format(fn=params['log_file'], rank=nest.Rank())
    with open(fn, 'w') as f:
        for key, value in d.items():
            f.write(key + ' ' + str(value) + '\n')

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
    run(record_conn=False)
