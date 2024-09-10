"""
Random balanced network with astrocyte_lr_1994 for HPC benchmark
------------------------------------------------------------------

This script creates and simulates random balanced network with the
astrocyte_lr_1994 model. This script is used for HPC benchmarks.

"""

import os
import time

import nest

M_INFO = 10
M_ERROR = 30

###############################################################################
# Set model parameters.

model_default = {
    "network_params": {
        "N_ex": 8000,  # number of excitatory neurons
        "N_in": 2000,  # number of inhibitory neurons
        "N_astro": 10000,  # number of astrocytes
        "p_third_if_primary": 0.5,  # probability of each created neuron-neuron connection to be paired with one astrocyte
        "pool_size": 10,  # astrocyte pool size for each target neuron
        "pool_type": "random",  # astrocyte pool will be chosen randomly for each target neuron
        "poisson_rate": 2000,  # Poisson input rate for neurons
    },
    "conn_params_e": {
        "rule": "pairwise_bernoulli",
        "p": 0.1,
    },
    "conn_params_i": {
        "rule": "pairwise_bernoulli",
        "p": 0.1,
    },
    "syn_params": {
        "w_a2n": 0.01,  # weight of astrocyte-to-neuron connection
        "w_e": 1.0,  # weight of excitatory connection in nS
        "w_i": -4.0,  # weight of inhibitory connection in nS
        "d_e": 2.0,  # delay of excitatory connection in ms
        "d_i": 1.0,  # delay of inhibitory connection in ms
    },
    "neuron_params_ex": {
        "tau_syn_ex": 2.0,  # excitatory synaptic time constant in ms
        "tau_syn_in": 4.0,  # inhibitory synaptic time constant in ms
    },
    "neuron_params_in": {
        "tau_syn_ex": 2.0,  # excitatory synaptic time constant in ms
        "tau_syn_in": 4.0,  # inhibitory synaptic time constant in ms
    },
    "astrocyte_params": {
        "delta_IP3": 0.5,  # Parameter determining the increase in astrocytic IP3 concentration induced by synaptic input
        "tau_IP3": 2.0,  # Time constant of the exponential decay of astrocytic IP3
    },
}

###############################################################################
# This function creates the nodes and build the network. The astrocytes only
# respond to excitatory synaptic inputs; therefore, only the excitatory
# neuron-neuron connections are paired with the astrocytes. The
# TripartiteConnect() function and the "tripartite_bernoulli_with_pool" rule
# are used to create the connectivity of the network.


def create_astro_network(network_params, neuron_params_ex, neuron_params_in, astrocyte_params, neuron_model, astrocyte_model, scale=1.0):
    """Create nodes for a neuron-astrocyte network."""
    print("Creating nodes ...")
    assert scale >= 1.0, "scale must be >= 1.0"
    nodes_ex = nest.Create(neuron_model, int(network_params["N_ex"] * scale), params=neuron_params_ex)
    nodes_in = nest.Create(neuron_model, int(network_params["N_in"] * scale), params=neuron_params_in)
    nodes_astro = nest.Create(astrocyte_model, int(network_params["N_astro"] * scale), params=astrocyte_params)
    nodes_noise = nest.Create("poisson_generator", params={"rate": network_params["poisson_rate"]})
    return nodes_ex, nodes_in, nodes_astro, nodes_noise


def connect_astro_network(nodes_ex, nodes_in, nodes_astro, nodes_noise, model_params, conn_params_e, conn_params_i):
    """Connect the nodes in a neuron-astrocyte network.
    The astrocytes are paired with excitatory connections only.
    """
    print("Connecting Poisson generator ...")
    nest.Connect(nodes_noise, nodes_ex + nodes_in, syn_spec={"weight": model_params["syn_params"]["w_e"]})
    print("Connecting neurons and astrocytes ...")
    # excitatory connections are paired with astrocytes
    conn_params_astro = {
        "rule": "third_factor_bernoulli_with_pool",
        "p": model_params["network_params"]["p_third_if_primary"],
        "pool_size": model_params["network_params"]["pool_size"],
        "pool_type": model_params["network_params"]["pool_type"],
    }
    syn_params_e = {
        "primary": {
            "synapse_model": "tsodyks_synapse",
            "weight": model_params["syn_params"]["w_e"],
            "tau_psc": model_params["neuron_params_ex"]["tau_syn_ex"],
            "delay": model_params["syn_params"]["d_e"],
        },
        "third_in": {
            "synapse_model": "tsodyks_synapse",
            "weight": model_params["syn_params"]["w_e"],
            "tau_psc": model_params["neuron_params_ex"]["tau_syn_ex"],
            "delay": model_params["syn_params"]["d_e"],
        },
        "third_out": {"synapse_model": "sic_connection", "weight": model_params["syn_params"]["w_a2n"]},
    }
    nest.TripartiteConnect(
        nodes_ex,
        nodes_ex + nodes_in,
        nodes_astro,
        conn_spec=conn_params_e,
        third_factor_conn_spec=conn_params_astro,
        syn_specs=syn_params_e,
    )
    # inhibitory connections are not paired with astrocytes
    syn_params_i = {
        "synapse_model": "tsodyks_synapse",
        "weight": model_params["syn_params"]["w_i"],
        "tau_psc": model_params["neuron_params_ex"]["tau_syn_in"],
        "delay": model_params["syn_params"]["d_i"],
    }
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_i, syn_params_i)


def build_network(params, model_params, neuron_model='aeif_cond_alpha_astro', astrocyte_model='astrocyte_lr_1994', record_conn=True):
    """Builds the network including setting of simulation and neuron
    parameters, creation of neurons and connections
    """

    tic = time.time()  # start timer on construction

    # set global kernel parameters
    nest.SetKernelStatus({'total_num_virtual_procs': params['nvp'],
                          'resolution': params['dt'],
                          'rng_seed': params['rng_seed'],
                          'overwrite_files': True,
                          'print_time': False})

    e, i, a, n = create_astro_network(
        model_params['network_params'],
        model_params['neuron_params_ex'],
        model_params['neuron_params_in'],
        model_params['astrocyte_params'],
        neuron_model,
        astrocyte_model,
        scale=params['scale']
    )

    nest.message(M_INFO, 'build_network', 'Creating excitatory spike recorder.')

    BuildNodeTime = time.time() - tic

    tic = time.time()

    connect_astro_network(
        e, i, a, n, model_params, model_params['conn_params_e'], model_params['conn_params_i'])

    # read out time used for building
    BuildEdgeTime = time.time() - tic

    d = {
        'py_time_create': BuildNodeTime,
        'py_time_connect': BuildEdgeTime,
        'N_ex': len(e),
        'N_in': len(i),
        'N_astro': len(a),
        'pool_size': model_params["network_params"]["pool_size"],
        'pool_type': 0 if model_params["network_params"]["pool_type"] == "random" else 1,
    }

    # Number of connections
    if record_conn:
        conn_data = nest.GetConnections()
        synapses = conn_data.get("synapse_model")
        connect_dict = {}
        for x in ["static_synapse", "tsodyks_synapse", "sic_connection"]:
            connect_dict[x] = synapses.count(x)
        d.update(connect_dict)

    return d, e, i, a

def run_simulation(params, model_update_dict):
    """Performs a simulation, including network construction"""

    nest.ResetKernel()
    nest.SyncProcesses()
    nest.set_verbosity(M_INFO)

    for key, value in model_update_dict.items():
        model_default[key].update(value)

    build_dict, nodes_ex, nodes_in, nodes_astro = build_network(
        params, model_default, neuron_model='aeif_cond_alpha_astro', astrocyte_model='astrocyte_lr_1994', record_conn=True)

    nest.Simulate(params['presimtime'])

    nest.Simulate(params['simtime'])

    d = {}
    d.update(build_dict)
    d.update(nest.GetKernelStatus())

    fn = '{fn}_{rank}.dat'.format(fn=params['log_file'], rank=nest.Rank())
    with open(fn, 'w') as f:
        for key, value in d.items():
            f.write(key + ' ' + str(value) + '\n')

