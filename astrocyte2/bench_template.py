"""
Random balanced network with astrocyte_lr_1994 for HPC benchmark
------------------------------------------------------------------

This script creates and simulates random balanced network with the
astrocyte_lr_1994 model. This script is used for HPC benchmarks.

"""

from network import model_default, run_simulation


###############################################################################
# Parameter section
# Define all relevant parameters: changes should be made here

params = {
    'model': '{model_name}',           # model name
    'nvp': {num_vps},                  # total number of virtual processes
    'scale': {scale_N},                # scaling factor of the network size
    'simtime': {model_time_sim},       # total simulation time in ms
    'presimtime': {model_time_presim}, # simulation time until reaching equilibrium
    'dt': 0.1,                         # simulation step
    'rng_seed': {rng_seed},            # random number generator seed
    'path_name': '.',                  # path where all files will have to be written
    'log_file': 'logfile',             # naming scheme for the log files
}

def run():
    # define model
    model = params["model"]

    # define model_update_dict according to specified model
    N_ex = model_default["network_params"]["N_ex"]
    N_in = model_default["network_params"]["N_in"]
    p = model_default["conn_params_e"]["p"]
    if model == "Bernoulli":
        model_update_dict = {
            "conn_params_e": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_i": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
        }
    elif model == "Synchronous":
        model_update_dict = {
            "conn_params_e": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_i": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "syn_params": {
                "w_a2n": 0.01,  # weight of astrocyte-to-neuron connection
                "w_e": 1.0,  # weight of excitatory connection in nS
                "w_i": -4.0,  # weight of inhibitory connection in nS
                "d_e": 2.0,  # delay of excitatory connection in ms
                "d_i": 2.0,  # delay of inhibitory connection in ms
            },
            "neuron_params_ex": {
                "tau_syn_ex": 2.0,  # excitatory synaptic time constant in ms
                "tau_syn_in": 2.0,  # inhibitory synaptic time constant in ms
            },
            "neuron_params_in": {
                "tau_syn_ex": 2.0,  # excitatory synaptic time constant in ms
                "tau_syn_in": 2.0,  # inhibitory synaptic time constant in ms
            },
        }
    elif model == "Fixed-indegree":
        model_update_dict = {
            "conn_params_e": {"rule": "fixed_indegree", "indegree": int(N_ex*p)},
            "conn_params_i": {"rule": "fixed_indegree", "indegree": int(N_in*p)},
        }
    elif model == "Fixed-outdegree":
        model_update_dict = {
            "conn_params_e": {"rule": "fixed_outdegree", "outdegree": int((N_ex+N_in)*p)},
            "conn_params_i": {"rule": "fixed_outdegree", "outdegree": int((N_ex+N_in)*p)},
        }
    elif model == "Fixed-total-number":
        model_update_dict = {
            "conn_params_e": {"rule": "fixed_total_number", "N": int(N_ex*(N_ex+N_in)*p)},
            "conn_params_i": {"rule": "fixed_total_number", "N": int(N_ex*(N_ex+N_in)*p)},
        }
    else:
        print("No correct model specified; use default (Beroulli).")
        model_update_dict = {
            "conn_params_e": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_i": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
        }

    # run simulation
    run_simulation(params, model_update_dict)

if __name__ == '__main__':
    run()
