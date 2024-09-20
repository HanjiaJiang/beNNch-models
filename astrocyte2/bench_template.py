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
    'pool_size': int({pool_size}),
    'pool_type': '{pool_type}',
    'default_astro': False,
}

def run():
    # define model
    model = params["model"]

    # define model_update_dict according to specified model
    N_ex = model_default["network_params"]["N_ex"]
    N_in = model_default["network_params"]["N_in"]
    p = model_default["network_params"]["p_primary"]
    p_third_if_primary = model_default["network_params"]["p_third_if_primary"]
    if model == "Bernoulli" or model == "Sparse":
        model_update_dict = {
            "conn_params_e": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_i": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
        }
    elif model == "Synchronous":
        model_update_dict = {
            "conn_params_e": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_i": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "syn_params": {
                "d_i": 2.0,  # delay of inhibitory connection in ms
            },
            "neuron_params_ex": {
                "tau_syn_in": 2.0,  # inhibitory synaptic time constant in ms
            },
            "neuron_params_in": {
                "tau_syn_in": 2.0,  # inhibitory synaptic time constant in ms
            },
        }
    elif model == "Surrogate":
        model_update_dict = {
            "network_params": {"astrocyte_model": "astrocyte_surrogate"},
            "conn_params_e": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_i": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
        }
    elif model == "No-tripartite":
        model_update_dict = {
            "network_params": {"no_tripartite": True},
            "conn_params_e": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_i": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_e_astro": {"rule": "pairwise_bernoulli", "p": p*p_third_if_primary/params["scale"]},
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
            "conn_params_e": {"rule": "fixed_total_number", "N": int(N_ex*(N_ex+N_in)*p*params["scale"])},
            "conn_params_i": {"rule": "fixed_total_number", "N": int(N_in*(N_ex+N_in)*p*params["scale"])},
        }
    elif model == "min-delay":
        model_update_dict = {
            "syn_params": {"d_a2n": 0.1},
            "conn_params_e": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_i": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
        }
    else:
        print("No correct model specified; use default (Beroulli).")
        model_update_dict = {
            "conn_params_e": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_i": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
        }

    if 'network_params' in model_update_dict:
        model_update_dict['network_params'].update({'pool_size': params['pool_size'], 'pool_type': params['pool_type']})
    else:
        model_update_dict['network_params'] = {'pool_size': params['pool_size'], 'pool_type': params['pool_type']}

    if params['default_astro']:
        model_update_dict['astrocyte_params'] = {'IP3': 0.4, 'delta_IP3': 0.0002, 'tau_IP3': 7142.0}
        if 'syn_params' in model_update_dict:
            model_update_dict['syn_params'].update({'w_a2n': 0.05})
        else:
            model_update_dict['syn_params'] = {'w_a2n': 0.05}

    # run simulation
    run_simulation(params, model_update_dict)

if __name__ == '__main__':
    run()
