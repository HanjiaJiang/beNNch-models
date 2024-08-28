"""
Random balanced network with astrocyte_lr_1994 for HPC benchmark
------------------------------------------------------------------

This script creates and simulates random balanced network with the
astrocyte_lr_1994 model. This script is used for HPC benchmarks.

"""

from network import model_default, build_network, run_simulation


###############################################################################
# Parameter section
# Define all relevant parameters: changes should be made here

params = {
    'nvp': {num_vps},                  # total number of virtual processes
    'scale': {N_SCALING},              # scaling factor of the network size
    'simtime': {model_time_sim},       # total simulation time in ms
    'presimtime': {model_time_presim}, # simulation time until reaching equilibrium
    'dt': 0.1,                         # simulation step
    'rng_seed': {rng_seed},            # random number generator seed
    'path_name': '.',                  # path where all files will have to be written
    'log_file': 'logfile',             # naming scheme for the log files
}

def run():
    # define conn_spec parameters
    model_update_dict = {
        "conn_params_e": {"rule": "pairwise_bernoulli", "p": 0.1/params["scale"]},
        "conn_params_i": {"rule": "pairwise_bernoulli", "p": 0.1/params["scale"]},
    }

    # run simulation
    run_simulation(params, model_update_dict)

if __name__ == '__main__':
    run()
