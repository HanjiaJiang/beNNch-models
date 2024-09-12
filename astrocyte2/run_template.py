###############################################################################
# Import all necessary modules for simulation and plotting.

import os
import sys
import random

import nest
import numpy as np
import pandas as pd

import plots
from network import model_default, build_network

###############################################################################
# Set simulation parameters.

nvp = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
model = sys.argv[2] if len(sys.argv) > 2 else 'Bernoulli'
params = {
    'model': model,      # model name and data path
    'nvp': nvp,                # total number of virtual processes
    'scale': 1,                # scaling factor of the network size
    'simtime': 10000,          # total simulation time in ms
    'presimtime': 1000,        # simulation time until reaching equilibrium
    'dt': 0.1,                 # simulation step
    'rng_seed': 1,             # random number generator seed
    'pool_size': 10,
    'pool_type': 'random',
}

###############################################################################
# Set number of cells sampled for analysis

N_analysis = 100

###############################################################################
# This function creates recording devices and connects them to the network.

def create_devices(exc, inh, astro):
    # create devices (multimeter default resolution = 1 ms)
    sr = nest.Create("spike_recorder")
    mm_neuron = nest.Create("multimeter", params={"record_from": ["I_SIC"]})
    mm_astro = nest.Create("multimeter", params={"record_from": ["IP3", "Ca_astro"]})
    # connect devices
    sampled_neurons = (exc + inh).tolist()
    astro_list = astro.tolist()
    assert len(sampled_neurons) >= N_analysis, f"Number of neurons < {N_analysis}!"
    assert len(astro_list) >= N_analysis, f"Number of astrocytes < {N_analysis}!"
    # connect all neurons to the spike recorder
    nest.Connect(sampled_neurons, sr)
    # connect N_analysis neurons and astrocytes to the multimeters
    sampled_neurons = sorted(random.sample(sampled_neurons, N_analysis))
    sampled_astrocytes = sorted(random.sample(astro_list, N_analysis))
    nest.Connect(mm_neuron, sampled_neurons)
    nest.Connect(mm_astro, sampled_astrocytes)
    return sr, mm_neuron, mm_astro

###############################################################################
# This function calculates the average neuronal firing rate

def calc_fr(events, n_neurons, start, end):
    mask = (events["times"]>=start)&(events["times"]<end)
    fr = 1000*len(events["times"][mask])/((end-start)*n_neurons)
    return fr

###############################################################################
# This function calculates the pairwise spike count correlations. For each pair
# of neurons, the correlation coefficient (Pearson's r) of their spike count
# histograms is calculated. The result of all pairs are returned.

def get_corr(hlist):
    coef_list = []
    n_pair_pass = 0
    n_pair_fail = 0
    for i, hist1 in enumerate(hlist):
        idxs = list(range(i + 1, len(hlist)))
        for j in idxs:
            hist2 = hlist[j]
            if np.sum(hist1) != 0 and np.sum(hist2) != 0:
                coef = np.corrcoef(hist1, hist2)[0, 1]
                coef_list.append(coef)
                n_pair_pass += 1
            else:
                n_pair_fail += 1
    if n_pair_fail > 0:
        print(f"n_pair_fail = {n_pair_fail}!")

    return coef_list, n_pair_pass, n_pair_fail


###############################################################################
# This function calculates the synchrony of neuronal firings.
# Histograms of spike counts of all neurons are obtained to evaluate local and
# global synchrony. The local synchrony is evaluated with average pairwise spike
# count correlation, and the global synchrony is evaluated with the variance of
# average spike count/average of variance of individual spike count.

def calc_synchrony(neuron_spikes, n_neurons, start, end, binwidth=10):
    # get data
    mask = (neuron_spikes["times"] >= start)&(neuron_spikes["times"] < end)
    senders = neuron_spikes["senders"][mask]
    times = neuron_spikes["times"][mask]
    rate = 1000 * len(senders) / ((end - start) * n_neurons)
    # sample neurons
    list_senders = list(set(senders))
    n_for_sync = len(list_senders)
    # make spike count histograms of individual neurons
    bins = np.arange(start, end + 0.1, binwidth)  # time bins
    hists = [np.histogram(times[senders == x], bins)[0].tolist() for x in set(senders)]
    # make spiking histogram of all sampled neurons
    hist_global = (np.histogram(times, bins)[0] / len(set(senders))).tolist()
    # calculate local and global synchrony
    coefs, n_pair_pass, n_pair_fail = get_corr(hists)  # local
    gsync = np.var(hist_global) / np.mean(np.var(hists, axis=1))  # global
    return rate, coefs, gsync, n_for_sync

###############################################################################
# This function plots the connections between neurons and astrocytes.

def plot_conn_distr(nodes_ex, nodes_astro, n_hist=None):
    for conn_name, source_nodes, target_nodes in zip(["n2n", "n2a", "a2n"], [nodes_ex, nodes_ex, nodes_astro], [nodes_ex, nodes_astro, nodes_ex]):
        n_hist_tmp = n_hist if isinstance(n_hist, int) else len(target_nodes)
        conns = nest.GetConnections(source_nodes, target_nodes[:n_hist_tmp])
        targets = conns.get("target")
        plots.plot_conn_hist(
            targets, subject=conn_name, save_path=model,
            xlabel=f"Number of {conn_name} connections per target",
            ylabel="Number of cases", title="Bernoulli")

###############################################################################
# This function updates the model parameters.

def update_model_parameters():
    # define model
    model = params["model"]

    # define model_update_dict according to specified model
    N_ex = model_default["network_params"]["N_ex"]
    N_in = model_default["network_params"]["N_in"]
    p = 0.1
    if model == "Bernoulli":
        model_update_dict = {
            "conn_params_e": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
            "conn_params_i": {"rule": "pairwise_bernoulli", "p": p/params["scale"]},
        }
    elif model == "Surrogate":
        model_update_dict = {
            "network_params": {"astrocyte_model": "astrocyte_surrogate"},
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

    if 'network_params' not in model_update_dict:
        model_update_dict['network_params'] = {}
    model_update_dict['network_params'].update({'pool_size': params['pool_size']})
    model_update_dict['network_params'].update({'pool_type': params['pool_type']})

    for key, value in model_update_dict.items():
        model_default[key].update(value)


###############################################################################
# This is the main function to run the simulation with.

def run():
    # update model parameters
    update_model_parameters()

    # make data folder if not exist, and copy python scripts
    path_name = params["model"]
    os.system(f"mkdir -p {path_name}")
    os.system(f"rsync -au *.py {path_name}")

    # use random seed for reproducible sampling
    random.seed(params["rng_seed"])

    # create and connect network and devices
    nest.ResetKernel()
    nest.SyncProcesses()
    nest.set_verbosity(10)

    # build network
    build_dict, nodes_ex, nodes_in, nodes_astro = build_network(params, model_default, record_conn=False)

    # create devices
    sr, mm_neuron, mm_astro = create_devices(nodes_ex, nodes_in, nodes_astro)

    # run simulation
    pre_sim_time, sim_time = params['presimtime'], params['simtime']
    nest.Simulate(pre_sim_time)
    nest.Simulate(sim_time)

    # calculate average SIC in neurons
    I_SIC = np.mean(mm_neuron.events["I_SIC"][mm_neuron.events["times"]>=pre_sim_time])

    # get spiking data of all neurons and calculate average firing rate
    events = sr.events
    neurons = (nodes_ex + nodes_in).tolist()
    rate_network = calc_fr(events, len(neurons), pre_sim_time, pre_sim_time+sim_time)
    print(f"Network average neuronal firing rate = {rate_network:.2f}")

    # create plots
    plots.plot_benchmark_model(len(neurons), events, mm_astro.events, mm_neuron.events, path_name)

    # synchrony analysis
    events_analysis = {}
    # filter by time
    mask_time = (events["times"]>=pre_sim_time)&(events["times"]<pre_sim_time+sim_time)
    for key in ["times", "senders"]:
        events_analysis[key] = events[key][mask_time]
    # sample (N_analysis) spiking neurons
    neurons_analysis = random.sample(list(set(events_analysis["senders"])), N_analysis)
    mask_neuron = np.isin(events_analysis["senders"], neurons_analysis)
    for key in ["times", "senders"]:
        events_analysis[key] = events_analysis[key][mask_neuron]
    # calculate and report synchrony
    rate, coefs, gsync, n_for_sync = calc_synchrony(
        events_analysis, N_analysis, pre_sim_time, pre_sim_time + sim_time
    )
    lsync, lsync_std = np.mean(coefs), np.std(coefs)
    print(f"Local synchrony = {lsync:.3f}+-{lsync_std:.3f}")
    print(f"Global synchrony = {gsync:.3f}")
    print(f"Firing rate of sampled neurons = {rate:.2f} spikes/s")
    print(f"(n = {n_for_sync} for synchrony analysis)")

    # save results
    data = {}
    data.update(params)
    for key, value in model_default.items():
        data.update(value)
    for key, value in data.items():
        data[key] = [value]
    data["fr"] = [rate_network]
    data["lsync"] = [lsync]
    data["gsync"] = [gsync]
    data["n_for_sync"] = [n_for_sync]
    data["I_SIC"] = [I_SIC]
    df = pd.DataFrame(data)
    df.to_csv(f"{path_name}/{path_name}.csv", index=False)

    # plot connections
    # when on PC, USE ONLY WHEN THE MODEL IS SMALL!
    # plot_conn_distr(nodes_ex, nodes_astro)

###############################################################################
# Run the script.

if __name__ == "__main__":
    # record output; only for debugging
    orig_stdout = sys.stdout
    path_name = params["model"]
    os.system(f"mkdir -p {path_name}")
    f = open(f'{path_name}/out.txt', 'w')
    sys.stdout = f

    run()

    # record output; only for debugging
    sys.stdout = orig_stdout
    f.close()
