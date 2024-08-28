import os
import pickle

import matplotlib
matplotlib.rcParams["font.size"] = 15
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def plot_benchmark_model(n_neurons_hist, events_sr, events_astro, events_neuro, save_path):
    # set plot geometry
    widths = [1]
    heights = [2, 1, 2, 1]
    fig = plt.figure(figsize=(6, 8))
    spec = gridspec.GridSpec(ncols=1, nrows=4, figure=fig, width_ratios=widths, height_ratios=heights)
    # pos = [0.3, 0.2, 0.5, 0.7]

    # set axes
    axe_raster = fig.add_subplot(spec[0, 0])
    axe_hist = fig.add_subplot(spec[1, 0])
    axe_astro = fig.add_subplot(spec[2, 0])
    axe_neuro = fig.add_subplot(spec[3, 0])

    # create plots
    plot_raster(axe_raster, events_sr)
    plot_hist(axe_hist, events_sr, n_neurons_hist)
    axe_hist.set_yticks([axe_hist.get_ylim()[1]])
    axe_calcium = plot_astro_dynamics(axe_astro, events_astro)
    plot_neuro_sic(axe_neuro, events_neuro)

    axe_raster.set_position(  [0.25, 0.82, 0.5, 0.15])
    axe_hist.set_position(    [0.25, 0.58, 0.5, 0.15])
    axe_astro.set_position(   [0.25, 0.34, 0.5, 0.15])
    axe_calcium.set_position( [0.25, 0.34, 0.5, 0.15])
    axe_neuro.set_position(   [0.25, 0.10, 0.5, 0.15])

    # save figure
    plt.savefig(f'{save_path}/benchmark_model.eps', format='eps', dpi=400)
    plt.close()

    # save data to pickle
    with open(f'{save_path}/data.pkl', 'wb') as f:
        pickle.dump([n_neurons_hist, events_sr, events_astro, events_neuro], f)

def plot_conn_hist(targets, subject="", bins=list(range(0, 1001, 10)), save_path=".", figsize=(4, 3), xlabel="X", ylabel="Y", title=""):
    hist, bin_edges = np.histogram(targets, bins=np.arange(min(targets), max(targets)+2))
    plt.hist(hist, bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{save_path}/conn_hist_{subject}.eps', dpi=400)
    plt.savefig(f'{save_path}/conn_hist_{subject}.png', dpi=400)
    plt.close()

def plot_astro_dynamics(axe, events, start=0.0, xlims=(None, None), ylims_1=(None, None), ylims_2=(None, None), twin=True):
    # astrocyte data
    mask = events["times"] > start
    ip3 = events["IP3"][mask]
    calcium = events["Ca_astro"][mask]
    times = events["times"][mask]
    times_set = list(set(times))
    ip3_means = np.array([np.mean(ip3[times == t]) for t in times_set])
    ip3_sds = np.array([np.std(ip3[times == t]) for t in times_set])
    calcium_means = np.array([np.mean(calcium[times == t]) for t in times_set])
    calcium_sds = np.array([np.std(calcium[times == t]) for t in times_set])
    tab_blue = "tab:blue"
    pale_blue = "#D1E5F0"
    tab_green = "tab:green"
    pale_green = "#D9F0D3"
    # astrocyte plot
    axe.set_ylabel(r"[IP$_{3}]$ ($\mu$M)", labelpad=0)
    axe.tick_params(axis="y", labelcolor=tab_blue)
    axe.fill_between(
        times_set, ip3_means + ip3_sds, ip3_means - ip3_sds, linewidth=0.0, color=pale_blue
    )
    axe.plot(times_set, ip3_means, linewidth=1.5, color=tab_blue)
    axe.patch.set_visible(False)
    axe.set_xlim(xlims)
    axe.set_ylim(ylims_1)
    if twin:
        axe_calcium = axe.get_figure().add_axes(axe.get_position())
        axe_calcium.set_ylabel(r"[Ca$^{2+}]$ ($\mu$M)", rotation=270, labelpad=20)
        axe_calcium.tick_params(axis="y", labelcolor=tab_green)
        axe_calcium.fill_between(
            times_set, calcium_means + calcium_sds, calcium_means - calcium_sds, linewidth=0.0, color=pale_green)
        axe_calcium.plot(times_set, calcium_means, linewidth=1.5, color=tab_green)
        axe_calcium.yaxis.tick_right()
        axe_calcium.yaxis.set_label_position("right")
        axe_calcium.patch.set_visible(False)
        axe_calcium.set_xlim(xlims)
        axe_calcium.set_ylim(ylims_2)
        axe_calcium.spines[["left", "top", "bottom"]].set_visible(False)
        axe.set_zorder(axe_calcium.get_zorder()+1)
    else:
        axe_calcium = None
    return axe_calcium

def plot_neuro_sic(axe, events, start=0.0, xlims=(None, None), ylims=(None, None)):
    # neuron data
    mask = events["times"] > start
    sic = events["I_SIC"][mask]
    times = events["times"][mask]
    times_set = list(set(times))
    sic_means = np.array([np.mean(sic[times == t]) for t in times_set])
    sic_sds = np.array([np.std(sic[times == t]) for t in times_set])
    tab_purple = "tab:purple"
    pale_purple = "#E7D4E8"
    # neuron plot
    axe.set_ylabel(r"$I_\mathrm{SIC}$ (pA)")
    axe.set_xlabel("Time (ms)")
    axe.fill_between(
        times_set, sic_means + sic_sds, sic_means - sic_sds, linewidth=0.0, color=pale_purple
    )
    axe.plot(times_set, sic_means, linewidth=1.5, color=tab_purple)
    axe.set_xlim(xlims)
    axe.set_ylim(ylims)

def plot_sync(axe, coefs, binwidth=0.01, xlims=(None, None), ylims=(None, None)):
    bins = np.arange(-1.0, 1.0+binwidth, binwidth)
    axe.hist(coefs, bins=bins, color="gray", edgecolor="k")
    axe.set_xlabel("Pairwise spike count correlation (Pearson's r)")
    axe.set_ylabel("number of pairs")
    axe.set_xlim(xlims)
    axe.set_ylim(ylims)

def filter_by_n(events, n_neurons_plot, first_neuron=0):
    # get spiking data of the first n_neurons_plot neurons for the raster plot and histogram
    events_raster = {}
    mask_raster = np.isin(events["senders"], np.arange(first_neuron, first_neuron+n_neurons_plot))
    for key in ["times", "senders"]:
        events_raster[key] = events[key][mask_raster]
    return events_raster

def plot_raster(axe, events, xlims=(None, None), ylims=(None, None), n_limit=100):
    # prepare data
    ts = events["times"]
    neurons = events["senders"]
    # filter
    events_raster = filter_by_n(events, n_limit)
    # raster plot
    axe.plot(events_raster["times"], events_raster["senders"], ".k", markersize=2)
    axe.set_ylabel("Neuron ID")
    axe.set_xlim(xlims)
    axe.set_ylim(ylims)

def plot_hist(axe, events, n_neurons, binwidth=0.5, xlims=(None, None), ylims=(None, None)):
    # prepare data and figure
    ts = events["times"]
    neurons = events["senders"]
    # histogram
    bins = np.arange(np.amin(ts), np.amax(ts), float(binwidth)).tolist()
    hist, _ = np.histogram(ts, bins=bins)
    heights = 1000 * hist / (binwidth * n_neurons)
    axe.bar(bins[:-1], heights, width=binwidth, color="k", edgecolor="k")
    # axe.set_xlabel("Time (ms)")
    axe.set_ylabel("Firing rate\n(spikes/s)")
    axe.set_xlim(xlims)
    axe.set_ylim(ylims)

def set_broken_axes(axe, axe_top, ylims, bottom_frac=0.8, broken_frac=0.05):
    """This functions sets broken axes for extremely large values."""
    pos = axe.get_position()
    # set the axes on bottom (the original one)
    axe.set_position([pos.x0, pos.y0, pos.width, pos.height*bottom_frac])
    axe.set_ylim((ylims[0], (ylims[1]-ylims[0])*bottom_frac))
    axe.spines.top.set_visible(False)
    # set the axes on top
    ylims_ = axe_top.get_ylim()
    axe_top.set_position(
        [pos.x0, pos.y0+pos.height*(bottom_frac+broken_frac), pos.width, pos.height*(1-bottom_frac-broken_frac)])
    axe_top.set_ylabel(None)
    axe_top.set_xticks([])
    axe_top.set_yticks([ylims_[1]])
    axe_top.set_ylim((ylims_[1] - (ylims_[1] - ylims_[0])*(1-bottom_frac-broken_frac), ylims_[1]))
    axe_top.spines.bottom.set_visible(False)
    axe_top.patch.set_visible(False)
    # define the "slash strike" marker
    d = 0.1  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    # slash strike at the bottom left of the top panel
    axe_top.plot([0], [0], transform=axe_top.transAxes, **kwargs)
    # slash strike at the top left of the bottom panel
    axe.plot([0], [1], transform=axe.transAxes, **kwargs)
    axe.yaxis.set_label_coords(-0.25, 0.5/bottom_frac, transform=axe.transAxes)
    # vertical strike at the bottom right of the top panel
    axe_top.plot(
        [1, 1], [0, -0.5], 'k', linewidth=matplotlib.rcParams['axes.linewidth'], transform=axe_top.transAxes,
        clip_on=False)

def make_panels_benchmark_model(
    n_neurons_hist_a, events_sr_a, events_astro_a, events_neuro_a, n_neurons_hist_b, events_sr_b, events_astro_b,
    events_neuro_b, broken_axis=True, save_path="make_figure_benchmark_model"):
    """This function makes separate panels for a external composite figure workflow."""
    # create save folder
    os.system(f"mkdir -p {save_path}")
    # xlims for all panels
    xlims = (0, 5000.0)
    # different ylims for different panels
    ylims_hist, ylims_sic = (0.0, 50.0), (-1.0, 31.0)
    ylims_ip3, ylims_calcium = (-0.1, 10.1), (-0.01, 1.01)
    # position of axes for all panels
    pos = [0.25, 0.2, 0.5, 0.7]
    # A = raster plot, sparse
    fig, axe = plt.subplots(1, 1, figsize=(4, 3))
    axe.set_position(pos)
    plot_raster(axe, events_sr_a, xlims=xlims)
    plt.savefig(f"{save_path}/benchmark_model_A.eps", dpi=400)
    plt.savefig(f"{save_path}/benchmark_model_A.png", dpi=400)
    plt.close()
    # B = raster plot, synchronous
    fig, axe = plt.subplots(1, 1, figsize=(4, 3))
    axe.set_position(pos)
    plot_raster(axe, events_sr_b, xlims=xlims)
    plt.savefig(f"{save_path}/benchmark_model_B.eps", format='eps', dpi=400)
    plt.savefig(f"{save_path}/benchmark_model_B.png", dpi=400)
    plt.close()
    # C = firing rate histogram, sparse
    fig, axe = plt.subplots(1, 1, figsize=(4, 1.5))
    axe.set_position(pos)
    plot_hist(axe, events_sr_a, n_neurons_hist_a, ylims=ylims_hist, xlims=xlims)
    plt.savefig(f"{save_path}/benchmark_model_C.eps", format='eps', dpi=400)
    plt.savefig(f"{save_path}/benchmark_model_C.png", dpi=400)
    plt.close()
    # D = firing rate histogram, synchronous
    fig, axe = plt.subplots(1, 1, figsize=(4, 1.5))
    axe.set_position(pos)
    plot_hist(axe, events_sr_b, n_neurons_hist_b, ylims=ylims_hist, xlims=xlims)
    if broken_axis:
        axe_top = fig.add_axes(axe.get_position())
        plot_hist(axe_top, events_sr_b, n_neurons_hist_b, xlims=xlims)
        set_broken_axes(axe, axe_top, ylims_hist)
    plt.savefig(f"{save_path}/benchmark_model_D.eps", format='eps', dpi=400)
    plt.savefig(f"{save_path}/benchmark_model_D.png", dpi=400)
    plt.close()
    # E = astrocytic dynamics, sparse
    fig, axe = plt.subplots(1, 1, figsize=(4, 3))
    axe.set_position(pos)
    plot_astro_dynamics(axe, events_astro_a, ylims_1=ylims_ip3, ylims_2=ylims_calcium, xlims=xlims)
    plt.savefig(f"{save_path}/benchmark_model_E.eps", format='eps', dpi=400)
    plt.savefig(f"{save_path}/benchmark_model_E.png", dpi=400)
    plt.close()
    # F = astrocytic dynamics, synchronous
    fig, axe = plt.subplots(1, 1, figsize=(4, 3))
    axe.set_position(pos)
    plot_astro_dynamics(axe, events_astro_b, ylims_1=ylims_ip3, ylims_2=ylims_calcium, xlims=xlims)
    if broken_axis:
        axe_top = fig.add_axes(axe.get_position())
        plot_astro_dynamics(axe_top, events_astro_b, xlims=xlims, twin=False)
        set_broken_axes(axe, axe_top, ylims_ip3, bottom_frac=0.6)
    plt.savefig(f"{save_path}/benchmark_model_F.eps", format='eps', dpi=400)
    plt.savefig(f"{save_path}/benchmark_model_F.png", dpi=400)
    plt.close()
    # G = SIC, sparse
    fig, axe = plt.subplots(1, 1, figsize=(4, 1.5))
    axe.set_position(pos)
    plot_neuro_sic(axe, events_neuro_a, ylims=ylims_sic, xlims=xlims)
    plt.savefig(f"{save_path}/benchmark_model_G.eps", format='eps', dpi=400)
    plt.savefig(f"{save_path}/benchmark_model_G.png", dpi=400)
    plt.close()
    # H = SIC, synchronous
    fig, axe = plt.subplots(1, 1, figsize=(4, 1.5))
    axe.set_position(pos)
    plot_neuro_sic(axe, events_neuro_b, ylims=ylims_sic, xlims=xlims)
    plt.savefig(f"{save_path}/benchmark_model_H.eps", format='eps', dpi=400)
    plt.savefig(f"{save_path}/benchmark_model_H.png", dpi=400)
    plt.close()

if __name__ == "__main__":
    with open(f"sparse/data.pkl", "rb") as f:
        a, b, c, d = pickle.load(f)
    with open(f"synchronous/data.pkl", "rb") as f:
        e, f, g, h = pickle.load(f)
    make_panels_benchmark_model(a, b, c, d, e, f, g, h)
    print("Figure is done!")
