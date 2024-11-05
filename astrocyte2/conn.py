import os
import sys
import pickle
import time
import random
random.seed(1)

import matplotlib
matplotlib.rcParams["font.size"] = 13
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def count_arr(df, n_mpi):
    cnt_arr = np.zeros((n_mpi, n_mpi))
    sources, targets = df["source"].values.astype(int), df["target"].values.astype(int)
    for i in range(n_mpi):
        for j in range(n_mpi):
            pass
            cnt_arr[i, j] = len(df[(df["source"]%n_mpi==i)&(df["target"]%n_mpi==j)])
    return cnt_arr

def conn_per_target(tlist, save_path, subject, figsize,
    source_name="astrocyte", target_name="neuron",
    xlabel=False, ylabel=False, xlims=(250, 550), ax_position=None):
    print("conn_per_target()...")
    fname = f"conn_per_target_{subject}"
    if os.path.isfile(f"{save_path}/{fname}.pkl"):
        with open(f"{save_path}/{fname}.pkl", "rb") as f:
            arr_cnt = pickle.load(f)
    else:
        arr_cnt = []
        print(len(set(tlist)))
        targets = np.array(tlist)
        for i, target in enumerate(list(set(tlist))):
            if i%10000 == 0:
                print(i)
            arr_cnt.append(np.count_nonzero(targets==target))
        with open(f"{save_path}/{fname}.pkl", "wb") as f:
            pickle.dump(arr_cnt, f)
    fig = plt.figure(figsize=figsize)
    if max(arr_cnt) - min(arr_cnt) > 20:
        bins = list(range(min(arr_cnt)-1, max(arr_cnt)+1))
    else:
        bins = list(range(min(arr_cnt)-1, min(arr_cnt)+21))
    # verify counts
    # print(np.histogram(arr_cnt, bins=(list(range(100)))))
    plt.hist(arr_cnt, bins, color='k', ec='k')
    if xlabel:
        plt.xlabel(f"Number of\n{source_name}-{target_name} connections\nper {target_name}")
    if ylabel:
        plt.ylabel("Number of\ncases", labelpad=0)
    plt.xlim(xlims)
    if isinstance(ax_position, list):
        ax = plt.gca()
        ax.set_position(ax_position)
    else:
        plt.tight_layout()
    plt.savefig(f'{save_path}/{fname}.eps', dpi=400)
    plt.savefig(f'{save_path}/{fname}.png', dpi=400)
    plt.close()

# first 100 astrocytes
def conn_num_distr(slist, tlist, save_path, subject, figsize,
    source_name="astrocyte", target_name="neuron", n_sample=100,
    xlabel=False, ylabel=False, xlims=(0, 100), ax_position=None):
    print("conn_num_distr()...")
    fname = f"conn_num_distr_{subject}"
    sset = list(set(slist))
    tset = list(set(tlist))
    random.shuffle(sset)
    sset_filter = sset[:n_sample]
    sources = [x for x in slist if x in sset_filter]
    targets = [tlist[i] for i, x in enumerate(slist) if x in sset_filter]
    if os.path.isfile(f"{save_path}/{fname}.pkl"):
        with open(f"{save_path}/{fname}.pkl", "rb") as f:
            arr_cnt = pickle.load(f)
    else:
        arr_s = np.array(sources)
        arr_t = np.array(targets)
        arr_st = []
        arr_cnt = []
        print(len(sources))
        t0 = t1 = t2 = t3 = 0
        for i, (source, target) in enumerate(zip(sources, targets)):
            if i%10000 == 0:
                print(i)
                print(int(t0), int(t1), int(t2), int(t3))
            t0a = time.time()
            if (source, target) not in arr_st:
                t1a = time.time()
                arr_st.append((source, target))
                t1 += time.time() - t1a
                t2a = time.time()
                mask = (arr_s == source) & (arr_t == target)
                t2 += time.time() - t2a
                t3a = time.time()
                arr_cnt.append(np.sum(mask))
                t3 += time.time() - t3a
            t0 += time.time() - t0a
        with open(f"{save_path}/{fname}.pkl", "wb") as f:
            pickle.dump(arr_cnt, f)
    fig = plt.figure(figsize=figsize)
    if max(arr_cnt) - min(arr_cnt) > 20:
        bins = list(range(min(arr_cnt)-1, max(arr_cnt)+1))
    else:
        bins = list(range(min(arr_cnt)-1, min(arr_cnt)+21))
    # verify counts
    # print(np.histogram(arr_cnt, bins=(list(range(100)))))
    plt.hist(arr_cnt, bins, color='k', ec='k')
    if xlabel:
        plt.xlabel(f"Number of connections per\n{source_name}-{target_name} pair")
    if ylabel:
        plt.ylabel("Number of\ncases", labelpad=0)
    plt.xlim(xlims)
    if isinstance(ax_position, list):
        ax = plt.gca()
        ax.set_position(ax_position)
    else:
        plt.tight_layout()
    plt.savefig(f'{save_path}/{fname}.eps', dpi=400)
    plt.savefig(f'{save_path}/{fname}.png', dpi=400)
    plt.close()

def conn_target_distr(sources, targets, save_path, subject, figsize,
    source_name="postsynaptic neurons", target_name="astrocyte",
    xlabel=False, ylabel=False, xlims=(0, 500), ax_position=None):
    print("conn_target_distr()...")
    fname = f"conn_target_distr_{subject}"
    if os.path.isfile(f"{save_path}/{fname}.pkl"):
        with open(f"{save_path}/{fname}.pkl", "rb") as f:
            arr_cnt = pickle.load(f)
    else:
        arr_s = np.array(sources)
        arr_t = np.array(targets)
        set_source = set(sources)
        print(len(set_source))
        arr_cnt = []
        for i, source in enumerate(set_source):
            if i%1000 == 0:
                print(i)
            arr_cnt.append(len(set(arr_t[arr_s==source])))
        with open(f"{save_path}/{fname}.pkl", "wb") as f:
            pickle.dump(arr_cnt, f)
    fig = plt.figure(figsize=figsize)
    if max(arr_cnt) - min(arr_cnt) > 20:
        bins = list(range(min(arr_cnt)-1, max(arr_cnt)+1))
    else:
        bins = list(range(min(arr_cnt)-1, min(arr_cnt)+21))
    plt.hist(arr_cnt, bins, color='k', ec='k')
    if xlabel:
        plt.xlabel(f"Number of connected {source_name}\nper {target_name}")
    if ylabel:
        plt.ylabel("Number of\ncases", labelpad=0)
    plt.xlim(xlims)
    if isinstance(ax_position, list):
        ax = plt.gca()
        ax.set_position(ax_position)
    else:
        plt.tight_layout()
    plt.savefig(f'{save_path}/{fname}.eps', dpi=400)
    plt.savefig(f'{save_path}/{fname}.png', dpi=400)
    plt.close()

def conn_source_distr(sources, targets, save_path, subject, figsize,
    source_name="astrocytes", target_name="postsynaptic neuron",
    xlabel=False, ylabel=True, xlims=(0, 500), ax_position=None):
    print("conn_source_distr()...")
    fname = f"conn_source_distr_{subject}"
    if os.path.isfile(f"{save_path}/{fname}.pkl"):
        with open(f"{save_path}/{fname}.pkl", "rb") as f:
            arr_cnt = pickle.load(f)
    else:
        arr_s = np.array(sources)
        arr_t = np.array(targets)
        set_target = set(targets)
        print(len(set_target))
        arr_cnt = []
        for i, target in enumerate(set_target):
            if i%1000 == 0:
                print(i)
            arr_cnt.append(len(set(arr_s[arr_t==target])))
        with open(f"{save_path}/{fname}.pkl", "wb") as f:
            pickle.dump(arr_cnt, f)
    fig = plt.figure(figsize=figsize)
    if max(arr_cnt) - min(arr_cnt) > 20:
        bins = list(range(min(arr_cnt)-1, max(arr_cnt)+1))
    else:
        bins = list(range(min(arr_cnt)-1, min(arr_cnt)+21))
    # verify counts
    # print(np.histogram(arr_cnt, bins=(list(range(100)))))
    plt.hist(arr_cnt, bins, color='k', ec='k')
    if xlabel:
        plt.xlabel(f"Number of connected {source_name}\nper {target_name}")
    if ylabel:
        plt.ylabel("Number of\ncases", labelpad=0)
    plt.xlim(xlims)
    if isinstance(ax_position, list):
        ax = plt.gca()
        ax.set_position(ax_position)
    else:
        plt.tight_layout()
    plt.savefig(f'{save_path}/{fname}.eps', dpi=400)
    plt.savefig(f'{save_path}/{fname}.png', dpi=400)
    plt.close()

def show_conn_distr(save_path, n=100, n_mpi=6, figsize=(2.5, 1.75)):
    for conn_name in ["n2n", "n2a", "a2n"]:
        print(conn_name + ":")
        with open(f"{save_path}/conn_{conn_name}_source.pkl", "rb") as f:
            print(f"opening {save_path}/conn_{conn_name}_source.pkl...")
            sources = pickle.load(f)
        with open(f"{save_path}/conn_{conn_name}_target.pkl", "rb") as f:
            print(f"opening {save_path}/conn_{conn_name}_target.pkl...")
            targets = pickle.load(f)
        if conn_name == "a2n":
            ax_position = [0.42, 0.2, 0.5, 0.7]
            if "bernoulli10_" in save_path:
                xlims1 = (0, 100)
                xlims2 = (0, 100)
                xlims3 = (0, 75)
            elif "bernoulli100_" in save_path:
                xlims1 = (0, 200)
                xlims2 = (0, 200)
                xlims3 = (0, 20)
            elif "bernoulli1000_" in save_path:
                xlims1 = (0, 500)
                xlims2 = (0, 500)
                xlims3 = (0, 20)
            else:
                xlims1 = (0, 500)
                xlims2 = (0, 500)
                xlims3 = (0, 20)
            conn_source_distr(sources, targets, save_path, conn_name, figsize, ax_position=ax_position, xlims=xlims1)
            conn_target_distr(sources, targets, save_path, conn_name, figsize, ax_position=ax_position, xlims=xlims2)
            conn_num_distr(sources, targets, save_path, conn_name, figsize=figsize, ax_position=ax_position, xlims=xlims3)
            conn_per_target(targets, save_path, conn_name, figsize=figsize, ax_position=ax_position)

        # plot links
        data = np.array([sources[::int(len(sources)/n)], targets[::int(len(sources)/n)]])
        ys = np.array([[1]*len(data[0]), [0]*len(data[1])])
        plt.figure(figsize=(10,10))
        plt.plot(data, ys)
        plt.xlim((0, max(sources + targets)))
        plt.savefig(os.path.join(save_path, "links_" + conn_name + ".png"),)
        plt.close()

        # print MPI matrix (not relevant)
        #df = pd.DataFrame(dict(source=sources, target=targets))
        #cnt_arr = count_arr(df, n_mpi)
        #print(cnt_arr)

if __name__ == "__main__":
    show_conn_distr(sys.argv[1])
