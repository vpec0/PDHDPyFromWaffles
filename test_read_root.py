import uproot as ur
import awkward as ak
import numpy as np

import sys, os



def run(argv) :
    global fname, NMAX

    # get data
    tree = ur.open(fname+':raw_waveforms')
    print(f'Input tree has {tree.num_entries} entries.')

    N_WFMS=50000

    do_sort = True
    counter = 0
    for t in tree.iterate(['adcs','channel'], step_size=N_WFMS, entry_stop=NMAX) :
        print(f'Starting processing batch of data: {counter}...')

        if do_sort :
            # sort by channel
            chan_sort = ak.argsort(t.channel)
            # channel runs
            chan_runs = ak.run_lengths(t.channel[chan_sort])
            channels  = ak.firsts(ak.unflatten(t.channel[chan_sort], chan_runs))
            wfms_all  = ak.unflatten(t.adcs[chan_sort], chan_runs)
            wfms_all  = ak.values_astype(wfms_all, np.float64)
            ak.copy(wfms_all)
        counter += 1


if __name__ == '__main__' :
    from sys import argv
    argv.pop(0)

    fname   = argv.pop(0)

    NMAX=-1
    if len(argv) > 0 :
        NMAX=int(argv.pop(0))

    run(argv)
