import uproot as ur
import awkward as ak
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('agg')

import pickle as pkl
import gzip

import sys, os



def run() :
    fname   = sys.argv[1]
    outpref = sys.argv[2]

    plot_charge  = False
    plot_avg     = False
    plot_2dhists = False

    # make sure the output directory exists
    print('Output prefix:', os.path.dirname(outpref))
    os.makedirs(os.path.dirname(outpref), exist_ok=True)
    subdirs=['avg', '2dhist', 'charge']
    for subd in subdirs :
        os.makedirs(os.path.dirname(outpref)+'/'+subd, exist_ok=True)



    N_WFMS = 50000
    MAX_WFMS = 50000
    if len(sys.argv) > 3 :
        if sys.argv[3] == -1 or sys.argv[3].lower() == 'none' :
            MAX_WFMS = None
        else:
            MAX_WFMS = int(sys.argv[3])


    avg_by_chan     = dict()
    counts_by_chan  = dict()
    hists_by_chan   = dict()
    qs_by_chan      = dict()
    # get data
    tree = ur.open(fname+':raw_waveforms')
    print(f'Input tree has {tree.num_entries} entries.')

    xedges = 0
    yedges = 0

    x_y_by_chan = dict()

    PRETRIGGER = 110

    # dump waforms by channel into a gzipped pickle file
    outfname = outpref + 'waveforms.pkl.gz'
    outf = OpenGZ(outfname)


    counter = 0
    wfm_counter = 0
    for t in tree.iterate(['adcs','channel'], step_size=N_WFMS, entry_stop=MAX_WFMS) :
        print(f'Starting processing batch of data: {counter}...')
        wfm_counter += ak.num(t.channel, axis=0)
        # sort by channel
        chan_sort = ak.argsort(t.channel)
        # channel runs
        chan_runs = ak.run_lengths(t.channel[chan_sort])
        channels  = ak.firsts(ak.unflatten(t.channel[chan_sort], chan_runs))
        wfms_all  = ak.unflatten(t.adcs[chan_sort], chan_runs)
        wfms_all  = ak.values_astype(wfms_all, np.float64)

        # clean pretrigger
        rms_mask = ak.nanstd(wfms_all[...,:PRETRIGGER],axis=-1) < 5
        wfms = wfms_all[rms_mask]
        # remove pedestal
        means = ak.nanmean(wfms[...,:PRETRIGGER],axis=-1)
        wfms = wfms - means

        # store selected waveforms
        pkl.dump((channels,wfms), outf)
        # wfms.show()
        # channels.show()

        counter += 1

    # close output wfm file
    outf.close()
    print(f'Processed {wfm_counter} waveforms.')




def SaveToGZ(fname, data) :
    with gzip.open(fname, 'wb', compresslevel=3) as outf :
        dump(data, outf)


def OpenGZ(fname) :
    return gzip.open(fname, 'wb', compresslevel=3)


if __name__ == '__main__' :
    run()
