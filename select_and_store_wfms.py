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

import Selections as sel

def run() :
    fname   = sys.argv[1]
    outpref = sys.argv[2]

    # make sure the output directory exists
    print('Output prefix:', os.path.dirname(outpref))
    outdir=os.path.dirname(outpref)
    if len(outdir) :
        os.makedirs(outdir, exist_ok=True)


    # How many waveforms to be read in a batch and how many in total
    N_WFMS = 50000
    MAX_WFMS = 50000
    if len(sys.argv) > 3 :
        if sys.argv[3] == -1 or sys.argv[3].lower() == 'none' :
            MAX_WFMS = None
        else:
            MAX_WFMS = int(sys.argv[3])


    # get data
    tree = ur.open(fname+':raw_waveforms')
    print(f'Input tree has {tree.num_entries} entries.')

    # dump waforms by channel into a gzipped pickle file
    outfname = outpref + 'waveforms.pkl.gz'
    outf = OpenGZ(outfname)

    # how many ticks to use to clean up waveform selection
    PRETRIGGER = 110

    counter     = 0
    wfm_counter = 0
    for t in tree.iterate(['adcs','channel'], step_size=N_WFMS, entry_stop=MAX_WFMS) :
        print(f'Starting processing batch of data: {counter}...')
        wfm_counter += ak.num(t.channel, axis=0)

        channels,wfms_all = sel.SortByChannel(t.channel, t.adcs)

        # clean pretrigger
        wfms = sel.CleanByPretrigRMS(wfms_all, pretrigger=PRETRIGGER, rmsthld=5)
        # remove pedestal
        wfms = sel.RemovePedestal(wfms, PRETRIGGER)

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
