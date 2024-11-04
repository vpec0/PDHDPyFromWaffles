import uproot as ur
import awkward as ak
import numpy as np

import pickle as pkl
import gzip

import sys, os



def run() :
    fname   = sys.argv[1]
    outfname = sys.argv[2]

    # make sure the output directory exists
    outdir=os.path.dirname(outfname)
    print('Output prefix:', outdir)
    if len(outdir) :
        os.makedirs(outdir, exist_ok=True)



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
    outf = OpenGZ(outfname, compresslevel=3)

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

        # store selected waveforms
        pkl.dump((channels,wfms_all), outf)
        counter += 1

    # close output wfm file
    outf.close()
    print(f'Processed {wfm_counter} waveforms.')




def SaveToGZ(fname, data) :
    with gzip.open(fname, 'wb', compresslevel=3) as outf :
        dump(data, outf)


def OpenGZ(fname, compresslevel=3) :
    return gzip.open(fname, 'wb', compresslevel=compresslevel)


if __name__ == '__main__' :
    run()
