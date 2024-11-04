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

qrange = (0,1e4)
outpref = ''

def run() :
    global outpref
    fname   = sys.argv[1]
    outpref = sys.argv[2]

    # make sure the output directory exists
    print('Output prefix:', os.path.dirname(outpref))
    os.makedirs(os.path.dirname(outpref), exist_ok=True)

    hists_by_chan  = dict()
    counts_by_chan = dict()

    # prepare output
    outf = OpenGZ(outpref+'selected_zeroed_wfms.pkl.gz')


    N_WFMS=50000
    NMAX=-1
    if len(sys.argv) > 3 :
        NMAX=int(sys.argv[3])

    do_sort = True
    counter       = 0
    total_wfms    = 0
    selected_wfms = 0
    # get data
    # tree = ur.open(fname+':raw_waveforms')
    # print(f'Input tree has {tree.num_entries} entries.')
    for t in ur.iterate(fname+':raw_waveforms',['adcs','channel'], step_size=N_WFMS, entry_stop=NMAX) :
        print(f'Starting processing batch of data: {counter}...')

        # sort by channel
        chan_sort = ak.argsort(t.channel)
        # channel runs
        chan_runs = ak.run_lengths(t.channel[chan_sort])
        channels  = ak.firsts(ak.unflatten(t.channel[chan_sort], chan_runs))
        wfms_all  = ak.unflatten(t.adcs[chan_sort], chan_runs)
        wfms_all  = ak.values_astype(wfms_all, np.float64)
        ProcessBatch(outf, hists_by_chan, counts_by_chan, channels, wfms_all)
        counter += 1
        total_wfms += ak.count(t.channel)
        selected_wfms += ak.sum(counts_by_chan.values())

    print(f'Number of input waveforms:    {total_wfms}')
    print(f'Number of selected waveforms: {selected_wfms}')
    print(f'Selection efficiency:         {selected_wfms/total_wfms:.3f}')

    #PlotHists(hists_by_chan, counts_by_chan)

def ProcessBatch(outf, hists_by_chan, counts_by_chan, channels, wfms) :
    global qrange

    # selection
    # clean pretrigger
    PRETRIGGER=125
    rms_mask = ak.nanstd(wfms[...,:PRETRIGGER],axis=-1) < 5
    wfms = wfms[rms_mask]
    # remove pedestal
    means = ak.nanmean(wfms[...,:PRETRIGGER],axis=-1)
    wfms = wfms - means


    # Get ROI
    start=PRETRIGGER
    stop =180
    wfms_for_charge = wfms[...,start:stop]

    # integrate
    charges = -ak.sum(wfms_for_charge, axis=-1)

    # for chan,qs in zip(channels, charges) :
    #     if len(qs) == 0 :
    #         continue
    #     hist,_ = np.histogram(qs.to_numpy().flatten(), bins=500, range=qrange)
    #     if chan not in hists_by_chan.keys() :
    #         hists_by_chan[chan]    = ak.zeros_like(hist)
    #         counts_by_chan[chan]   = 0

    #     hists_by_chan[chan]    = hists_by_chan[chan] + hist
    #     counts_by_chan[chan]   = counts_by_chan[chan] + len(qs)

    pkl.dump({'channels':channels, 'wfms':wfms, 'charges':charges}, outf)

def PlotHists(hists_by_chan, counts_by_chan):
    global qrange
    global outpref

    fig, ax = plt.subplots()

    print('Plotting charge histograms...')
    for chan,qh in hists_by_chan.items() :
        ax.cla()
        plt.title(f'Channel {chan} made of {counts_by_chan[chan]} waveforms')
        plt.xlabel(r'Integrated charge [ADC$\times$tick]')
        plt.ylabel('Count')
        plt.stairs(qh, np.histogram_bin_edges(None, len(qh), qrange))
        fig.savefig(outpref+f'charge_{chan}.png')
    print('Done.')

def OpenGZ(fname, compresslevel=3) :
    return gzip.open(fname, 'wb', compresslevel=compresslevel)


if __name__ == '__main__' :
    run()
