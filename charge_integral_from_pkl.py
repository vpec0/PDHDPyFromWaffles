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
    with gzip.open(fname, 'rb') as f :
        counter = 0
        while True:
            # if counter > 0 :
            #     break
            try:
                channels,wfms = pkl.load(f)
                print(f'Histogramming partial data... {counter}')
                ProcessBatch(hists_by_chan, counts_by_chan, channels, wfms)
            except EOFError:
                break
            counter += 1

    PlotHists(hists_by_chan, counts_by_chan)

def ProcessBatch(hists_by_chan, counts_by_chan, channels, wfms) :
    global qrange

    # Get ROI
    roi = (125,200)
    wfms = wfms[...,roi[0]:roi[1]]

    thld = 10
    thld_mask = wfms < -thld
    # integrate, limit ROI to first 200 ticks
    charges = -ak.sum(wfms[thld_mask], axis=-1)

    for chan,qs in zip(channels, charges) :
        if len(qs) == 0 :
            continue
        hist,_ = np.histogram(qs.to_numpy().flatten(), bins=500, range=qrange)
        if chan not in hists_by_chan.keys() :
            hists_by_chan[chan]    = ak.zeros_like(hist)
            counts_by_chan[chan]   = 0

        hists_by_chan[chan]    = hists_by_chan[chan] + hist
        counts_by_chan[chan]   = counts_by_chan[chan] + len(qs)

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


def SaveToGZ(fname, data) :
    with gzip.open(fname, 'wb', compresslevel=3) as outf :
        dump(data, outf)


def OpenGZ(fname) :
    return gzip.open(fname, 'wb', compresslevel=3)


if __name__ == '__main__' :
    run()
