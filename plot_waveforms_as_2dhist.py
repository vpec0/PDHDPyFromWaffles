import gzip
import pickle
import sys, os
import uproot as ur
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import awkward as ak

import Selections as sel
import Tools as tls

mpl.use('agg')


outpref = ''

def run() :
    global outpref
    fname = sys.argv[1]
    outpref = sys.argv[2]

    # make sure the output directory exists
    print('Output prefix:', os.path.dirname(outpref))
    os.makedirs(os.path.dirname(outpref), exist_ok=True)


    # How many waveforms to be read in a batch and how many in total
    N_WFMS = 50000
    MAX_WFMS = 50000
    if len(sys.argv) > 3 :
        if sys.argv[3] == -1 or sys.argv[3].lower() == 'none' :
            MAX_WFMS = None
        else:
            MAX_WFMS = int(sys.argv[3])

    hists_by_chan = dict()
    bins_by_chan  = dict()
    counter = 0
    for channels, wfms in tls.RetrieveData(fname, maxwfms=MAX_WFMS) :
        print(f'Histogramming partial data... {counter}')
        wfms = FilterWfms(wfms)
        AddData(hists_by_chan, bins_by_chan, channels, wfms)
        counter += 1

    SaveToGZ(outpref+'all_2dhists.pkl.gz', (hists_by_chan, bins_by_chan))
    PlotHists(hists_by_chan, bins_by_chan)


def FilterWfms(wfms) :
    wfms = sel.CleanByPretrigRMS(wfms, pretrigger=120, rmsthld=5)
    # wfms = sel.CleanByTailRMS(wfms,tail_start=400,tail_end=600,rmsthld=5)
    wfms = sel.RemovePedestal(wfms,pretrigger=120)
    wfms = sel.CleanByMeanRMS(wfms,start=100,stop=400,rmsthld=5)

    return wfms


def AddData(hists_by_chan, bins_by_chan, channels, wfms) :
    print(f'Processing {ak.num(ak.flatten(wfms,axis=1),axis=0)} waveforms')
    # wfms.show()
    # channels.show()

    hists, bins = tls.Create2DHists(wfms, channels, bins=(600,600), range=((0,600),(-500,100)))
    tls.Add2DHists(hists, bins, hists_by_chan, bins_by_chan)



def PlotHists(hists_by_chan, bins) :
    global outpref

    fig, ax = plt.subplots()
    ax.set_xlabel('Time ticks')
    ax.set_ylabel('ADC')
    fig.tight_layout()
    cb = fig.colorbar(mpl.cm.ScalarMappable())
    print('Plotting 2d waveform hists...')
    counter = 0
    for chan,hist in hists_by_chan.items() :
        x,y = bins[chan]

        ax.cla()
        #max_loc  = np.max(hist[110:400,:])
        max_glob = np.max(hist)
        #print(f'Plotting 2d hist for channel {chan}. {max_loc = }, {max_glob = }')
        ax.set_title(f'Channel {chan}')

        #cm = plt.pcolormesh(xedges,yedges,hist.T, cmap='RdBu', vmin=-max, vmax=max)
        cm = plt.pcolormesh(x,y,hist.T, cmap='summer', norm=mcolors.LogNorm(vmin=1, vmax=max_glob))
            #plt.pcolormesh(xedges, yedges, hist)
        cb.update_normal(cm)
        fig.savefig(outpref+f'2dhist_wfm_{chan}.png')

        counter += 1

    print('Done.')

def SaveToGZ(fname, data) :
    from pickle import dump as pkldump
    with gzip.open(fname, 'wb', compresslevel=3) as outf :
        pkldump(data, outf)



if __name__ == '__main__' :
    run()
