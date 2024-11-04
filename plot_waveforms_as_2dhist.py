import gzip
import pickle
import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import awkward as ak

outpref = ''

def run() :
    global outpref
    fname = sys.argv[1]
    outpref = sys.argv[2]

    # make sure the output directory exists
    print('Output prefix:', os.path.dirname(outpref))
    os.makedirs(os.path.dirname(outpref), exist_ok=True)


    NMAX=-1
    if len(sys.argv) > 3 :
        NMAX=int(sys.argv[3])


    hists_by_chan = dict()
    bins_by_chan  = dict()
    counter = 0
    with gzip.open(fname, 'rb') as f :
        while True:
            if NMAX > -1 and counter >= NMAX :
                break
            try:
                data = pickle.load(f)
                if isinstance(data, dict) :
                    channels = data['channels']
                    wfms = data['wfms']
                else :
                    channels,wfms = data

                print(f'Histogramming partial data... {counter}')
                AddData(hists_by_chan, bins_by_chan, channels, wfms)
            except EOFError:
                break
            counter += 1
    SaveToGZ(outpref+'all_2dhists.pkl.gz', (hists_by_chan, bins_by_chan))
    #PlotHists(hists_by_chan, bins_by_chan)

def AddData(hists_by_chan, bins_by_chan, channels, wfms) :
    print(f'Processing {ak.num(ak.flatten(wfms,axis=1),axis=0)} waveforms')
    # wfms.show()
    # channels.show()

    for chan,chwfms in zip(channels, wfms) :
        nwfms = len(chwfms)
        if nwfms == 0  :
            continue
        size = len(chwfms[0])
        x = np.repeat( [np.arange(size,dtype=np.float64)], len(chwfms), axis=0 )
        hist, xedges, yedges = np.histogram2d( x.flatten(), chwfms.to_numpy().flatten(), bins=[size,600], range=((0,size),(-500,100)) )
        if chan not in hists_by_chan.keys() :
            hists_by_chan[chan] = np.zeros_like(hist)
            bins_by_chan[chan]   = (xedges, yedges)
        hists_by_chan[chan] = hists_by_chan[chan] + hist


def PlotHists(hists_by_chan, bins) :
    global outpref

    fig, ax = plt.subplots()
    print('Plotting 2d waveform hists...')
    counter = 0
    for chan,hist in hists_by_chan.items() :
        x,y = bins[chan]

        fig.clf()
        max_loc  = np.max(hist[110:400,:])
        max_glob = np.max(hist)
        print(f'Plotting 2d hist for channel {chan}. {max_loc = }, {max_glob = }')
        plt.title(f'Channel {chan}')
        plt.xlabel('Time ticks')
        plt.ylabel('ADC')

        #cm = plt.pcolormesh(xedges,yedges,hist.T, cmap='RdBu', vmin=-max, vmax=max)
        cm = plt.pcolormesh(x,y,hist.T, cmap='summer', norm=mcolors.LogNorm(vmin=1, vmax=max_glob))
        plt.colorbar()
        plt.tight_layout()
        #plt.pcolormesh(xedges, yedges, hist)
        fig.savefig(outpref+f'2dhist_wfm_{chan}.png')

        counter += 1

    print('Done.')

def SaveToGZ(fname, data) :
    from pickle import dump as pkldump
    with gzip.open(fname, 'wb', compresslevel=3) as outf :
        pkldump(data, outf)



if __name__ == '__main__' :
    run()
