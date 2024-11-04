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
                print(f'Histogramming partial data... {counter}')
                PlotHists(*pickle.load(f))
            except EOFError:
                break
            counter += 1

def PlotHists(hists_by_chan, bins) :
    global outpref

    fig, ax = plt.subplots()
    print('Plotting 2d waveform hists...')
    counter = 0
    for chan,hist in hists_by_chan.items() :
        x,y = bins[chan]

        fig.clf()
        #max_loc  = np.max(hist[110:400,:])
        max_glob = np.max(hist)
        #print(f'Plotting 2d hist for channel {chan}. {max_loc = }, {max_glob = }')
        print(f'Plotting 2d hist for channel {chan}. {max_glob = }')
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
