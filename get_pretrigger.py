import uproot as ur
import awkward as ak
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('agg')

import pickle as pkl

import sys, os


fname   = sys.argv[1]
outpref = sys.argv[2]

# make sure the output directory exists
print('Output prefix:', os.path.dirname(outpref))
os.makedirs(os.path.dirname(outpref), exist_ok=True)
subdirs=['avg', '2dhist']
for subd in subdirs :
    os.makedirs(os.path.dirname(outpref)+'/'+subd, exist_ok=True)
    


N_WFMS = 10000
MAX_WFMS = 50000
if len(sys.argv) > 3 :
    if sys.argv[3] == -1 or sys.argv[3].lower() == 'none' :
        MAX_WFMS = None
    MAX_WFMS = int(sys.argv[3])


avg_by_chan     = dict()
counts_by_chan  = dict()
hists_by_chan   = dict()
# get data
tree = ur.open(fname+':raw_waveforms')
#print(f'Input tree has {tree.num_entries} entries. Will process {N_WFMS}.')

xedges = 0
yedges = 0

x_y_by_chan = dict()

PRETRIGGER = 100

counter = 0
for t in tree.iterate(['adcs','channel'], step_size=N_WFMS, entry_stop=MAX_WFMS) :
    # sort by channel
    chan_sort = ak.argsort(t.channel)
    # channel runs
    chan_runs = ak.run_lengths(t.channel[chan_sort])
    channels  = ak.firsts(ak.unflatten(t.channel[chan_sort], chan_runs))
    wfms      = ak.unflatten(t.adcs[chan_sort][...,:PRETRIGGER], chan_runs)

    for chan,chwfms in zip(channels, wfms) :
        if chan in avg_by_chan.keys() :
            avg_by_chan[chan]    = avg_by_chan[chan] + ak.sum(chwfms, axis=0)
            counts_by_chan[chan] = counts_by_chan[chan] + len(chwfms)
        else:
            avg_by_chan[chan] = ak.sum(chwfms, axis=0)
            counts_by_chan[chan] = len(chwfms)
    
    print(f'Histogramming partial data... {counter}')
    for chan,chwfms in zip(channels, wfms) :
        x = np.array(list(range(PRETRIGGER))*len(chwfms))
        hist, xedges, yedges = np.histogram2d( x, chwfms.to_numpy().flatten(), bins=[PRETRIGGER,500], range=((0,PRETRIGGER),(7500,8500)) )
        if chan in hists_by_chan.keys() :
            hists_by_chan[chan] = hists_by_chan[chan] + hist
        else :
            hists_by_chan[chan] = hist
            x_y_by_chan[chan] = (xedges, yedges)


    counter += 1


to_scale = avg_by_chan
for chan,wfm in to_scale.items() :
    avg_by_chan[chan] = wfm / counts_by_chan[chan]
    
fig, ax = plt.subplots()
    
plot_avg=True
if plot_avg :
    print('Plotting average waveforms...')
    for chan,wfm in avg_by_chan.items() :
        ax.cla()
        plt.plot(ak.to_numpy(wfm))
        fig.savefig(outpref+f'avg/avg_wfm_{chan}.png')
    print('Done.')



plot_2dhists = True
# plot coloured 2D hists
if plot_2dhists : 
    print('Plotting 2d waveform hists...')
    for chan,hist in hists_by_chan.items() :
        #aprint(f'Plotting 2d hist for channel {chan}')    
        fig.clf()
        x,y = x_y_by_chan[chan]
        #max = np.max(hist)*1.1
        #cm = plt.pcolormesh(xedges,yedges,hist.T, cmap='RdBu', vmin=-max, vmax=max)
        cm = plt.pcolormesh(x,y,hist.T, cmap='summer', norm=mcolors.LogNorm())
        plt.colorbar()
        #plt.pcolormesh(xedges, yedges, hist)
        fig.savefig(outpref+f'2dhist/2dhist_wfm_{chan}.png')
    print('Done.')


data = dict()
for chan,hist in hists_by_chan.items() :
    x,y = x_y_by_chan[chan]
    data[chan] = (hist, x, y)

import gzip as gz
with gz.open('pretrigger_2dhists.pkl.gz', 'wb', compresslevel=3) as outpkl :
    pkl.dump(data, outpkl)

