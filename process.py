import uproot as ur
import awkward as ak
import numpy as np
from matplotlib import pyplot as plt
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

counter = 0
for t in tree.iterate(['adcs','channel'], step_size=N_WFMS, entry_stop=MAX_WFMS) :
    # sort by channel
    chan_sort = ak.argsort(t.channel)
    # channel runs
    chan_runs = ak.run_lengths(t.channel[chan_sort])
    channels  = ak.firsts(ak.unflatten(t.channel[chan_sort], chan_runs))
    wfms      = ak.unflatten(t.adcs[chan_sort], chan_runs)

    for chan,chwfms in zip(channels, wfms) :
        if chan in avg_by_chan.keys() :
            avg_by_chan[chan]    = avg_by_chan[chan] + ak.sum(chwfms, axis=0)
            counts_by_chan[chan] = counts_by_chan[chan] + len(chwfms)
        else:
            avg_by_chan[chan] = ak.sum(chwfms, axis=0)
            counts_by_chan[chan] = len(chwfms)
    
    print(f'Histogramming partial data... {counter}')
    for chan,chwfms in zip(channels, wfms) :
        x = np.array(list(range(1024))*len(chwfms))
        hist, xedges, yedges = np.histogram2d( x, chwfms.to_numpy().flatten(), bins=[1024,500], range=((0,1024),(5000,10000)) )
        if chan in hists_by_chan.keys() :
            hists_by_chan[chan] = hists_by_chan[chan] + hist
        else :
            hists_by_chan[chan] = hist
            x_y_by_chan[chan] = (xedges, yedges)


    counter += 1



fig, ax = plt.subplots()
    
plot_avg=False
if plot_avg :
    print('Plotting average waveforms...')
    for chan,wfm in zip(channels, avg) :
        ax.cla()
        fig.plot(ak.to_numpy(wfm))
        fig.savefig(outpref+f'avg/avg_wfm_{chan}.png')
    print('Done.')



plot_2dhists = False
# plot coloured 2D hists
if plot_2dhists : 
    print('Plotting 2d waveform hists...')
    for chan,hist in hists_by_chan.items() :
        print(f'Plotting 2d hist for channel {chan}')    
        fig.clf()
        x,y = x_y_by_chan[chan]
        #max = np.max(hist)*1.1
        #cm = plt.pcolormesh(xedges,yedges,hist.T, cmap='RdBu', vmin=-max, vmax=max)
        cm = plt.pcolormesh(x,y,hist.T, cmap='hot')
        plt.colorbar()
        #plt.pcolormesh(xedges, yedges, hist)
        fig.savefig(outpref+f'2dhist/2dhist_wfm_{chan}.png')
    print('Done.')

#n_weird_wfms = ak.sum(ak.num(wfms, axis=-1) != 1024)

data = dict()
for chan,hist in hists_by_chan.items() :
    x,y = x_y_by_chan[chan]
    data[chan] = (hist, x, y)

import gzip as gz
with gz.open('2dhists.pkl.gz', 'wb', compresslevel=3) as outpkl :
    pkl.dump(data, outpkl)


    
#channels.show()
# print(f'NChannels = {len(chan_runs)}')
# #t.channel[chan_sort].show()
# #chan_runs.show()
# print(f'Total read waveforms: {ak.sum(chan_runs)}')
# print(f'Number of waveforms not 1024 in length: {n_weird_wfms}')
# #wfms.show()
