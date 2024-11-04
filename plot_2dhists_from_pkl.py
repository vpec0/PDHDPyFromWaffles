import gzip
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

fname = sys.argv[1]
outpref = sys.argv[2]

NMAX=-1
if len(sys.argv) > 3 :
    NMAX=sys.argv[3]

with gzip.open(fname, 'rb') as f :
    data = pickle.load(f)


fig, ax = plt.subplots()


print('Plotting 2d waveform hists...')
counter = 0
for chan,(hist, x, y) in data.items() :
    if NMAX != -1 and counter >= NMAX :
        break
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
