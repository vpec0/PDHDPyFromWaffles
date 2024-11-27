import gzip
import pickle
import sys, os, argparse
import uproot as ur
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import awkward as ak
from scipy.stats import norm

import Selections as sel
import Tools as tls

mpl.use('agg')


outpref = ''

def run() :
    global outpref

    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    parser.add_argument('outpref', nargs='?', default='')
    parser.add_argument('-n','--nmax', default=10, type=int)

    args = parser.parse_args()


    fname = args.fname
    outpref = args.outpref

    # make sure the output directory exists
    print('Output prefix:', outpref)
    outdir=os.path.dirname(outpref)
    if len(outdir) :
        os.makedirs(outdir, exist_ok=True)


    # How many waveforms to be read in a batch and how many in total
    N_WFMS = 50000
    MAX_WFMS = args.nmax

    long_wfm = 0

    nwfms = 50

    counter = 0
    for channels, wfms in tls.RetrieveData(fname, maxwfms=MAX_WFMS) :
        for ch,chwfms in zip(channels,wfms) :
            if len(chwfms[0]) > 1024 :
                print(f'Channel {ch} has waveforms of length {len(chwfms[0])}')

            if len(chwfms) < nwfms :
                continue

            long_wfm = ak.concatenate(chwfms[:nwfms])
            break
        break


    template = -np.loadtxt('templates/SPE_DAPHNE2_HPK_2024.dat')


    template = np.pad(template,(0,len(long_wfm)-len(template)) , constant_values=0.)
    template_fft = np.fft.rfft(template)
    wfm_fft      = np.fft.rfft(long_wfm)

    print(template)
    print(len(template_fft))
    print(len(wfm_fft))

    filter = norm(len(long_wfm)*0.5, 4.).pdf(np.arange(len(long_wfm)))
    filter_fft = abs(np.fft.rfft(filter))


    deco_fft = wfm_fft* (filter_fft/template_fft)

    print(len(deco_fft))
    print(deco_fft)

    deco = np.fft.irfft(deco_fft)


    plt.cla()
    plt.plot(template)
    plt.savefig(outpref+'template.png')

    plt.cla()
    plt.plot(template_fft)
    plt.savefig(outpref+'template_fft.png')

    plt.cla()
    plt.plot(filter)
    plt.savefig(outpref+'filter.png')
    plt.cla()
    plt.plot(filter_fft)
    plt.savefig(outpref+'filter_fft.png')

    plt.cla()
    plt.plot(wfm_fft)
    plt.savefig(outpref+'long_wfm_fft.png')
    plt.cla()
    plt.plot(deco_fft)
    plt.savefig(outpref+'long_wfm_fft_deco.png')

    plt.cla()
    plt.plot(long_wfm)
    plt.savefig(outpref+'long_wfm.png')

    plt.cla()
    plt.plot(deco)
    plt.savefig(outpref+'long_wfm_deco.png')


    with ur.recreate(outpref+'deconv_hist.root') as outf :
        outf['wfm_deco'] = np.histogram(np.arange(len(deco)), bins=np.arange(len(deco)+1), weights=deco)


if __name__ == '__main__' :
    run()
