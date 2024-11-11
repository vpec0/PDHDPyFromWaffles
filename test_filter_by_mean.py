import sys, os

import awkward as ak
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
matplotlib.use('agg')


from Tools import *
from Selections import *

def run() :
    fname = sys.argv[1]
    outpref = sys.argv[2]

    # make sure the output directory exists
    print('Output prefix:', outpref)
    outdir=os.path.dirname(outpref)
    if len(outdir) :
        os.makedirs(outdir, exist_ok=True)


    for channels, wfms in RetrieveData(fname, maxwfms=1000) :
        wfmsroi = wfms[...,100:400]
        # rescale wfms
        scale = abs(ak.min(wfmsroi, axis=-1))
        wfmsroi_scaled = wfmsroi / scale
        # get temp mean
        n_wfms = ak.num(wfmsroi,axis=1)
        wfmsroi_mean = ak.mean(wfmsroi_scaled,axis=1,keepdims=True)

        plt.plot(ak.ravel(wfmsroi_mean[0]))
        plt.savefig(outpref+'mean_wfm.png')

        wfmsroi_mean[0].show()

        wfm_res = wfmsroi-wfmsroi_mean*scale

        plt.cla()
        plt.plot(wfm_res[0][0])
        plt.savefig(outpref+'wfm_res.png')


        tot_wfms = ak.count(wfms)

        # clean pretrigger
        for rmsthld in range(5,16,2) :
            rms_mask = ak.nanstd(wfm_res,axis=-1) < rmsthld

            filtered_wfs = ak.count(wfms[rms_mask])

            plt.clf()
            for wfm in wfms[rms_mask][0,:5] :
                plt.plot(wfm[0:500])
            plt.savefig(outpref+f'filtered_wfms_rmsthld{rmsthld}.png')

            print(f'Eff of RMS {rmsthld} thld is {filtered_wfs/tot_wfms}')



if __name__ == '__main__' :
    run()
