
import awkward as ak
import numpy as np


def SortByChannel(channels,wfms):
    # sort by channel
    chan_sort = ak.argsort(channels)
    # channel runs
    chan_runs = ak.run_lengths(channels[chan_sort])
    chan_list  = ak.firsts(ak.unflatten(channels[chan_sort], chan_runs))
    # Waveforms sorted by channel
    wfms_sorted  = ak.unflatten(wfms[chan_sort], chan_runs)
    wfms_sorted  = ak.values_astype(wfms_sorted, np.float64)

    return (chan_list, wfms_sorted)


def CleanByPretrigRMS(wfms, pretrigger=50, rmsthld=5) :
    # clean pretrigger
    rms_mask = ak.nanstd(wfms[...,:pretrigger],axis=-1) < rmsthld
    return wfms[rms_mask]

def CleanByTailRMS(wfms, tail_start=400, tail_end=600, rmsthld=5) :
    # clean pretrigger
    rms_mask = ak.nanstd(wfms[...,tail_start:tail_end],axis=-1) < rmsthld
    return wfms[rms_mask]

def CleanByMeanRMS(wfms, start=100, stop=400, rmsthld=5) :
    wfmsroi = wfms[...,start:stop]
    # rescale wfms
    scale = ak.max(abs(wfmsroi), axis=-1)
    wfmsroi_scaled = wfmsroi / scale
    # get temp mean
    wfmsroi_mean = ak.mean(wfmsroi_scaled,axis=1,keepdims=True)

    # clean pretrigger
    rms_mask = ak.nanstd(wfmsroi-wfmsroi_mean*scale,axis=-1) < rmsthld
    return wfms[rms_mask]



def RemovePedestal(wfms, pretrigger=50) :
    means = ak.nanmean(wfms[...,:pretrigger],axis=-1)
    return wfms - means
