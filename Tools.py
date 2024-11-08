
import awkward as ak
import numpy as np
import gzip
import pickle
import uproot as ur
from pathlib import Path


import Selections as sel

class RetrieveData :
    def __init__(self, fname, niter=None, maxwfms=None) :
        self.type = 'unknown'
        self.ext = Path(fname).suffix
        if self.ext == '.root' :
            self.type = 'root'
        elif self.ext == '.gz' :
            self.type = 'gzip'

        self.niter = niter
        self.maxwfms = maxwfms
        self.counter = 0
        self.wfmcounter = 0

        if self.type == 'root' :
            # print out available data
            with ur.open(fname+':raw_waveforms') as tree :# this should not be reading data in, only the size
                print(f'Input tree has {tree.num_entries} entries.')
            self.iterable = ur.iterate(fname+':raw_waveforms', ['adcs','channel'], step_size=50000, entry_stop=maxwfms)

        elif self.type == 'gzip' :
            self.f =  gzip.open(fname, 'rb')


    def __iter__(self) :
        return self


    def __next__(self) :
        if self.niter is not None and self.counter >= self.niter :
            raise StopIteration
        if self.maxwfms is not None and self.wfmcounter >= self.maxwfms :
            raise StopIteration
        self.counter += 1

        if self.type == 'root' :
            t = next(self.iterable)
            self.wfmcounter += ak.count(t.channel)

            return sel.SortByChannel(t.channel, t.adcs)

        elif self.type == 'gzip' :
            try:
                data = pickle.load(self.f)
                if isinstance(data, dict) :
                    channels = data['channels']
                    wfms = data['wfms']
                else :
                    channels,wfms = data
                    self.wfmcounter += ak.sum(ak.num(wfms,axis=1), axis = 0)
                return channels, wfms
            except EOFError:
                raise StopIteration

        else :
            print(f'Unkknown type of file on the input: {self.ext}')
            raise StopIteration

    def __del__(self) :
        if self.type == 'gzip' :
            self.f.close()


def Create2DHists(wfms, channels, bins=None, range=None) :
    '''
    Expects:
      wfms - 3D array wfms[chanIt][wfmIt][wfmDigitIt] of ADCs.
      channels - list of channel IDs: channels[chanIt]
    '''

    hists_by_chan = dict()
    bins_by_chan  = dict()
    for chan,chwfms in zip(channels, wfms) :
        nwfms = len(chwfms)
        if nwfms == 0  :
            continue
        size = len(chwfms[0])
        # determine number of bins and the range of the histogram
        if bins is None :
            # by default, create bin for each TDC tick and 600 bins for the ADC range
            bins  = (size,600)
        if range == None :
            # by default, use full TDC range and zoomed ADC range: ((TDCLow,TDCHi), (ADCLow,ADCHi))
            range = ((0,size), (-500,100))
        # build array of TDC ticks
        x = np.repeat( [np.arange(size,dtype=np.float64)], len(chwfms), axis=0 )
        # creates histogram
        hist, xedges, yedges = np.histogram2d( x.flatten(), chwfms.to_numpy().flatten(), bins=bins, range=range )
        hists_by_chan[chan] = hist
        bins_by_chan[chan]  = (xedges, yedges)

    return hists_by_chan, bins_by_chan


def Add2DHists( new_hists_by_chan,new_bins_by_chan, hists_by_chan,bins_by_chan ) :
    for chan in new_hists_by_chan.keys() :
        h = new_hists_by_chan[chan]
        b = new_bins_by_chan[chan]
        if chan not in hists_by_chan.keys() :
            hists_by_chan[chan] = np.zeros_like(h)
            bins_by_chan[chan]  = b
            # FIXME: check that this hist has the same bins as the one stored
        hists_by_chan[chan] = hists_by_chan[chan] + h
