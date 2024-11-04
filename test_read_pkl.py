import awkward as ak
import numpy as np

import pickle as pkl
import gzip

import sys, os

#mpl.use('agg')
#print(mpl.get_backend())


def run(argv) :
    global fname, NMAX
    counter = 0
    with gzip.open(fname, 'rb') as f :
        while True:
            if NMAX != -1 and counter >= NMAX :
                break
            try:
                data = pkl.load(f)
                print(f'Reading partial waveform data... {counter}')
            except EOFError:
                break
            counter += 1




if __name__ == '__main__' :
    from sys import argv
    argv.pop(0)

    fname   = argv.pop(0)

    NMAX=-1
    if len(argv) > 0 :
        NMAX=int(argv.pop(0))

    run(argv)
