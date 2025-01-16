import uproot as ur
import sys


data=0
with ur.open(sys.argv[1]+':raw_waveforms') as tree :
    data = tree.arrays(['adcs','channel'],entry_stop=int(sys.argv[3]))


with ur.recreate(sys.argv[2]) as outf :
    outf['raw_waveforms'] = {'adcs':data.adcs, 'channel':data.channel}
