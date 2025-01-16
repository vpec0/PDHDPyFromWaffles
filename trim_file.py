import uproot as ur
import sys

BATCH_SIZE=20000
MAX_SIZE=int(sys.argv[3])

outf = ur.recreate(sys.argv[2])

stored = 0
iteration = 0
for data in  ur.iterate(sys.argv[1]+':raw_waveforms',['adcs','channel'],step_size=BATCH_SIZE) :
    if stored >= MAX_SIZE :
        break

    print(f'Storing batch {iteration} of size {len(data.channel)}')

    if stored == 0 :
        outf['raw_waveforms'] = {'adcs':data.adcs, 'channel':data.channel}
    else :
        outf['raw_waveforms'].extend({'adcs':data.adcs, 'channel':data.channel})

    stored += len(data.channel)
    iteration += 1

outf.close()
