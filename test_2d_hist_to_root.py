import awkward as ak
import numpy as np
import uproot as ur


contents = np.reshape(np.arange(25), (5,5))
edges = np.arange(6)

with ur.recreate('test_2d_hist.root') as f :
    f['h2d'] = (contents, edges, edges)
