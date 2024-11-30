from parsing_tools import parse_coinco_with_attributes, matching
from bert_dewatermarking import replacement_coinco
import numpy as np
all_dists_bad = []
all_dists_good = []

_ , cdps = parse_coinco_with_attributes('coinco.xml')

for cdp in cdps[:1000]:
    g, b = replacement_coinco(cdp)
    all_dists_bad.extend(b)
    all_dists_good.extend(g)
    
print(len(all_dists_good))
print(len(all_dists_bad))

np.save('dists_good.npy', np.array(all_dists_good))
np.save('dists_bad.npy', np.array(all_dists_bad))