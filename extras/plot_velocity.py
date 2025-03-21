import json
import matplotlib
from matplotlib.pyplot import legend

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from pathlib import Path
plt.style.use(['science','ieee'])

def _get_vels(glob_pattern):
    vels = []
    for j_file in Path.cwd().glob(glob_pattern):
        with open(j_file, 'r') as f:
            data = json.load(f)
        # don't care about individuals, just the group
        for v in data:
            vels.append(data[v])

    flat_vels = list(np.concatenate(vels))
    return flat_vels

def get_normalised_bins(glob_pattern, frame_rate_correction=1):
    vels = _get_vels(glob_pattern)
    c,b = np.histogram(vels,bins=30,range=(0,50))
    b = b*frame_rate_correction
    c = c/sum(c)
    return c,b


#flat_vels_dance = get_vels('output/dance*.json')
#flat_vels_mot = get_vels('output/MOT20*.json')
#flat_vels_mot17 = get_vels('output/MOT17*.json')
#counts, bins = np.histogram(flat_vels_dance,bins=30,range=(0,50))
#counts_mot, bins_mot = np.histogram(flat_vels_mot,bins=30,range=(0,50))
#counts_mot17, bins_mot17 = np.histogram(flat_vels_mot17,bins=30,range=(0,50))

#bins_mot = bins_mot*20/25  # correct for frame rate
#bins_mot17 = bins_mot17*20/25
#counts = counts /sum(counts)
#counts_mot = counts_mot /sum(counts_mot)
#counts_mot17 = counts_mot17/sum(counts_mot17)

fig = plt.figure(figsize= (6.4, 4.8))
glob_list = ['output/dance*.json', 'output/MOT20*.json', 'output/MOT17*.json','output/sport*.json']
correction = [1,1,1,1]
label = ['DanceTrack','MOT20','MOT17','SportsMOT']
#frame_rate = [25,20,14,25]
for g,lab in zip(glob_list,label):
    corr=1
    print(f'Processing {g}, correction {corr}')
    counts,bins = get_normalised_bins(g,corr)
    plt.stairs(counts, bins,label=lab)

#plt.stairs(counts_mot, bins_mot)
#plt.stairs(counts_mot17, bins_mot17)

plt.xlabel('Pixel Velocity pixel/frame')
plt.ylabel('Normalised counts')
plt.legend()
plt.show()
fig.savefig('awesome_figure.png', dpi=300, bbox_inches='tight')