import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from pathlib import Path
plt.style.use(['science','ieee'])

def get_vels(glob_pattern):
    vels = []
    for j_file in Path.cwd().glob(glob_pattern):
        with open(j_file, 'r') as f:
            data = json.load(f)
        # don't care about individuals, just the group
        for v in data:
            vels.append(data[v])

    flat_vels = list(np.concatenate(vels))
    return flat_vels

flat_vels_dance = get_vels('output/dance*.json')
flat_vels_mot = get_vels('output/MOT*.json')

counts, bins = np.histogram(flat_vels_dance,bins=30,range=(0,50))
counts_mot, bins_mot = np.histogram(flat_vels_mot,bins=30,range=(0,50))

bins_mot = bins_mot*20/25  # correct for frame rate
counts = counts /sum(counts)
counts_mot = counts_mot /sum(counts_mot)
fig = plt.figure(figsize= (6.4, 4.8))
plt.stairs(counts, bins)
plt.stairs(counts_mot, bins_mot)
plt.xlabel('Pixel Velocity pixel/frame')
plt.ylabel('Normalised counts')
plt.show()
fig.savefig('awesome_figure.png', dpi=300, bbox_inches='tight')