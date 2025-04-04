import json
import matplotlib
from matplotlib.pyplot import legend

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from pathlib import Path
plt.style.use(['science','ieee'])




with open('output/v2.json') as f:
    data = json.load(f)
stub_list  = ['dancetrack','MOT20','MOT17','sports_']
legend_lab = ['Dancetrack','MOT20','MOT17','SportsMOT']
fig = plt.figure(figsize= (6.4, 4.8))
for stub,lab in zip(stub_list,legend_lab):
    vels = data[stub]['velocities']
    corr = 1
    print(f'Processing {stub}, correction {corr}')
    counts, bins = np.histogram(np.array(vels), bins=30, range=(0,50))
    counts = counts / sum(counts)
    plt.stairs(counts, bins, label=lab)

plt.xlabel('Pixel Velocity pixel/frame')
plt.ylabel('Normalised counts')
plt.legend()
fig.savefig('vel_figure.png', dpi=300, bbox_inches='tight')

fig = plt.figure(figsize= (6.4, 4.8))
for stub,lab in zip(stub_list,legend_lab):
    vels = data[stub]['angles']
    corr = 1
    print(f'Processing {stub}, correction {corr}')

    counts, bins = np.histogram(np.rad2deg(np.array(vels)), bins=30, range=(0,200))
    counts = counts / sum(counts)
    plt.stairs(counts, bins, label=lab)

plt.xlabel('Delta Angle (degrees)')
plt.ylabel('Normalised counts')
plt.legend()
fig.savefig('angle_figure.png', dpi=300, bbox_inches='tight')
plt.show()

