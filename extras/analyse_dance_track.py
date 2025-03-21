import cv2
import numpy as np
from pathlib import Path
import math
import json
import configparser

from skimage.color.rgb_colors import steelblue

#from extras.plot_velocity import frame_rate

base_path=Path('/home/phil/datasets/MOT/dancetrack/train/train1')
base_path=Path('/home/phil/datasets/MOT/MOT20/train')
#base_path=Path('/home/phil/datasets/MOT/MOT17/train')
#base_path=Path('/home/phil/datasets/MOT/SportsMOT/sportsmot_publish/dataset/train')
stub_name = 'sports_'  # set to '' to use just path.stem
stub_name='MOT20'
for paths in base_path.iterdir():
    print(paths)
    imgs =  paths/Path('img1')
    ground_truth_file = paths/Path('gt/gt.txt')
    print(f'Opening {ground_truth_file}')
    with open(ground_truth_file, 'r') as f:
        gt_s = f.readlines()
    gt = [[int(float(val.rstrip())) for val in line.split(',')] for line in gt_s]

    # get the frame rate
    seqinfo_file = paths/Path('seqinfo.ini')
    config = configparser.ConfigParser()
    config.read_file(open(seqinfo_file))
    fps = float(config['Sequence']['frameRate'])
    # convert to dict, with frame number as key
    data={}
    for fr in gt:
        if fr[0] in data:
            data[fr[0]].append(fr)
        else:
            data[fr[0]]=[fr]
    centre_loc={}
    for gt1_indx in data:
        print(gt1_indx)
        img_path = paths/Path(f'img1/{gt1_indx:06}.jpg')
        print(img_path)
        if img_path.is_file():
            draw_img = True
            img = cv2.imread(str(img_path))
        else:
            draw_img = False
        bboxes = data[gt1_indx]
        centres = []
        for bbox in bboxes:
            center = (bbox[1], bbox[2]+bbox[4]//2,bbox[3]+bbox[5]//2)  # id, x, y
            centres.append(center)
            if draw_img:
                img = cv2.rectangle(img, (bbox[2], bbox[3]), (bbox[2] + bbox[4], bbox[3] + bbox[5]), (0, 255, 0), 2)
                img = cv2.circle(img,center[1:],5,(0,0,255),-1)
        centre_loc[gt1_indx]=centres
        if draw_img:
            cv2.imshow('img',img)
            if cv2.waitKey(10)==ord('q'):
                cv2.destroyAllWindows()
                break

    velocity={}
    for gt1_indx in centre_loc:
        print(gt1_indx)
        if gt1_indx>1:
            vels=[]
            for ind in range(len(centre_loc[gt1_indx])):
                vel1 = centre_loc[gt1_indx][ind]
                print(f'vel1 {vel1}')
                #vel2 = centre_loc[gt1_indx-1][ind]
                try:
                    vel2_ind = [x[0] for x in centre_loc[gt1_indx-1]].index(vel1[0])
                except ValueError:
                    print('No matching index')
                    continue
                vel2 = centre_loc[gt1_indx-1][vel2_ind]
                print(f'vel2 {vel2}')
                vel = math.sqrt((vel1[1]-vel2[1])**2+(vel1[2]-vel2[2])**2)
                vel = vel / fps * 25
                vels.append(vel)
            velocity[gt1_indx]=vels
            print(velocity[gt1_indx])

    out_file = Path('output')/Path(f'{stub_name}{paths.stem}_dance_track.json')
    with open(out_file, 'w') as f:
        json.dump(velocity,f)


