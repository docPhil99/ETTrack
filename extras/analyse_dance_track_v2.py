import cv2
import numpy as np
from pathlib import Path
import math
import json
import configparser

from skimage.color.rgb_colors import steelblue

#from extras.plot_velocity import frame_rate

draw_flag = True


path_list = [Path('/home/phil/python/ETTrack/ETTrack/datasets/dancetrack/train') , Path('/home/phil/python/ETTrack/ETTrack/datasets/MOT20/train'),Path('/home/phil/python/ETTrack/ETTrack/datasets/MOT17/train'),Path('/home/phil/python/ETTrack/ETTrack/datasets/SportsMOT/sportsmot_publish/dataset/train')]
stub_list  = ['dancetrack','MOT20','MOT17','sports_']


# gt format
# <frame>,  <id>, <bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>

results = {}
for base_path,stub_name in zip(path_list,stub_list):
    angles =[]
    velocities =[]
    for paths in base_path.iterdir():
        print(paths)
        imgs =  paths/Path('img1')
        ground_truth_file = paths/Path('gt/gt.txt')
        print(f'Opening {ground_truth_file}')
        try:
            with open(ground_truth_file, 'r') as f:
                gt_s = f.readlines()
        except:
            print(f'Could open {ground_truth_file}')
            continue
        gt = [[int(float(val.rstrip())) for val in line.split(',')] for line in gt_s]

        # get the frame rate
        seqinfo_file = paths/Path('seqinfo.ini')
        config = configparser.ConfigParser()
        config.read_file(open(seqinfo_file))
        fps = float(config['Sequence']['frameRate'])
        # convert to dict, with ID as key
        data={}
        for fr in gt:
            centre = (fr[2] +fr[4] / 2, fr[3] - fr[5] / 2)
            dat = {'frame':fr[0],'centre':centre}
            if fr[1] in data:   # check if ID exists, if not make new key, else append
                data[fr[1]].append(dat)
            else:
                data[fr[1]]=[dat]

        for id, val in data.items():
            for indx in range(2,len(val)):
                c1=val[indx]
                cc1 =  c1['centre']
                c2 = val[indx-1]
                cc2 = c2['centre']
                c3 = val[indx - 2]
                cc3 = c3['centre']

                print(indx)
                velocity = math.sqrt((cc1[0]-cc2[0])**2+(cc1[1]-cc2[1])**2)
                velocity = velocity/fps*25
                ang1 = math.atan2(cc1[1]-cc2[1],cc1[0]-cc2[0])
                ang2 = math.atan2(cc2[1] - cc3[1], cc2[0] - cc3[0])
                delta_ang  = ang2-ang1
                angles.append(delta_ang)
                velocities.append(velocity)
    results[stub_name]={'angles':angles,'velocities':velocities}
out_file = Path('output') / Path(f'v2.json')
with open(out_file, 'w') as f:
    json.dump(results,f)




        #     bboxes = data[gt1_indx]
        #     centres = []
        #     for bbox in bboxes:
        #         center = (bbox[1], bbox[2]+bbox[4]//2,bbox[3]+bbox[5]//2)  # id, x, y
        #         centres.append(center)
        #         if draw_img:
        #             img = cv2.rectangle(img, (bbox[2], bbox[3]), (bbox[2] + bbox[4], bbox[3] + bbox[5]), (0, 255, 0), 2)
        #             img = cv2.circle(img,center[1:],5,(0,0,255),-1)
        #     centre_loc[gt1_indx]=centres
        #     if draw_img:
        #         cv2.imshow('img',img)
        #         if cv2.waitKey(10)==ord('q'):
        #             cv2.destroyAllWindows()
        #             break
        #
        # velocity={}
        # for gt1_indx in centre_loc:
        #     print(gt1_indx)
        #     if gt1_indx>1:
        #         vels=[]
        #         for ind in range(len(centre_loc[gt1_indx])):
        #             vel1 = centre_loc[gt1_indx][ind]
        #             print(f'vel1 {vel1}')
        #             #vel2 = centre_loc[gt1_indx-1][ind]
        #             try:
        #                 vel2_ind = [x[0] for x in centre_loc[gt1_indx-1]].index(vel1[0])
        #             except ValueError:
        #                 print('No matching index')
        #                 continue
        #             vel2 = centre_loc[gt1_indx-1][vel2_ind]
        #             print(f'vel2 {vel2}')
        #             vel = math.sqrt((vel1[1]-vel2[1])**2+(vel1[2]-vel2[2])**2)
        #             vel = vel / fps * 25
        #             vels.append(vel)
        #         velocity[gt1_indx]=vels
        #         print(velocity[gt1_indx])
        #
        #
        # def angle_between_vectors(u, v):
        #     dot_product = sum(i * j for i, j in zip(u, v))
        #     norm_u = math.sqrt(sum(i ** 2 for i in u))
        #     norm_v = math.sqrt(sum(i ** 2 for i in v))
        #     cos_theta = dot_product / (norm_u * norm_v)
        #     angle_rad = math.acos(cos_theta)
        #     angle_deg = math.degrees(angle_rad)
        #     return angle_rad, angle_deg
        #
        # angles = {}
        # for gt1_indx in centre_loc:
        #     print(gt1_indx)
        #     if gt1_indx > 1:
        #         vels = []
        #         for ind in range(len(centre_loc[gt1_indx])):
        #             cnt1 = centre_loc[gt1_indx][ind]
        #             #print(f'vel1 {vel1}')
        #             # vel2 = centre_loc[gt1_indx-1][ind]
        #             try:
        #                 vel2_ind = [x[0] for x in centre_loc[gt1_indx - 1]].index(vel1[0])
        #             except ValueError:
        #                 print('No matching index')
        #                 continue
        #             cnt2 = centre_loc[gt1_indx - 1][vel2_ind]
        #             try:
        #                 vel3_ind = [x[0] for x in centre_loc[gt1_indx - 2]].index(vel1[0])
        #             except ValueError:
        #                 print('No matching index')
        #                 continue
        #             except KeyError:
        #                 print('Key error')
        #                 continue
        #             cnt3 = centre_loc[gt1_indx - 2][vel3_ind]
        #             BA = (cnt1[0]-cnt2[0],cnt1[1]-cnt2[1])
        #             BC = (cnt3[0]-cnt2[0],cnt3[1]-cnt2[1])
        #             # try:
        #             #     ang_r, ang_d = angle_between_vectors(BA,BC)
        #             # except ZeroDivisionError:
        #             #     print(f'Division by zero from {BA} and {BC}')
        #             #     continue
        #             # except ValueError as e:
        #             #     print(f'Error {e}')
        #             #     continue
        #             ang = math.atan2(cnt1[1]-cnt2[1],cnt1[0]-cnt2[0])
        #             ang2 = math.atan2(cnt2[1] - cnt3[1], cnt2[0] - cnt3[0])
        #             angle_deg = math.degrees(ang) - math.degrees(ang2)
        #             vels.append(angle_deg)
        #
        #         angles[gt1_indx] = vels
        #         print(velocity[gt1_indx])
        #
        # out_file = Path('output') / Path(f'{stub_name}{paths.stem}_angle_track.json')
        # with open(out_file, 'w') as f:
        #     json.dump(angles, f)
        #
        # out_file = Path('output')/Path(f'{stub_name}{paths.stem}_dance_track.json')
        # with open(out_file, 'w') as f:
        #     json.dump(velocity,f)
        #
        #
