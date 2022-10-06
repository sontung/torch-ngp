import pickle
import cv2
import random
import numpy as np
import scipy.linalg
import torch
from tqdm import tqdm
from test_solver import solve_for


from colmap_db_read import extract_colmap_matches, extract_id2name, extract_colmap_sift


def concat_images_different_sizes(images):
    # get maximum width
    ww = max([du.shape[0] for du in images])

    # pad images with transparency in width
    new_images = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        w1 = img.shape[0]
        img = cv2.copyMakeBorder(
            img,
            0,
            ww - w1,
            0,
            0,
            borderType=cv2.BORDER_CONSTANT,
            value=(
                0,
                0,
                0,
                0))
        new_images.append(img)

    # stack images vertically
    result = cv2.hconcat(new_images)
    return result


def visualize_matching_pairs(image1, image2, _pairs):
    image = concat_images_different_sizes([image1, image2])
    for idx in range(0, len(_pairs), 10):
        pair = _pairs[idx]
        color = (
            random.random() * 255,
            random.random() * 255,
            random.random() * 255)
        fid1, fid2 = pair[:2]
        x1, y1 = map(int, fid1)
        cv2.circle(image, (x1, y1), 5, color, 2)

        x2, y2 = map(int, fid2)
        cv2.circle(image, (x2 + image1.shape[1], y2), 5, color, 2)
        cv2.line(image, (x1, y1), (x2 + image1.shape[1], y2), color, 1)
    return image


def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return ta, tb, denom


with open('hack.pickle', 'rb') as handle:
    data = pickle.load(handle)
[all_ro, all_rd, all_depths, all_names, all_sizes] = data
db_path = "/home/n11373598/work/nerf-vloc/data/horn/images/colmap.db"
img_path = "/home/n11373598/work/nerf-vloc/data/horn/images/db_images"
matches = extract_colmap_matches(db_path)
keypoints, _, id2name = extract_colmap_sift(db_path)
name2id = {v: k for k, v in id2name.items()}
name2index = {name.split("/")[-1]: index for index, name in enumerate(all_names)}

results = []
logs = {}
for (img_id1, img_id2) in tqdm(matches):
    matches_arr = matches[(img_id1, img_id2)]
    img_name1 = id2name[img_id1]
    img_name2 = id2name[img_id2]
    img1 = cv2.imread(f"{img_path}/{img_name1}")
    img2 = cv2.imread(f"{img_path}/{img_name2}")

    pairs = []
    for kp1, kp2 in matches_arr:
        u1, v1 = keypoints[img_id1][kp1]
        u2, v2 = keypoints[img_id2][kp2]
        (u1, v1) = map(int, (u1, v1))
        (u2, v2) = map(int, (u2, v2))
        k1 = u1*v1
        k2 = u2*v2
        ro1 = all_ro[name2index[img_name1]][k1]
        rd1 = all_rd[name2index[img_name1]][k1]

        ro2 = all_ro[name2index[img_name2]][k2]
        rd2 = all_rd[name2index[img_name2]][k2]

        t1_pred = all_depths[name2index[img_name1]][k1]
        t2_pred = all_depths[name2index[img_name2]][k2]

        # t1, t2, _ = solve_for(ro1, ro1+rd1, ro2, ro2+rd2)
        t1, t2, w = closest_point_2_lines(ro1, rd1, ro2, rd2)
        if w < 0.00001:
            continue
        p1 = ro1+t1*rd1
        p2 = ro2+t2*rd2
        score = np.abs(t1/t1_pred-t2/t2_pred)
        vote1 = str(round(t1/t1_pred, 2))
        vote2 = str(round(t2/t2_pred, 2))
        vote3 = str(round(t2/t2_pred, 1))
        if vote1 == vote2:
            if vote3 not in logs:
                logs[vote3] = 1
            else:
                logs[vote3] += 1

keys = list(logs.keys())
keys = sorted(keys, key=lambda du: logs[du])
for k in keys:
    print(k, logs[k])
