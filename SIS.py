import numpy as np
import cv2
import time
from datetime import datetime


# _________________________________________________________________________________________________
# Formulation of 1D similarity
def oneDim_sim(ref_1D, eval_1D, length):
    if np.size(ref_1D) <= np.size(eval_1D):
        diff = 0
        eval_ind = np.array([], dtype=np.int16)
        for point in ref_1D:
            diff = diff + np.min(np.abs(eval_1D - point))
            eval_ind = np.append(eval_ind, np.argmin(np.abs(eval_1D - point)))

        eval_1D = np.delete(eval_1D, eval_ind)
        ref_1D_border = np.append(ref_1D, [0, length])
        for point in eval_1D:
            diff = diff + 2 * np.min(np.abs(ref_1D_border - point))
        metric = (length - diff) / length
        metric = np.max([metric, 0])
    else:
        diff = 0
        ref_ind = np.array([], dtype=np.int16)
        if np.size(eval_1D) == 0:
            metric = 0
        else:
            for point in eval_1D:
                diff = diff + np.min(np.abs(ref_1D - point))
                ref_ind = np.append(ref_ind, np.argmin(np.abs(ref_1D - point)))

            ref_1D = np.delete(ref_1D, ref_ind)
            for point in ref_1D:
                diff = diff + 2 * np.min(np.abs(eval_1D - point))
            metric = (length - diff) / length
            metric = np.max([metric, 0])

    return metric


# _________________________________________________________________________________________________
# SIS calcuation: set orthogonal scanline pairs, investigate intersecions, and calculate 1D similartiy
def SIS(map1, map2, num_scanline, scan_len):
    map_size = map1.shape
    size_h = map_size[1]
    size_v = map_size[0]

    scan_length = scan_len
    intersection_counts = np.zeros((2, num_scanline))
    similarity = np.zeros((1, num_scanline))

    for it in range(0, num_scanline):

        # Select points to define scanline
        h = np.random.choice(range(0, size_h))
        v = np.random.choice(range(0, size_v))

        # Record intersection count .
        intersect_count1_v = 0
        intersect_count2_v = 0
        intersect_count1_h = 0
        intersect_count2_h = 0
        # Record intersection position
        intersect_pos1_v = np.array([])
        intersect_pos2_v = np.array([])
        intersect_pos1_h = np.array([])
        intersect_pos2_h = np.array([])

        # Investigate intersections for vertical scanline
        v_start = int(max(v - scan_length / 2, 0))
        v_end = int(min(v + scan_length / 2, size_v))
        # Index for distinguishing continuous intersections, and variables for saving positions for start & end of
        # continuous intersections
        intersect_ind1, intersect_ind2 = 0, 0
        pos_start1, pos_end1, pos_start2, pos_end2 = 0, 0, 0, 0
        for j in range(v_start, v_end):
            if np.any(map1[j, h]) and intersect_ind1 == 0:
                intersect_ind1 = 1
                pos_start1 = j - (v - scan_length / 2)
            elif (~np.any(map1[j, h]) or j == v_end - 1) and intersect_ind1 == 1:
                intersect_ind1 = 0
                pos_end1 = j - 1 - (v - scan_length / 2)
                intersect_pos1_v = np.append(intersect_pos1_v, (pos_end1 + pos_start1) / 2)
                intersect_count1_v += 1

            if np.any(map2[j, h]) and intersect_ind2 == 0:
                intersect_ind2 = 1
                pos_start2 = j - (v - scan_length / 2)
            elif (~np.any(map2[j, h]) or j == v_end - 1) and intersect_ind2 == 1:
                intersect_ind2 = 0
                pos_end2 = j - 1 - (v - scan_length / 2)
                intersect_pos2_v = np.append(intersect_pos2_v, (pos_end2 + pos_start2) / 2)
                intersect_count2_v += 1

        # Investigate intersections for horizontal scanlines
        h_start = int(max(h - scan_length / 2, 0))
        h_end = int(min(h + scan_length / 2, size_h))
        # Initialize indices and variables
        intersect_ind1, intersect_ind2 = 0, 0
        pos_start1, pos_end1, pos_start2, pos_end2 = 0, 0, 0, 0
        for i in range(h_start, h_end):
            if np.any(map1[v, i]) and intersect_ind1 == 0:
                intersect_ind1 = 1
                pos_start1 = i - (h - scan_length / 2)
            elif (~np.any(map1[v, i]) or i == h_end - 1) and intersect_ind1 == 1:
                intersect_ind1 = 0
                pos_end1 = i - 1 - (h - scan_length / 2)
                intersect_pos1_h = np.append(intersect_pos1_h, (pos_end1 + pos_start1) / 2)
                intersect_count1_h += 1

            if np.any(map2[v, i]) and intersect_ind2 == 0:
                intersect_ind2 = 1
                pos_start2 = i - (h - scan_length / 2)
            elif (~np.any(map2[v, i]) or i == h_end - 1) and intersect_ind2 == 1:
                intersect_ind2 = 0
                pos_end2 = i - 1 - (h - scan_length / 2)
                intersect_pos2_h = np.append(intersect_pos2_h, (pos_end2 + pos_start2) / 2)
                intersect_count2_h += 1

        # Intersection counts of scanline
        intersection_counts[0][it] = (intersect_count1_v + intersect_count1_h)
        intersection_counts[1][it] = (intersect_count2_v + intersect_count2_h)

        # Calculate 1D similarity for investigated scanline pair
        oneDim_sim_h = oneDim_sim(intersect_pos1_h, intersect_pos2_h, scan_length)
        oneDim_sim_v = oneDim_sim(intersect_pos1_v, intersect_pos2_v, scan_length)
        similarity[0, it] = oneDim_sim_h * oneDim_sim_v

    standard_deviation = np.std(similarity)
    SIS_value = np.mean(similarity)

    # Calculate trace frequency
    frequency = np.average(intersection_counts)

    return SIS_value, standard_deviation, frequency


# _________________________________________________________________________________________________

# Import trace maps for SIS calculation
# Directory of trace maps
map1_dir = 'D:/SIS/Validation/Rockmechanics related params/fracture density/fmap 0.png'
map2_dir = 'D:/SIS/Validation/Rockmechanics related params/fracture density/fmap d+2 (2).png'

# Since the imported images have 3 channels, the channels are compressed into a single channel.
# Data type is converted to uint8 for processing.
fmap1 = cv2.imread(map1_dir)
fmap1 = np.median(fmap1, axis=2)
fmap1 = np.array(fmap1.astype(np.uint8))
fmap1 = np.where(fmap1 > 0, fmap1, 0)

fmap2 = cv2.imread(map2_dir)
fmap2 = np.median(fmap2, axis=2)
fmap2 = np.array(fmap2.astype(np.uint8))
fmap2 = np.where(fmap2 > 0, fmap2, 0)

map_shape = np.shape(fmap1)
x_len, y_len = map_shape[1], map_shape[0]
scanline_length_i = np.min([x_len, y_len]) / 2

np.random.seed(0)

# Preliminary calculation with 100 scanlines
time_start = time.process_time()
SIS_val, std, freq = SIS(fmap1, fmap2, 100, scanline_length_i)

scanline_length = np.floor(scanline_length_i * 4 / freq)
scanline_length = np.min([np.min([x_len, y_len]), scanline_length])
scanline_n = int((1.96 * std / 0.01 ** 2))

if scanline_n <= 100:
    print('Number of scanlines: 100')
    print('SIS: ' + str(SIS_val))
else:
    time_elasped_pre = (time.process_time() - time_start)
    print('Number of scanlines: ' + str(scanline_n))
    print('Estimated calculation time: ' + str(time_elasped_pre * scanline_n / 100 / 60) + 'min')
    print(datetime.now().time())

    SIS_val, _, _ = SIS(fmap1, fmap2, scanline_n, scanline_length)

    print('\n SIS: ' + str(SIS_val))
    time_elasped = time.process_time() - time_start - time_elasped_pre
    print('Calculation time: ' + str(time_elasped / 60) + 'min')
