from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import cv2
import os 
from matplotlib import pyplot as plt
import tqdm
import json
import argparse
from collections import defaultdict


def point_is_in_boxes(point, bboxes):
    for i, bbox in enumerate(bboxes):
        if bbox[0] <= point[0] < bbox[2] and bbox[1] <= point[1] < bbox[3]:
            return True
    return False


def get_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def compute_coverage_and_overlap(image_size, bboxes):
    covered, overlap = 0, 0
    for x in range(image_size[1]):
        for y in range(image_size[0]):
            if point_is_in_boxes([x, y], bboxes):
                covered += 1
    if covered == 0 or image_size[0]==0 or image_size[1]==0:
        return 0, 0, 0    
    else:
        coverage = covered / (image_size[0] * image_size[1])
        areas = [get_area(bbox) for bbox in bboxes]
        overlap = sum(areas) / covered
        return coverage, overlap, areas


def compress_bbox(bbox, scale=2):
    if type(bbox) == list:
        for i in range(len(bbox)):
            bbox[i] = bbox[i] // scale
    elif type(bbox) == np.ndarray:
        bbox = bbox // scale
    else:
        raise TypeError(f"bbox type {type(bbox)} is unexpected")
    
    return bbox


def compute_metrics(df, compress_scale, keys=None):
    all_coverage, all_overlap, all_areas, all_bbox_num = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for i in tqdm.tqdm(range(len(df))):
        sample = df.loc[i]
        # image_size = sample['annotations'][0]['segmentation']['size'] // compress_scale
        image_size = sample["hw"][0] // compress_scale, sample["hw"][1] // compress_scale

        for key in keys:
            bboxes = np.array(sample[key]).tolist()
            if len(bboxes) == 0:
                continue
            bboxes = [compress_bbox(bbox, scale=compress_scale) for bbox in bboxes]

            coverage, overlap, areas = compute_coverage_and_overlap(image_size, bboxes)
            if coverage == 0:
                continue
            all_coverage[key].append(coverage)
            all_overlap[key].append(overlap)
            all_areas[key].extend(areas)
            all_bbox_num[key].append(len(bboxes))

    # from IPython import embed; embed()
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    for key in keys:
        print(f"######## {key} ########")
        print(f"bbox_num: {sum(all_bbox_num[key]) / len(all_bbox_num[key])}")
        print(f"bbox_num_quantile: {quantiles}: {np.quantile(np.array(all_bbox_num[key]), quantiles)}")
        print(f"coverage: {sum(all_coverage[key]) / len(all_coverage[key])}")
        print(f"coverage_quantile: {quantiles}: {np.quantile(np.array(all_coverage[key]), quantiles)}")
        print(f"overlap: {sum(all_overlap[key]) / len(all_overlap[key])}")
        print(f"overlap_quantile: {quantiles}: {np.quantile(np.array(all_overlap[key]), quantiles)}")
        print(f"bbox_area_quantile: {quantiles}: {np.quantile(np.array(all_areas[key]), quantiles)}")
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="reservoir/processed_data/cropped_bboxes.parquet")
    parser.add_argument('--baseline_key', type=str, default="cluster_centers")
    parser.add_argument('--exp_key', type=str, default="cropped_boxes")
    parser.add_argument('--compress_scale', type=int, default=2)
    args = parser.parse_args()
    df = pd.read_parquet(args.file_path)
    compute_metrics(df, args.compress_scale, args.baseline_key, args.exp_key)
