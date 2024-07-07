from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import cv2
import os 
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import argparse
import time
import yaml
from detail_caption_construction.utils.bbox_statistics import compute_metrics

def convert_bbox(bboxes, mode):
    bbox = []
    for b in bboxes:
        if mode == "xywh":
            x, y, w, h = b
            x1, y1, x2, y2 = x, y, x+w, y+h
        elif mode == "xyxy":
            x1, y1, x2, y2 = b
        bbox.append([x1, y1, x2, y2])
    return bbox

def convert_bbox_area(bboxes, mode):
    area = []
    for b in bboxes:
        if mode == "xywh":
            x, y, w, h = b
        elif mode == "xyxy":
            x1, y1, x2, y2 = b
            w, h = x2-x1, y2-y1
        area.append(w*h)
    return area

def show_box(box, ax, lw):
    color = np.random.rand(3)
    if isinstance(box[0], (int, float)):
        box = [box]
    for b in box:
        x0, y0 = b[0], b[1]
        w, h = b[2] - b[0], b[3] - b[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=lw))    

def draw_bbox(sample, img_key, bboxes, save_path, linewidth=4):
    image_bytes = sample[img_key]
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_h, image_w = image.shape[:2]
    # print(image_h, image_w)

    fig, ax = plt.subplots(figsize=(image_w/100, image_h/100), dpi=100)

    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(image)

    for i in range(len(bboxes)):
        show_box(bboxes[i], ax, linewidth)
        
    plt.savefig(os.path.join(save_path, f'{int(sample["item_id"])}.jpg'), format='jpeg')
    plt.close()


def compute_iou(bboxes1, bboxes2, type="iou"):

    bbox1_array = np.array(bboxes1)
    bbox2_array = np.array(bboxes2)

    bbox1_area = (bbox1_array[:, 2] - bbox1_array[:, 0]) * (bbox1_array[:, 3] - bbox1_array[:, 1])
    bbox2_area = (bbox2_array[:, 2] - bbox2_array[:, 0]) * (bbox2_array[:, 3] - bbox2_array[:, 1])

    intersection_tl = np.maximum(bbox1_array[:, None, :2], bbox2_array[:, :2])
    intersection_br = np.minimum(bbox1_array[:, None, 2:], bbox2_array[:, 2:])
    intersection_wh = np.maximum(0, intersection_br - intersection_tl)
    intersection_area = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]
    union_area = bbox1_area[:, None] + bbox2_area - intersection_area
    iou = intersection_area / union_area
    if type == "iou":
        return iou

    cir_tl = np.minimum(bbox1_array[:, None, :2], bbox2_array[:, :2])
    cir_br = np.maximum(bbox1_array[:, None, 2:], bbox2_array[:, 2:])
    cir_wh = np.maximum(0, cir_br - cir_tl)
    cir_area = cir_wh[:, :, 0] * cir_wh[:, :, 1]
    
    giou = (cir_area - union_area) / cir_area
    iou = iou - giou
    if type == "giou":
        return iou, giou

def filter_bbox(bboxes, areas, hw):
    res_bbox = []
    res_area = []
    if len(bboxes) == 0:
        return res_bbox, res_area
    h, w = hw
    for bbox, area in zip(bboxes, areas):
        bbox_h, bbox_w = bbox[3]-bbox[1], bbox[2]-bbox[0]
        if area/(h*w) < 0.001:
            continue
        if area/(h*w) > 0.9:
            continue
        # if  bbox_h/h < 0.01 or bbox_h/h > 0.9:
        #     continue
        # if  bbox_w/w < 0.01 or bbox_w/w > 0.9:
        #     continue
        res_bbox.append(bbox)
        res_area.append(area)
    return res_bbox, res_area

def merge_bbox(bboxes, merge_threshold):
    # from IPython import embed; embed()
    sorted_indices = np.argsort((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]))
    bboxes = np.array(bboxes[sorted_indices])

    merged_bboxes = []
    while len(bboxes) > 0:
        bbox = bboxes[0]
        merge_indices = [0]
        
        for i in range(1, len(bboxes)):
            iou, giou = compute_iou([bbox], [bboxes[i]], "giou")
            if iou > merge_threshold or giou < 0.1:
                merge_indices.append(i)
        
        merged_bbox = [int(min(bboxes[merge_indices][:, 0])),
                       int(min(bboxes[merge_indices][:, 1])),
                       int(max(bboxes[merge_indices][:, 2])),
                       int(max(bboxes[merge_indices][:, 3]))]
        merged_bboxes.append(merged_bbox)
        
        bboxes = np.array([bboxes[i] for i in range(len(bboxes)) if i not in merge_indices])
    
    return merged_bboxes

def init_cluster_centers(k, points):
    points = np.array(points)
    centers = [points[np.random.choice(len(points))]]

    while len(centers) < k:
        distances = np.linalg.norm(points[:, np.newaxis, :] - centers, axis=2)
        min_distances = np.min(distances, axis=1)
        probabilities = min_distances / min_distances.sum()
        next_centers_index = np.random.choice(len(points), p=probabilities)
        centers.append(points[next_centers_index])
    
    return centers

def iou_distance(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_area = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    union_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection_area
    iou = intersection_area / union_area

    cir_area = max(0, max(x2, x4) - min(x1, x3)) * max(0, max(y2, y4) - min(y1, y3))
    
    giou = (cir_area - union_area) / cir_area
    iou = iou - giou
    return 1 - iou


def point_is_in_boxes(point, bboxes):
    times = 0
    for i, bbox in enumerate(bboxes):
        if bbox[0] <= point[0] < bbox[2] and bbox[1] <= point[1] < bbox[3]:
            return True
    return False


def get_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def maximalRectangle(matrix):
    continuous = [[0 for _ in range(len(matrix[0]))] for __ in range(len(matrix))]
    for i in range(len(matrix)):
        cur = 0
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                cur += 1
            else:
                cur = 0
            continuous[i][j] = cur

    def maxRec(nums):
        stack = []
        right_bound = [0 for _ in range(len(nums))]
        for i in range(len(nums)):
            if len(stack) == 0 or nums[i] >= nums[stack[-1]]:
                stack.append(i)
            else:
                while len(stack) != 0 and nums[i] < nums[stack[-1]]:
                    top = stack.pop()
                    right_bound[top] = i - 1
                stack.append(i)

        while len(stack) != 0:
            top = stack.pop()
            right_bound[top] = len(nums) - 1

        left_bound = [0 for _ in range(len(nums))]
        for i in range(len(nums) - 1, -1, -1):
            if len(stack) == 0 or nums[i] >= nums[stack[-1]]:
                stack.append(i)
            else:
                while len(stack) != 0 and nums[i] < nums[stack[-1]]:
                    top = stack.pop()
                    left_bound[top] = i + 1
                stack.append(i)

        while len(stack) != 0:
            top = stack.pop()
            left_bound[top] = 0

        best, best_left_bound, best_right_bound = 0, 0, 0
        # print(left_bound)
        # print(right_bound)
        for i in range(len(left_bound)):
            if nums[i] * (right_bound[i] - left_bound[i] + 1) > best:
                best = nums[i] * (right_bound[i] - left_bound[i] + 1)
                best_left_bound, best_right_bound = left_bound[i], right_bound[i]
        return best, best_left_bound, best_right_bound

    best = 0
    bbox = [0, 0, 0, 0]
    for j in range(len(matrix[0])):
        nums = [continuous[i][j] for i in range(len(matrix))]
        # from IPython import embed; embed()
        this_best, this_best_left_bound, this_best_right_bound = maxRec(nums)
        if best < this_best:
            width = this_best // (this_best_right_bound - this_best_left_bound + 1)
            best, bbox[0], bbox[1], bbox[2], bbox[3] = this_best, j - width + 1, this_best_left_bound, j, this_best_right_bound
            
    return best, bbox


def compress_pixels(bbox_map, scale=2):
    compressed_bbox_map = np.zeros([bbox_map.shape[0] // scale, bbox_map.shape[1] // scale], dtype=int)
    for x in range(compressed_bbox_map.shape[1]):
        for y in range(compressed_bbox_map.shape[0]):
            if bbox_map[y * scale: y * scale + scale, x * scale: x * scale + scale].sum() >= scale ** 2 / 2:
                compressed_bbox_map[y, x] = 1
    return compressed_bbox_map


def recover_compressed_bbox(bbox, scale=2):
    for i in range(len(bbox)):
        bbox[i] = bbox[i] * scale
    
    return bbox


def cluster(df, name, config):
    new_path = {}
    if config['Draw']['draw_bbox']:
        for key, val in config['Draw.path'].__dict__.items():
            path = val.split("/")
            new_path[key] = "/".join(path[:-1] + [name] + [path[-1]])
            os.makedirs(new_path[key], exist_ok=True)

    df["bbox"] = df["bboxes"].apply(convert_bbox, mode="xyxy")
    df["area"] = df["bboxes"].apply(convert_bbox_area, mode="xyxy")

    cluster_centers, merged_cluster_centers, cropped_boxes = [], [], []
    k_means_times, merge_times, crop_times = [], [], []
    for i in tqdm(range(len(df))):
        start_time_stamp = time.time()
        bbox_list, area_list = filter_bbox(df["bbox"][i], df["area"][i], df["hw"][i])
        if len(bbox_list) == 0:
            cluster_centers.append([])
            merged_cluster_centers.append([])
            cropped_boxes.append([])
            continue
        points = [bbox+[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2] for bbox in bbox_list]
        # points = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2] for bbox in bbox_list]
        
        sorted_bbox_indices = np.argsort(area_list)[::-1]
        initial_centers = [points[j] for j in sorted_bbox_indices[:min(config['kmeans_center_num'], len(sorted_bbox_indices))]]
        km = KMeans(n_clusters=len(initial_centers), n_init=10, init="k-means++", random_state=0)
        
        km.fit(points)
        k_means_time_stamp = time.time()
        k_means_times.append(time.time() - start_time_stamp)

        labels = km.labels_
        cluster_center = km.cluster_centers_
        clustered_bboxes = []
        new_cluster_center = []
        for j in range(len(initial_centers)):
            cur_bbox = np.array(bbox_list)[labels==j]
            if len(cur_bbox) == 0:
                continue
            clustered_bboxes.append(cur_bbox.tolist())
            new_cluster_center.append([int(min(cur_bbox[:, 0])), int(min(cur_bbox[:, 1])), int(max(cur_bbox[:, 2])), int(max(cur_bbox[:, 3]))])
        
        cluster_centers.append(new_cluster_center)
        bboxes_for_each_center = np.array(new_cluster_center)
        
        merged_center = merge_bbox(bboxes_for_each_center, config['merge_threshold'])
        merged_cluster_centers.append(merged_center)

        merge_time_stamp = time.time()
        merge_times.append(merge_time_stamp - k_means_time_stamp)

        area_merged_center = [[get_area(bbox), bbox] for bbox in merged_center]
        area_merged_center.sort(key=lambda x: x[0], reverse=True)
        
        bboxes_to_be_cropped = [item[1] for item in area_merged_center[:config['bbox_to_be_cropped_num']]]
        remain_bboxes = [item[1] for item in area_merged_center[config['bbox_to_be_cropped_num']:]]
        for kk, bbox in enumerate(bboxes_to_be_cropped):
            bbox_map = np.zeros([int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0])], dtype=int)
            for x in range(int(bbox[2]) - int(bbox[0])):
                for y in range(int(bbox[3]) - int(bbox[1])):
                    point = [bbox[0] + x, bbox[1] + y]
                    if not point_is_in_boxes(point, remain_bboxes):
                        bbox_map[y][x] = 1
            compressed_bbox_map = compress_pixels(bbox_map, scale=config['compress_scale'])
            for _ in range(config['expected_cropped_bbox_num_per_bbox']):
                if compressed_bbox_map.shape[0] == 0 or compressed_bbox_map.shape[1] == 0:
                    break
                if compressed_bbox_map.sum() / (compressed_bbox_map.shape[0] * compressed_bbox_map.shape[1]) < 0.3:
                    break
                __, best_bbox = maximalRectangle(compressed_bbox_map)
                compressed_bbox_map[best_bbox[1]: best_bbox[3], best_bbox[0]: best_bbox[2]] = 0
                if get_area(best_bbox) / (compressed_bbox_map.shape[0] * compressed_bbox_map.shape[1]) < 0.3:
                    break
                best_bbox = recover_compressed_bbox(best_bbox, scale=config['compress_scale'])
                best_bbox[0], best_bbox[1], best_bbox[2], best_bbox[3] = \
                    best_bbox[0] + bbox[0], best_bbox[1] + bbox[1], best_bbox[2] + bbox[0], best_bbox[3] + bbox[1] 
                remain_bboxes.append(best_bbox)
        cropped_boxes.append(remain_bboxes)
        crop_time_stamp = time.time()
        crop_times.append(crop_time_stamp - merge_time_stamp)

        if config['Draw']['draw_bbox']:
            draw_bbox(df.loc[i], config['img_key'], clustered_bboxes, save_path=new_path["clustered_bboxes"], linewidth=6)
            draw_bbox(df.loc[i], config['img_key'], new_cluster_center, save_path=new_path["cluster_center"], linewidth=6)
            draw_bbox(df.loc[i], config['img_key'], merged_center, save_path=new_path["merge"], linewidth=6)
            draw_bbox(df.loc[i], config['img_key'], remain_bboxes, save_path=new_path["cropped"], linewidth=6)

    df["cluster_centers"] = cluster_centers
    df["merged_cluster_centers"] = merged_cluster_centers
    df["cropped_boxes"] = cropped_boxes

    print(f"kmeans time {sum(k_means_times) / len(k_means_times)}")
    print(f"merge time {sum(merge_times) / len(merge_times)}")
    print(f"crop time {sum(crop_times) / len(crop_times)}")
    
    return df
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args() 

    name = "SAM"
    with open(args.config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    config = config['cluster']

    data_source = config['data']['sourcesam']
    img_key = config['img_key']

    data_paths = ["detail_caption_construction/data/source_data/detailcaps_100_frame.parquet"]

    for data_path in data_paths:
        df = pd.read_parquet(data_path)
        # df = df[:20]
        bboxes = []
        hw = []
        for i in tqdm(range(len(df))):
            bbox = []
            for j in range(len(df["annotations"][i])):
                x, y, w, h = df["annotations"][i][j]["bbox"].tolist()
                bbox.append([x, y, x+w, y+h])
            bboxes.append(bbox)
            hw.append(df["annotations"][i][0]["segmentation"]["size"].tolist())

        print("### Doing clustering ###")
        config['img_key'] = img_key
        item_id = df["item_id"].tolist() if "item_id" in df.columns else [i for i in range(len(df))]
        cluster_info = {"item_id": df["item_id"].tolist(), "frame": df["frame"].tolist(), "bboxes": bboxes, "hw": hw}
        df_cluster = pd.DataFrame(cluster_info)
        df_cluster = cluster(df_cluster, name, config)

        print("### Doing evaluation ###")
        compute_metrics(df_cluster, config['compress_scale'], keys=["cluster_centers", "merged_cluster_centers", "cropped_boxes"])
        df["hw"] = hw
        df[f"{name}_bboxes"] = bboxes
        df[f"{name}_cluster_centers"] = df_cluster["cluster_centers"]
        df[f"{name}_merged_cluster_centers"] = df_cluster["merged_cluster_centers"]
        df[f"{name}_cropped_boxes"] = df_cluster["cropped_boxes"]

        from IPython import embed; embed()
