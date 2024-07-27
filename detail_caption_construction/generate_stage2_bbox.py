"""
    Copyright (2024) CAPTURE project Authors 

    Licensed under the Apache License, Version 2.0 (the "License"); 
    you may not use this file except in compliance with the License. 
    You may obtain a copy of the License at 

        http://www.apache.org/licenses/LICENSE-2.0 

    Unless required by applicable law or agreed to in writing, software 
    distributed under the License is distributed on an "AS IS" BASIS, 
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
    See the License for the specific language governing permissions and 
    limitations under the License.
"""


from collections import defaultdict
from email.policy import default
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# sys.path.append('tagore/module/detection/CoDETR')
sys.path.append('tagore/module/detection/SAM/sam_train_eval_example')

import cv2
import yaml
import torch
import argparse
import pandas as pd
import numpy as np
from cruise.data_module.utils import parse_data_source

from PIL import Image
from abc import ABC, abstractmethod
from tqdm import tqdm


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from pycocotools import mask as mask_utils
from matplotlib import pyplot as plt
from collections import defaultdict

from detail_caption_construction.utils.bbox_cluster import cluster, convert_bbox
from detail_caption_construction.utils.bbox_statistics import compute_metrics
from detail_caption_construction.utils.utils import get_data_files


class BBoxPredictor(ABC):
    def __init__(self, config):
        self.config = config
        self.init_predictor()
    
    @abstractmethod
    def init_predictor(self, **kwargs):
        pass

    @abstractmethod
    def process_input(self, inputs, **kwargs):
        pass

    @abstractmethod
    def predict(self, batch, **kwargs):
        pass

    @abstractmethod
    def process_output(self, outputs, **kwargs):
        pass


class SAMPredictor(BBoxPredictor):

    def init_predictor(self):
        self.device = f"cuda:{os.environ.get('RANK', 0)}"
        local_checkpoint = 'detail_caption_construction/ckpt/sam/' + os.path.basename(self.config['checkpoint'])
        print(f"loading {local_checkpoint}")
        sam = sam_model_registry[self.config['visual_encoder']](
            checkpoint=local_checkpoint)
        sam.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=self.config['mask']['points_per_side'],      # point num per size, 
            pred_iou_thresh=self.config['mask']['pred_iou_thresh'],   # iou thresh
            stability_score_thresh=self.config['mask']['stability_score_thresh'],    # 
            crop_n_layers=self.config['mask']['crop_n_layers'],
            crop_n_points_downscale_factor=self.config['mask']['crop_n_points_downscale_factor'],
            min_mask_region_area=self.config['mask']['min_mask_region_area'],  # Requires open-cv to run post-processing
            )

    def visualize(self, image, bboxes, out_file, linewidth=2):

        image_h, image_w = image.shape[:2]
        fig, ax = plt.subplots(figsize=(image_w/100, image_h/100), dpi=100)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(image)

        for i in range(len(bboxes)):
            color = np.random.rand(3)
            bbox = bboxes[i]
            x0, y0, w, h = bbox
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=linewidth))
        
        plt.savefig(out_file, format='jpeg')
        plt.close()

    def process_input(self, inputs, img_key, order="RGB"):
        batch_images = []
        batch_imagehw = []
        for i, sample in inputs.iterrows():
            img_np = np.frombuffer(sample[img_key], np.uint8)
            image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if image is None:
                batch_images.append(None)
                batch_imagehw.append((0,0))
                continue
            if order == "RGB":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif order == "BGR":
                pass
            batch_images.append(image)
            batch_imagehw.append(image.shape[:2])
        return {'image': batch_images, 'hw': batch_imagehw}

    def predict(self, batch):
        outputs = []
        for image in batch["image"]:
            try:
                mask = self.mask_generator.generate(image)
            except Exception as e:
                mask = None
            outputs.append(mask)
        batch["outputs"] = outputs
        torch.cuda.empty_cache()
        return batch

    def process_output(self, outputs):
        result = defaultdict(list)
        output = outputs["outputs"]
        result["hw"] = outputs["hw"]
        bboxes = []
        for i in range(len(output)):
            bbox = []
            if output[i] is None:
                result["sam"].append(None)
                bboxes.append(bbox)
                continue
            for j in range(len(output[i])):
                output[i][j]["segmentation"] = mask_utils.encode(output[i][j]["segmentation"])
                output[i][j]["segmentation"]["counts"] = output[i][j]["segmentation"]["counts"].decode()
                x, y, w, h = output[i][j]["bbox"]
                bbox.append([x, y, x+w, y+h])
            bboxes.append(bbox)
            result["sam"].append(output[i])
        result["SAM_bboxes"] = bboxes
        return result


def main(config_path, chunk_index, chunk_num, node_index, node_num):
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    do_cluster, do_crop, do_eval = config['do_cluster'], config['do_crop'], config['do_eval']
    img_key = config['img_key']

    model_hparams = config['model']
    model = SAMPredictor(config=model_hparams)

    batch_size = config['batch_size']
    source_data_files, target_data_files = get_data_files(config, node_index=node_index, node_num=node_num)

    for source_data_file in source_data_files:
        if f"{source_data_file.split('/')[-1].split('.')[0]}_processed" in target_data_files:
            print(f"file {source_data_file} processed, skipping")
            continue
        print(f"processing {source_data_file}")

        df = pd.read_parquet(source_data_file)
        start, end = chunk_index * (len(df) // chunk_num), (chunk_index + 1) * (len(df) // chunk_num) - 1
        if len(df) - end < len(df) // chunk_num:
            end = len(df) - 1
        df = df.loc[start: end]

        processed_data = defaultdict(list)
        for offset in tqdm(range(0, len(df), batch_size)):
            offset += start
            inputs = df.loc[offset: offset + batch_size - 1]
            batch = model.process_input(inputs, img_key=img_key)
            outputs = model.predict(batch)
            results = model.process_output(outputs)

            for key, val in results.items():
                processed_data[key].extend(val)
        
        if do_cluster:
            print("### Doing clustering ###")
            item_id = df["item_id"].tolist() if "item_id" in df.columns else [i+start for i in range(len(df))]
            cluster_info = {"item_id": item_id, img_key: df[img_key].tolist(), "bboxes": processed_data[f"{'SAM'}_bboxes"], "hw": processed_data["hw"]}
            df_cluster = pd.DataFrame(cluster_info)
            df_cluster = cluster(df_cluster, 'SAM', config['cluster'])
            print("### Clustering completed ###")

            if do_eval:
                print("### Doing evaluation ###")
                compute_metrics(df_cluster, config['cluster']['compress_scale'], keys=["cluster_centers", "merged_cluster_centers", "cropped_boxes"])

            processed_data[f"{'SAM'}_cluster_centers"] = df_cluster["cluster_centers"]
            processed_data[f"{'SAM'}_merged_cluster_centers"] = df_cluster["merged_cluster_centers"]
            if do_crop:
                processed_data[f"{'SAM'}_cropped_boxes"] = df_cluster["cropped_boxes"]
        
        df = df.reset_index(drop=True)
        for key, val in processed_data.items():
            df[key] = val
        
        if not do_cluster and do_eval:
            print("### Doing evaluation ###")
            compute_metrics(df, config['cluster']['compress_scale'], baseline_key="bboxes")
            
        # from IPython import embed; embed()
        base_path = os.path.basename(source_data_file)
        output_path = f"detail_caption_construction/data/processed_data/{base_path.split('.')[0]}_chunk{chunk_index}.parquet"
        print(df)
        print(output_path)
        print(df.columns)
        df.to_parquet(output_path)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--chunk_index', type=int, default=0)
    parser.add_argument('--chunk_num', type=int, default=1)
    parser.add_argument('--node_index', type=int, default=0)
    parser.add_argument('--node_num', type=int, default=10)

    args = parser.parse_args()
    main(
        config_path=args.config_path, 
        chunk_index=args.chunk_index, 
        chunk_num=args.chunk_num,
        node_index=args.node_index, 
        node_num=args.node_num,
    )
        