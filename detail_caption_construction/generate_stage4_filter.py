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


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import io
import torch
import pandas as pd
import yaml
import tqdm
import argparse
from PIL import Image
from transformers import AutoProcessor, Owlv2ForObjectDetection
from nltk.tokenize import sent_tokenize
import re

from capture.FactualSceneGraph.src.factual_scene_graph.parser.scene_graph_parser import SceneGraphParser
from detail_caption_construction.utils.utils import get_data_files


stop_words = []
with open('capture/diffed_objects.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i > 513:
            break
        if not line.startswith('#'):
            stop_words.append(line.strip().split(':')[0])
stop_words = set(stop_words)


def get_phrases(parse_res, caption):
    objects = [entity['head'] for entity in parse_res['entities']]
    attributes = [(adj, entity['head']) for entity in parse_res['entities'] for adj in entity['attributes']]
 
    all_phrases = []

    # first run: find {adj} {noun} phrases
    for attr_index, attribute in enumerate(attributes):
        phrase = f"{attribute[0]} {attribute[1]}"
        # all_phrases.append(phrase)
        if attribute[1] in objects:
            objects.remove(attribute[1])
            
    # second run: find remaining noun phrases
    for obj_index, object in enumerate(objects):
        phrase = f"{object}"
        all_phrases.append(phrase)
    
    all_phrases = [phrase for phrase in all_phrases if phrase in caption]

    return all_phrases


def get_elements(parse_res, caption):
    objects = [(entity['head'], entity['head']) for entity in parse_res['entities']]
    attributes = [(adj, f"{adj} {entity['head']}") for entity in parse_res['entities'] for adj in entity['attributes']]
 
    all_elements = [element for element in objects + attributes if element[0] in caption]
    all_phrases = [element[1] for element in all_elements]
    ori_words = [element[0] for element in all_elements]

    return all_phrases, ori_words


def ground_attributes_to_sentence(parse_res, sentence):
    objects = [entity['head'] for entity in parse_res['entities']]
    attributes = [(adj, entity['head']) for entity in parse_res['entities'] for adj in entity['attributes']]
    match_info = {}
    attribute_shot_indices = set()
    object_shot_indices = set()
    temp_sentence = sentence

    # first run: find {adj} {noun} phrases
    for attr_index, attribute in enumerate(attributes):
        phrase = f"{attribute[0]} {attribute[1]}"
        start = sentence.find(phrase)
        if start != -1:
            match_info[phrase] = [start, start + len(phrase)]
            attribute_shot_indices.add(attr_index)
            if attribute[1] in objects:
                objects.remove(attribute[1])
            temp_sentence = temp_sentence.replace(phrase, '#'*len(phrase))

    # second run: find {noun} {adj} phrases
    for attr_index, attribute in enumerate(attributes):
        phrase = f"{attribute[1]} {attribute[0]}"
        start = sentence.find(phrase)
        if start != -1:
            match_info[phrase] = [start, start + len(phrase)]
            attribute_shot_indices.add(attr_index)
            if attribute[1] in objects:
                objects.remove(attribute[1])
            temp_sentence = temp_sentence.replace(phrase, '#'*len(phrase))

    # third run: find remaining noun phrases
    for obj_index, object in enumerate(objects):
        phrase = f"{object}"
        start = sentence.find(phrase)
        if start != -1:
            match_info[phrase] = [start, start + len(phrase)]
            object_shot_indices.add(obj_index)
            temp_sentence = temp_sentence.replace(phrase, '#'*len(phrase))

    return match_info
    

def process_batch(model, processor, parser, batch, threshold, nms_threshold):
    images = [Image.open(io.BytesIO(batch.loc[i, 'frame'])).convert("RGB") for i in range(len(batch))]

    all_local_caption = [eval(sample['local_caption']) for _, sample in batch.iterrows()]
    filtered_all_local_caption = []
    for sample_idx, local_caption in enumerate(all_local_caption):
        graph_obj = parser.parse([caption for bbox, caption in local_caption], beam_size=5, return_text=False, max_output_len=128)
        all_phrases, all_ori_words = [], []
        for bbox_idx, res in enumerate(graph_obj):
            phrases = get_phrases(res, local_caption[bbox_idx][1])
            all_phrases.append(phrases)
        
        for phrases_idx, phrases in enumerate(all_phrases):
            phrases.append(local_caption[phrases_idx][1])
            all_phrases[phrases_idx] = [' '.join(phrase.split(' ')) if ' ' in phrase else phrase for phrase in phrases ]
            
        sample_images = [images[sample_idx] for _ in range(len(all_phrases))]
        results = []
        for mini_batch_start in range(0, len(all_phrases), 8):
            mini_batch_all_phrases = all_phrases[mini_batch_start: mini_batch_start+8]
            mini_batch_sample_images = sample_images[mini_batch_start: mini_batch_start+8]
        
            if len(mini_batch_all_phrases) == 0 or sum([len(phrases) for phrases in mini_batch_all_phrases]) == 0:
                mini_batch_results = [{
                    'scores': torch.tensor([]),
                    'labels': torch.tensor([]),
                    'boxes': torch.tensor([]),
                } for _ in range(len(mini_batch_all_phrases))]
                results.extend(mini_batch_results)
                continue

            input_tensor = processor(text=mini_batch_all_phrases, images=mini_batch_sample_images, truncation=True, return_tensors="pt")
            input_tensor = input_tensor.to("cuda")

            with torch.no_grad():
                outputs = model(**input_tensor)

            padded_image_size = [max(images[sample_idx].size), max(images[sample_idx].size)]
            target_sizes = [padded_image_size for _ in range(len(mini_batch_all_phrases))]

            if nms_threshold < 1.0:
                mini_batch_results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, nms_threshold=nms_threshold, threshold=threshold)
            else:
                mini_batch_results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

            results.extend(mini_batch_results)

        filtered_local_caption = []
        for bbox_idx, (phrases, result) in enumerate(zip(all_phrases, results)):
            label_scores = [0 for _ in range(len(phrases))]
            try:
                for score_idx in range(result['scores'].shape[0]):
                    label_idx = result['labels'][score_idx]
                    if label_idx < len(label_scores):
                        label_scores[label_idx] = max(label_scores[label_idx], result['scores'][score_idx])
            except:
                from IPython import embed; embed()

            this_local_caption = list(all_local_caption[sample_idx][bbox_idx])

            for phrase_idx, phrase in enumerate(phrases[:-1]):
                if label_scores[phrase_idx] < threshold:
                    this_local_caption[1] = re.sub(pattern=rf"([a-z]*){phrase}([a-z]*)", repl='', string=this_local_caption[1])
            filtered_local_caption.append(this_local_caption)  
        
        final_filtered_local_caption = filtered_local_caption
        filtered_all_local_caption.append(final_filtered_local_caption)

    all_overall_caption = [sample['overall_caption'] for _, sample in batch.iterrows()]
    filtered_all_overall_caption = []
    for sample_idx, caption in enumerate(all_overall_caption):
        sentences = sent_tokenize(caption)
        graph_obj = parser.parse(sentences, beam_size=5, return_text=False, max_output_len=128)
        all_phrases, all_ori_words = [], []
        for sent, res in zip(sentences, graph_obj):
            phrases_se = ground_attributes_to_sentence(res, sent)
            all_phrases.append(list(phrases_se.keys()))
        
        sample_images = [images[sample_idx] for _ in range(len(all_phrases))]
        results = []
        for mini_batch_start in range(0, len(all_phrases), 8):
            mini_batch_all_phrases = all_phrases[mini_batch_start: mini_batch_start+8]
            mini_batch_sample_images = sample_images[mini_batch_start: mini_batch_start+8]
        
            if len(mini_batch_all_phrases) == 0 or sum([len(phrases) for phrases in mini_batch_all_phrases]) == 0:
                mini_batch_results = [{
                    'scores': torch.tensor([]),
                    'labels': torch.tensor([]),
                    'boxes': torch.tensor([]),
                } for _ in range(len(mini_batch_all_phrases))]
                results.extend(mini_batch_results)
                continue

            input_tensor = processor(text=mini_batch_all_phrases, images=mini_batch_sample_images, truncation=True, return_tensors="pt")
            input_tensor = input_tensor.to("cuda")

            with torch.no_grad():
                outputs = model(**input_tensor)

            padded_image_size = [max(images[sample_idx].size), max(images[sample_idx].size)]
            target_sizes = [padded_image_size for _ in range(len(mini_batch_all_phrases))]

            if nms_threshold < 1.0:
                mini_batch_results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, nms_threshold=nms_threshold, threshold=threshold)
            else:
                mini_batch_results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

            results.extend(mini_batch_results)
            
        filtered_caption = []
        for sent_idx, (sent, phrases, result) in enumerate(zip(sentences, all_phrases, results)):
            label_scores = [0 for _ in range(len(phrases))]
            try:
                for score_idx in range(result['scores'].shape[0]):
                    label_idx = result['labels'][score_idx]
                    if label_idx < len(label_scores):
                        label_scores[label_idx] = max(label_scores[label_idx], result['scores'][score_idx])
            except:
                from IPython import embed; embed()

            for phrase, score in zip(phrases, label_scores):
                if score < threshold:
                    sent = sent.replace(phrase, "")

            filtered_caption.append(sent)

        filtered_caption = ' '.join(filtered_caption)
        filtered_all_overall_caption.append(filtered_caption)

    batch_processed_data = []
    for sample_idx in range(len(batch)):
        processed_sample = batch.loc[sample_idx].to_dict()
        processed_sample['filtered_local_caption'] = repr(filtered_all_local_caption[sample_idx])
        processed_sample['filtered_overall_caption'] = filtered_all_overall_caption[sample_idx]
        batch_processed_data.append(processed_sample)

    return batch_processed_data


def prepare_model_for_inference(model, dtype):
    model.cuda()
    model.eval()
    if dtype is not None:
        model.to(dtype)


def main(config_path, chunk_index, chunk_num, node_index, node_num):
    with open(config_path) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    processor = AutoProcessor.from_pretrained(f"{config['ckpt_path']}/{config['model_path']}")
    model = Owlv2ForObjectDetection.from_pretrained(f"{config['ckpt_path']}/{config['model_path']}").to("cuda")
    parser = SceneGraphParser('capture/ckpt/flan-t5-base-VG-factual-sg', device='cuda')

    source_data_files, target_data_files = get_data_files(config, node_index=node_index, node_num=node_num)

    for source_data_file in source_data_files:
        if f"{source_data_file.split('/')[-1].split('.')[0]}_processed" in target_data_files:
            print(f"file {source_data_file} processed, skipping")
            continue
        print(f"processing {source_data_file}")
        batch_size = config['batch_size']
        processed_data = []
        df = pd.read_parquet(source_data_file)

        start, end = chunk_index * (len(df) // chunk_num), (chunk_index + 1) * (len(df) // chunk_num) - 1
        if len(df) - end < len(df) // chunk_num:
            end = len(df) - 1
        df = df.loc[start: end]
        for offset in tqdm.trange(0, len(df), batch_size):
            offset += start
            batch = df.loc[offset: offset + batch_size - 1].reset_index(drop=True)

            with torch.inference_mode():
                batch_processed_data = process_batch(model, processor, parser, batch, config['threshold'], config['nms_threshold'])
                processed_data.extend(batch_processed_data)

        processed_df = pd.DataFrame(processed_data).reset_index(drop=True)
        base_path = os.path.basename(source_data_file)
        output_path = f"detail_caption_construction/data/processed_data/{base_path.split('.')[0]}_chunk{chunk_index}.parquet"
        processed_df.to_parquet(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--chunk_index', type=int, default=0)
    parser.add_argument('--chunk_num', type=int, default=4)
    parser.add_argument('--node_index', type=int, default=0)
    parser.add_argument('--node_num', type=int, default=10)
    args = parser.parse_args()

    main(
        config_path=args.config_path,
        chunk_index=args.chunk_index, 
        chunk_num=args.chunk_num,
        node_index=args.node_index,
        node_num=args.node_num
    )







            # image = Image.open(io.BytesIO(image)).convert("RGB")













