# Benchmarking and Improving Detail Image Caption
[![License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/foundation-multimodal-models/CAPTURE)
<a href='https://arxiv.org/abs/2405.19092'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
[![Dataset](https://img.shields.io/badge/Dataset-Huggingface%204.0-yellow)](https://huggingface.co/datasets/foundation-multimodal-models/DetailCaps-4870)


Code and data for paper: 

*Benchmarking and Improving Detail Image Caption*. 
Hongyuan Dong\*, Jiawen Li\*, Bohong Wu, Jiacong Wang, Yuan Zhang, Haoyuan Guo (* Equal Contribution)
	
<!-- If you find this work useful and use it on your own research, please cite our paper. --> 
Our paper is now available on [arXiv](https://arxiv.org/abs/2405.19092). 

	
## Overview
Image captioning has long been regarded as a fundamental task in visual understanding. 
Recently, however, few large vision-language model (LVLM) research discusses model's image captioning performance because of the outdated short-caption benchmarks and unreliable evaluation metrics. 
In this work, we propose to benchmark detail image caption task by curating high-quality evaluation datasets annotated by human experts, GPT-4V and Gemini-1.5-Pro.
We also design a more reliable caption evaluation metric called **CAPTURE** (CAPtion evaluation by exTracting and coUpling coRE information).
CAPTURE extracts visual elements, e.g., objects, attributes and relations from captions, and then matches these elements through three stages, achieving the highest consistency with expert judgements over other rule-based or model-based caption metrics. 
The proposed benchmark and metric provide reliable evaluation for LVLM's detailed image captioning ability. 
Guided by this evaluation, we further explore to unleash LVLM's detail caption capabilities by synthesizing high-quality data through a five-stage data construction pipeline. 
Our pipeline only uses a given LVLM itself and other open-source tools, without any human or GPT-4V annotation in the loop.
Experiments show that the proposed data construction strategy significantly improves model-generated detail caption data quality for LVLMs with leading performance, and the data quality can be further improved in a self-looping paradigm.
<p align="center">
<img src="images/intro.png"/>
</p>


## Detail Image Caption Benchmark
We release the DetailCaps-4870 benchmark, which contains 4870 images with high-quality reference captions annotated by GPT-4V&Gemini-1.5-Pro. 
The statistics of DetailCaps-4870 compared with other image caption benchmarks of comparables sizes is shown below:

| Benchmark | Data source | Annt. expert | Img num | ref num | Avg len | Uni. 2-gram |
| --- | --- | --- | --- | --- | --- | --- |
| **COCO<sub>test</sub>** | COCO | Human | $5000$ | $25,010$ | $10.59$ | $61,448$ |
| **Nocaps<sub>val</sub>** | Openimages | Human | $4500$ | $45,000$ | $11.49$ | $116,969$ |
| **DetailCaps-100** | COCO, SAM, LAION, CC, SBU | GPT-4V, Human | $100$ | $100$ | $175.96$ | $10,858$ |
| **DetailCaps-4870** | COCO, SAM, LAION, CC, SBU, Coyo, Flickr | GPT-4V, GPT4O, Gemini-1.5-Pro | $4870$ | $14610$ | $122.06$ | $533,201$ |

The evaluation dataset will soon be available on [Huggingface](https://huggingface.co/). 
Please download the dataset and put it under the `datasets` folder.

## Detail Image Caption Evaluation Metric: CAPTURE
The proposed metric **CAPTURE** (CAPtion evaluation by exTracting and coUpling coRE information) achieves the highest consistency with expert judgements on DetailCaps benchmarks. 
We show the average consistency scores on DetailCaps-100 and DetailCaps-4870 benchmarks in the table below.

| Caption metric | PCC $\rho$ $\uparrow$ | $1-R^2$ $\downarrow$ | Kendall's $\tau$ $\uparrow$ | Sample $\tau$ $\uparrow$ |
| --- | --- | --- | --- | --- |
| **BLEU** | $0.2608$ | $54.75$ | $0.1866$ | $0.2462$ |
| **ROUGE-L** | $0.2951$ | $134.12$ | $0.2149$ | $0.3383$ |
| **CIDEr** | $0.1148$ | $2.6e^7$ | $0.1165$ | $0.0991$ |
| **METEOR** | $0.4022$ | $290.38$ | $0.2927$ | $0.4062$ |
| **SPICE** | $0.4386$ | $155.95$ | $0.3244$ | $0.4718$ |
| **CLIPScore** | $0.3558$ | $21.46$ | $0.2479$ | $0.3841$ |
| **CAPTURE** | $0.5091$ | $8.29$ | $0.3861$ | $0.6018$ |

We evaluate SOTA open-source LVLMs' detail captioning abilities with our benchmark and metric.
The results are listed below.

| Model | Language Model | Caption Data | Resolution | CAPTURE |
| --- | --- | --- | --- | --- |
| **CogVLM** | Vicuna-7B | Human Annt. | $490^2$ | $60.06$ |
| **ShareCaptioner-7B** | Vicuna-7B | GPT-4V Annt. | $448^2$ | $59.80$ |
| **LLaVA-1.5-7B** | Vicuna-7B | Synthesized | $336^2$ | $51.05$ |
| **LLaVA-1.5-13B** | Vicuna-13B | Synthesized | $336^2$ | $51.20$ |
| **LLaVA-NEXT-7B** | Vicuna-7B | GPT-4V Annt. | $336^2$*{1-5} | $58.61$ |
| **LLaVA-NEXT-13B** | Vicuna-13B | GPT-4V Annt. | $336^2$*{1-5} | $59.01$ |
| **LLaVA-NEXT-34B** | Hermes-2-Yi-34B | GPT-4V Annt. | $336^2$*{1-5} | $59.20$ |
| **Mini-Gemini-HD-7B** | Vicuna-7B | GPT-4V Annt. | $336^2$*5 | $57.95$ |
| **Mini-Gemini-HD-13B** | Vicuna-13B | GPT-4V Annt. | $336^2$*5 | $58.66$ |
| **Intern-XComposerV2** | Vicuna-7B | GPT-4V Annt. | $490^2$ | $59.86$ |
| **InternVL-V1.2-PLUS-40B** | Hermes-2-Yi-34B | GPT-4V Annt. | $448^2$ | $60.69$ |
| **InternVL-V1.5-26B** | InternLM-20B | GPT-4V Annt. | $448^2$*{1-41} | $63.42$ |


## Detail Image Caption Construction
We construct a data construction pipeline to unleash LVLM's detail image captioning ability with open-source vision and language tools.
We show the performance of the performance of the proposed data construction pipeline with different LVLM bachbones below. 

| Caption | DetailCaps-100 | DetailCaps-4870 | Average |
| --- | --- | --- | --- |
| **LLaVA-1.5-7B self** | $51.23$ | $51.05$ | $51.14$ |
| **LLaVA-1.5-7B syn** | $57.11$ | $56.25$ | $56.68$ |
| **LLaVA-1.5-13B self** | $51.76$ | $51.20$ | $51.48$ |
| **LLaVA-1.5-13B syn** | $57.36$ | $57.05$ | $57.20$ |
| **LLaVA-NEXT-7B self** | $61.48$ | $58.61$ | $60.73$ |
| **LLaVA-NEXT-7B syn** | $62.24$ | $60.39$ | $61.31$ |
| **Mini-Gemini-7B-HD self** | $59.51$ | $57.95$ | $58.73$ |
| **Mini-Gemini-7B-HD syn** | $60.44$ | $59.07$ | $59.75$ |



## Quick Start

### Environment
Run the following scripts to prepare the environment for CAPTURE and the data construction pipeline.
```bash
conda create -n detailcaption python=3.9
conda activate detailcaption
bash prepare.sh
```

### Detail Image Caption Evaluation
We have wrapped the proposed CAPTURE evaluation metric into pip package, and you can install it as follows:
```bash
pip3 install capture_metric
```
After installation, CAPTURE metric can be used in the same way as other caption evaluation metrics implemented in [pycocoevalcap](https://github.com/sks3i/pycocoevalcap), such as BLEU, CIDEr, METEOR, ROUGE, etc.
Here is an example:
```python
from capture_metric.capture import CAPTURE
refs = {
  <sample_key>: [ref_0, ref_1, ...],
  ...
}
preds = {
  <sample_key>: [pred_caption],
  ...
}

evaluator = CAPTURE()
score = evaluator.compute_score(refs, preds)
print(f"CAPTURE score: {score}")
```

You can now use [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to evaluate you LVLM's detail image caption performance on the DetailCaps-4870 benchmark with CAPTURE metric. 
We refer to [lmms detailcaps](https://github.com/EvolvingLMMs-Lab/lmms-eval) for more details.


### Detail Image Caption Construction
For detail image caption construction, first download SAM, Owlv2, LLaVA-v1.5 (or other LVLM), LLaMA-2 and place them under `ckpt` folder: 
```
ckpt
├─sam
|  ├─sam_vit_h_4b8939.pth
|  └─sam_vit_l_0b3195.pth
├─owlv2-large-patch14-ensemble
├─llava-v1.5-13b
├─llava-v1.5-7b
├─llava-v1.5-13b
├─Llama-2-7b-chat-hf
└─Llama-2-13b-chat-hf
```
Then organize your image data in `.parquet` format with binary image stored in the `frame` field.
Run the followig script to generate annotations for your parquet data files stored in `<source_path>`.
`<model_size>` should be set as either `7b` or `13b`, corresponding to pipelines for different model size. 
```bash
bash generate_all_annotations.sh <model_size> <source_path>
```


## Citation
```bibtex
@article{dong2024benchmarking,
  title={Benchmarking and Improving Detail Image Caption},
  author={Dong, Hongyuan and Li, Jiawen and Wu, Bohong and Wang, Jiacong and Zhang, Yuan and Guo, Haoyuan},
  journal={arXiv preprint arXiv:2405.19092},
  year={2024}
}
```


