# Benchmarking and Improving Detail Image Caption

Code and data for paper: 

*Benchmarking and Improving Detail Image Caption*. 
Hongyuan Dong\*, Jiawen Li\*, Bohong Wu, Jiacong Wang, Yuan Zhang, Haoyuan Guo (* Equal Contribution)
	
<!-- If you find this work useful and use it on your own research, please cite our paper. --> 
Our paper is now available on [arXiv](https://github.com/foundation-multimodal-models/detail_image_caption). 

	
## Overview
Image captioning has long been regarded as a fundamental task in visual understanding. 
Recently, however, few large vision-language model (LVLM) research discusses model's image captioning performance because of the outdated short-caption benchmarks and unreliable evaluation metrics. 
In this work, we propose to benchmark detail image caption task by curating high-quality evaluation datasets annotated by human experts, GPT-4V and Gemini-1.5-Pro.
We also design a more reliable caption evaluation metric called \textbf{CAPTURE} (CAPtion evaluation by exTracting and coUpling coRE information).
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
| **DetailCaps-100** | COCO, SAM, LAION, CC, SBU | GPT-4V&Human | $100$ | $100$ | $175.96$ | $10,858$ |
| **DetailCaps-4870** | COCO, SAM, LAION, CC, SBU, Coyo, Flickr | GPT-4V&Gemini-1.5-Pro | $4870$ | $9740$ | $122.06$ | $377,184$ |

The evaluation dataset will soon be available on [Huggingface](https://huggingface.co/). 
Please download the dataset and put it under the `datasets` folder.

## Detail Image Caption Evaluation Metric: CAPTURE
The proposed metric **CAPTURE** (CAPtion evaluation by exTracting and coUpling coRE information) achieves the highest consistency with expert judgements on DetailCaps benchmarks. 
We show the average consistency scores on DetailCaps-100 and DetailCaps-4870 benchmarks in the table below.

| Caption metric | PCC $\rho$ $\uparrow$ | $1-R^2$ $\downarrow$ | Kendall's $\tau$ $\uparrow$ | Sample $\tau$ $\uparrow$ |
| --- | --- | --- | --- | --- |
| **BLEU** | $0.2625$ | $60.57$ | $0.1879$ | $0.2488$ |
| **ROUGE-L** | $0.2923$ | $138.06$ | $0.2127$ | $0.3312$ |
| **CIDEr** | $0.1024$ | $1.99e^7$ | $0.1034$ | $0.0756$ |
| **METEOR** | $0.4015$ | $289.02$ | $0.2922$ | $0.4075$ |
| **SPICE** | $0.4368$ | $128.85$ | $0.3230$ | $0.4687$ |
| **CLIPScore** | $0.3498$ | $32.46$ | $0.2423$ | $0.3519$ |
| **CAPTURE** | $0.5051$ | $8.20$ | $0.3822$ | $0.5927$ |

## Improving Detail Image Caption
We construct a data construction pipeline to unleash LVLM's detail image captioning ability with open-source vision and language tools.
We show the performance of the performance of the proposed data construction pipeline with different LVLM bachbones below. 

| Caption | DetailCaps-100 | DetailCaps-4870 | Average |
| --- | --- | --- | --- |
| **LLaVA-1.5-7B self** | $51.23$ | $51.27$ | $51.25$ |
| **LLaVA-1.5-7B syn** | $57.11$ | $56.18$ | $56.64$ |
| **LLaVA-1.5-13B self** | $51.76$ | $51.45$ | $51.61$ |
| **LLaVA-1.5-13B syn** | $57.36$ | $56.83$ | $57.09$ |
| **LLaVA-NEXT-7B self** | $61.48$ | $59.86$ | $60.67$ |
| **LLaVA-NEXT-7B syn** | $62.24$ | $60.10$ | $61.17$ |
| **Mini-Gemini-7B-HD self** | $59.51$ | $57.68$ | $58.59$ |
| **Mini-Gemini-7B-HD syn** | $60.44$ | $58.64$ | $59.54$ |



## Quick Start

### Environment
Run the following scripts to prepare the environment for CAPTURE and the data construction pipeline.
```bash
conda create -n detailcaption python=3.9
conda activate detailcaption
bash prepare.sh
```

### Detail Image Caption Evaluation
To evaluate the performance of a LVLM on DetailCaps-4870, run the following scripts. 
```bash
bash evaluate.sh <model_prediction>
```
`<model_prediction>` is the path of the model-generated caption file.
Please organize your results in the following format:
```
{
    'id': '0001',
    'caption': 'A man is walking on the street.'
}, 
......
```

### Detail Image Caption Construction
Organize your image data in `.parquet` format with binary image stored in the `frame` field.
Run the followig script to generate annotations for your parquet data files stored in `<source_path>`.
```bash
bash generate_all_annotations.sh <model_size> <source_path>
```


## Citation
```bibtex
@article{}
```


