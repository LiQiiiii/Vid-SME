<div align="center">
<h1><a href="https://arxiv.org/abs/2506.03179v1" target="_blank">Vid-SME: Membership Inference Attacks against Large Video Understanding Models</a></h1>

  
<div>
<a target="_blank" href="https://arxiv.org/abs/2506.03179v1">
  <img src="https://img.shields.io/badge/arXiv-2506.03179v1-b31b1b.svg" alt="arXiv Paper"/>
</a>
<a href="https://huggingface.co/LIQIIIII/Vid-SME" target="_blank">
  <img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"/>
</a>
<a href="https://huggingface.co/datasets/LIQIIIII/Vid-SME-Eval" target="_blank">
  <img
    alt="HuggingFace dataset (coming soon)"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace--Dataset%20coming%20soon-ffbd45?logo=huggingface&logoColor=white"
  />
</a>
</div>

<div>
Qi Li&emsp;Runpeng Yu&emsp;Xinchao Wang<sup>&dagger;</sup>
</div>
<div>
    <a href="https://sites.google.com/view/xml-nus/people?authuser=0" target="_blank">xML-Lab</a>, National University of Singapore&emsp;
    <sup>&dagger;</sup>corresponding author 
</div>
</div>
</div>

------------------
TL;DR (1) - Introduce Vid-SME, the first dedicated method for video membership inference attacks against large video understanding models.

TL;DR (2) - Benchmarking MIA performance by training three VULLMs, each on a distinct dataset, using different representative training strategies.

## Overview

<div align="center">
    <div style="max-width: 100%; text-align: left; margin-bottom: 20px;">
        <img src="assets/main_pipeline.jpg" alt="Diagram 2" style="display: block; margin: 0 auto; width: 100%;">
    </div>
</div>
<strong>Figure 1.</strong> Vid-SME against Video Understanding Large Language Models (VULLMs). <strong>Left:</strong> An example of the video instruction context used in our experiments. <strong>Middle:</strong> The overall pipeline of Vid-SME. <strong>Right:</strong> The detailed illustration of the membership score calculaiton of Vid-SME.
            
## Installation & Preparation

1. Please follow the instructions provided in [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA) to build the virtual environment.

2. Prepare the dataset and models. Download the [models](https://huggingface.co/LIQIIIII/Vid-SME) and move them into `./checkpoints`. For the [datasets](https://huggingface.co/datasets/LIQIIIII/Vid-SME), the json files are given in the `./video_json` folder, download the related videos and move them into `./video_json/videos`.


## Evaluation

Run Vid-SME on each model via the corresponding script:

```
python Vid_SME_main_CinePile.py
```


## Citation

If you finding our work interesting or helpful to you, please cite as follows:

```
@misc{li2025vidsmemembershipinferenceattacks,
      title={Vid-SME: Membership Inference Attacks against Large Video Understanding Models}, 
      author={Qi Li and Runpeng Yu and Xinchao Wang},
      year={2025},
      eprint={2506.03179},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.03179}, 
}
```
