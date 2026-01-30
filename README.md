# Flora: Effortless Context Construction to Arbitrary Length and Scale (AAAI'26 Oral)

[Flora: Effortless Context Construction to Arbitrary Length and Scale](https://arxiv.org/pdf/2507.19786?) (AAAI'26 Oral)


<p align="center" width="40%">
<a ><img src="images/icon.png" alt="overview" style="width: 40%; min-width: 300px; display: block; margin: auto;"></a>
</p>

This is the repo for the Flora project, which introduces an LLM/Human-free augmentation method for infinite Long-Context SFT data construction, which concurrently **improves the LLM long-context understanding and retrieval performances** and **significantly mitigates the short-context performance degredation**. 



## Contents
- [Overview](#overview)
- [Run Code](#run-code)
- [Citation](#citation)
- [Our Related Works](#our-related-works)

## Overview

Effectively handling long contexts is challenging for Large Language Models (LLMs) due to the rarity of long texts, high computational demands, and substantial forgetting of short-context abilities. Recent approaches have attempted to construct long contexts for instruction tuning, but these methods often require LLMs or human interventions, which are both costly and limited in length and diversity. Also, the drop in short-context performances of present long-context LLMs remains significant. In this paper, we introduce Flora, an effortless (human/LLM-free) long-context construction strategy. Flora can markedly enhance the long-context performance of LLMs by arbitrarily assembling short instructions based on categories and instructing LLMs to generate responses based on long-context meta-instructions. This enables Flora to produce contexts of arbitrary length and scale with rich diversity, while only slightly compromising short-context performance. Experiments on Llama3-8B-Instruct and QwQ-32B show that LLMs enhanced by Flora excel in three long-context benchmarks while maintaining strong performances in short-context tasks.

<p align="center" width="90%">
<a ><img src="images/method.png" alt="overview" style="width: 90%; min-width: 300px; display: block; margin: auto;"></a>
</p>




## Run Code

```
python Flora/data_construct_flora.py \
    --data_path data/infinity_instruct_data.json \
    --save_path infinity_instruct_flora.json \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --epo_num 1 \
    --max_length 128000 \
    --wrap_max_num 10 \
    --version all
```

```--data_path```: Input data, in alpaca format. <br>
```--save_path```: Save data path. <br>
```--model_name_or_path```: The model used to calculate the token counts. <br>
```--epo_num```: The times of random mosaic process to be run. <br>
```--max_length```: Max token length. <br>
```--version```: Which Strategy to use. <br>

## Training

We use the prompt and code base from [LLAMA-Factory](https://github.com/hiyouga/LlamaFactory):


## Citation

Please consider citing our papers if you think our codes, data, or models are useful. Thank you! <br>

```
@article{chen2025flora,
  title={Flora: Effortless Context Construction to Arbitrary Length and Scale},
  author={Chen, Tianxiang and Tan, Zhentao and Bo, Xiaofan and Wu, Yue and Gong, Tao and Chu, Qi and Ye, Jieping and Yu, Nenghai},
  journal={arXiv preprint arXiv:2507.19786},
  year={2025}
}
```

## Our Related Works

If you are a NLPer and interested in **LLM Continual Pretraining** for Domain-specific Knowledge Injection, please see [Cherry_LLM](https://github.com/MingLiiii/Cherry_LLM). <br>
If you are a CVer and interested in **Medical AI**, please see [Zig-RiR](https://github.com/tianyi-lab/Mosaic-IT). <br>









# Flora
Official Implementation of Human/LLM-free SFT data construction strategy for long-context LLMs: "Flora: Effortless Context Construction to Arbitrary Length and Scale"

The Paper is Under Review, Code will come soon
