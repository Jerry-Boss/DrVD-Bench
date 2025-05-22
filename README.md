<h1 align="center">DrVD-Bench: Do Vision-Language Models Reason Like Human Doctors in Medical Image Diagnosis?</h1>

<p align="center">
  <a href="#">paper</a> ｜ <a href="https://www.kaggle.com/datasets/tianhongzhou/drvd-bench/data">kaggle</a> ｜ <a href="https://huggingface.co/datasets/jerry1565/DrVD-Bench">huggingface</a> ｜ <a href="https://github.com/1565220678/DrVD-Bench">github</a>
</p>

This repository is the official implementation of the paper: **DrVD-Bench: Do Vision-Language Models Reason Like Human Doctors in Medical Image Diagnosis?**

## Introduction
Vision–language models (VLMs) exhibit strong zero-shot generalization on natural images and show early promise in interpretable medical image analysis. However, existing benchmarks do not systematically evaluate whether these models truly reason like human clinicians or merely imitate superficial patterns.  
To address this gap, we propose DrVD-Bench, the first multimodal benchmark for clinical visual reasoning. DrVD-Bench consists of three modules: *Visual Evidence Comprehension*, *Reasoning Trajectory Assessment*, and *Report Generation Evaluation*, comprising **7 789** image–question pairs.  
Our benchmark covers **20 task types**, **17 diagnostic categories**, and **five imaging modalities**—CT, MRI, ultrasound, X-ray, and pathology. DrVD-Bench mirrors the clinical workflow from modality recognition to lesion identification and diagnosis.  
We benchmark **19 VLMs** (general-purpose & medical-specific, open-source & proprietary) and observe that performance drops sharply as reasoning complexity increases. While some models begin to exhibit traces of human-like reasoning, they often rely on shortcut correlations rather than grounded visual understanding. DrVD-Bench therefore provides a rigorous framework for developing clinically trustworthy VLMs.

<div align="center">
  <img src="images/cover_image.png" alt="cover image" />
</div>

## Quick Start

### Prepare Environment
~~~bash
pip3 install -r requirements.txt
~~~

### Obtain DeepSeek API Key
Report generation will use **DeepSeek** to extract report keywords, and instruction-following weaker models can also leverage DeepSeek to extract answers from their outputs.  
You can apply for an API key on the [DeepSeek platform](https://platform.deepseek.com/).  
For more details, please refer to the official documentation: [DeepSeek API Docs](https://api-docs.deepseek.com/zh-cn/).

### Obtain Model Outputs
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tianhongzhou/drvd-bench/data) or [Hugging Face](https://huggingface.co/datasets/jerry1565/DrVD-Bench).  
2. Run inference with your model and append the results to the `model_response` field in the corresponding files.  
3. **`model_response` format requirements**  
   - **visual_evidence_qa.jsonl / independent_qa.jsonl**: Single letter `A` / `B` / `C` …  
   - **joint_qa.jsonl**: List containing only letters, separated by commas, e.g., `['B','D','A']`  
   - **report_generation.jsonl**: Full string  

#### Inference Example Using Qwen-2.5-VL-72B API
The Qwen-2.5-VL-72B API can be obtained on the Alibaba Cloud Bailian platform ([link](https://bailian.console.aliyun.com/?tab=model#/model-market)).

· task - joint_qa.jsonl
~~~bash
python qwen2.5vl_example.py \
  --API_KEY="your_qwen_api_key" \
  --INPUT_PATH="/path/to/joint_qa.jsonl" \
  --OUTPUT_PATH="/path/to/result.jsonl" \
  --IMAGE_ROOT='path/to/benchmark/data/root' \
  --type="joint"
~~~

· other tasks
~~~bash
python qwen2.5vl_example.py \
  --API_KEY="your_qwen_api_key" \
  --INPUT_PATH="/path/to/qa.jsonl" \
  --OUTPUT_PATH="/path/to/result.jsonl" \
  --IMAGE_ROOT='path/to/benchmark/data/root' \
  --type="single"
~~~

#### Mapping Script
Applicable for instruction-following weaker models; if your model cannot standardize outputs according to the above format, you can use the following script to extract option answers from the `model_response` field:
~~~bash
python map.py \
  --API_KEY="your_deepseek_api_key" \
  --INPUT_FILE="/path/to/model_result.jsonl" \
  --OUTPUT_FILE="/path/to/model_result_mapped.jsonl"
~~~

### Compute Metrics

#### task - visual_evidence_qa.jsonl / independent_qa.jsonl
~~~bash
python compute_choice_metric.py \
  --json_path="/path/to/results.jsonl" \
  --type='single'
~~~

#### task - joint_qa.jsonl
~~~bash
python compute_choice_metric.py \
  --json_path="/path/to/results.jsonl" \
  --type='joint'
~~~

#### task - report_generation.jsonl
~~~bash
python report_generation_metric.py \
  --API_KEY='your_deepseek_api_key' \
  --JSON_PATH='/path/to/results.jsonl'
~~~

## Contact
- **Tianhong Zhou**   · <zth24@mails.tsinghua.edu.cn>  
- **Yin Xu** · <xuyin23@mails.tsinghua.edu.cn>  
- **Yingtao Zhu** · <zhuyt22@mails.tsinghua.edu.cn>
