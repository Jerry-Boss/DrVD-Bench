<h1 align="center" style="border-bottom:none; padding-bottom:0; margin-bottom:0;">
  DrVD-Bench: Do Vision-Language Models Reason Like Human Doctors in Medical Image Diagnosis?
</h1>

<h1 align="center" style="color:#d00;">
  NeurIPS 2024 D&B Track
</h1>

<p align="center">
  <a href="https://jerry-boss.github.io/DrVD-Bench.github.io/">homepage</a> ｜ <a href="https://arxiv.org/abs/2505.24173">arxiv</a> ｜ <a href="https://www.kaggle.com/datasets/tianhongzhou/drvd-bench/data">kaggle</a> ｜ <a href="https://pypi.org/project/drvd-bench/0.2.2/">pypi</a> ｜ <a href="https://huggingface.co/datasets/jerry1565/DrVD-Bench">huggingface</a>
</p>

This repository is the official implementation of the paper: **DrVD-Bench: Do Vision-Language Models Reason Like Human Doctors in Medical Image Diagnosis?**

## 0. Introduction
Vision–language models (VLMs) exhibit strong zero-shot generalization on natural images and show early promise in interpretable medical image analysis. However, existing benchmarks do not systematically evaluate whether these models truly reason like human clinicians or merely imitate superficial patterns.  
To address this gap, we propose DrVD-Bench, the first multimodal benchmark for clinical visual reasoning. DrVD-Bench consists of three modules: *Visual Evidence Comprehension*, *Reasoning Trajectory Assessment*, and *Report Generation Evaluation*, comprising **7 789** image–question pairs.  
Our benchmark covers **20 task types**, **17 diagnostic categories**, and **five imaging modalities**—CT, MRI, ultrasound, X-ray, and pathology. DrVD-Bench mirrors the clinical workflow from modality recognition to lesion identification and diagnosis.  
We benchmark **19 VLMs** (general-purpose & medical-specific, open-source & proprietary) and observe that performance drops sharply as reasoning complexity increases. While some models begin to exhibit traces of human-like reasoning, they often rely on shortcut correlations rather than grounded visual understanding. DrVD-Bench therefore provides a rigorous framework for developing clinically trustworthy VLMs.

<div align="center">
  <img src="images/cover_image.png" alt="cover image" />
</div>

## 1. Quick Start

### 1.1. Prepare Environment
~~~bash
pip3 install -r requirements.txt
~~~

### 1.2. Obtain DeepSeek API Key
Report generation will use **DeepSeek** to extract report keywords, and instruction-following weaker models can also leverage DeepSeek to extract answers from their outputs.  
You can apply for an API key on the [DeepSeek platform](https://platform.deepseek.com/).  
For more details, please refer to the official documentation: [DeepSeek API Docs](https://api-docs.deepseek.com/zh-cn/).

### 1.3. Obtain Model Outputs
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

### 1.4. Compute Metrics

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

## 2. DrVD-Bench PyPi Usage Instructions
Tutorial refer to [qwen2.5vl_7b_demo_pypi.ipynb](qwen2.5vl_7b_demo_pypi.ipynb)
### 2.0. Installation

```bash
pip install drvd-bench
```

---

### 2.1. Function `get_drvd_data`

#### Functionality
Reads JSONL, generates the final prompt, and returns an iterator of (img_path, prompt, origin_data).

#### Note
This code only returns an iterator for retrieving data and does not include model inference.

#### Parameters
- `jsonl_path (str | Path)`: Path to the dataset JSONL file  
- `image_root (str | Path)`: Root directory of the dataset  
- `data_type ({“single”, “joint”})`: Question type, default “single”. Use “joint” only for `joint_qa.jsonl`; all other JSONL files should use “single”  
- `verbose (bool)`: Whether to show a progress bar; default is True

#### Returns
Iterator[(img_path:str, prompt:str, record:dict)]

#### Example
```python
from drvd_bench import get_drvd_data
for img, prompt, original_data in get_drvd_data(
    "test_drvd_pypi/data/visual_evidence_qa.jsonl",
    "test_drvd_pypi/data",
    data_type="single"
):
    print(img, prompt[:80] + "…")
```

---

### 2.2. Function `map_result`

#### Functionality
Calls the DeepSeek Chat API to map open-ended responses to options and writes back to JSONL.

#### Note
Use only when the model instruction-following capability is insufficient and for non-report generation tasks. Not needed if the model instruction-following is strong.

#### Parameters
- `api_key (str)`: DeepSeek API Key; refer to the DeepSeek official website for details  
- `input_path (str | Path)`: Path to the original prediction JSONL file  
- `output_path (str | Path)`: Path to save the mapped JSONL file  
- `base_url (str)`: Private deployment URL; typically does not need modification  
- `show_preview (int)`: Preview the first N results; default is 5

#### Example
```python
from drvd_bench import map_result
map_result(
    api_key="sk-…",
    input_path="pred.jsonl",
    output_path="mapped.jsonl"
)
```

---

### 2.3. Function `compute_choice_metric`

#### Functionality
Calculates accuracy for multiple-choice questions.

#### Note
Your JSONL file must contain an `answer` key for the original data’s answer. The model’s result must be stored in the key `model_response` and should be a single option, such as A, B, C, D, etc.

#### Parameters
- `jsonl_path (str | Path)`: Path to the prediction result JSONL file  
- `mode ({“single”, “joint”})`: Question type, default “single”. Use “joint” only for `joint_qa.jsonl`; all other multiple-choice JSONL result files should use “single”

#### Example
```python
from drvd_bench import compute_choice_metric
compute_choice_metric(
    "visual_evidence_qa_pred.jsonl",
    mode="single"
)
```

---

### 2.4. Function `compute_report_generation_metric`

#### Functionality
Computes BLEU and BERTScore F1 for report generation tasks.

#### Note
Your JSONL file must contain an `answer` key for the original data’s answer. The model’s result must be stored in the key `model_response`.

#### Parameters
- `api_key (str)`: DeepSeek API Key  
- `json_path (str | Path)`: Path to the JSONL result file containing `answer` / `model_response` fields for report generation  
- `base_url (str)`: Private deployment URL; typically does not need modification

#### Example
```python
from drvd_bench import compute_report_generation_metric
compute_report_generation_metric(
    api_key="sk-…",
    json_path="report_generation_preds.jsonl"
)
```

## Contact
- **Tianhong Zhou**   · <zth24@mails.tsinghua.edu.cn>  
- **Yin Xu** · <xuyin23@mails.tsinghua.edu.cn>  
- **Yingtao Zhu** · <zhuyt22@mails.tsinghua.edu.cn>
