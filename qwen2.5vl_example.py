#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import time
import base64
import mimetypes
import re
import io
import os
from io import BytesIO
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from openai import OpenAI

# ==== Configuration ====
MODEL_NAME = "qwen2.5-vl-72b-instruct"
BASE_URL   = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds

# System prompts
SYSTEM_PROMPT_ANALYSIS = "You are a helpful medical image analysis assistant."
SYSTEM_PROMPT_DISCRETE = "Only respond with one capital letter."
SYSTEM_PROMPT_JOINT    = "Respond with four or five capital letters (A-H), separated by commas."

# ==== Image handling ====
def compress_image(path: str, max_dim: int = 1024, quality: int = 85, size_limit_mb: int = 10) -> bytes:
    """Compress image to JPEG under size limit (MB)."""
    img = Image.open(path)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    data = buf.getvalue()
    # account for base64 inflation (~1.37x)
    if len(data) * 1.37 > size_limit_mb * 1024 * 1024:
        return compress_image(path, int(max_dim * 0.9), int(quality * 0.9), size_limit_mb)
    return data


def encode_image(path: str, max_dim: int = 1024, quality: int = 85) -> str:
    """Return data URL of compressed image."""
    try:
        img_bytes = compress_image(path, max_dim, quality)
    except Exception:
        with open(path, 'rb') as f:
            img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = 'image/jpeg'
    return f"data:{mime};base64,{b64}"

# ==== Generic API call ====
def api_infer(prompt: str,
              system_prompt: str,
              image_path: str,
              client: OpenAI,
              max_dim: int = 1024,
              quality: int = 85,
              size_limit_mb: int = 10,
              max_tokens: int = None,
              temperature: float = None) -> str:
    data_url = encode_image(image_path, max_dim, quality)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]}
    ]
    last_exc = None
    for i in range(1, MAX_RETRIES + 1):
        try:
            params = {"model": MODEL_NAME, "messages": messages}
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if temperature is not None:
                params["temperature"] = temperature
            resp = client.chat.completions.create(**params)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_exc = e
            print(f"[Warning] attempt {i}/{MAX_RETRIES} failed: {e}")
            if i < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise last_exc


# ==== Entry point ====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--API_KEY',     required=True)
    parser.add_argument('--INPUT_PATH',  required=True)
    parser.add_argument('--OUTPUT_PATH', required=True)
    parser.add_argument('--IMAGE_ROOT',  required=True)
    parser.add_argument('--type',        required=True, choices=['single','joint'])
    args = parser.parse_args()

    client = OpenAI(api_key=args.API_KEY, base_url=BASE_URL)

    if args.type == 'single':
        # read all lines
        with open(args.INPUT_PATH, 'r', encoding='utf-8') as f:
            lines = [l for l in f if l.strip()]
        first = json.loads(lines[0])
        mode = 'analysis' if 'task' in first else 'discrete'

        # single-mode processing
        stats = defaultdict(lambda: {'total':0,'correct':0})
        with open(args.OUTPUT_PATH, 'w', encoding='utf-8') as fout:
            for line in tqdm(lines, desc=f"Processing {mode}"):
                rec = json.loads(line)
                img_path = os.path.join(args.IMAGE_ROOT, rec.get('image_paths'))
                if mode == 'analysis':
                    # report or choice
                    prompt = (rec['question'] + '\n\nWrite the image report as a single coherent paragraph, no more than 500 words.') if rec['task']=='report_generation' else ("Question: " + rec['question'] + "\nOptions:\n" + "\n".join(rec['options']) + "\n\nProvide only the letter as your answer.")
                    try:
                        ans = api_infer(prompt, SYSTEM_PROMPT_ANALYSIS, img_path, client)
                        rec['model_response'] = ans
                    except Exception as e:
                        rec['error'] = str(e)
                else:
                    # discrete QA
                    prompt = (
                        "You are participating in an educational exercise based on visual information.\n\n"
                        f"Question:\n{rec['question']}\nOptions:\n" + "\n".join(rec['options']) + "\n\nPlease answer with a single letter (A-H) only."
                    )
                    try:
                        reply = api_infer(prompt, SYSTEM_PROMPT_DISCRETE, img_path, client,
                                          max_dim=512, quality=75,
                                          max_tokens=50, temperature=0)
                        rec['modal_response']      = reply
                    except Exception as e:
                        rec['error'] = str(e)
                fout.write(json.dumps(rec, ensure_ascii=False)+'\n')

        print(f"✅ Output written to {args.OUTPUT_PATH}")

    else:
        # joint mode
        items = []
        with open(args.INPUT_PATH,'r',encoding='utf-8') as f:
            for l in f:
                if l.strip(): items.append(json.loads(l))
        results = []
        for rec in tqdm(items, desc="Processing joint"):
            img_path = os.path.join(args.IMAGE_ROOT, rec.get('image_paths'))
            try:
                reply = api_infer(rec['joint_prompt'], SYSTEM_PROMPT_JOINT, img_path, client, max_tokens=100, temperature=0)
                rec['model_response'] = reply
            except Exception as e:
                rec['modal_response']={k:'Error' for k in ['modality','body_region','organ','lesion','diagnosis']}
                rec['response_status']=f"Error: {e}"
            finally:
                results.append(rec)
        with open(args.OUTPUT_PATH,'w',encoding='utf-8') as fout:
            for rec in results:
                fout.write(json.dumps(rec, ensure_ascii=False)+'\n')
        print(f"✅ Output written to {args.OUTPUT_PATH}")

if __name__ == '__main__':
    main()
