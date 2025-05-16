#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# ========== Configuration ==========
def parse_args():
    parser = argparse.ArgumentParser(
        description="Map model responses to option letters using DeepSeek API."
    )
    parser.add_argument(
        "--API_KEY",
        required=True,
        help="Your DeepSeek API key"
    )
    parser.add_argument(
        "--INPUT_FILE",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--OUTPUT_FILE",
        required=True,
        help="Path to output JSONL file"
    )
    return parser.parse_args()

# ========== Initialize DeepSeek ==========
def init_client(api_key: str):
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

# ========== Load and Save JSONL ==========
def load_jsonl(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_jsonl(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

# ========== Map model response to option via DeepSeek ==========
def map_answer_to_option(client, question, options, original_answer):
    opt_text = "\n".join(options)
    prompt = (
        "You are a medical assistant. Given the question, the list of options, and a response from another model, "
        "please map the response to the best matching option letter (e.g., A, B, C, D). Only output the letter.\n\n"
        f"Question: {question}\n\nOptions:\n{opt_text}\n\n"
        f"Other model's answer: {original_answer}\n\n"
        "Instruction: Only output the letter (A, B, C, or D) that best matches the model's response. Do not add any other text."
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"❗ DeepSeek request failed: {e}")
        return None

# ========== Main process ==========
def main():
    args = parse_args()
    client = init_client(args.API_KEY)
    input_path = Path(args.INPUT_FILE)
    output_path = Path(args.OUTPUT_FILE)

    dataset = load_jsonl(input_path)
    mapped_results = []

    total_mapped = 0
    failed_mapped = 0
    preview_count = 0

    for sample in tqdm(dataset, desc="Processing and Mapping"):
        task = sample.get("task", "")
        # skip mapping for free-form caption tasks
        if task == "caption":
            mapped_results.append(sample)
            continue

        options = sample.get("options", [])
        raw = sample.get("model_response", "").strip()

        pred_letter = map_answer_to_option(
            client,
            sample.get("question", ""),
            options,
            raw
        )
        letter_to_option = {opt[0]: opt for opt in options if opt}
        if pred_letter in letter_to_option:
            sample["model_response"] = letter_to_option[pred_letter]
        else:
            sample["model_response"] = None
            failed_mapped += 1

        total_mapped += 1
        # print first few samples for sanity check
        if preview_count < 5:
            print(sample)
            preview_count += 1

        mapped_results.append(sample)

    # save results
    save_jsonl(output_path, mapped_results)

    # ========== Statistics ==========
    print(f"\n✅ Mapping complete. Processed {total_mapped} samples")
    print(f"❌ Failed to map (no matching option): {failed_mapped}")
    if total_mapped > 0:
        success_rate = (total_mapped - failed_mapped) / total_mapped * 100
        print(f"✅ Success rate: {round(success_rate, 2)}%")
    print(f"\n📁 Saved to: {output_path}")

if __name__ == "__main__":
    main()