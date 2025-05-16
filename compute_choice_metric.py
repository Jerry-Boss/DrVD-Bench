import json
import argparse
from collections import defaultdict

def calculate_single_metrics(jsonl_file):
    """
    Compute accuracy per modality and task for single-choice results.
    Skips items with task 'caption'.
    Expects fields: 'modality', 'task', 'answer', 'huatuo_vision_result'.
    """
    stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # skip caption tasks
            if data.get('task') == 'caption':
                continue
            modality = data.get('modality')
            task = data.get('task')
            answer = data.get('answer', '')
            model_result = data.get('huatuo_vision_result')
            # extract option letter
            correct_option = answer.split('.')[0].strip()
            stats[modality][task]['total'] += 1
            if model_result == correct_option:
                stats[modality][task]['correct'] += 1
    # print results
    for modality, tasks in stats.items():
        print(f"Modality: {modality}")
        for task, counts in tasks.items():
            total = counts['total']
            correct = counts['correct']
            acc = correct / total if total else 0
            print(f"  Task: {task}\n    Accuracy: {acc:.2%} ({correct}/{total})")
        print()


def compute_joint_metrics(jsonl_file):
    """
    Compute multi-level accuracy for joint-inference results.
    Expects fields: 'response_status', 'modal_response', and '<level>_answer' for each level.
    Levels: modality, body_region, organ, lesion, diagnosis
    """
    levels = ['modality', 'body_region', 'organ', 'lesion', 'diagnosis']
    total_cnt = defaultdict(int)
    correct_cnt = defaultdict(int)
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get('response_status') != 'OK':
                continue
            modal_resp = data.get('modal_response', {})
            for key in levels:
                if key in modal_resp:
                    total_cnt[key] += 1
                    if modal_resp[key] == data.get(f"{key}_answer"):
                        correct_cnt[key] += 1
    # print report
    print("\n✅ Accuracy Report:")
    for key in levels:
        tot = total_cnt.get(key, 0)
        if tot:
            corr = correct_cnt.get(key, 0)
            acc = corr / tot
            print(f"{key:>12}: {acc:.2%} ({corr}/{tot})")


def main():
    parser = argparse.ArgumentParser(
        description='Compute choice metrics in single or joint mode.'
    )
    parser.add_argument(
        '--json_path', '-j', required=True,
        help='Path to the JSONL results file.'
    )
    parser.add_argument(
        '--type', '-t', choices=['single', 'joint'], required=True,
        help="Type of metric to compute: 'single' for per-task, 'joint' for multi-level joint output."
    )
    args = parser.parse_args()

    if args.type == 'single':
        calculate_single_metrics(args.json_path)
    else:
        compute_joint_metrics(args.json_path)

if __name__ == '__main__':
    main()