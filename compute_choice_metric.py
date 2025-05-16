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
            modality = data.get('modality')
            if 'task' in data:
                task = data.get('task')
            else:
                task = data.get('level')
            answer = data.get('answer', '')
            model_result = data.get('model_response')
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
            print(f"  {task}: {acc:.2%} ({correct}/{total})")
        print()


def compute_joint_metrics(jsonl_file):
    """
    Compute hierarchical accuracy for joint‐inference results,
    grouping by modality and then reporting accuracy at each level.
    Expects each record to have:
      - model_response: string like "A,B,C,D"
      - '<level>_answer' fields for each level, e.g. modality_answer="C. CT"
      - a 'modality' field for grouping
    """
    levels = ['modality', 'organ', 'lesion', 'diagnosis']
    # metrics[modality][level] = {'total': int, 'correct': int}
    metrics = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

    # Tally totals and corrects per modality and level
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            mod = data.get('modality', 'Unknown')
            # split the string "C,A,C,D" into list ['C','A','C','D']
            resp_tokens = data.get('model_response', '')
            if isinstance(resp_tokens, str):
                resp_list = resp_tokens.split(',')
            else:
                resp_list = []

            for idx, level in enumerate(levels):
                metrics[mod][level]['total'] += 1
                # correct answer is like "C. CT" → take first char
                ans_field = data.get(f"{level}_answer", "")
                correct_letter = ans_field[0] if isinstance(ans_field, str) and ans_field else ""
                if idx < len(resp_list) and resp_list[idx] == correct_letter:
                    metrics[mod][level]['correct'] += 1

    # Print the report
    print("\n✅ Hierarchical Accuracy Report by Modality:")
    for mod, level_stats in metrics.items():
        print(f"\nModality: {mod}")
        for level in levels:
            tot = level_stats[level]['total']
            corr = level_stats[level]['correct']
            acc = corr / tot if tot else 0.0
            print(f"  {level:>12}: {acc:.2%} ({corr}/{tot})")


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