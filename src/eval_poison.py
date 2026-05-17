import argparse
import json
import os

from src.utils.evaluate import get_attack_metrics_webqsp


def main():
    parser = argparse.ArgumentParser(description='Evaluate WebQSP poison attack metrics.')
    parser.add_argument('--prediction_path', required=True)
    parser.add_argument('--poison_record_path', default='dataset/webqsp_ours/poison_records.jsonl')
    parser.add_argument('--output_path', default='')
    parser.add_argument('--attack_match', choices=['substring', 'exact', 'word'], default='substring')
    args = parser.parse_args()

    metrics = get_attack_metrics_webqsp(
        args.prediction_path,
        args.poison_record_path,
        match_mode=args.attack_match,
    )

    attack = metrics['attack']
    spread = metrics['spread']
    print('Attack manipulation:')
    for key in [
        'A-Precision',
        'A-H@1',
        'A-MRR',
        'covered_attack_samples',
        'targeted_parent_samples',
        'targeted_subquestion_samples',
        'retrieval_exposed_subquestion_samples',
    ]:
        print(f'{key}: {attack[key]:.4f}' if isinstance(attack[key], float) else f'{key}: {attack[key]}')

    print('Spread / chain:')
    for key in ['shared_parent_spread_rate', 'overall_parent_spread_rate', 'chain_success@k', 'dependency_ASR']:
        print(f'{key}: {spread[key]:.4f}')
    print(f'breakpoint_histogram: {json.dumps(spread["breakpoint_histogram"], sort_keys=True)}')
    print(f'poison_subquestions: {spread["poison_subquestions"]}')
    print(f'dependency_subquestions: {spread["dependency_subquestions"]}')

    output_path = args.output_path
    if not output_path:
        output_path = os.path.splitext(args.prediction_path)[0] + '_attack_metrics.json'
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write('\n')
    print(f'attack metrics path: {output_path}')


if __name__ == '__main__':
    main()
