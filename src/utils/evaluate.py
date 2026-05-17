import json
import os
import pandas as pd
import re
import string
from collections import Counter, defaultdict


def get_accuracy_gqa(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        if label in pred:
            correct += 1
    return correct / len(df)


def get_metrics_gqa(path):
    acc = get_accuracy_gqa(path)
    return {"Test Acc": acc}


def get_accuracy_expla_graphs(path):
    df = pd.read_json(path, lines=True)
    # compute accuracy
    correct = 0
    for pred, label in zip(df["pred"], df["label"]):
        matches = re.findall(r"support|Support|Counter|counter", pred.strip())
        if len(matches) > 0 and matches[0].lower() == label:
            correct += 1

    return correct / len(df)


def get_metrics_expla_graphs(path):
    acc = get_accuracy_expla_graphs(path)
    return {"Test Acc": acc}


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def eval_hits_at_1(prediction, answer):
    prediction = [p for p in prediction if str(p).strip()]
    if len(prediction) == 0:
        return 0
    return eval_hit(prediction[0], answer)


def eval_exact_match(prediction, answer):
    pred_set = {normalize(str(p)) for p in prediction if normalize(str(p))}
    answer_set = {normalize(str(a)) for a in answer if normalize(str(a))}
    return int(len(pred_set) > 0 and pred_set == answer_set)


def _split_predictions(prediction):
    prediction = str(prediction).replace("|", "\n")
    return [p.strip() for p in prediction.split("\n") if p.strip()]


def attack_target_match(prediction, target, mode='substring'):
    pred = normalize(str(prediction))
    tgt = normalize(str(target))
    if not pred or not tgt:
        return False
    if mode == 'exact':
        return pred == tgt
    if mode == 'word':
        return re.search(rf"(?<!\w){re.escape(tgt)}(?!\w)", pred) is not None
    return tgt in pred


def get_metrics_webqsp(path):
    df = pd.read_json(path, lines=True)

    # Load results
    acc_list = []
    hit_list = []
    hits_at_1_list = []
    em_list = []
    f1_list = []
    precission_list = []
    recall_list = []

    for prediction, answer in zip(df.pred.tolist(), df.label.tolist()):

        answer = answer.split("|")

        prediction = _split_predictions(prediction)
        f1_score, precision_score, recall_score = eval_f1(prediction, answer)
        f1_list.append(f1_score)
        precission_list.append(precision_score)
        recall_list.append(recall_score)
        prediction_str = " ".join(prediction)
        acc = eval_acc(prediction_str, answer)
        hit = eval_hit(prediction_str, answer)
        hits_at_1 = eval_hits_at_1(prediction, answer)
        em = eval_exact_match(prediction, answer)
        acc_list.append(acc)
        hit_list.append(hit)
        hits_at_1_list.append(hits_at_1)
        em_list.append(em)

    acc = sum(acc_list) * 100 / len(acc_list)
    hit = sum(hit_list) * 100 / len(hit_list)
    hits_at_1 = sum(hits_at_1_list) * 100 / len(hits_at_1_list)
    em = sum(em_list) * 100 / len(em_list)
    f1 = sum(f1_list) * 100 / len(f1_list)
    pre = sum(precission_list) * 100 / len(precission_list)
    recall = sum(recall_list) * 100 / len(recall_list)

    return {
        "Accuracy": acc,
        "Hit": hit,
        "Precision": pre,
        "Recall": recall,
        "F1": f1,
        "Hits@1": hits_at_1,
        "EM": em,
        "Test Acc": hit,
    }


def get_accuracy_webqsp(path):
    metrics = get_metrics_webqsp(path)
    for name in ["Accuracy", "Hit", "Precision", "Recall", "F1", "Hits@1", "EM"]:
        print(f"{name}: {metrics[name]:.4f}")

    return metrics["Hit"]


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _iter_poison_targets(row):
    target = row.get('poison_target')
    if isinstance(target, str) and target.strip():
        yield target
    for target in row.get('adversarial_candidates', []) or []:
        if isinstance(target, str) and target.strip():
            yield target
    attack_meta = row.get('attack_meta', {}) or {}
    if isinstance(attack_meta, dict):
        target = attack_meta.get('target_answer') or attack_meta.get('poison_target')
        if isinstance(target, str) and target.strip():
            yield target


def get_attack_metrics_webqsp(prediction_path, poison_record_path, match_mode='substring'):
    pred_rows = {str(row.get('id', '')): row for row in _load_jsonl(prediction_path)}
    poison_rows = _load_jsonl(poison_record_path)

    targets_by_parent = defaultdict(set)
    sub_records = []
    retrieval_exposed = 0
    for row in poison_rows:
        targets = {normalize(t) for t in _iter_poison_targets(row) if normalize(t)}
        if not targets:
            continue
        if row.get('retrieval_target_in_desc'):
            retrieval_exposed += 1
        parent_id = str(row.get('parent_id', row.get('id', '')))
        if parent_id:
            targets_by_parent[parent_id].update(targets)
        sub_records.append({
            'id': str(row.get('id', '')),
            'parent_id': parent_id,
            'sub_id': row.get('sub_id', 0),
            'question': row.get('question', ''),
            'needs_prev_answer': bool(row.get('needs_prev_answer', False)),
            'dep_type': row.get('dep_type', 'none'),
            'targets': targets,
        })

    a_precision = []
    a_h1 = []
    a_mrr = []
    hit_by_sub = {}
    hit_by_parent = {}
    parent_hits = defaultdict(list)
    for rec in sub_records:
        pred_row = pred_rows.get(rec['id'])
        if pred_row is None:
            pred_row = pred_rows.get(rec['parent_id'])
        if pred_row is None:
            continue
        targets = rec['targets']
        preds = _split_predictions(pred_row.get('pred', ''))
        hit_count = 0
        reciprocal_rank = 0.0
        for rank, pred in enumerate(preds, start=1):
            matched = any(attack_target_match(pred, target, match_mode) for target in targets)
            if matched:
                hit_count += 1
                if reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / rank
        a_precision.append(hit_count / max(1, len(preds)))
        a_h1.append(1 if preds and any(attack_target_match(preds[0], target, match_mode) for target in targets) else 0)
        a_mrr.append(reciprocal_rank)
        sub_hit = hit_count > 0
        hit_by_sub[rec['id']] = sub_hit
        parent_hits[rec['parent_id']].append(sub_hit)

    for parent_id in targets_by_parent:
        hits = parent_hits.get(parent_id, [])
        hit_by_parent[parent_id] = any(hits) if hits else False

    n_attack = max(1, len(a_precision))
    attack_metrics = {
        'A-Precision': sum(a_precision) * 100 / n_attack,
        'A-H@1': sum(a_h1) * 100 / n_attack,
        'A-MRR': sum(a_mrr) * 100 / n_attack,
        'covered_attack_samples': len(a_precision),
        'targeted_parent_samples': len(targets_by_parent),
        'targeted_subquestion_samples': len(sub_records),
        'retrieval_exposed_subquestion_samples': retrieval_exposed,
    }

    shared_groups = defaultdict(list)
    dep_hits = []
    by_parent = defaultdict(list)
    for rec in sub_records:
        parent_hit = hit_by_parent.get(rec['parent_id'], False)
        rec['hit'] = parent_hit
        by_parent[rec['parent_id']].append(rec)
        shared_groups[normalize(rec['question'])].append(rec)
        if rec['needs_prev_answer']:
            dep_hits.append(parent_hit)

    parent_spread = sum(hit_by_parent.values()) * 100 / max(1, len(hit_by_parent))
    shared_rates = []
    for rows in shared_groups.values():
        parents = {row['parent_id'] for row in rows}
        if len(parents) <= 1:
            continue
        shared_rates.append(sum(hit_by_parent.get(parent, False) for parent in parents) / max(1, len(parents)))

    chain_scores = []
    breakpoints = Counter()
    for rows in by_parent.values():
        deps = sorted([row for row in rows if row['needs_prev_answer']], key=lambda x: int(x.get('sub_id') or 0))
        if not deps:
            continue
        first_break = None
        for row in deps:
            if not row['hit']:
                first_break = row.get('sub_id')
                break
        chain_scores.append(1 if first_break is None else 0)
        breakpoints['none' if first_break is None else str(first_break)] += 1

    spread_metrics = {
        'shared_parent_spread_rate': sum(shared_rates) * 100 / max(1, len(shared_rates)),
        'overall_parent_spread_rate': parent_spread,
        'chain_success@k': sum(chain_scores) * 100 / max(1, len(chain_scores)),
        'dependency_ASR': sum(dep_hits) * 100 / max(1, len(dep_hits)),
        'breakpoint_histogram': dict(breakpoints),
        'poison_subquestions': len(sub_records),
        'dependency_subquestions': len(dep_hits),
    }

    return {'attack': attack_metrics, 'spread': spread_metrics}


def save_metrics(dataset, path):
    metrics = metric_funcs[dataset](path)
    for name in ["Accuracy", "Hit", "Precision", "Recall", "F1", "Hits@1", "EM"]:
        if name in metrics:
            print(f"{name}: {metrics[name]:.4f}")

    metrics_path = os.path.splitext(path)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"metrics path: {metrics_path}")

    return metrics


eval_funcs = {
    "expla_graphs": get_accuracy_expla_graphs,
    "scene_graphs": get_accuracy_gqa,
    "scene_graphs_baseline": get_accuracy_gqa,
    "webqsp": get_accuracy_webqsp,
    "webqsp_baseline": get_accuracy_webqsp,
    "webqsp_ours": get_accuracy_webqsp,
    "webqsp_ours_papercfg": get_accuracy_webqsp,
    "webqsp_rand": get_accuracy_webqsp,
}

metric_funcs = {
    "expla_graphs": get_metrics_expla_graphs,
    "scene_graphs": get_metrics_gqa,
    "scene_graphs_baseline": get_metrics_gqa,
    "webqsp": get_metrics_webqsp,
    "webqsp_baseline": get_metrics_webqsp,
    "webqsp_ours": get_metrics_webqsp,
    "webqsp_ours_papercfg": get_metrics_webqsp,
    "webqsp_rand": get_metrics_webqsp,
}
