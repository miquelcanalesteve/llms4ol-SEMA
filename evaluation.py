import json
from pathlib import Path
import pandas as pd

def load_ground_truth(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {(item["parent"].strip(), item["child"].strip()) for item in data}

def load_predictions(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {(item["parent"].strip(), item["child"].strip()) for item in data.values()}

def compute_scores(gt_pairs, pred_pairs):
    tp_set = set()
    fp_set = set()
    fn_set = set()

    for pred in pred_pairs:
        if pred in gt_pairs:
            tp_set.add(pred)
        else:
            fp_set.add(pred)

    for gt in gt_pairs:
        if gt not in pred_pairs:
            fn_set.add(gt)

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return tp, fp, fn, precision, recall, f1

if __name__ == "__main__":
    # === Set paths ===
    gt_path = Path("task_c/MatOnto/train_v2/matonto_test_pairs.json")

    pred_paths = [

        Path("task_c_llama_3_1b_ct_train_v2_r6_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_df_all_e1_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_df_all_e2_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_df_q1_e1_true.json"),
        Path("task_c_llama_3_1b_ct_train_v3_r6_base_true.json"),
        Path("task_c_llama_3_1b_ct_train_v3_r6_base_2_true.json"),
        Path("task_c_llama_3_1b_ct_train_v3_r6_df_all_e1_true.json"),
        Path("task_c_llama_3_1b_ct_train_v3_r6_base_owl_wiki_true.json"),
        Path("task_c_llama_3_1b_ct_train_v3_r6_df_all_e1_wiki_owl_true.json"),
        Path("task_c_llama_3_1b_ct_train_v4_r6_base_wiki_true.json"),
        Path("task_c_llama_3_1b_ct_train_v4_r6_base_wiki_S_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_proco_2941_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_proco_2941_without_eot_true.json"),
        Path("real_random_true.json"),
        Path("all_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_fs_1_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_fs_1_one_T_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_fs_2_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_fs_2_one_T_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_fs_3_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_fs_3_one_T_true.json"),
        # Path("task_c_llama_3_1b_ct_train_v2_r6_fs_4_true.json"),
        # Path("task_c_llama_3_1b_ct_train_v2_r6_fs_4_one_T_true.json"),
        # Path("task_c_llama_3_1b_ct_train_v2_r6_fs_5_true.json"),
        # Path("task_c_llama_3_1b_ct_train_v2_r6_fs_5_one_T_true.json"),
        # # Path("task_c_llama_3_1b_ct_train_v2_r6_fs_6_true.json"),
        # Path("task_c_llama_3_1b_ct_train_v2_r6_fs_6_one_T_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_child_class_true.json"),
        Path("task_c_llama_3_1b_ct_train_v4_top10_20_r6_test_v2_true.json"),
        Path("task_c_llama_3_1b_ct_train_v3_top10_30_r6_test_v2_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_fsfs_3_one_T_3360_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_fsfs_3_one_T_6720_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_domain_true.json"),
        Path("task_c_llama_3_1b_ct_train_v2_r6_domain_2_true.json")

        # Path("type_combinations_task_c_output_1b_base_v2_true.json"),
        # Path("type_combinations_task_c_output_1b_instruct_true.json"),
        # Path("type_combinations_task_c_output_8b_instruct_true.json"),

        # Add more files if needed
    ]

    # === Load ground truth ===
    ground_truth = load_ground_truth(gt_path)

    # === Evaluate each prediction file ===
    results = []
    for pred_path in pred_paths:
        if not pred_path.exists():
            print(f"⚠️ File not found: {pred_path}")
            continue
        predictions = load_predictions(pred_path)
        tp, fp, fn, precision, recall, f1 = compute_scores(ground_truth, predictions)
        results.append({
            "file": pred_path.name,
            "tp": round(tp,4),
            "fp": round(fp,4),
            "fn": round(fn,4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        })

    # === Save results to Excel ===
    df = pd.DataFrame(results)
    output_file = "evaluation_summary.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n📊 Results saved to {output_file}")
