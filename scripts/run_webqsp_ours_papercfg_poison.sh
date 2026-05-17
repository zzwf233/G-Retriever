#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
PREPROCESS_GPU="${PREPROCESS_GPU:-${GPU%%,*}}"
SEED="${SEED:-0}"
LLM_FROZEN="${LLM_FROZEN:-False}"
MAX_TXT_LEN="${MAX_TXT_LEN:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
PLANNER_MODE="${PLANNER_MODE:-llm}"
PLANNER_MODEL_NAME="${PLANNER_MODEL_NAME:-deepseek-ai/DeepSeek-V3.2}"
PLANNER_API_BASE="${PLANNER_API_BASE:-https://api.siliconflow.cn/v1}"
PLANNER_API_KEY="${PLANNER_API_KEY:-${SILICONFLOW_API_KEY:-}}"
PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE:-0.0}"
PLANNER_MAX_TOKENS="${PLANNER_MAX_TOKENS:-512}"
TARGET_TOP_K="${TARGET_TOP_K:-5}"
MAX_TARGET_TRIALS="${MAX_TARGET_TRIALS:-5}"
ATTACK_MATCH="${ATTACK_MATCH:-substring}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
MAX_MEMORY="${MAX_MEMORY:-}"
FORCE="${FORCE:-1}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export G_RETRIEVER_EMB_BATCH_SIZE="${G_RETRIEVER_EMB_BATCH_SIZE:-16}"
export G_RETRIEVER_EMB_DATA_PARALLEL="${G_RETRIEVER_EMB_DATA_PARALLEL:-0}"
export G_RETRIEVER_FORCE_POISON_DESC="${G_RETRIEVER_FORCE_POISON_DESC:-0}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="output/webqsp/model_name_graph_llm_llm_model_name_7b_llm_frozen_${LLM_FROZEN}_max_txt_len_${MAX_TXT_LEN}_max_new_tokens_${MAX_NEW_TOKENS}_gnn_model_name_gt_patience_2_num_epochs_${NUM_EPOCHS}_seed${SEED}_checkpoint_best.pth"
fi

DECOMPOSED_FILE="dataset/webqsp_ours_papercfg/decomposed_subquestions.jsonl"
PREDICTION_PATH="output/webqsp_ours_papercfg/model_name_graph_llm_llm_model_name_7b_llm_frozen_${LLM_FROZEN}_max_txt_len_${MAX_TXT_LEN}_max_new_tokens_${MAX_NEW_TOKENS}_gnn_model_name_gt_patience_2_num_epochs_${NUM_EPOCHS}_seed${SEED}.csv"

if [[ "${PLANNER_MODE}" == "llm" && -z "${PLANNER_API_KEY}" ]]; then
  echo "PLANNER_API_KEY or SILICONFLOW_API_KEY is required for PLANNER_MODE=llm." >&2
  exit 1
fi

echo "[1/5] Dependency-aware subquestion decomposition"
DECOMP_ARGS=(--output_file "${DECOMPOSED_FILE}" --split test)
if [[ "${FORCE}" == "1" ]]; then
  DECOMP_ARGS+=(--force)
fi
python -m src.dataset.preprocess.webqsp_decompose "${DECOMP_ARGS[@]}"

echo "[2/5] Ours paper-config poisoning: 5 adversarial candidates x 4 triples = 20 triples/question"
POISON_ARGS=(
  --decomposed_file "${DECOMPOSED_FILE}"
  --planner_mode "${PLANNER_MODE}"
  --planner_model_name "${PLANNER_MODEL_NAME}"
  --planner_api_base "${PLANNER_API_BASE}"
  --planner_temperature "${PLANNER_TEMPERATURE}"
  --planner_max_tokens "${PLANNER_MAX_TOKENS}"
  --target_top_k "${TARGET_TOP_K}"
  --max_target_trials "${MAX_TARGET_TRIALS}"
  --seed "${SEED}"
)
if [[ -n "${PLANNER_API_KEY}" ]]; then
  POISON_ARGS+=(--planner_api_key "${PLANNER_API_KEY}")
fi
if [[ "${FORCE}" == "1" ]]; then
  POISON_ARGS+=(--force)
fi
CUDA_VISIBLE_DEVICES="${PREPROCESS_GPU}" python -m src.dataset.preprocess.webqsp_ours_papercfg "${POISON_ARGS[@]}"

echo "[3/5] Build G-Retriever cached retrieval graphs/descriptions"
if [[ "${FORCE}" == "1" ]]; then
  rm -rf dataset/webqsp_ours_papercfg/cached_graphs dataset/webqsp_ours_papercfg/cached_desc
  rm -f dataset/webqsp_ours_papercfg/subquestion_q_embs.pt
fi
CUDA_VISIBLE_DEVICES="${PREPROCESS_GPU}" python -m src.dataset.webqsp_ours_papercfg

echo "[4/5] Evaluate clean checkpoint on paper-config ours-poisoned test graphs"
TRAIN_ARGS=(
  --dataset webqsp_ours_papercfg
  --model_name graph_llm
  --llm_model_name 7b
  --llm_frozen "${LLM_FROZEN}"
  --batch_size 1
  --eval_batch_size 1
  --grad_steps 1
  --num_epochs "${NUM_EPOCHS}"
  --max_txt_len "${MAX_TXT_LEN}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --seed "${SEED}"
  --eval_only True
  --checkpoint_path "${CHECKPOINT_PATH}"
)
if [[ -n "${MAX_MEMORY}" ]]; then
  TRAIN_ARGS+=(--max_memory "${MAX_MEMORY}")
fi
CUDA_VISIBLE_DEVICES="${GPU}" python train.py "${TRAIN_ARGS[@]}"

echo "[5/5] Evaluate attack manipulation and cascade spread metrics"
python -m src.eval_poison \
  --prediction_path "${PREDICTION_PATH}" \
  --poison_record_path dataset/webqsp_ours_papercfg/poison_records.jsonl \
  --attack_match "${ATTACK_MATCH}"

echo "Done."
echo "Prediction file: ${PREDICTION_PATH}"
echo "Utility metrics: ${PREDICTION_PATH%.csv}_metrics.json"
echo "Attack metrics: ${PREDICTION_PATH%.csv}_attack_metrics.json"
