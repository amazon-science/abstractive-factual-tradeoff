#! /usr/bin/sh

input_file=${1}
output_dir=${2}
model_checkpoint=${3}
gpus=${4}
keys=${5:-article,summary}

if [ "$#" -ne 4 ] && [ "$#" -ne 5 ]; then
    echo "command: sh run_factcc.sh input_file output_dir model_checkpoint gpu keys"
    echo "example: sh run_factcc.sh input.jsonl output factcc-checkpoint 0,1,2,3 article,summary"
    exit 1
fi

set -e
set -x

# TODO: change FACTCC_DIR according to your local setting
FACTCC_DIR=${HOME}/factCC
PYTHONPATH=${FACTCC_DIR}:${FACTCC_DIR}/modeling:$PYTHONPATH

FACTCC_WRAPPER_DIR=$(cd -P $(dirname $0) && pwd)

mkdir -p ${output_dir}

python ${FACTCC_WRAPPER_DIR}/create_sentence_based_input.py -k ${keys} -o ${output_dir}/data-dev.jsonl -i ${input_file}

python ${FACTCC_WRAPPER_DIR}/eval.py \
  --task_name factcc_annotated \
  --do_lower_case \
  --per_gpu_eval_batch_size 64 \
  --model_type bert \
  --model_name_or_path bert-base-uncased  \
  --data_dir ${output_dir}/ \
  --checkpoint_dir ${model_checkpoint} \
  --output_dir ${output_dir}/ \
  --visible_gpus ${gpus}

python ${FACTCC_WRAPPER_DIR}/aggregate_factcc_prediction_score.py \
  --prediction_score ${output_dir}/prediction_score.csv \
  --input ${output_dir}/data-dev.jsonl \
  --output ${output_dir}

cp ${output_dir}/sentence_level_mean.csv ${output_dir}/factcc.csv
