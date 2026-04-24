#!/bin/bash

declare -A model_langs=(
    ["BabyLM-community/GPT-BERT-baseline-babylm-zho-nld-equal"]="nld zho"
    ["BabyLM-community/GPT-BERT-baseline-babylm-eng-zho-equal"]="eng zho"
    ["BabyLM-community/GPT-BERT-baseline-babylm-eng-nld-equal"]="eng nld"
    ["BabyLM-community/GPT-BERT-baseline-babylm-eng-nld-zho-equal"]="eng nld zho"
    ["BabyLM-community/GPT-BERT-baseline-babylm-zho"]="zho"
    ["BabyLM-community/GPT-BERT-baseline-babylm-nld"]="nld"
    ["BabyLM-community/GPT-BERT-baseline-babylm-Strict-Small"]="eng"
    ["BabyLM-community/GPT-BERT-baseline-babylm-Strict"]="eng"
)

for model_name in "${!model_langs[@]}"; do
    echo "Evaluating ${model_name} (all revisions)"
    bash zeroshot_model_fast_all.sh --model_name ${model_name} --langs "${model_langs[$model_name]}"
done
