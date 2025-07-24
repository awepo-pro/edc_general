#!/usr/bin/bash -e
# MODEL=gpt-3.5-turbo
MODEL=deepseek-chat

OIE_LLM=${MODEL}
SD_LLM=${MODEL}
SC_LLM=${MODEL}
EE_LLM=${MODEL}

EMBEDDER=hf.co/second-state/E5-Mistral-7B-Instruct-Embedding-GGUF:Q8_0
# EMBEDDER=intfloat/e5-mistral-7b-instruct

SC_EMBEDDER=${EMBEDDER}
SR_EMBEDDER=${EMBEDDER}

DATASET=example

uv run --active run.py \
    --oie_llm $OIE_LLM \
    --oie_few_shot_example_file_path "./few_shot_examples/${DATASET}/oie_few_shot_examples.txt" \
    --sd_llm $SD_LLM \
    --sd_few_shot_example_file_path "./few_shot_examples/${DATASET}/sd_few_shot_examples.txt" \
    --sc_llm $SC_LLM \
    --sc_embedder $SC_EMBEDDER \
    --sr_embedder $SR_EMBEDDER \
    --ee_llm $EE_LLM \
    --input_text_file_path "./datasets/${DATASET}.txt" \
    --output_dir "./output/${DATASET}_target_alignment" \
    --enrich_schema \
    --output_dir "./output/${DATASET}_self_canonicalization" \
    --logging_debug \
    --target_schema_path "./schemas/${DATASET}_schema.csv" \
    --zh
