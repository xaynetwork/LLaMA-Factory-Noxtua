python scripts/vllm_infer.py \
  --model_name_or_path zai-org/GLM-4.5-Air-Base \
  --adapter_name_or_path /home/mryakhovskiy/LLaMA-Factory-Noxtua/saves/GLM-4.5-Air-Base/lora/16-dpo \
  --dataset alpaca_en_demo \
  --cutoff_len 8096 \
  --max_samples 20 \
  --enable_thinking false \
  --batch_size 4 \
  --max_new_tokens 4096 \
  --pipeline_parallel_size 1 \
  --save_name "generated_predictions-dpo1.jsonl" \
  --vllm_config '{"gpu_memory_utilization": 0.90, "enable_expert_parallel": true, "dtype": "bfloat16"}'

