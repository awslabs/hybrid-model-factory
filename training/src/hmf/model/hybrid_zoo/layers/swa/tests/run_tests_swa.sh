MODEL_CONFIG="/path/to/your/swa/model/config.json"

torchrun --nproc-per-node 8 test_swa.py --seqlen 4096 --dtype bf16 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_swa.py --seqlen 8192 --dtype bf16 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_swa.py --seqlen 16384 --dtype bf16 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_swa.py --seqlen 32768 --dtype bf16 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_swa.py --seqlen 65536 --dtype bf16 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_swa.py --seqlen 131072 --dtype bf16 --model_config "$MODEL_CONFIG"

# Hybrid Ministral3's SWA, which also scales its queries by position (Llama 4-style). Its small
# original_max_position_embeddings (1024) makes the scale vary within these sequence lengths.
MINISTRAL3_CONFIG="$(dirname "$0")/ministral3_swa_config.json"
torchrun --nproc-per-node 8 test_swa.py --seqlen 4096 --dtype bf16 --model_config "$MINISTRAL3_CONFIG"
torchrun --nproc-per-node 8 test_swa.py --seqlen 16384 --dtype bf16 --model_config "$MINISTRAL3_CONFIG"
