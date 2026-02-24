MODEL_CONFIG="/path/to/your/gated_deltanet/config.json"

torchrun --nproc-per-node 8 test_gdn.py --seqlen 4096 --dtype bf16 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_gdn.py --seqlen 8192 --dtype bf16 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_gdn.py --seqlen 16384 --dtype bf16 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_gdn.py --seqlen 32768 --dtype bf16 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_gdn.py --seqlen 65536 --dtype bf16 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_gdn.py --seqlen 131072 --dtype bf16 --model_config "$MODEL_CONFIG"
