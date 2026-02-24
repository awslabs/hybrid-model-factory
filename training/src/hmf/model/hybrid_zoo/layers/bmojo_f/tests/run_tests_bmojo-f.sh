MODEL_CONFIG="/path/to/your/bmojo_model/config.json"

torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 4096 --dtype bf16 --window_size 127 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 8192 --dtype bf16 --window_size 127 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 16384 --dtype bf16 --window_size 127 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 32768 --dtype bf16 --window_size 127 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 65536 --dtype bf16 --window_size 127 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 131072 --dtype bf16 --window_size 127 --model_config "$MODEL_CONFIG"

torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 16384 --dtype bf16 --window_size 511 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 32768 --dtype bf16 --window_size 511 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 65536 --dtype bf16 --window_size 511 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 131072 --dtype bf16 --window_size 511 --model_config "$MODEL_CONFIG"

torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 65536 --dtype bf16 --window_size 2047 --model_config "$MODEL_CONFIG"
torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 131072 --dtype bf16 --window_size 2047 --model_config "$MODEL_CONFIG"

torchrun --nproc-per-node 8 test_bmojo-f_sp.py --seqlen 131072 --dtype bf16 --window_size 4095 --model_config "$MODEL_CONFIG"