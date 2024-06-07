
data_root=/home/v-leiyuxuan/working_dir/activation-beacon-new
model=namespace-Pt/activation-beacon-llama2-7b-chat
data=activation-beacon:lm/codeparrot.json

# language modeling perplexity
CUDA_VISIBLE_DEVICES=0 python -m main.eval_lm \
    --data_root $data_root \
    --stride 0 \
    --model_name_or_path $model \
    --max_length 32768 \
    --num_eval_tokens 256 \
    --num_eval_length 4096 \
    --eval_data $data

CUDA_VISIBLE_DEVICES=0 python -m main.eval_lm \
    --data_root $data_root \
    --stride 0 \
    --model_name_or_path $model \
    --max_length 32768 \
    --num_eval_tokens 256 \
    --num_eval_length 8192 \
    --eval_data $data

CUDA_VISIBLE_DEVICES=0 python -m main.eval_lm \
    --data_root $data_root \
    --stride 0 \
    --model_name_or_path $model \
    --max_length 32768 \
    --num_eval_tokens 256 \
    --num_eval_length 32768 \
    --eval_data $data
