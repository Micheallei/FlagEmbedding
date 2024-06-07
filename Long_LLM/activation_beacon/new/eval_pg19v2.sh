
data_root=/home/v-leiyuxuan/working_dir/FlagEmbedding/Long_LLM/activation_beacon/new/data/activation-beacon-new
model=namespace-Pt/activation-beacon-llama2-7b-chat

# language modeling perplexity
python -m main.eval_lm \
    --data_root $data_root \
    --stride 0 \
    --model_name_or_path $model \
    --max_length 4096 \
    --num_eval_tokens 4096 \
    --num_eval_length 4096 \
    --eval_data activation-beacon:lm/pg19.json

python -m main.eval_lm \
    --data_root $data_root \
    --stride 0 \
    --model_name_or_path $model \
    --max_length 8192 \
    --num_eval_tokens 8192 \
    --num_eval_length 8192 \
    --eval_data activation-beacon:lm/pg19.json

python -m main.eval_lm \
    --data_root $data_root \
    --stride 0 \
    --model_name_or_path $model \
    --max_length 32768 \
    --num_eval_tokens 32768 \
    --num_eval_length 32768 \
    --eval_data activation-beacon:lm/pg19.json
