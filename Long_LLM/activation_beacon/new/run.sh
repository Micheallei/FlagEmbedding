cd new

torchrun --nproc_per_node 4 -m main.train \
--data_root /home/aiscuser/FlagEmbedding/Long_LLM/activation_beacon/new/data/activation-beacon-new \
--output_dir /home/v-leiyuxuan/blob/MemoryLLM/output/20240520/activation-beacon-llama2-chat-7b \
--model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--train_data activation-beacon:redpajama/train.json[200000] \
--max_length 8192 \
--min_length 1200 \
--enable_beacon \
--beacon_window 1024 \
--beacon_stride 1024 \
--beacon_attn step-expansion \
--beacon_attend_prev True \
--beacon_sink_size 1 \
--beacon_ratio 2 2 2 2 2 4 4 4 4 4 8 8 16 16 32 32 64 128 \
--beacon_ratio_mix step-random \
--beacon_param q k v o \
--gradient_checkpointing \
--save_strategy steps \
--max_steps 10000 \
--save_steps 10000 \
--logging_steps 50 \
--chat_template llama-2 \
--group_by_stride strict > /home/aiscuser/FlagEmbedding/Long_LLM/activation_beacon/new/train2.log 2>&1
# --deepspeed data/deepspeed/stage3.json