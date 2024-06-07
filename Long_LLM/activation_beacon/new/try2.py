import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.llama.modeling_llama import LlamaForCausalLM

model_id = "/home/v-leiyuxuan/blob/MemoryLLM/output/20240520/activation-beacon-llama2-chat-7b-full/checkpoint-10000"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print(model.config.beacon_ratio)

model = model.cuda().eval()

with torch.no_grad():
    # short context
    messages = [{"role": "user", "content": "Tell me about yourself."}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print(f"Input Length: {inputs['input_ids'].shape[1]}")
    print(f"Output:       {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    # reset memory before new generation task
    model.memory.reset()

    # long context
    perfix = "There is an important info hidden inside a lot of irrelevant text before. Find it and memorize them. I will quiz you about the important information there. "
    suffix = "What is the pass key?"
    context = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    key = "The pass key is mylove55. Remember it. mylove55 is the pass key. "
    # x, y = 100, 235
    text = perfix + context*200 + key + context*180 + suffix
    text2 = perfix + context*235 + key + context*100 + suffix
    text3 = perfix + context*300 + key + context*35 + suffix
    
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    outputs = model.generate(**inputs, do_sample=False, top_p=1, temperature=1, max_new_tokens=20)[:, inputs["input_ids"].shape[1]:]
    print("*"*20)
    print(f"Input Length: {inputs['input_ids'].shape[1]}")
    # print(f"Answers:      {example['answer']}")
    print(f"Prediction:   {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# "beacon_ratio": [
#     2,
#     2,
#     2,
#     2,
#     2,
#     4,
#     4,
#     4,
#     4,
#     4,
#     8,
#     8,
#     16,
#     16,
#     32,
#     32,
#     64,
#     128
#   ],