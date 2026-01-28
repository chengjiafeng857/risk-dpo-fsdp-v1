import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random
from datasets import load_dataset
from tqdm import tqdm
import os
import yaml
import json
from dataset_process_hh import build_HH_dataset
import argparse

# load yaml config
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# control randomness
def random_controler(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# load dpo trained model
def load_the_dpo_model(model_path, use_bf16):
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",         
    )
    return policy

# generate response
@ torch.no_grad()
def generate_response(model, tokenizer, prompt, max_prompt_length, max_new_tokens, do_sample, temperature, top_p):
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_prompt_length
        )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # repetition_penalty=1.1,
        # no_repeat_ngram_size=4,
        # num_return_sequences=1,
    )
    
    # only take the new generated tokens as our response
    prompt_len = inputs["input_ids"].shape[1]
    response_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).lstrip()
    return response

# generate
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_bf16 = config['precision'] == 'bf16'

    # load policy and base model
    policy_model_path = config['dpo_training']['save_dir']
    policy = load_the_dpo_model(policy_model_path, use_bf16)
    policy.eval()
    policy.config.use_cache = True

    base_model_name = config['base_model_name']
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model.eval()
    base_model.config.use_cache = True

    test_cfg = config['HH_test']
    seed = test_cfg['seed']
    random_controler(seed)

    # load test data
    num_sample = test_cfg['num_sample']
    ds_name = test_cfg['hh_test']
    raw_hh = load_dataset(ds_name, split='test')
    hh_test = build_HH_dataset(raw_hh)
    hh = hh_test.shuffle(seed=seed).select(range(num_sample))

    # generation setup
    max_prompt_length = int(test_cfg['max_prompt_length'])
    max_new_tokens = int(test_cfg['max_new_tokens'])
    temperature = float(test_cfg['temperature'])
    top_p = float(test_cfg['top_p'])
    do_sample = test_cfg['do_sample']

    # save dir
    policy_out_dir = test_cfg['HH_test_model1_out']
    os.makedirs(os.path.dirname(policy_out_dir) or ".", exist_ok=True)

    base_out_dir = test_cfg['HH_test_model2_out']
    os.makedirs(os.path.dirname(base_out_dir) or ".", exist_ok=True)

    # generate and save
    with open(policy_out_dir, "w", encoding="utf-8") as f_pol, open(base_out_dir,   "w", encoding="utf-8") as f_base:
        for i, ex in enumerate(tqdm(hh, total=len(hh))):
            prompt = ex["prompt"]
            chosen = ex["chosen"]

            # policy response
            policy_response = generate_response(
                model=policy,
                tokenizer=tokenizer,
                prompt=prompt,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            rec = {
                "idx": i,
                "prompt": prompt,
                "response": policy_response,
                "HH_response": chosen,
                "model_id": "policy_model",
                "dataset": "Anthropic/hh-rlhf",
                "split": "test",
                "max_prompt_length": max_prompt_length,
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(do_sample),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
            f_pol.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # base model response
            base_response = generate_response(
                model=base_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            rec = {
                "idx": i,
                "prompt": prompt,
                "response": base_response,
                "HH_response": chosen,
                "model_id": "base_model",
                "dataset": "Anthropic/hh-rlhf",
                "split": "test",
                "max_prompt_length": max_prompt_length,
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(do_sample),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }
            f_base.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print("Generation completed.")

if __name__ == "__main__":
    main()