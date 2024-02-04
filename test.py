import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import json
import torch
import argparse
import datetime
import data_utils
import torch.nn as nn
import bitsandbytes as bnb
from tqdm.notebook import tqdm
from peft import LoraConfig, PeftModel
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForCausalLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset: ['SQA', 'ECQA', 'ARC', 'GSM8K', 'MATH']
    parser.add_argument('--dataset', default='MATH', type=str)
    parser.add_argument('--batch_size', default='10', type=int)
    parser.add_argument('--base_model', default='mistralai/Mistral-7B-Instruct-v0.2', type=str)
    parser.add_argument('--lora_model', default='checkpoints/MAGDi_MATH', type=str)
    parser.add_argument('--cache_dir', default='', type=str)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--max_new_tokens', default=600, type=int)
    args = parser.parse_args()

    print(f"=== evaluating {args.lora_model} on {args.dataset} dataset ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.lora_model, 
        cache_dir=args.cache_dir if args.cache_dir else './hf_models', 
        device_map='auto')

    # model = PeftModel.from_pretrained(model, args.lora_model)
    # model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side='left', add_eos_token=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    test_samples, test_batch = data_utils.prepare_test_data(args.dataset)

    result = []
    for idx in range(0, len(test_batch), args.batch_size):
        
        test_data = test_batch[idx: idx+args.batch_size]
        test_data = tokenizer(test_data, return_tensors='pt', padding=True).to('cuda')
        output_tokens = model.generate(**test_data,
                                    do_sample=True,
                                    top_p=0.9,
                                    top_k=50,
                                    temperature=args.temperature,
                                    pad_token_id=tokenizer.eos_token_id,
                                    max_new_tokens=args.max_new_tokens,
                                    eos_token_id=tokenizer.eos_token_id,
                                    num_return_sequences=1)

        generated_txts = tokenizer.batch_decode(output_tokens)
        result.extend(generated_txts)
        pred_ans = []
        for o in result:
            pred_ans.append(data_utils.parse_answer(args.dataset, o))
        acc = data_utils.calc_acc(args.dataset, pred_ans, test_samples)
        print(f"{datetime.datetime.now().strftime('%y-%m-%d %H:%M')} | samples evaluated: {len(result)} | accuracy: {acc}")

    # print(pred_ans[:5])
    # print("====")
    # print(result[:5])