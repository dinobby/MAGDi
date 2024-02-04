import json
import torch
import utils
import data_utils
import pickle
import numpy as np
np.random.seed(0)

import networkx as nx
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    AutoPeftModelForCausalLM
)

import argparse
from model import MAGDi, MAGDiTrainer
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset: ['SQA', 'ECQA', 'ARC', 'GSM8K', 'MATH']
    parser.add_argument('--dataset', default='SQA', type=str)
    parser.add_argument('--model_name', default='mistralai/Mistral-7B-Instruct-v0.2', type=str)
    parser.add_argument('--gcn_in_channels', default=4096, type=int)
    parser.add_argument('--gcn_hidden_channels', default=512, type=int)
    parser.add_argument('--gcn_out_channels', default=3, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--num_train_samples', default=1000, type=int)
    parser.add_argument('--max_node_num', default=12, type=int)    
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-6, type=float)
    args = parser.parse_args()

    with open(f"node_emb/{args.dataset}_node_emb.pkl", "rb") as f:
        node_embeddings = pickle.load(f)

    with open(f"MAG/{args.dataset}_1000.json", "r") as f:
        all_result = json.load(f)
    all_result = all_result[:args.num_train_samples]

    model = MAGDi(model_name=args.model_name,
                gcn_in_channels=args.gcn_in_channels,
                gcn_hidden_channels=args.gcn_hidden_channels,
                gcn_out_channels=args.gcn_out_channels,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma)

    node_embeddings = node_embeddings.reshape(args.num_train_samples, args.max_node_num, model.decoder.config.hidden_size)
    node_embeddings = torch.tensor(node_embeddings)
    node_embeddings = node_embeddings[:args.num_train_samples, :, :]
    node_embeddings.size()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                            padding_side='left',
                                            add_eos_token=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=["GCN", "MistralDecoderLayer"],
        dtype='float16',
        low_zero=False,
    )

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["GCN", "MistralDecoderLayer"],
        dtype='float16'
    )

    model = dispatch_model(model, device_map=device_map)

    for param in model.decoder.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model.decoder.gradient_checkpointing_enable()
    model.decoder.enable_input_require_grads()
    model.decoder = get_peft_model(model.decoder, config)
    model.decoder.lm_head = utils.CastOutputToFloat(model.decoder.lm_head)
    training_batch = utils.prepare_batch(tokenizer, all_result, args.num_train_samples, args.max_node_num)

    graphs = utils.construct_graphs(all_result, node_embeddings, args.num_train_samples, args.max_node_num)
    training_batch, graphs = utils.pad_graphs(training_batch, graphs)
    print(len(training_batch), len(graphs))

    trainer = MAGDiTrainer(
        model=model, 
        train_dataset=training_batch,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4, 
            gradient_accumulation_steps=4,
            warmup_steps=100, 
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=10, 
            output_dir='outputs',
            remove_unused_columns=False,
            save_strategy="no"
        ),
        data_collator=data_utils.MAGDiDataCollator(tokenizer)
    )

    trainer.train()
    model.decoder.save_pretrained("MAGDi_ARC")