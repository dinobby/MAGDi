import json
import torch
import pickle
import numpy as np
from tqdm.notebook import tqdm
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForCausalLM

with open("MAG_ARC.json", "r") as f:
    MAGs = json.load(f)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          padding_side='left',
                                          add_eos_token=True)

tokenizer.pad_token_id = tokenizer.eos_token_id
ordered_list, labels = utils.generate_ordered_list(MAGs)

node_embeddings = None
batch_size = 50

for i in tqdm(range(0, len(ordered_list), batch_size)):
    batch = ordered_list[i: i+batch_size]
    tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    weights_for_non_padding = tokens.attention_mask * torch.arange(start=1, end=last_hidden_state.shape[1] + 1).to(tokens.attention_mask.device).unsqueeze(0)
    weights_for_non_padding = weights_for_non_padding.to(last_hidden_state.device)
    sum_node_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
    num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
    emb = (sum_node_embeddings / num_of_none_padding_tokens).detach().cpu().numpy()
        
    if node_embeddings is None:
        node_embeddings = emb
    else:
        node_embeddings = np.concatenate([node_embeddings, emb])
print(node_embeddings.shape)

with open("ARC_node_emb.pkl", "wb") as f:
    pickle.dump(node_embeddings, f)