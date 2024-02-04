import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import random
from itertools import product
import numpy as np
from itertools import cycle

class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)

def generate_ordered_list(all_result):
    ordered_list = []
    labels = []
    for e, result in enumerate(all_result):
        q = result['question']
        for i in range(4):
            if result.get(f'claude_output_{i}'):
                exp = result.get(f'claude_output_{i}')['reasoning']
                ans = result.get(f'claude_output_{i}')['answer']
                if result.get(f'claude_output_{i}')['answer'] == result['gold_answer']:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                exp = "None"
                labels.append(2)
            exp = remove_agreement(exp)
            full_sent = f"[INST] ### Question: {q}[/INST] ### Answer: {exp} So the answer is {ans}"
            ordered_list.append(full_sent)

        for i in range(4):
            if result.get(f'gpt4_output_{i}'):
                exp = result.get(f'gpt4_output_{i}')['reasoning']
                ans = result.get(f'gpt4_output_{i}')['answer']
                if result.get(f'gpt4_output_{i}')['answer'] == result['gold_answer']:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                exp = "None"
                labels.append(2)
            exp = remove_agreement(exp)
            full_sent = f"[INST] ### Question: {q}[/INST] ### Answer: {exp} So the answer is {ans}"
            ordered_list.append(full_sent)

        for i in range(4):
            if result.get(f'bard_output_{i}'):
                exp = result.get(f'bard_output_{i}')['reasoning']
                ans = result.get(f'bard_output_{i}')['answer']
                if result.get(f'bard_output_{i}')['answer'] == result['gold_answer']:
                    labels.append(1)
                else:
                    labels.append(0)
            else:
                exp = "None"
                labels.append(2)
            exp = remove_agreement(exp)
            full_sent = f"[INST] ### Question: {q}[/INST] ### Answer: {exp} So the answer is {ans}"
            ordered_list.append(full_sent)
    return ordered_list, labels

def construct_graph():

    # construct a single graph w/ conversational structure
    # node 0, 1, 2, 3    => agent 0's round 0, 1, 2, 3
    # node 4, 5, 6, 7    => agent 1's round 0, 1, 2, 3
    # node 8, 9, 10, 11  => agent 2's round 0, 1, 2, 3
    
    edge_index = torch.tensor([[0, 1],[1, 2],[2, 3],
                               [4, 5],[5, 6],[6, 7],
                               [8, 9],[9, 10],[10, 11],
                               [0, 5], [5, 8],
                               [1, 4], [1, 8],
                               [0, 9], [4, 9],
                               [2, 5], [2, 9],
                               [1, 6], [6, 9],
                               [1, 10], [5, 10],
                               [3, 6], [3, 10],
                               [2, 7], [7, 10],
                               [2, 11], [6, 11]
                               ], dtype=torch.long)

    data = Data(edge_index=edge_index.t().contiguous())
    return data

def construct_graphs(all_result, embeddings, num_train_samples, max_node_num):
    # construct list of graphs with node embeddings and node labels
    _, labels = generate_ordered_list(all_result)
    graphs = [construct_graph() for _ in range(num_train_samples)] 
    labels = torch.tensor(labels, dtype=torch.long)
    labels = labels.reshape(num_train_samples, max_node_num)
    for g, emb, y in zip(graphs, embeddings, labels):
        g.x = emb
        g.y = y
    return graphs

def pad_graphs(training_batch, graphs):
    pool = cycle(graphs)
    graphs = [next(pool) for _ in range(len(training_batch))]

    for tb, g in zip(training_batch, graphs):
        tb['graph'] = g

    return training_batch, graphs


def prepare_contrastive_samples(samples, labels):
    """
    Prepare contrastive learning samples, maximizing the usage by replicating the minority set.

    Args:
    samples (list): The list of samples.
    labels (list): The list of labels corresponding to the samples.

    Returns:
    (list, list): Tuple of two lists - positive samples and negative samples.
    """
    if len(samples) != len(labels):
        raise ValueError("Samples and labels must be of the same length.")

    positive_samples = [sample for sample, label in zip(samples, labels) if label == 1]
    negative_samples = [sample for sample, label in zip(samples, labels) if label == 0]

    if len(negative_samples) == 0:
        negative_samples = ["NA"] # padding for negatives
        
    if len(positive_samples) == 0:
        return None
        
    if len(positive_samples) > len(negative_samples):
        negative_samples = (negative_samples * ((len(positive_samples) // len(negative_samples)) + 1))[:len(positive_samples)]
    elif len(negative_samples) > len(positive_samples):
        positive_samples = (positive_samples * ((len(negative_samples) // len(positive_samples)) + 1))[:len(negative_samples)]

    return positive_samples, negative_samples

def prepare_batch(tokenizer, all_result, num_train_samples, max_node_num):

    ordered_list, labels = generate_ordered_list(all_result)
    ordered_list = np.array(ordered_list)
    ordered_list = ordered_list.reshape(num_train_samples, max_node_num)
    labels = np.array(labels)
    labels = labels.reshape(num_train_samples, max_node_num)

    result = []
    for sentence, label in zip(ordered_list, labels):
        pairs = prepare_contrastive_samples(sentence, label)
        if pairs:
            positive_samples, negative_samples = pairs
            pos_enc = tokenizer(positive_samples)
            neg_enc = tokenizer(negative_samples)
            for pi, pa, ni, na in zip(pos_enc.input_ids, pos_enc.attention_mask, neg_enc.input_ids, neg_enc.attention_mask):
                result.append({
                    'pos_input_ids': pi,
                    'pos_attention_mask': pa,
                    'pos_labels': pi,
                    'neg_input_ids': ni,
                    'neg_attention_mask': na,
                    'neg_labels': ni
                })
    return result

def remove_agreement(text, phrase=["I disagree", "I agree"]):
    sentences = text.split('. ')
    sentences = [sentence for sentence in sentences if not (sentence.startswith(phrase[0]) or sentence.startswith(phrase[1]))]
    result_text = '. '.join(sentences)
    return result_text