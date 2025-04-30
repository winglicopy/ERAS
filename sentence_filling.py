import json
import numpy as np


with open('./candidates_words.json', 'r') as f:
    candidates = json.load(f)

with open('./templates.json', 'r') as f:
    templates = json.load(f)

amb_templates = templates['amb_templates']
nonamb_templates = templates['nonamb_templates']

def flatten_dict(dictionaty):
    item_list = []
    for key, item in dictionaty.items():
        if isinstance(item, list):
            
            if len(item_list) > 0:
                if len(item_list[0]) != len(item[0]):
                    item_list = [item_list, item]
                else:
                    item_list+=item
            else:
                item_list+=item
        elif isinstance(item, dict):
            item_list+=flatten_dict(item)
    return item_list
def find_list(target_key, dictionaty):
    target_list = []
    for key in dictionaty.keys():
        if key == target_key:
            if isinstance(dictionaty[target_key], dict):
                return flatten_dict(dictionaty[target_key])
            elif isinstance(dictionaty[target_key], list):
                return dictionaty[target_key]
        if isinstance(dictionaty[key], dict) and len(target_list)==0:
            target_list = find_list(target_key, dictionaty[key])
    return target_list
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn.functional import softmax

# Load the pre-trained model and tokenizer
model_name = "bert-base-chinese"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

# Function to calculate the probability of a candidate
def get_candidate_probability(list_candidate_tokens, input_text, mask_indices):
    
    # Tokenize the input sentence
    tokenized_text = tokenizer.tokenize(input_text)
    mask_token_index = mask_indices[0]

    # Replace the masked token with the candidate tokens
    
    all_candidate_ids = []
    id_len = 0
    test_prev = []
    for candidate_tokens in list_candidate_tokens:
        tokenized_candidate = ["[CLS]"] + tokenized_text[:mask_token_index] + candidate_tokens + tokenized_text[mask_token_index + 1:]
        input_ids = tokenizer.convert_tokens_to_ids(tokenized_candidate)
        all_candidate_ids.append(input_ids)
        test_prev = tokenized_candidate
    
    #tokenized_candidate = ["[CLS]"] + tokenized_text[:mask_token_index] + candidate_tokens + tokenized_text[mask_token_index + 1:]
    

    # Convert input IDs to tensors
    input_tensor = torch.tensor(all_candidate_ids).to(device)

    # Get the logits from the model
    with torch.no_grad():
        logits = model(input_tensor).logits
    
    
    # Calculate the probability of the candidate word
    probs = softmax(logits, dim=-1)
    
    ori_probs = []
    for prob, input_ids in zip(probs, all_candidate_ids):
        ori_prob = prob[range(len(input_ids)), input_ids].clone()
        ori_probs.append(ori_prob)
    ori_probs = torch.stack(ori_probs, dim=0)
    probs = probs[:,range(len(input_ids)), input_ids]
    #print(torch.prod(probs[:,1:mask_token_index+1]).size())
    #print(ori_probs)
    prob = (torch.prod(ori_probs[:,1:mask_token_index+1], dim = 1)* torch.prod(ori_probs[:,mask_token_index+len(candidate_tokens)+1:], dim = 1))
    for ind in mask_indices[1:]:
        prob/=ori_probs[:,ind+len(candidate_tokens)]
    #prob = torch.prod(ori_probs[:, mask_token_index+1:mask_token_index+len(candidate_tokens)+1], dim = 1)

    return prob.tolist()

def generate4pair_temps(template, amb_template):
    tokens = tokenizer.tokenize(template)
    amb_tokens = tokenizer.tokenize(amb_template)
    candidates_in_order = []
    mask_inds = []
    for i, token in enumerate(tokens):
        if not is_chinese(token) and token != "。":
            list_cands = find_list(token, candidates)
            #print(token, list_cands)
            tokens[i] = "[MASK]"
            amb_tokens[i] = "[MASK]"
            mask_inds.append(i)
            candidates_in_order.append(list_cands)
    mask_sentence = "".join(tokens)
    selected_sentece = [mask_sentence]
    amb_mask_sentence = "".join(amb_tokens)
    amb_selected_sentece = [amb_mask_sentence]
    ori_mask_inds = np.array(mask_inds)
    mask_inds = np.array(mask_inds)

    #print(mask_sentence, mask_inds)
    count = 0
    for cands, mask_ind in zip(candidates_in_order, mask_inds):
        #print()
        
        temp_sentences = []
        amb_sentences = []
        threshold = 0.74+ count*0.05
        for j, mask_sentence in enumerate(selected_sentece):
            #print(mask_sentence)
            tokens = tokenizer.tokenize(mask_sentence)
            amb_tokens = tokenizer.tokenize(amb_selected_sentece[j])
            if isinstance(cands[0], str):
                #print(mask_sentence)
                list_candidate_tokens = []
                for cand in cands:
                    candidate_token = tokenizer.tokenize(cand)
                    list_candidate_tokens.append(candidate_token)
                #print(mask_inds[count:])
                prob = get_candidate_probability(list_candidate_tokens, mask_sentence, mask_inds[count:])
                prob = np.array(prob)
                #print(prob)
                prob /= np.max(prob)
                #print(prob)
                select_inds = prob>threshold
                #print(select_inds, threshold)
                select_inds = select_inds.nonzero()

                for ind in select_inds[0]:
                    #print(ind, mask_ind)
                    tokens[mask_ind] = cands[ind]
                    temp_sentences.append("".join(tokens))
                    
                    amb_tokens[mask_ind] = cands[ind]
                    amb_sentences.append("".join(amb_tokens))
            elif isinstance(cands[0], list):
                #print(mask_sentence)
                for len_candidate in cands:
                    list_candidate_tokens = []
                    for cand in len_candidate:
                        candidate_token = tokenizer.tokenize(cand)
                        list_candidate_tokens.append(candidate_token)
                    #print(mask_inds[count:])
                    prob = get_candidate_probability(list_candidate_tokens, mask_sentence, mask_inds[count:])
                    prob = np.array(prob)
                    prob /= np.max(prob)
                    select_inds = prob>threshold
                    select_inds = select_inds.nonzero()

                    for ind in select_inds[0]:
                        tokens[mask_ind] = len_candidate[ind]
                        temp_sentences.append("".join(tokens))
                        
                        amb_tokens[mask_ind] = len_candidate[ind]
                        amb_sentences.append("".join(amb_tokens))
        count+=1
        mask_inds+=(len(cands[0])-1)
        #print(selected_sentece, select_inds)
        selected_sentece = temp_sentences
        amb_selected_sentece = amb_sentences
    return selected_sentece, amb_selected_sentece

all_samples = []
for template, amb_template in zip(nonamb_templates, amb_templates):
    selected_sentece, amb_selected_sentece =  generate4pair_temps(template, amb_template)
    generate_pairs = []
    for sentence, amb_sentence in zip(selected_sentece, amb_selected_sentece):
        generate_pairs.append([amb_sentence, sentence])
    all_samples.append(generate_pairs)

for i, sample in enumerate(all_samples):
    if len(sample) > 40000:
        all_samples[i] = sample[:40000]

import json
with open('./template_sentences_w_cand.json', 'w') as f:
    json.dump(all_samples, f)