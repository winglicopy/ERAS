# Basic import statements
import json
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="Erlangshen", type=str, required=False)

args = parser.parse_args()

model_name = args.model_name

# Load dataset
with open("./template_sentences_w_cand.json", "r") as f:
    dataset = json.load(f)

target_sentiments = [ 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1,  ]

file_ind= 7

def process(paradigm: List[List[float]], care_score:int) -> pd.DataFrame:
    # Dim 1: test sentence, control sentence
    # Dim 2: yhat_1, yhat_0
    #print(np.array(paradigm).shape)
    paradigm = np.array(paradigm).reshape(-1, 2, 2)
    return pd.DataFrame({"Test Score": paradigm[:, 0, care_score],
                         "Control Score": paradigm[:, 1, care_score]})
    
result_file = f"./sent_results/{model_name}_result.json"

results = []
with open(result_file, "r") as f:
    scores = json.load(f)
    for i in range(0, len(scores), 2):
        if i < 18:
            results.append(process(scores[i], 1))
            results.append(process(scores[i+1], 0))
        elif i < 40:
            results.append(process(scores[i], 0))
        elif i < 60:
            results.append(process(scores[i], 1))

accuracies = 0
all_accuracy = []
#exclude = [1, 13,25,32]
exclude = []
for i, result in enumerate(results):
    if i in exclude:
        continue
    accuracy = len(result[result["Control Score"]<=result["Test Score"]])/len(result)
    accuracies+=accuracy
    all_accuracy.append(accuracy)

occu_fil = f"./sent_results/{model_name}_occu.json"

filter_occu_scores = []
with open(occu_fil, "r") as f:
    occu_scores = json.load(f)
    for i in range(0, len(occu_scores), 2):
        filter_occu_scores.append(occu_scores[i])
        if i<18:
            filter_occu_scores.append(occu_scores[i+1])

Neg_accu = 0
Pos_accu = 0

Neg_count = 0
Pos_count = 0

True_Neg_count = 0
True_Pos_count = 0

for result, occus in zip(results, filter_occu_scores):
    diff = result["Control Score"] - result["Test Score"]    
    
    for dif, occu in zip(diff, occus):
        if isinstance(occu, list):
            occu = occu[0]
        if occu>=0:
            Neg_count += 1
            if dif >= 0:
                True_Neg_count+=1
        else:
            Pos_count += 1
            if dif<0:
                True_Pos_count+=1

recall = True_Neg_count/Neg_count

Neg_accu = 0
Pos_accu = 0

Neg_count = 0
Pos_count = 0

True_Neg_count = 0
True_Pos_count = 0

for result, occus in zip(results, filter_occu_scores):
    diff = result["Control Score"] - result["Test Score"]
    
    for dif, occu in zip(diff, occus):
        if isinstance(occu, list):
            occu = occu[0]
        if dif>=0:
            Neg_count += 1
            if occu >= 0:
                True_Neg_count+=1
        else:
            Pos_count += 1
            if occu < 0:
                True_Pos_count+=1
prec = True_Neg_count/Neg_count

GPER = (1-accuracies/39) * prec

print(model_name,"has overall accuracy of:", accuracies/39, "and necessity of:", prec, "and sufficiency of:", recall, "and GPER of:", GPER)