import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from transformers import AutoTokenizer, AutoModel, utils
import torch
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Net(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")

        self.fc = nn.Linear(768, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.device = device

    def forward(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64
        '''
        x = x.to(device)
        y = y.to(device)
        
        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
            enc = self.dropout(enc)
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]
        
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Net(vocab_size=3)
model = nn.DataParallel(model)
model.load_state_dict(torch.load("./seg_models/MSR_trained_model.pth"))
model.to(device)
model.eval()

from transformers import BertTokenizer

def get_segmentation(sentences):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    input_ids = tokenizer(sentences, padding=True, return_tensors="pt").input_ids

    logits, _, label=model(input_ids, torch.Tensor(1))
    return label

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

with open('./template_sentences_w_cand.json', 'r') as f:
    template_sentences = json.load( f)
    
batch_num = 1000
all_seg_results = []
for sentences in template_sentences:
    segmentation_results = []
    all_sentence = []
    for pair_sentence in sentences:
        all_sentence += pair_sentence
    num_batches = (len(all_sentence)-1)// batch_num +1
    for i in range(num_batches):
        start = i * batch_num
        end = (i+1) * batch_num
        if end > len(all_sentence):
            end = len(all_sentence)
        segmentation_results += get_segmentation(all_sentence[start:end]).cpu().tolist()

    all_seg_results.append(segmentation_results)
    del segmentation_results

with open('./MSR_all_seg_results.json', 'w') as f:
    json.dump(all_seg_results, f)