import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="MSR", type=str, required=False)

args = parser.parse_args()

model_name = args.model_name

with open('./templates.json', 'r') as f:
    templates = json.load(f)

amb_templates = templates['amb_templates']
nonamb_templates = templates['nonamb_templates']

with open('./key_word_locations.json', 'r') as f:
    key_word_locations = json.load( f)

with open('./template_sentences_w_cand.json', 'r') as f:
    template_sentences = json.load(f)


with open(f'./seg_results/{model_name}_all_seg_results.json', 'r') as f:
    all_seg_results = json.load( f)

ind = 0
accuracies = []
control_accuracies = []
overall_accuracy = 0
control_overall_accuracy = 0
partial_overall_accuracy = 0
all_count = 0
exclude = [1, 13,32,46]
exclude = []
all_words = []
for key_word_location, template, seg_scores in zip(key_word_locations, template_sentences, all_seg_results):
    
    if ind > 17 and ind % 2 == 1 :
        ind+=1
        continue
    
    if ind in exclude:
        ind+=1
        continue
    key_word = ""
    key_seg = []
    branch = key_word_location[-1]
    accuracy = 0
    partial_accuracy = 0
    control_accuracy = 0
    
    for i, pair in enumerate(template):
        all_count+=1
        test_seg_tag = seg_scores[2*i]
        control_seg_tag = seg_scores[2*i+1]
        sentence = pair[0]
        if len(key_word_location)==6 and len(pair[0]) == 9:
            sentence = sentence[1:]
            test_seg_tag = test_seg_tag[1:]
        if key_word != sentence[key_word_location[0]-1:key_word_location[1]-1]:
            key_word = sentence[key_word_location[0]-1:key_word_location[1]-1]
            all_words.append([key_word[branch:branch+2], key_word[(1-branch):(1-branch)+2]])
            
        if branch == 0:
            key_seg = [key_word[:2], key_word[2:]]
        else:
            key_seg = [key_word[:1], key_word[1:]]
        
        if test_seg_tag[key_word_location[0]+1] == 1+branch:
            accuracy+=1
        else:
            if test_seg_tag[key_word_location[0]+2] == 2-branch:
                accuracy+=1
        if control_seg_tag[key_word_location[0]+1] == 1+branch:
            control_accuracy+=1
        else:
            if control_seg_tag[key_word_location[0]+2] == 2-branch:
                control_accuracy+=1
        if test_seg_tag[key_word_location[0]] != 2 or test_seg_tag[key_word_location[0]+1] != 2 or test_seg_tag[key_word_location[0]+2] != 2:
            partial_accuracy+=1
        

    accuracy /= len(template)
    control_accuracy /= len(template)
    partial_accuracy /= len(template)
    #print(partial_accuracy)
    
    overall_accuracy += accuracy
    control_overall_accuracy += control_accuracy
    partial_overall_accuracy += partial_accuracy
    
    #print(key_word, key_seg, i, accuracy)
    accuracies.append(accuracy)
    control_accuracies.append(control_accuracy)
    
    ind+=1
overall_accuracy/=39
control_overall_accuracy/=39
partial_overall_accuracy/=39

print(f"(without theoratically neutral ones) test overall average accuracy for {model_name}:", overall_accuracy, ", control overall average accuracy:", control_overall_accuracy)