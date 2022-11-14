from transformers import BertTokenizer
import torch
from model import bert
from keras.utils import pad_sequences
import json

def predictions(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512
    label2id = {0:'tech', 1: 'business', 2: 'sport', 3: 'entertainment', 4: 'politics'}
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    model = bert.BertClassifier()
    model.load_state_dict(torch.load("../model_training/bert_best_model"))
    model.to(device)
    
    tokenized_text = tokenizer(text,padding = "max_length",max_length = max_length, truncation = True, return_tensors = "pt")
    
    input_ids = pad_sequences(tokenized_text['input_ids'],maxlen = max_length,dtype = torch.Tensor, truncating = "post",padding = "post")
    input_ids = input_ids.astype(dtype='int64')
    input_ids = torch.tensor(input_ids)


    attention_mask = pad_sequences(tokenized_text['attention_mask'],maxlen = max_length,dtype = torch.Tensor, truncating = "post",padding = "post")
    attention_mask = attention_mask.astype(dtype='int64')
    attention_mask = torch.tensor(attention_mask)
    
    input_ids = input_ids.squeeze(1).to(device)
    attention_mask = attention_mask.to(device)
    
    outputs = model(input_ids,attention_mask)
    
    #probs = F.softmax(outputs,dim = -1).cpu().detach().numpy().tolist()
    prediction = torch.argmax(outputs,dim=-1)
    return json.dumps({"text": text, "category": label2id[int(prediction)]})
