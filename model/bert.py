from transformers import BertModel
from torch import nn

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768,5)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids = input_ids, attention_mask = attention_mask, return_dict = False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        
        return final_layer
        