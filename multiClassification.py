import sys
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import json
from multiModalDataset import getMultiModalDataset
from baseline_model.runUtils import train, val, predict
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig
from baseline_model.imgClassification import ImgModel
from baseline_model.textClassification import TextModel

# sys.path.append(config.root_path)

# config.setup_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        self.attention_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_heads)])
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs):
        # inputs: (batch_size, num_modalities, input_size)
        batch_size = inputs.size(0)
        
        # Query, Key, Value projections
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)
        
        # Attention scores for each head
        attention_scores = []
        for head in self.attention_heads:
            scores = head(query)
            attention_scores.append(scores)
        
        # Concatenate attention scores
        attention_scores = torch.cat(attention_scores, dim=2)
        attention_scores = self.softmax(attention_scores)
        
        # Weighted sum of values for each head
        weighted_sums = []
        for i in range(self.num_heads):
            weights = attention_scores[:, :, i].unsqueeze(2)
            weighted_sum = torch.matmul(weights.transpose(1, 2), value)
            weighted_sums.append(weighted_sum)
        
        # Concatenate weighted sums
        weighted_sums = torch.cat(weighted_sums, dim=2)
        weighted_sums = weighted_sums.view(batch_size, -1)
        
        return weighted_sums
    
class MultiModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_model = ImgModel(config).to(device)
        for param in self.img_model.efnet.parameters():
            param.requires_grad_(True)

        self.text_model = TextModel().to(device)
        for param in self.text_model.xlmroberta.parameters():
            param.requires_grad_(True)
        self.attn = RobertaModel(RobertaConfig.from_pretrained('roberta-base'))

        self.attention = MultiHeadAttention(input_size=2048, num_heads=8, hidden_size=256).to(device)
        self.classifier = nn.Linear(2048, 3).to(device)
        self.linear = nn.Linear(1280,768)
        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(768 * 2, 3)

    def forward(self, data):
        text_out, text_features = self.text_model(data)
        img_out, img_features = self.img_model(data)
        
        # print(img_features)
        # print(img_out.toTensor().shape)
        
        if self.config['fuse_model'] == 'attention':
            img_features = img_features.squeeze(-1).squeeze(-1)
            text_features = F.pad(text_features, (0,1280))
            img_features = F.pad(img_features, (0,768))
            text_features = torch.unsqueeze(text_features,1)
            img_features = torch.unsqueeze(img_features,1)
            print(img_features.shape)
            print(text_features.shape)
            text_img_features = torch.cat([text_features, img_features], dim=1).to(device)
            print(text_img_features.shape)
            fused_features = self.attention(text_img_features)
            print(fused_features.shape)
            output = self.classifier(fused_features)
            print(output.shape)
        elif self.config['fuse_model'] == 'concat':
            img_features = img_features.squeeze(-1).squeeze(-1)
            img_features = self.linear(img_features)
            text_img_features = torch.cat([text_features, img_features], dim=1)
            output = self.fc(self.dp(text_img_features))
        return output, None

def run(config):
    print(device)
    model = MultiModel(config)
    model.to(device)

    xlmroberta_params = list(map(id, model.text_model.xlmroberta.parameters()))
    resnet_params = list(map(id, model.img_model.efnet.parameters()))
    down_params = filter(lambda p: id(p) not in xlmroberta_params + resnet_params, model.parameters())
    optimizer = AdamW([
        {'params': model.text_model.xlmroberta.parameters(), 'lr': config['xlmroberta_lr']},
        {'params': model.img_model.efnet.parameters(), 'lr': config['efnet_lr']},
        {'params': down_params, 'lr': config['multi_lr']}
    ])
    dataset = getMultiModalDataset(config['train_data_json'])
    # print(len(dataset))
    train_dataset = Subset(dataset, range(0, 3500))
    val_dataset = Subset(dataset, range(3500, 4000))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    train(model, optimizer, train_loader, val_loader, config, config['saved_model_path'])


def testNow(config):
    model = torch.load(config['saved_model_path'], map_location=device)
    dataset = getMultiModalDataset(config['train_data_json'])
    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    print('final validation accuracy:', val(model, val_loader, config))


def predictNow(config):
    model = torch.load(config['saved_model_path'], map_location=device)
    test_loader = DataLoader(getMultiModalDataset(config['test_data_json']), batch_size=config['batch_size'], shuffle=False)
    predict(model, test_loader, config, config['prediction_path'])


if __name__ == '__main__':
    with open('config.json','r', encoding='utf-8') as f:
        config = json.load(f)
    run(config)
    testNow(config)
    predictNow(config)