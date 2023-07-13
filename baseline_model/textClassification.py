import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import json
import config
from baseline_model.runUtils import train, val, predict
from transformers import XLMRobertaModel
from baseline_model.data_pipelines import getTextDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.xlmroberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        for param in self.xlmroberta.parameters():
            param.requires_grad_(True)

        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(768, 3)

    def forward(self, data):
        # print(len(data['text']))
        # encoded_texts=data['text'].toTensor()
        # print(encoded_texts.shape)
        input_ids = data['text']['input_ids'].to(device)
        attention_mask = data['text']['attention_mask'].to(device)
        # token_type_ids = data['text']['token_type_ids'].to(device)
        # print(input_ids.shape)
        # print(attention_mask.shape)
        out = self.xlmroberta(
            input_ids=input_ids.squeeze(1),
            attention_mask=attention_mask.squeeze(1)
        )
        # print(out)
        hidden_states = out.pooler_output
        # print(hidden_states.shape)
        out = self.fc(self.dp(hidden_states))
        return out, hidden_states


def run(config):
    model = TextModel()
    model.to(device)

    xlmroberta_params = list(map(id, model.xlmroberta.parameters()))
    down_params = filter(lambda p: id(p) not in xlmroberta_params, model.parameters())
    optimizer = AdamW([
        {'params': model.xlmroberta.parameters(), 'lr': config['xlmroberta_lr']},
        {'params': down_params, 'lr': config['text_lr']}
    ])

    dataset = getTextDataset(config, config['train_data_json'])
    # print(len(dataset))
    train_dataset = Subset(dataset, range(0, 3500))
    val_dataset = Subset(dataset, range(3500, 4000))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    train(model, optimizer, train_loader, val_loader, config, config['saved_text_model_path'])


def testNow(config):
    model = torch.load(config['saved_text_model_path'], map_location=device)
    dataset = getTextDataset(config, config['train_data_json'])
    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    print('final validation accuracy:', val(model, val_loader, config))


def predictNow(config):
    model = torch.load(config['saved_text_model_path'], map_location=device)
    test_loader = DataLoader(getTextDataset(config, config['test_data_json']), batch_size=config['batch_size'], shuffle=False)
    predict(model, test_loader, config, config['text_only_prediction_path'])


if __name__ == '__main__':
    with open('../config.json','r', encoding='utf-8') as f:
        config = json.load(f)
    run(config)
    testNow(config)
    predictNow(config)