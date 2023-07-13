from torch.utils.data import Dataset, DataLoader
import torch 
from torchvision import transforms
import pandas as pd
import json
from PIL import Image
import os
from transformers import AutoTokenizer
from baseline_model.data_pipelines import preprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('config.json', 'r', encoding='utf8') as f:
    config = json.load(f)
tags = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
    "": 3
}
class MultiModalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        # self.vocab = vocab
        # self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        guid = self.data[index]['guid']
        text = self.data[index]['text']
        # print(text)
        # if self.tokenizer:
        text = preprocess(text)
        text = self.tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors='pt')
        img_path = os.path.join(config['img_texts']+self.data[index]['img'])
        img = Image.open(img_path).convert('RGB')
        img = self.img_transform(img)
        tag = self.data[index]['tag']
        return {
            'guid': guid,
            'text': text,
            'img': img,
            'tag': tags[tag]
        }
# _dataset = MultiModalDataset(config['train_data_csv'])
# print(_dataset[1])

def getMultiModalDataset(data_dir):
    with open(data_dir, 'r', encoding='utf-8') as fs:
        data = json.load(fs)
    return MultiModalDataset(data)

# print(getMultiModalDataset(config['train_data_csv']))