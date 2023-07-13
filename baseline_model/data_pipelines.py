import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
import os
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tags = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
    "": 3
}

def preprocess(text):
    # print(text)
    #remove mention
    text = re.sub("@[A-Za-z0-9_]+","", text)
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style retext text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    text = re.sub(r'^rt[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'^https[\s]+', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = re.sub(r'%', '', text)
    text = re.sub(r'\.\.\.', ' ', text)
    # #remove angka
    text = re.sub('[0-9]+', '', text)
    text = re.sub(r':', '', text)
    #remove space
    text = text.strip()
    #remove double space
    text = re.sub('\s+',' ',text)
    # print(text)
    return text

class TextDataset(Dataset):
    def __init__(self, data: list, config):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.config = config

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        guid = item['guid']
        text = item['text']
        text = preprocess(text)
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tag = item['tag']
        tag = torch.tensor(tags[tag], dtype=torch.long)
        
        return {
            'guid': guid,
            'text': encoded_text,
            'tag': tag
        }


class ImgDataset(Dataset):
    def __init__(self, data: list, config):
        self.data = data
        self.config = config
        if config['img_model'] == 'ResNet':
            self.img_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif config['img_model'] == 'EFNet':
            self.img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transform = None
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        guid = item['guid']
        img = item['img']
        tag = item['tag']
        tag = torch.tensor(tags[tag], dtype=torch.long)
        img_path = os.path.join(self.config['img_texts'], img)
        img = self.img_transform(Image.open(img_path))
        return {
            'guid': guid,
            'img': img,
            'tag': tag
        }


def getTextDataset(config, data_dir):
    with open(data_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return TextDataset(data, config)


def getImgDataset(config, data_dir):
    with open(data_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return ImgDataset(data, config)