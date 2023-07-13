import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import json
import torchvision.models as models
import torch.nn.modules as nn
import torchvision.models as cv_models
import math
from sklearn.model_selection import KFold
from baseline_model.data_pipelines import getImgDataset
from baseline_model.runUtils import train, val, predict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# config.setup_seed()

class ReluOrGelu(nn.Module):
    def __init__(self, activate_type: str):
        super(ReluOrGelu, self).__init__()
        self.activate_type = activate_type

    def forward(self, x):
        if self.activate_type == 'relu':
            return torch.relu(x)
        elif self.activate_type == 'gelu':
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ImgModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['img_model'] == 'ResNet':
            self.resnet = cv_models.resnet34(pretrained=True)
            # self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))
            # self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])
            # self.output_dim = self.resnet_encoder[7][2].conv3.out_channels
            for param in self.resnet.parameters():
                param.requires_grad_(True)
            self.classifier = nn.Sequential(
                nn.Linear(512, 3),
                ReluOrGelu("gelu")
            )
        elif config['img_model'] == 'EFNet':
            self.efnet = models.efficientnet_b0(pretrained=True)
            for param in self.efnet.parameters():
                param.requires_grad_(True)
                
        # self.dropout = nn.Dropout(0.1)
        # self.fc = nn.Linear(1280, 3)
            self.classifier = nn.Sequential(
                nn.Linear(1280, 3),
                ReluOrGelu("gelu")
            )

    def forward(self, data):
        imgs = data['img'].to(device)
        if self.config['img_model'] =='ResNet':
            # image_encoder = self.resnet_encoder(imgs)
            # # image_encoder = self.conv_output(image_encoder)
            # image_cls = self.resnet_avgpool(image_encoder)
            # image_cls = torch.flatten(image_cls, 1)
            # return image_encoder, image_cls
            resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])
            features = resnet_features(imgs)
            # lstm = nn.LSTM(input_size=512, hidden_size=768, num_layers=1, batch_first=True).to(device)
            # print(out.shape)
            # fc_layer = nn.Linear(2048, 768).to(device)
            # out = out.view(out.size(0), out.size(1), -1)
            # out = out.squeeze(-1)
            # print(out.shape)
            # out, _ = lstm(out)
            # print(out.shape)
            # out = out[:, -1, :]
            # print(out.shape) #32*1000
            # out1 = self.resnet_features(imgs)
            # print(out1.shape)
            # self.fc = nn.Linear(768, 3)
            features = features.squeeze(-1).squeeze(-1)
            # print(features.shape)
            out = self.classifier(features)
            features = nn.Linear(512, 768)
            # out = self.fc(self.dropout(out))
        elif self.config['img_model'] == 'EFNet':
            # Remove the last fully connected layer
            efnet_features = nn.Sequential(*list(self.efnet.children())[:-1])
            features = efnet_features(imgs)
            # print(features.shape)
            out = torch.flatten(features, start_dim=1).to(device)
            # print(out.shape)
            # self.fc = nn.Linear(out.shape[1], 3).to(device)
            # out = self.fc(self.dropout(out))
            print(out.shape)
            out = self.classifier(out)
            # features = nn.Linear(1280, 3)
        return out, features


def run(config):
    model = ImgModel(config).to(device)
    # model.to(device)
    if config['img_model'] == 'ResNet':
        resnet_params = list(map(id, model.resnet.parameters()))
        down_params = filter(lambda p: id(p) not in resnet_params, model.parameters())
        optimizer = AdamW([
            {'params': model.resnet.parameters(), 'lr': config['resnet_lr']},
            {'params': down_params, 'lr': config['img_lr']}
        ])
    elif config['img_model'] =='EFNet':
        efnet_params = list(map(id, model.efnet.parameters()))
        down_params = filter(lambda p: id(p) not in efnet_params, model.parameters())
        optimizer = AdamW([
            {'params': model.efnet.parameters(), 'lr': config['efnet_lr']},
            {'params': down_params, 'lr': config['img_lr']}
        ])

    dataset = getImgDataset(config, config['train_data_json'])
    # print(len(dataset))
    # kf = KFold(n_splits=5, shuffle=True)
    # for train_indices, val_indices in kf.split(dataset):
    #     train_dataset = Subset(dataset, train_indices)
    #     val_dataset = Subset(dataset, val_indices)
    train_dataset = Subset(dataset, range(0, 3500))
    val_dataset = Subset(dataset, range(3500, 4000))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    train(model, optimizer, train_loader, val_loader, config, config['saved_img_model_path'])


def testNow(config):
    model = torch.load(config['saved_img_model_path'], map_location=device)
    dataset = getImgDataset(config, config['train_data_json'])
    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    print('final validation accuracy:', val(model, val_loader, config))


def predictNow(config):
    model = torch.load(config['saved_img_model_path'], map_location=device)
    test_loader = DataLoader(getImgDataset(config, config['test_data_json']), batch_size=config['batch_size'], shuffle=False)
    predict(model, test_loader, config, config['img_only_prediction_path'])


if __name__ == '__main__':
    with open('../config.json','r', encoding='utf-8') as f:
        config = json.load(f)
    run(config)
    testNow(config)
    predictNow(config)