import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labelDict = {
    0: 'positive',
    1: 'negative',
    2: 'neutral'
}


def train(model, optimizer, train_loader, val_loader, config, save_dir):
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    test_label, test_output = [], []
    for e in range(config['epoch']):
        for i, data in enumerate(tqdm(train_loader)):
            model.train()
            labels = data['tag'].to(device)
            test_label.extend(labels.cpu().tolist())
            optimizer.zero_grad()
            out, _ = model(data)
            # print(labels.device)
            # print(out.device)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            out = out.argmax(dim=1)
            test_output.extend(out.detach().cpu().tolist())
            accuracy = (out == labels).sum().item() / len(labels)
            print('epoch:', e + 1, 'step:', i + 1, 'loss:', loss.item(), 'train accuracy:', accuracy)

            if (i + 1) % 10 == 0:
                print('validation:')
                accuracy = val(model, val_loader, config)
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model, save_dir)
                    print('saved the model as', save_dir)
        accuracy = val(model, val_loader, config)
        print('epoch:', e + 1, 'validation accuracy:', accuracy)
        accuracy = accuracy_score(test_label, test_output)
        precision = precision_score(test_label, test_output, average='macro')
        recall = recall_score(test_label, test_output, average='macro')
        f1 = f1_score(test_label, test_output, average='macro')

        print('Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}'.format(accuracy, precision, recall, f1))


def val(model, data_loader, config):
    test_label, test_output = [], []
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            labels = data['tag'].to(device)
            test_label.extend(labels.cpu().tolist())
            out, _ = model(data)
            out = out.argmax(dim=1)
            test_output.extend(out.detach().cpu().tolist())
            correct += (out == labels).sum().item()
            total += len(labels)
    accuracy = accuracy_score(test_label, test_output)
    precision = precision_score(test_label, test_output, average='macro')
    recall = recall_score(test_label, test_output, average='macro')
    f1 = f1_score(test_label, test_output, average='macro')
    print('Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}'.format(accuracy, precision, recall, f1))
    return accuracy

def predict(model, data_loader, config, predict_dir):
    model.eval()
    result = dict()
    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            out, _ = model(data)
            out = out.argmax(dim=1)
            ids = data['guid']
            for j in range(len(ids)):
                guid = str(ids[j].item())
                tag = labelDict[out[j].item()]
                result[guid] = tag
                # print(guid, tag)
    with open(predict_dir, 'w', encoding='utf-8') as wfs:
        wfs.write('guid,tag\n')
        with open(config['test_without_label'], 'r', encoding='utf-8') as rfs:
            rfs.readline()
            for line in rfs:
                guid = line[0: line.find(',')]
                wfs.write(guid + ',' + result[guid] + '\n')


if __name__ == '__main__':
    print(device)