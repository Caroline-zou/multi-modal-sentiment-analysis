import json
import os
import config
import pandas as pd
import csv
import re

def getEncoding(path):
    try:
        with open(path, 'r', encoding='utf-8') as fs:
            fs.readline()
            return 'utf-8'
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='ANSI') as fs:
                fs.readline()
                return 'ANSI'
        except UnicodeDecodeError:
            exit(-1)


def run():
    train_data_json = list()
    test_data_json = list()
    labels = dict()
    train_data_csv = list()
    test_data_csv = list()
    with open(config['train_data'], 'r', encoding='utf-8') as fs:
        for line in fs:
            line = line.strip()
            if line[0] != 'g':
                idx = line.find(',')
                labels[int(line[0: idx])] = line[idx + 1:]

    with open(config['test_data'], 'r', encoding='utf-8') as fs:
        for line in fs:
            line = line.strip()
            if line[0] != 'g':
                idx = line.find(',')
                labels[int(line[0: idx])] = ''

    for root, _, files in os.walk(config['img_texts']):
        for f in files:
            if f[-1] == 't':
                # print(f)
                path = os.path.join(root, f)
                encoding = getEncoding(path)
                with open(path, 'r', encoding=encoding) as fs:
                    text = fs.read()
                    guid = int(f[0: f.find('.')])
                    # print(guid, encoding)
                    # print(text)

                    tag = labels.get(guid)
                    data = {
                        'guid': guid,
                        'text': text.strip(),
                        'tag': tag,
                        'img': str(guid) + '.jpg'
                    }
                    data1 = []
                    data1.append(int(guid))
                    data1.append(text.strip())
                    data1.append(tag)
                    data1.append(str(guid) + '.jpg')
                    if tag is not None:
                        if tag != '':
                            train_data_json.append(data)
                            train_data_csv.append(data1)
                        else:
                            test_data_json.append(data)
                            test_data_csv.append(data1)
                    # print(text)

    print(len(train_data_json))
    print(len(test_data_json))

    with open(config['train_data_json'], 'w', encoding='utf-8') as f:
        json.dump(train_data_json, f, ensure_ascii=False)
    with open(config['test_data_json'], 'w', encoding='utf-8') as f:
        json.dump(test_data_json, f, ensure_ascii=False)

    with open(config['train_data_csv'], 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['GUID', 'TEXT', 'TAG', 'IMG'])
        for row in train_data_csv:
            writer.writerow(row)
    with open(config['test_data_csv'], 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['GUID', 'TEXT', 'TAG', 'IMG'])
            for row in test_data_csv:
                writer.writerow(row)

def remove(text):
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
    #remove coma
    # text = re.sub(r',','',text)
    # #remove angka
    text = re.sub('[0-9]+', '', text)
    text = re.sub(r':', '', text)
    #remove space
    text = text.strip()
    #remove double space
    text = re.sub('\s+',' ',text)
    # print(text)
    return text

def clean_data():
    
    
    with open(config['train_data_csv'], 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    header = rows[0]
    print(header)
    sorted_rows = sorted(rows[1:], key=lambda row: int(row[0]))

    with open(config['train_data_csv'], 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(sorted_rows)
    train_df = pd.read_csv(config['train_data_csv'])
    train_df['CLEANED_TEXT'] = train_df['TEXT'].apply(lambda x: remove(x.lower()))
    train_df.to_csv(config['train_data_csv'], index=False)
    with open(config['test_data_csv'], 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    header = rows[0]
    print(rows[1][0])
    sorted_rows = sorted(rows[1:], key=lambda row: int(row[0]))

    with open(config['test_data_csv'], 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(sorted_rows)
    test_df = pd.read_csv(config['test_data_csv'])
    test_df['CLEANED_TEXT'] = test_df['TEXT'].apply(lambda x: remove(x.lower()))
    test_df.to_csv(config['test_data_csv'], index=False)


if __name__ == '__main__':
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    run()
    clean_data()
    df = pd.read_csv(config['train_data_csv'])
    print(df.head())
    df = pd.read_csv(config['test_data_csv'])
    print(df.head())