import json
import os
import argparse
from baseline_model import imgClassification, textClassification
import multiClassification
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='img_and_text')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--predict', action='store_true')

    
    parser.add_argument('--prediction_path', type=str)
    parser.add_argument('--img_only_prediction_path', type=str)
    parser.add_argument('--text_only_prediction_path', type=str)
    parser.add_argument('--saved_text_model_path', type=str)
    parser.add_argument('--saved_img_model_path', type=str)
    parser.add_argument('--saved_model_path', type=str)
    
    parser.add_argument('--img_model', type=str)
    parser.add_argument('--text_model', type=str)
    parser.add_argument('--fuse_model', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--multi_lr', type=float)
    parser.add_argument('--img_lr', type=float)
    parser.add_argument('--text_lr', type=float)
    parser.add_argument('--efnet_lr', type=float)
    parser.add_argument('--xlmroberta_lr', type=float)
    parser.add_argument('--resnet_lr', type=float)
    parser.add_argument('--epoch', type=int)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open('config.json','r', encoding='utf-8') as f:
        config = json.load(f)

    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.multi_lr:
        config['multi_lr'] = args.multi_lr
    if args.img_lr:
        config['img_lr'] = args.img_lr
    if args.text_lr:
        config['text_lr'] = args.text_lr
    if args.efnet_lr:
        config['efnet_lr'] = args.efnet_lr
    if args.xlmroberta_lr:
        config['xlmroberta_lr'] = args.xlmroberta_lr
    if args.resnet_lr:
        config['resnet_lr'] = args.resnet_lr
    if args.prediction_path:
        config['prediction_path'] = args.prediction_path
    if args.img_only_prediction_path:
        config['img_only_prediction_path'] = args.img_only_prediction_path
    if args.text_only_prediction_path:
        config['text_only_prediction_path'] = args.text_only_prediction_path
    if args.saved_img_model_path:
        config['saved_img_model_path'] = args.saved_img_model_path
    if args.saved_text_model_path:
        config['saved_text_model_path'] = args.saved_text_model_path
    if args.saved_model_path:
        config['saved_model_path'] = args.saved_model_path
    if args.img_model:
        config['img_model'] = args.img_model
    if args.text_model:
        config['text_model'] = args.text_model
    if args.fuse_model:
        config['fuse_model'] = args.fuse_model
    if args.epoch:
        config['epoch'] = args.epoch
    
    if args.mode == 'img_and_text':
        print('q')
        if args.train:
            print('1')
            multiClassification.run(config)
        if args.test:
            multiClassification.testNow(config)
        if args.predict:
            multiClassification.predictNow(config)
    elif args.mode == 'img_only':
        if args.train:
            imgClassification.run(config)
        if args.test:
            imgClassification.testNow(config)
        if args.predict:
            imgClassification.predictNow(config)
    elif args.mode == 'text_only':
        if args.train:
            textClassification.run(config)
        if args.test:
            textClassification.testNow(config)
        if args.predict:
            textClassification.predictNow(config)