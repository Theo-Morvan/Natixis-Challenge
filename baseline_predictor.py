import pandas as pd
import numpy as np
import os
import json
import argparse

def EMA(prices, alpha):
    ema = prices[0]
    for x in prices[1:]:
        ema = alpha*x + (1-alpha)* ema
    return ema

def create_baseline(data):

    pred_reg = []
    pred_classif = []

    for data_dict in data:
        stock_series = data_dict['stock']
        EMA_returns = EMA(stock_series, 2/(len(stock_series)+1))
        classif_target = 1 if stock_series[-1] > EMA_returns else 0
        pred_classif.append(classif_target)
        pred_reg.append(stock_series[-1])
    
    return pred_reg, pred_classif


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", type=str, default = 'data/',
                help="Path to training data.")
    parser.add_argument("-sp", "--split", type=str, default = 'dev',
                help="Split: dev or test.")
    parser.add_argument("-s", "--savepath", type=str, default = 'res/answer/',
                help="Path to save the results.")
    parser.add_argument("-i", "--index", type=str, default = 'all',
                help="Which index data to evaluate. Use 'all' for all indexes in the training data directory.")

    args = parser.parse_args()

    split = args.split
    if args.index =='all':
        indexes = [index.split('.')[0] for index in os.listdir(f'{args.datapath}/{split}')]
    else:
        indexes = [args.index]

    for index in indexes:
        save_path = os.path.join(args.savepath, index)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(f'{args.datapath}/{split}/{index}.json', 'r') as fp:
            data = json.load(fp)
        pred_reg, pred_classif = create_baseline(data)

        with open(os.path.join(save_path, 'pred_reg.txt'),'w') as f:
            f.write('\n'.join(list(map(str, pred_reg))))

        with open(os.path.join(save_path, 'pred_classif.txt'),'w') as f:
            f.write('\n'.join(list(map(str, pred_classif))))
