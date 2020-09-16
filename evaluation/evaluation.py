import os
import numpy as np
import argparse

DATASET_PATH = 'data/NIA-2020_PATHOLOGY'

def evaluation_metrics (label, pred):
    metrics = _get_metrics (label, pred)
    return np.round(np.mean(list(metrics.values())), 4)

def _confusion_matrix(label, pred):  
    ### TN (0,0) / FN (0,1)/ FP (1,0) / TP (1,1)
    TN, FN, FP, TP = 0, 0, 0, 0

    for y_hat,y in zip(pred,label):
        if y == 0:
            if y_hat ==0:
                    TN = TN + 1
            else:
                    FN = FN + 1
        elif y == 1:
            if y_hat == 0:
                FP = FP +1
            else:
                TP = TP +1
    return TN, FN, FP, TP

def _get_metrics (label, pred):
    metrics = dict()
    SMOOTH = 1e-3
    
    num_P, num_N = len(np.where(label==1)), len(np.where(label==0))
    TN, FN, FP, TP = _confusion_matrix(label, pred)


    metrics['acc'] = (TP + TN) / (TP + FN + FP + TN + SMOOTH)
    metrics['prec'] = TP / (TP+FP + SMOOTH) ## ppv
    metrics['recall'] = TP / (TP+FN + SMOOTH) ## sensitivive
    metrics['spec'] = TN / (TN + FP + SMOOTH) ## 
    metrics['npv'] = TN / (TN + FN + SMOOTH) ## 
    metrics['f1'] = 2*(metrics['prec']*metrics['recall'])/(metrics['prec']+ metrics['recall'] + SMOOTH)

    return metrics



def label_loader (root_path):
    labels = {}
    with open (os.path.join(root_path,'test_label'), 'rt') as f :
        for row in f:
            row = row.split()
            labels[int(row[0])] = (int(row[1]))
    return labels


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--prediction',type=str,default='pred.txt')
    args.add_argument('--test_label_path',type=str)
    config = args.parse_args() 

    label_dict = label_loader(os.path.join(DATASET_PATH,'test'))

    keys = []
    preds = []
    with open (config.prediction,'rt')as f:
        for txt in f: 
            row = txt.split()
            keys.append(int(row[0]))
            preds.append(int(row[1]))

    labels = [label_dict[x] for x in keys]
    
    print(evaluation_metrics(labels, preds))