import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, average_precision_score

def cal_metrics_test(preds, tars):
    try:
        preds = np.concatenate(preds, axis=0)
        tars = np.concatenate(tars, axis=0)

        auc_macro = roc_auc_score(tars, preds, average='macro')  # macro_auc
        auc_micro = roc_auc_score(tars, preds, average='micro')  # micro_auc
        mAP_macro = average_precision_score(tars, preds, average='macro')
        mAP_micro = average_precision_score(tars, preds, average='micro')

        label_score_dict = {}
        label_score_dict['mAP_macro'] = mAP_macro  # Core indicators
        label_score_dict['mAP_micro'] = mAP_micro
        label_score_dict['auc_macro'] = auc_macro
        label_score_dict['auc_micro'] = auc_micro

        return label_score_dict

    except Exception as e:
        print(e, flush=True)


     