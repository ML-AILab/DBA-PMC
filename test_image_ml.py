import torch
import warnings
import argparse
import torchmetrics
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from utils.metrics import cal_metrics_test
from models.get_cvmodel import build_cvmodel
from dataloader.dataloader_cvmodel_ml_2 import CVModel_ML_2_Dataset, get_files_fromtxt, collate_fn
from dataloader.image_transforms import Image_Transforms
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str, default='')
parser.add_argument('--backbone', type=str, default='convnexttiny_ml_sl2')
parser.add_argument('--dataloader_id', type=int, default=1)
parser.add_argument('--validfile', type=str, default='test_mini.txt')
parser.add_argument('--num_classes', type=int, default=7)
parser.add_argument('--squaresize', type=int, default=256)
parser.add_argument('--cropsize', type=int, default=224)
args = parser.parse_args()


def evaluate(val_loader, model):
    preds, tars, preds_2, tars_2 = [], [], [], []

    # model.cuda()
    model.eval()
    with torch.no_grad():
        for i,(input, target, target_2) in enumerate(val_loader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).float()).cuda()
            target_2 = Variable(torch.from_numpy(np.array(target_2)).float()).cuda()

            output, output_2 = model(input)
            output = torch.sigmoid(output)
            output_2 = torch.sigmoid(output_2)

            preds.append(output.cpu().numpy())
            tars.append(target.cpu().numpy().astype(np.int64))
            preds_2.append(output_2.cpu().numpy()>=0.5)
            tars_2.append(target_2.cpu().numpy().astype(np.int64))

    preds_2 = np.concatenate(np.array(preds_2), axis=0)
    tars_2 = np.concatenate(np.array(tars_2), axis=0)

    return preds, tars, preds_2, tars_2


def main():
    val_data = './data/images_label/' + args.validfile

    print(args.modelpath)

    try:
        model = build_cvmodel(backbone=args.backbone, num_classes=args.num_classes, is_pretrained=False)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(args.modelpath, map_location=lambda storage, loc: storage))
    except:
        model = build_cvmodel(backbone=args.backbone, num_classes=args.num_classes, is_pretrained=False)
        model.load_state_dict(torch.load(args.modelpath, map_location=lambda storage, loc: storage))

    model.cuda()

    val_data_list = get_files_fromtxt(val_data, "val", args.num_classes)
    val_transforms = Image_Transforms(mode='val', dataloader_id=args.dataloader_id, square_size=args.squaresize, crop_size=args.cropsize).get_transforms()
    val_dataloader = DataLoader(CVModel_ML_2_Dataset(val_data_list, transforms=val_transforms, val=True), batch_size=128,
                                 shuffle=False, collate_fn=collate_fn, pin_memory=False, num_workers=16)

    print('**************************************************************')
    print('evaluate_val')
    preds, tars, preds_2, tars_2 = evaluate(val_dataloader, model)

    label_score_dict_val = cal_metrics_test(preds, tars)
    
    macro_auc = label_score_dict_val['auc_macro']
    micro_auc = label_score_dict_val['auc_micro']
    mAP_macro = label_score_dict_val['mAP_macro'] # Core indicators
    mAP_micro = label_score_dict_val['mAP_micro']

    print('mAP_macro:', round(mAP_macro, 4)*100) # Core indicators
    print('mAP_micro:', round(mAP_micro, 4)*100)
    print('auc_macro:', round(macro_auc, 4)*100)
    print('auc_micro:', round(micro_auc, 4)*100)

    preds_np = np.concatenate(preds, axis=0)
    tars_np = np.concatenate(tars, axis=0)
    preds_2_np = preds_2
    tars_2_np = tars_2
    w_txt = open(args.modelpath.split('/')[-2]+'.txt', 'w',encoding='utf-8')
    for idx, imgname in enumerate(val_data_list['filename']):
        pred_prob = preds_np[idx]
        target = tars_np[idx]
        pred_2_prob = str(preds_2_np[idx][0])
        target_2 = str(tars_2_np[idx][0])

        target_str, pred_str = '', ''
        for t in target:
            target_str += str(t)+'_'
        target_str = target_str[:-1]
        for p in pred_prob:
            pred_str += str(round(p, 4)) + '_'
        pred_str = pred_str[:-1]

        final_inf = imgname + '\t' + target_str + '\t' + pred_str + '\t' + target_2 + '\t' + pred_2_prob + '\n'

        w_txt.write(final_inf)

    print('**************************************************************')



main()
