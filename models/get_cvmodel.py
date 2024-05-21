import torch
import torch.nn as nn
import torchvision
import numpy as np
import timm
from models.cv_models.convnextv1.convnext import convnext_tiny, convnext_small, convnext_base
from models.cv_models.swinv1.swin_transformer import swin
from models.cv_models.convnextv2.convnextv2 import convnextv2_tiny, convnextv2_base
from models.cv_models.resnet.resnet_sl2 import resnet50_ml_sl2,resnet101_ml_sl2
from models.cv_models.convnextv2.convnextv2_ml_sl2 import convnextv2_ml_sl2_tiny,convnextv2_ml_sl2_base
from models.cv_models.convnextv1.convnext_ml_sl2 import convnext_tiny_ml_sl2,convnext_small_ml_sl2,convnext_base_ml_sl2
from models.cv_models.swinv1.swinv1_ml_sl2 import swin_ml_sl2

pretrained_models_dir = {
        'resnet50': '/vepfs/chenchucheng/pretrained_models/image_cls/resnet50-19c8e357.pth',
        'resnet101': '/vepfs/chenchucheng/pretrained_models/image_cls/resnet101-5d3b4d8f.pth',
        'convnexttiny': '/vepfs/chenchucheng/pretrained_models/image_cls/fb_convnext/convnext_tiny_22k_1k_224.pth',
        'convnextbase': '/vepfs/chenchucheng/pretrained_models/image_cls/fb_convnext/convnext_base_22k_1k_224.pth',
        'convnextv2tiny': '/vepfs/chenchucheng/pretrained_models/image_cls/fb_convnext/convnextv2_tiny_22k_224_ema.pt',
        'convnextv2base': '/vepfs/chenchucheng/pretrained_models/image_cls/fb_convnext/convnextv2_base_22k_224_ema.pt',
        'swintiny': '/vepfs/chenchucheng/pretrained_models/image_cls/ms_swin/swin_tiny_patch4_window7_224_22kto1k_finetune.pth',
        'swinbase': '/vepfs/chenchucheng/pretrained_models/image_cls/ms_swin/swin_base_patch4_window7_224_22kto1k.pth',

        'resnet50_ml_sl2': '/vepfs/chenchucheng/pretrained_models/image_cls/resnet50-19c8e357.pth',
        'resnet101_ml_sl2': '/vepfs/chenchucheng/pretrained_models/image_cls/resnet101-5d3b4d8f.pth',
        'convnexttiny_ml_sl2': '/vepfs/chenchucheng/pretrained_models/image_cls/fb_convnext/convnext_tiny_22k_1k_224.pth',
        'convnextbase_ml_sl2': '/vepfs/chenchucheng/pretrained_models/image_cls/fb_convnext/convnext_base_22k_1k_224.pth',
        'convnextv2tiny_ml_sl2': '/vepfs/chenchucheng/pretrained_models/image_cls/fb_convnext/convnextv2_tiny_22k_224_ema.pt',
        'convnextv2base_ml_sl2': '/vepfs/chenchucheng/pretrained_models/image_cls/fb_convnext/convnextv2_base_22k_224_ema.pt',
        'swintiny_ml_sl2': '/vepfs/chenchucheng/pretrained_models/image_cls/ms_swin/swin_tiny_patch4_window7_224_22kto1k_finetune.pth',
        'swinbase_ml_sl2': '/vepfs/chenchucheng/pretrained_models/image_cls/ms_swin/swin_base_patch4_window7_224_22kto1k.pth'
    }

def build_cvmodel(backbone='swinbase', num_classes=2, is_pretrained=False):
    try:
        pretrained_model_path = pretrained_models_dir[backbone]
        print(pretrained_model_path)
    except Exception as e:
        print(e)
        pass

    if backbone == 'resnet50':
        model = torchvision.models.resnet50()
        if is_pretrained:
            model.load_state_dict(torch.load(pretrained_model_path, map_location=lambda storage, loc: storage))
        model.fc = nn.Sequential(nn.Linear(2048, num_classes))
    elif backbone == 'resnet101':
        model = torchvision.models.resnet101()
        if is_pretrained:        
            model.load_state_dict(torch.load(pretrained_model_path, map_location=lambda storage, loc: storage))
        model.fc = nn.Sequential(nn.Linear(2048, num_classes))
    elif backbone == 'convnexttiny':
        model = convnext_tiny()
        if is_pretrained:
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["model"])
        model.head = nn.Linear(768, num_classes)
    elif backbone == 'convnextbase':
        model = convnext_base()
        if is_pretrained:
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            for k in ['head.weight', 'head.bias']:  # 22k分类的预训练模型跟目前定义模型时候的1k分类不一样
                print('del key:', k, flush=True)
                del checkpoint['model'][k]
            model.load_state_dict(checkpoint["model"], strict=False)
        model.head = nn.Linear(1024, num_classes)
    elif backbone == 'convnextv2tiny':
        model = convnextv2_tiny()
        if is_pretrained:
            state_dict = model.state_dict()
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            model.load_state_dict(checkpoint["model"], strict=False)
        model.head = nn.Linear(768, num_classes)
    elif backbone == 'convnextv2base': 
        model = convnextv2_base()
        if is_pretrained:
            state_dict = model.state_dict()
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint[k]
            model.load_state_dict(checkpoint["model"], strict=False)
        model.head = nn.Linear(1024, num_classes)
    elif backbone == 'swintiny':
        model = swin(type='swintiny')
        if is_pretrained:
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=False)
        model.head = nn.Linear(768, num_classes)
    elif backbone == 'swinbase':
        model = swin(type='swinbase')
        if is_pretrained:
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            for k in ['head.weight', 'head.bias']:  # 22k分类的预训练模型跟目前定义模型时候的1k分类不一样
                print('del key:', k, flush=True)
                del checkpoint['model'][k]
            model.load_state_dict(checkpoint['model'], strict=False)
        model.head = nn.Linear(1024, num_classes)
    
    elif backbone == 'resnet50_ml_sl2':
        model = resnet50_ml_sl2()
        if is_pretrained:
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            del checkpoint['fc.weight'], checkpoint['fc.bias']
            model.load_state_dict(checkpoint, strict=False)
        model.fc = nn.Linear(2048, num_classes)  
        model.fc_2 = nn.Linear(2048, 1) 
    elif backbone == 'resnet101_ml_sl2':
        model = resnet101_ml_sl2()
        if is_pretrained:
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            del checkpoint['fc.weight'], checkpoint['fc.bias']
            model.load_state_dict(checkpoint, strict=False)
        model.fc = nn.Linear(2048, num_classes)  
        model.fc_2 = nn.Linear(2048, 1) 
    
    elif backbone == 'convnexttiny_ml_sl2':
        model = convnext_tiny_ml_sl2()
        if is_pretrained:
            state_dict = model.state_dict()
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            for k in ['head.weight', 'head.bias']:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint['model'][k]
            model.load_state_dict(checkpoint["model"], strict=False)
        model.head = nn.Linear(768, num_classes)  
        model.head_2 = nn.Linear(768, 1) 
    elif backbone == 'convnextbase_ml_sl2':
        model = convnext_base_ml_sl2()
        if is_pretrained:
            state_dict = model.state_dict()
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            for k in ['head.weight', 'head.bias']:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint['model'][k]
            model.load_state_dict(checkpoint["model"], strict=False)
        model.head = nn.Linear(1024, num_classes)  
        model.head_2 = nn.Linear(1024, 1) 

    elif backbone == 'convnextv2tiny_ml_sl2'  :
        model = convnextv2_ml_sl2_tiny()
        if is_pretrained:
            state_dict = model.state_dict()
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            for k in ['head.weight', 'head.bias']:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint['model'][k]
            model.load_state_dict(checkpoint["model"], strict=False)
        model.head = nn.Linear(768, num_classes)  
        model.head_2 = nn.Linear(768, 1) 
    elif backbone == 'convnextv2base_ml_sl2':
        model = convnextv2_ml_sl2_base()
        if is_pretrained:
            state_dict = model.state_dict()
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            for k in ['head.weight', 'head.bias']:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint['model'][k]
            model.load_state_dict(checkpoint["model"], strict=False)
        model.head = nn.Linear(1024, num_classes)  
        model.head_2 = nn.Linear(1024, 1) 

    elif backbone == 'swintiny_ml_sl2':
        model = swin_ml_sl2(type='swintiny')
        if is_pretrained:
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=False)
        model.head = nn.Linear(768, num_classes)
        model.head_2 = nn.Linear(768, 1)
    elif backbone == 'swinbase_ml_sl2':
        model = swin_ml_sl2(type='swinbase')
        if is_pretrained:
            checkpoint = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
            for k in ['head.weight', 'head.bias']:  # 22k分类的预训练模型跟目前定义模型时候的1k分类不一样
                print('del key:', k, flush=True)
                del checkpoint['model'][k]
            model.load_state_dict(checkpoint['model'], strict=False)
        model.head = nn.Linear(1024, num_classes)
        model.head_2 = nn.Linear(1024, 1)
   
    else:
        model = timm.create_model(backbone[5:], pretrained=True, num_classes=num_classes)
        print(backbone)

    return model
