import timm
import torch
import torch.nn as nn

import config

def build_model(model_cfg):
    name       = model_cfg["name"]
    timm_id    = model_cfg["timm_id"]
    pretrained = model_cfg["pretrained"]

    print(f"\n Building model: {name}")
    model = timm.create_model(
        timm_id,
        pretrained=pretrained,
        num_classes=config.NUM_CLASSES,
    )
    model = model.to(config.DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model