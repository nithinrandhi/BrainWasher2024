# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:37:06 2024

@author: TEJA
"""


import torch 

from BrainWasher_algorithm import BrainWasher
from unlearner_data_loader import get_dataset
#from utils_inceptionresnetv2 import InceptionResNetV2
from models import FaceNetModel






BrainWasher_Inception=BrainWasher()
#model = InceptionResNetV2(10572)
model=FaceNetModel()

trained_model_path='/kaggle/working/log/fc_finetune.pth'
trained_model=torch.load(trained_model_path)
model.load_state_dict(trained_model)

retain_loader,forget_loader,validation_loader= get_dataset(64)
model_forget=BrainWasher_Inception.unlearning(model, retain_loader, forget_loader, validation_loader)
forget_state=model_forget.state_dict()
torch.save(forget_state,'/kaggle/working/models/pins_unlearned_model.pth')


