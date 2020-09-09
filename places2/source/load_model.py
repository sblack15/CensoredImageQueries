# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision

import torchvision.models as models
import torch

def load_resnet18_model(weights):
	model = models.__dict__['resnet18'](num_classes=365)
	checkpoint = torch.load(weights, map_location=lambda storage, loc: storage)
	state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
	model.load_state_dict(state_dict)
	model.eval()
	return model

