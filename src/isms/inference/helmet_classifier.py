"""Classify an already cropped head as helmet or no helmet."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, mobilenet_v3_small

from src.inference.vendor.pp_lcnet import PPLCNet_x0_5


class HelmetClassifier:
    def __init__(self, weights, device=None):
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(weights, map_location=self.device, weights_only=True)
        self.labels = checkpoint['classes']
        architecture = checkpoint.get('architecture')
        if architecture == 'pplcnet_x0_5':
            self.model = PPLCNet_x0_5(class_num=len(self.labels)).to(self.device)
        elif architecture == 'efficientnet_b0':
            self.model = efficientnet_b0(weights=None, num_classes=len(self.labels)).to(self.device)
        elif architecture == 'mobilenet_v3_small':
            self.model = mobilenet_v3_small(weights=None, num_classes=len(self.labels)).to(self.device)
        else:
            raise ValueError(f'Unsupported classifier architecture: {architecture}')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        norm = checkpoint['normalization']
        self.transform = transforms.Compose([
            transforms.Resize((checkpoint['input_size'], checkpoint['input_size'])),
            transforms.ToTensor(), transforms.Normalize(norm['mean'], norm['std']),
        ])

    @torch.inference_mode()
    def classify(self, bgr_crop):
        if not isinstance(bgr_crop, np.ndarray) or bgr_crop.ndim != 3 or bgr_crop.shape[2] != 3 or bgr_crop.size == 0:
            raise ValueError('Expected a nonempty BGR image')
        image = Image.fromarray(np.ascontiguousarray(bgr_crop[:, :, ::-1]))
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probabilities = self.model(tensor).softmax(dim=1)[0].cpu().tolist()
        return dict(zip(self.labels, probabilities))
