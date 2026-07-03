import os

import numpy as np
import pandas as pd
import torch
from PIL import Image


image_size_dict = {
    'inception_v1': 299,
    'inception_v2': 299,
    'inception_v3': 299,
    'tf_inception_v3': 299,
    'inception_v4': 299,
    'tf_inception_v4': 299,
    'inception_resnet_v2': 299,
    'tf_inception_resnet_v2': 299,
    'inception_resnet_v2.tf_in1k': 299,
    'resnet50': 224,
    'resnet_v1_50': 224,
    'resnet_v1_101': 224,
    'resnet_v1_152': 224,
    'resnet_v1_200': 224,
    'resnet_v2_50': 299,
    'resnet_v2_101': 299,
    'resnet_v2_152': 299,
    'resnet_v2_200': 299,
    'vgg16': 224,
    'vgg_16': 224,
    'vgg_19': 224,
}


mean_dict = {
    'inception_v1': [0.5, 0.5, 0.5],
    'inception_v2': [0.5, 0.5, 0.5],
    'inception_v3': [0.485, 0.456, 0.406],
    'tf_inception_v3': [0.5, 0.5, 0.5],
    'inception_v4': [0.5, 0.5, 0.5],
    'tf_inception_v4': [0.5, 0.5, 0.5],
    'inception_resnet_v2': [0.5, 0.5, 0.5],
    'tf_inception_resnet_v2': [0.5, 0.5, 0.5],
    'inception_resnet_v2.tf_in1k': [0.5, 0.5, 0.5],
    'resnet50': [0.485, 0.456, 0.406],
    'resnet_v1_50': [0.485, 0.456, 0.406],
    'resnet_v1_101': [0.485, 0.456, 0.406],
    'resnet_v1_152': [0.485, 0.456, 0.406],
    'resnet_v1_200': [0.485, 0.456, 0.406],
    'resnet_v2_50': [0.5, 0.5, 0.5],
    'resnet_v2_101': [0.5, 0.5, 0.5],
    'resnet_v2_152': [0.5, 0.5, 0.5],
    'resnet_v2_200': [0.5, 0.5, 0.5],
    'vgg16': [0.485, 0.456, 0.406],
    'vgg_16': [0.485, 0.456, 0.406],
    'vgg_19': [0.485, 0.456, 0.406],
}


std_dict = {
    'inception_v1': [0.5, 0.5, 0.5],
    'inception_v2': [0.5, 0.5, 0.5],
    'inception_v3': [0.229, 0.224, 0.225],
    'tf_inception_v3': [0.5, 0.5, 0.5],
    'inception_v4': [0.5, 0.5, 0.5],
    'tf_inception_v4': [0.5, 0.5, 0.5],
    'inception_resnet_v2': [0.5, 0.5, 0.5],
    'tf_inception_resnet_v2': [0.5, 0.5, 0.5],
    'inception_resnet_v2.tf_in1k': [0.5, 0.5, 0.5],
    'resnet50': [0.229, 0.224, 0.225],
    'resnet_v1_50': [0.229, 0.224, 0.225],
    'resnet_v1_101': [0.229, 0.224, 0.225],
    'resnet_v1_152': [0.229, 0.224, 0.225],
    'resnet_v1_200': [0.229, 0.224, 0.225],
    'resnet_v2_50': [0.5, 0.5, 0.5],
    'resnet_v2_101': [0.5, 0.5, 0.5],
    'resnet_v2_152': [0.5, 0.5, 0.5],
    'resnet_v2_200': [0.5, 0.5, 0.5],
    'vgg16': [0.229, 0.224, 0.225],
    'vgg_16': [0.229, 0.224, 0.225],
    'vgg_19': [0.229, 0.224, 0.225],
}


class AdvDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        model_name,
        input_dir=None,
        output_dir=None,
        targeted=False,
        target_class=None,
        eval=False,
        load_num=None,
    ):
        self.targeted = targeted
        self.target_class = target_class
        self.data_root = input_dir
        self.image_size = image_size_dict.get(model_name, 299)
        self.f2l = self.load_labels(os.path.join(self.data_root, 'labels.csv'))

        if load_num is not None:
            self.f2l = dict(list(self.f2l.items())[:load_num])

        if eval:
            self.data_dir = output_dir
            print(f'=> Eval mode: evaluating on {self.data_dir} (Resize to {self.image_size})')
        else:
            self.data_dir = os.path.join(self.data_root, 'images')
            print(f'=> Train mode: training on {self.data_dir} (Resize to {self.image_size})')
            if output_dir:
                print(f'Save images to {output_dir}')

    def __len__(self):
        return len(self.f2l)

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]
        filepath = os.path.join(self.data_dir, filename)

        image = Image.open(filepath).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        label = self.f2l[filename]
        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        f2l = {}
        for i in range(len(dev)):
            img_filename = str(dev.iloc[i]['ImageId']) + '.png'
            true_label = int(dev.iloc[i]['TrueLabel']) - 1

            if self.targeted:
                if self.target_class:
                    target_label = int(self.target_class) - 1
                else:
                    target_label = int(dev.iloc[i]['TargetClass']) - 1
                f2l[img_filename] = [true_label, target_label]
            else:
                f2l[img_filename] = true_label

        return f2l


def save_images(output_dir, adversaries, filenames):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    adversaries = torch.clamp(adversaries, 0.0, 1.0)
    adversaries = (adversaries.detach().permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))
