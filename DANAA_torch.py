import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import timm
from torch.utils.data import DataLoader

from utils_torch import AdvDataset, save_images


def get_opt_layers(model, layer_name_str):
    base_model = model[1] if isinstance(model, nn.Sequential) else model
    mapping = {
        "mixed_5e": "features.9",
        "mixed_5b": "features.4",
        "features.6": "features.6",
        "conv2d_4a": "features.4",
        "InceptionV3/InceptionV3/Mixed_5b/concat": "Mixed_5b",
        "InceptionV4/InceptionV4/Mixed_5e/concat": "features.9",
        "InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu": "conv2d_4a"
    }
    layer_names = [name.strip() for name in layer_name_str.split(',')]
    modules = []
    for name in layer_names:
        target_path = mapping.get(name, name)
        mod = base_model
        try:
            for p in target_path.split('.'):
                mod = getattr(mod, p)
            modules.append(mod)
        except AttributeError:
            raise ValueError(f"Unable to locate feature layer: {name}")
    return modules


def load_model(model_name):
    if model_name == 'tf_inception_v3':
        model = timm.create_model('inception_v3.tf_in1k', pretrained=True)
        preprocess = PreprocessingModel(299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return nn.Sequential(preprocess, model)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        preprocess = PreprocessingModel(299, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return nn.Sequential(preprocess, model)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        preprocess = PreprocessingModel(224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return nn.Sequential(preprocess, model)
    elif model_name == 'tf_inception_v4':
        model = timm.create_model('inception_v4.tf_in1k', pretrained=True)
        preprocess = PreprocessingModel(299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return nn.Sequential(preprocess, model)
    elif model_name in ['tf_inception_resnet_v2', 'inception_resnet_v2.tf_in1k']:
        model = timm.create_model('inception_resnet_v2.tf_in1k', pretrained=True)
        preprocess = PreprocessingModel(299, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return nn.Sequential(preprocess, model)
    elif model_name == 'vgg_16' or model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        preprocess = PreprocessingModel(224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return nn.Sequential(preprocess, model)
    elif model_name == 'resnet_v1_152':
        model = models.resnet152(pretrained=True)
        preprocess = PreprocessingModel(224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return nn.Sequential(preprocess, model)
    else:
        raise ValueError(f"Model {model_name} not supported.")


def get_NAA_loss(adv_features, base_features, weights_list):
    total_loss = 0
    gamma_loss = 1.0
    for adv_feat, base_feat, weights in zip(adv_features, base_features, weights_list):
        attribution = (adv_feat - base_feat) * weights
        blank = torch.zeros_like(attribution)
        positive = torch.where(attribution >= 0, attribution, blank)
        negative = torch.where(attribution < 0, attribution, blank)
        balance_attribution = positive + gamma_loss * negative
        total_elements = adv_feat.numel() * 2
        total_loss += torch.sum(balance_attribution) / total_elements
    return total_loss / len(adv_features)


def normalize(grad, opt=2):
    if opt == 0:
        return grad
    if opt == 1:
        abs_sum = torch.sum(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        return grad / (abs_sum + 1e-12)
    elif opt == 2:
        square = torch.sum(torch.square(grad), dim=(1, 2, 3), keepdim=True)
        return grad / (torch.sqrt(square) + 1e-12)
    return grad


def project_kern(kern_size):
    kern = torch.ones((1, 1, kern_size, kern_size)) / (kern_size ** 2 - 1)
    kern[0, 0, kern_size // 2, kern_size // 2] = 0.0
    return kern, kern_size // 2


def project_noise(x, P_kern, kern_size_half):
    channels = x.shape[1]
    kernel = P_kern.expand(channels, 1, P_kern.shape[2], P_kern.shape[3]).to(x.device)
    x = F.conv2d(x, kernel, stride=1, padding=kern_size_half, groups=channels)
    return x


def input_diversity(x, use_DI, prob, image_resize=None, pad_value=0.5):
    if not use_DI or torch.rand(1).item() > prob:
        return x
    img_size = x.shape[-1]
    resize_dim = int(image_resize) if image_resize is not None else int(img_size * 1.1)
    resize_dim = max(resize_dim, img_size + 1)
    rnd = torch.randint(img_size, resize_dim, (1,)).item()
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='nearest')
    h_rem = resize_dim - rnd
    pad_top = torch.randint(0, h_rem, (1,)).item() if h_rem > 0 else 0
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, h_rem, (1,)).item() if h_rem > 0 else 0
    pad_right = h_rem - pad_left
    if torch.is_tensor(pad_value):
        pad_value = pad_value.to(device=x.device, dtype=x.dtype)
        padded = pad_value.expand(x.shape[0], -1, resize_dim, resize_dim).clone()
        padded[:, :, pad_top:pad_top + rnd, pad_left:pad_left + rnd] = rescaled
    else:
        padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=float(pad_value))
    ret = F.interpolate(padded, size=[img_size, img_size], mode='nearest')
    return ret


class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.input_size = resize
        self.register_buffer('input_mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.resize = transforms.Resize((resize, resize))
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, x):
        return self.normalize(self.resize(x))


class DANAA(nn.Module):
    def __init__(self, model_name, epsilon, alpha, epoch, targeted=False, device=None,
                 layer_name='Mixed_5b', ens=30, prob=0.7, use_DI=False, use_PIM=False,
                 amplification_factor=2.5, gamma=0.5, Pkern_size=3, scale=0.125,
                 baseline_step=0.00125, momentum=1.0, image_resize=331):
        super(DANAA, self).__init__()
        self.model_name = model_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = epoch
        self.targeted = targeted
        self.device = device if device is not None else torch.device('cuda')
        self.layer_name = layer_name
        self.ens = ens
        self.scale = scale
        self.baseline_step = baseline_step
        self.prob = prob
        self.use_DI = use_DI
        self.use_PIM = use_PIM
        self.amplification_factor = amplification_factor
        self.gamma = gamma
        self.Pkern_size = Pkern_size
        self.momentum = momentum
        self.image_resize = image_resize

        self.model = load_model(model_name).to(self.device)
        self.model.eval()
        self.opt_operations = get_opt_layers(self.model, self.layer_name)
        preprocess = self.model[0] if isinstance(self.model, nn.Sequential) else None
        self.di_pad_value = getattr(preprocess, 'input_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1))

        if self.use_PIM:
            self.P_kern, self.kern_size_half = project_kern(self.Pkern_size)

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if labels.ndim > 1:
            labels = labels[:, 0]
        labels = labels.long()

        features = []

        def hook_fn(module, input, output):
            features.append(output)

        handles = [mod.register_forward_hook(hook_fn) for mod in self.opt_operations]

        # DANAA difference 1: replace NAA's black baseline path with a dynamic baseline.
        dynamic_base = images.clone().detach()
        agg_grads = None

        if self.ens == 0:
            features.clear()
            x_step = dynamic_base.clone().detach().requires_grad_(True)
            x_step_di = input_diversity(x_step, self.use_DI, self.prob, self.image_resize, self.di_pad_value)
            logits = self.model(x_step_di)
            probs = F.softmax(logits, dim=1)
            target_response = probs.gather(1, labels.view(-1, 1)).squeeze(1)
            grad_fs = torch.autograd.grad(target_response.sum(), features, retain_graph=False)
            agg_grads = [grad_f.detach() for grad_f in grad_fs]
        else:
            for l in range(int(self.ens)):
                features.clear()
                x_step = dynamic_base + torch.randn_like(images) * self.scale  # 加高斯噪声
                x_step = x_step.detach().requires_grad_(True)
                x_step_di = input_diversity(x_step, self.use_DI, self.prob, self.image_resize, self.di_pad_value)
                logits = self.model(x_step_di)
                probs = F.softmax(logits, dim=1)
                target_response = probs.gather(1, labels.view(-1, 1)).squeeze(1)

                # DANAA difference 2: use the same feature-gradient weights as NAA,
                # while also taking the input gradient of -p_y to update the baseline.
                grad_fs = torch.autograd.grad(target_response.sum(), features, retain_graph=True)  # 计算softmax输出对特征层梯度
                # 计算输出对输入的梯度，注意这里用的是每一次中间结果x_step，并且是计算-softmax对x_step求梯度，对应原代码x_grad，实际上是-py（x）
                input_grad = torch.autograd.grad((-target_response).sum(), x_step, retain_graph=False)[0]

                if agg_grads is None:
                    agg_grads = [torch.zeros_like(grad_f) for grad_f in grad_fs]
                for k, grad_f in enumerate(grad_fs):
                    agg_grads[k] += grad_f / self.ens  # 除以积分步长

                dynamic_base = dynamic_base + self.baseline_step * torch.sign(input_grad.detach())  # 生成基线，对应原论文△xt
                dynamic_base = dynamic_base.detach()

        weights_list = [-normalize(ag.detach(), opt=2) for ag in agg_grads]  # 归一化

        features.clear()
        with torch.no_grad():
            dynamic_base_di = input_diversity(dynamic_base, self.use_DI, self.prob, self.image_resize, self.di_pad_value)
            self.model(dynamic_base_di)
            base_features = [f.clone().detach() for f in features]

        delta = torch.zeros_like(images, requires_grad=True)
        momentum_state = 0
        amplification_update = torch.zeros_like(images).to(self.device)

        for i in range(self.num_iter):
            features.clear()
            if self.use_DI:
                with torch.no_grad():
                    dynamic_base_di = input_diversity(dynamic_base, self.use_DI, self.prob, self.image_resize, self.di_pad_value)
                    self.model(dynamic_base_di)
                    iter_base_features = [f.clone().detach() for f in features]
                features.clear()
            else:
                iter_base_features = base_features

            adv_images = images + delta
            adv_images_di = input_diversity(adv_images, self.use_DI, self.prob, self.image_resize, self.di_pad_value)
            self.model(adv_images_di)
            current_adv_features = features[:]

            loss = get_NAA_loss(current_adv_features, iter_base_features, weights_list)
            if self.targeted:
                loss = -loss

            grad = torch.autograd.grad(loss, delta)[0]
            grad_norm = grad / (torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-12)
            momentum_state = self.momentum * momentum_state + grad_norm

            if self.use_PIM:
                alpha_beta = self.alpha * self.amplification_factor
                gamma_alpha = self.gamma * alpha_beta

                amplification_update += alpha_beta * torch.sign(momentum_state)
                cut_noise = torch.clamp(torch.abs(amplification_update) - self.epsilon, 0.0, None) * torch.sign(
                    amplification_update)

                projection = gamma_alpha * torch.sign(project_noise(cut_noise, self.P_kern, self.kern_size_half))
                amplification_update += projection

                delta = delta + alpha_beta * torch.sign(momentum_state) + projection
            else:
                delta = delta + self.alpha * torch.sign(momentum_state)

            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            delta = delta.detach().requires_grad_(True)

        for handle in handles:
            handle.remove()
        return delta


def main():
    parser = argparse.ArgumentParser(description="DANAA Attack in PyTorch")
    parser.add_argument('--model_name', type=str, default='tf_inception_resnet_v2')
    parser.add_argument('--attack_method', type=str, default='DANAA_PI_DI')
    parser.add_argument('--layer_name', type=str, default='InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu')
    parser.add_argument('--input_dir', type=str, default=r'D:\pycharm\Project_1\attribution\NAA\NAA-master\image')
    parser.add_argument('--output_dir', type=str, default=r'./adv/DANAA_inc_Res_v2_PD')
    parser.add_argument('--max_epsilon', type=float, default=16.0)
    parser.add_argument('--num_iter', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=1.07)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--momentum', type=float, default=1.0)
    parser.add_argument('--GPU_ID', type=str, default='0')
    parser.add_argument('--scale', type=float, default=0.25)
    parser.add_argument('--baseline_step', type=float, default=0.00125)
    parser.add_argument('--prob', type=float, default=0.7)
    parser.add_argument('--amplification_factor', type=float, default=2.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--Pkern_size', type=int, default=3)
    parser.add_argument('--ens', type=int, default=30)
    parser.add_argument('--image_resize', type=int, default=331)
    parser.add_argument('--load_num', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2013)  # 控制随机性，但实际并没有用到
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eps = args.max_epsilon / 255.0
    alpha = args.alpha / 255.0

    print(f"=> Initialize dataset (Target Model: {args.model_name})...")
    dataset = AdvDataset(
        model_name=args.model_name,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        targeted=False,
        eval=False,
        load_num=args.load_num
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    attacker = DANAA(
        model_name=args.model_name,
        epsilon=eps,
        alpha=alpha,
        epoch=args.num_iter,
        device=device,
        layer_name=args.layer_name,
        ens=args.ens,
        prob=args.prob,
        use_DI=('DI' in args.attack_method),
        use_PIM=('PI' in args.attack_method),
        amplification_factor=args.amplification_factor,
        gamma=args.gamma,
        Pkern_size=args.Pkern_size,
        momentum=args.momentum,
        scale=args.scale,
        baseline_step=args.baseline_step,
        image_resize=args.image_resize
    )

    print("=> Starting DANAA attack...")
    for i, (images, labels, filenames) in enumerate(dataloader):
        print(f"   [Batch {i + 1}] Attacking...")
        delta = attacker(images, labels)
        adv_images = torch.clamp(images.to(device) + delta, 0.0, 1.0)
        save_images(args.output_dir, adv_images, filenames)

    print("=> Attack completed! All adversarial examples have been saved to:", args.output_dir)


if __name__ == '__main__':
    main()
