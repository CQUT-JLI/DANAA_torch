# PyTorch Implementation of DANAA: Double Adversarial Neuron Attribution for Transferable Attacks

This repository provides a PyTorch implementation of **DANAA** converted from the original TensorFlow implementation of *DANAA: Towards Transferable Attacks with Double Adversarial Neuron Attribution*.

The implementation follows the original DANAA framework and keeps the main attack pipeline consistent with NAA. In particular, the data preprocessing, surrogate model interface, feature-layer extraction, NAA attribution loss, DI, PI, momentum update, perturbation clipping, and adversarial image saving pipeline are inherited from the PyTorch NAA implementation. The main algorithmic difference lies in the attribution-weight estimation stage, where the original static black-baseline integration path in NAA is replaced by a dynamically updated adversarial attribution path.

---

## Prerequisites

The implementation requires the following packages:

```text
Python >= 3.8
PyTorch
torchvision
timm
NumPy
Pandas
Pillow
tqdm
```

A typical installation command is:

```bash
pip install torch torchvision timm numpy pandas pillow tqdm
```

---

## Dataset

This implementation follows the NIPS 2017 adversarial transfer setting.

The expected dataset structure is:

```text
dataset/
├── images/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── labels.csv
```

The `labels.csv` file should contain the image id and label information, for example:

```text
ImageId,TrueLabel,TargetClass
```

During loading, images are resized according to the surrogate model input size. For Inception-style surrogate models, the input size is `299 × 299`.

The dataset loader performs the following preprocessing:

```python
image = Image.open(filepath).convert('RGB')
image = image.resize((image_size, image_size), Image.BILINEAR)
image = np.array(image).astype(np.float32) / 255.0
image = torch.from_numpy(image).permute(2, 0, 1)
```

Therefore, images are represented as tensors with shape:

```text
[B, 3, 299, 299]
```

and value range:

```text
[0, 1]
```

The adversarial perturbation is also maintained in this `[0,1]` image space.

Since PyTorch and timm ImageNet models use class indices from `0` to `999`, while the NIPS 2017 labels follow the `1` to `1000` convention, labels are shifted by `-1` during loading:

```python
label = TrueLabel - 1
```

---

## Supported Surrogate Models

This implementation mainly uses TensorFlow-style pretrained Inception models from `timm`:

| Repository Model Name | timm Model |
|---|---|
| `tf_inception_v3` | `inception_v3.tf_in1k` |
| `tf_inception_v4` | `inception_v4.tf_in1k` |
| `tf_inception_resnet_v2` | `inception_resnet_v2.tf_in1k` |

For these models, the input size is `299 × 299`, and the model-side preprocessing uses:

```python
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

Thus, although the image and perturbation are maintained in `[0,1]` space, the actual input to the Inception model is:

```text
x_model = (x - 0.5) / 0.5 = 2x - 1
```

This matches the input convention of the original TensorFlow Inception models.

---

## Feature Layer Mapping

The original TensorFlow implementation obtains intermediate feature tensors through graph operation names. In this PyTorch implementation, intermediate features are extracted using forward hooks.

The TensorFlow layer names are mapped to PyTorch/timm module names as follows:

```python
{
    "InceptionV3/InceptionV3/Mixed_5b/concat": "Mixed_5b",
    "InceptionV4/InceptionV4/Mixed_5e/concat": "features.9",
    "InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu": "conv2d_4a"
}
```

During the forward pass, hooks are registered on the selected modules:

```python
handles = [mod.register_forward_hook(hook_fn) for mod in self.opt_operations]
```

The captured feature maps are then used to compute the DANAA attribution loss.

---

## Difference Between NAA and DANAA

DANAA shares most of the attack framework with NAA. The major difference is the attribution path used to estimate neuron attribution weights.

### NAA

NAA estimates neuron attribution weights along a static path between the original image and a black baseline:

```text
original image  →  black baseline
```

The attribution weight is obtained by aggregating gradients of the original class probability with respect to the selected feature layer along this path.

### DANAA

DANAA replaces the static black-baseline path with a dynamically updated adversarial attribution path. Starting from the original image, DANAA repeatedly updates the path point using the input gradient of the negative original-class probability:

```text
x_{t+1} = x_t + step · sign(∇_x(-p_y(x_t)))
```

At each path point, DANAA computes the gradient of the original class probability with respect to the selected feature layer and aggregates these gradients as neuron attribution weights.

Therefore, compared with NAA, DANAA changes the attribution-weight estimation stage while keeping the remaining attack optimization framework consistent.

---

## Attack Objective

After the DANAA attribution weights are estimated, the attack uses the same attribution-guided loss form as NAA:

```python
attribution = (adv_feat - base_feat) * weights
```

where:

- `adv_feat` is the feature map of the adversarial image;
- `base_feat` is the feature map of the dynamically generated baseline;
- `weights` are the DANAA neuron attribution weights.

The perturbation is optimized by backpropagating this attribution loss with respect to `delta`:

```python
grad = torch.autograd.grad(loss, delta)[0]
```

The update follows a momentum-based iterative process, with optional DI and PI enhancements.

---

## Running DANAA Attacks

Use the following command format to generate adversarial examples:

```bash
python DANAA_torch.py \
    --model_name tf_inception_v3 \
    --attack_method DANAA_PI_DI \
    --layer_name InceptionV3/InceptionV3/Mixed_5b/concat \
    --input_dir ./dataset \
    --output_dir ./adv/DANAA_inc_v3 \
    --max_epsilon 16 \
    --num_iter 15 \
    --alpha 1.07 \
    --batch_size 16 \
    --ens 30
```

For Inception-v4:

```bash
python DANAA_torch.py \
    --model_name tf_inception_v4 \
    --attack_method DANAA_PI_DI \
    --layer_name InceptionV4/InceptionV4/Mixed_5e/concat \
    --input_dir ./dataset \
    --output_dir ./adv/DANAA_inc_v4 \
    --max_epsilon 16 \
    --num_iter 15 \
    --alpha 1.07 \
    --batch_size 16 \
    --ens 30
```

For Inception-ResNet-v2:

```bash
python DANAA_torch.py \
    --model_name tf_inception_resnet_v2 \
    --attack_method DANAA_PI_DI \
    --layer_name InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu \
    --input_dir ./dataset \
    --output_dir ./adv/DANAA_inc_res_v2 \
    --max_epsilon 16 \
    --num_iter 15 \
    --alpha 1.07 \
    --batch_size 16 \
    --ens 30
```

The generated adversarial examples will be saved to the directory specified by `--output_dir`.

---

## Important Parameters

| Parameter | Description |
|---|---|
| `--model_name` | Surrogate model used to generate adversarial examples |
| `--attack_method` | Attack variant, e.g. `DANAA`, `DANAA_DI`, `DANAA_PI_DI` |
| `--layer_name` | Intermediate feature layer used for attribution attack |
| `--input_dir` | Dataset root directory containing `images/` and `labels.csv` |
| `--output_dir` | Directory used to save adversarial examples |
| `--max_epsilon` | Maximum perturbation budget in pixel scale |
| `--num_iter` | Number of attack iterations |
| `--alpha` | Step size in pixel scale |
| `--batch_size` | Batch size |
| `--ens` | Number of aggregated attribution samples |
| `--prob` | Probability of applying input diversity |
| `--image_resize` | Resize dimension used in input diversity |
| `--amplification_factor` | PI amplification factor |
| `--gamma` | PI projection coefficient |
| `--Pkern_size` | Kernel size used in PI projection |
| `--momentum` | Momentum factor |

For TensorFlow-style Inception surrogate models, `max_epsilon` and `alpha` are defined in the `[0,1]` image space in PyTorch and are internally equivalent to the normalized-space perturbation budget used in the TensorFlow implementation.

---

## Evaluation

After generating adversarial examples, they can be evaluated on surrogate or black-box target models using an evaluation script.

A typical evaluation command is:

```bash
python verify.py \
    --ori_path ./dataset/images/ \
    --adv_path ./adv/DANAA_inc_v3/
```

The evaluation script should report clean accuracy, attack success rate, and transfer attack success rate on the selected models.

---

## Notes on PyTorch Reproduction

This implementation is adapted to PyTorch's dynamic computation graph. Therefore, it is not a line-by-line translation of the TensorFlow code.

The main implementation differences are:

1. TensorFlow uses placeholders and forwards a concatenated `2B` batch of clean and adversarial images, while PyTorch forwards baseline images, path samples, and adversarial images separately.
2. TensorFlow extracts feature tensors from graph operation names, while PyTorch uses forward hooks.
3. TensorFlow updates adversarial images directly in the normalized input space, while PyTorch maintains a perturbation variable `delta` in `[0,1]` image space.
4. For DI, TensorFlow pads with value `0` in normalized space. The PyTorch implementation pads with the corresponding mean value in `[0,1]` space, so the normalized padding value remains consistent.
5. PI is implemented with grouped convolution in PyTorch, corresponding to depthwise convolution in TensorFlow.

Despite these engineering differences, the core DANAA mechanism and the main attack hyperparameters are kept consistent with the TensorFlow implementation for TensorFlow-style Inception surrogate models.

---

## Citation

If you use this implementation or the DANAA method in your research, please cite the original paper:

```bibtex
@inproceedings{jin2023danaa,
  title={DANAA: Towards transferable attacks with double adversarial neuron attribution},
  author={Jin, Zhibo and Zhu, Zhiyu and Wang, Xinyi and Zhang, Jiayu and Shen, Jun and Chen, Huaming},
  booktitle={International Conference on Advanced Data Mining and Applications},
  pages={456--470},
  year={2023},
  organization={Springer}
}
```

---

## Reference

This PyTorch implementation is based on the TensorFlow implementation of DANAA and follows the NAA-style attribution attack framework.
