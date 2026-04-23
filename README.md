# DCLGAN

This fork adapts the original [DCLGAN repo](https://github.com/JunlinHan/DCLGAN) for sim-to-real style transfer on the `pushBlockWide` RL environment (84x84 images), with modernized tooling for running on the ComputeCanada cluster.

<img width="789" height="570" alt="Screenshot 2026-04-23 at 2 21 14 AM" src="https://github.com/user-attachments/assets/9cbcbe53-30c5-4274-8634-a591020c03f6" />


## Changes from the upstream repo

### Repo-level changes (apply to all runs)

- **PyTorch 2.x compatibility fix** — `models/dcl_model.py` and `models/simdcl_model.py`: changed `torch.nn.L1Loss('sum')` to `torch.nn.L1Loss(reduction='sum')`. The positional-string form crashes on modern PyTorch.
- **Weights & Biases logging replaces visdom** — `options/train_options.py` adds a `--use_wandb` flag; `util/visualizer.py` initializes a wandb run and logs scalar losses (via `print_current_losses`) and image panels (via `display_current_results`). Visdom is no longer required.
- **Robust loading of corrupt images** — `data/unaligned_dataset.py`: on `PIL.Image.open` failure, skip to the next valid image instead of crashing. The `train_sim` split contained ~454 empty files.

### New files

- `run_train.sh` — SLURM sbatch script for the grumpifycat smoke-test run (H100 MIG 1g.10gb, short wall time).
- `run_sim2real_dcl.sh` — sbatch script for baseline DCL training on sim→real (H100 MIG 2g.20gb, 10h).
- `run_sim2real_cut.sh` — same, with `--model cut`.
- `run_sim2real_dcl_tuned.sh` — DCL with tuned loss weights to prioritize structural fidelity: `--lambda_GAN 0.5 --lambda_NCE 10.0 --lambda_IDT 5.0`.
- `run_sim2real_cyclegan.sh` — CycleGAN baseline for comparison: `--model cycle_gan --gan_mode lsgan`.

All sim2real scripts use 84x84-appropriate settings: `--load_size 96 --crop_size 84 --netG resnet_6blocks --batch_size 4 --print_freq 1000 --display_freq 1000`. Datasets are symlinked under `datasets/sim2real/{trainA,trainB,testA,testB}`.

### Running on ComputeCanada

Environment setup (one-time):
```bash
module load python/3.11.5 cuda/12.6
python -m venv ~/envs/dclgan
source ~/envs/dclgan/bin/activate
pip install --no-index torch torchvision wandb dominate
```

Put your W&B key in `.env` at the repo root as `WANDB_API_KEY=...`. Submit jobs with `sbatch run_sim2real_<variant>.sh`.

---

The original upstream README follows below.

---

[arXiv](https://arxiv.org/abs/2104.07689)  |  [Video](https://youtu.be/w0oltXvLgmI)  |  [Slide](imgs/DCLGAN_slide.pptx)

# Dual Contrastive Learning Adversarial Generative Networks (DCLGAN)

We provide our PyTorch implementation of DCLGAN, which is a simple yet powerful model for unsupervised Image-to-image translation. Compared to CycleGAN, DCLGAN performs geometry changes with more realistic results. Compared to CUT, DCLGAN is usually more robust and achieves better performance. A viriant, SimDCL (Similarity DCLGAN) also avoids mode collapse using a new similarity loss.

DCLGAN is a general model performing ***all kinds of Image-to-Image translation tasks***. It achieves ***SOTA*** performances in most tasks that we have tested.

[Dual Contrastive Learning for Unsupervised Image-to-Image Translation](imgs/han_dualcontrastive.pdf)<br>
[Junlin Han](https://junlinhan.github.io/), Mehrdad Shoeiby, Lars Petersson, Mohammad Ali Armin<br>
DATA61-CSIRO and Australian National University<br>
In NTIRE, CVPRW 2021.
 
Our pipeline is quite straightforward. The main idea is a dual setting with two encoders to capture the variability in two distinctive domains. 
<img src='imgs/dclgan.png' align="right" width=950>

## Example Results

### Unpaired Image-to-Image Translation
Qualitative results:

<img src="imgs/results.png" width="800px"/>

Quantitative results:

<img src="imgs/table.png" width="800px"/>

More visual results:

<img src="imgs/results2.png" width="800px"/>

## Prerequisites
Python 3.6 or above.

For packages, see requirements.txt.

### Getting started

- Clone this repo:
```bash
git clone https://github.com/JunlinHan/DCLGAN.git
```

- Install PyTorch 1.6 or above and other dependencies (e.g., torchvision, visdom, dominate, gputil).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.


### DCLGAN and SimDCL Training and Test

- Download the `grumpifycat` dataset 
```bash
bash ./datasets/download_cut_dataset.sh grumpifycat
```
The dataset is downloaded and unzipped at `./datasets/grumpifycat/`.

- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.

Train the DCL model:
```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_DCL 
```

Or train the SimDCL model:

```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_SimDCL --model simdcl
```

We also support CUT:

```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_cut --model cut
```

and fastCUT:

```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_fastcut --model fastcut
```

and CycleGAN:

```bash
python train.py --dataroot ./datasets/grumpifycat --name grumpycat_cyclegan --model cycle_gan
```

The checkpoints will be stored at `./checkpoints/grumpycat_DCL/`.

- Test the DCL model:
```bash
python test.py --dataroot ./datasets/grumpifycat --name grumpycat_DCL
```

The test results will be saved to an html file here: `./results/grumpycat_DCL/latest_test/`.

### DCLGAN, SimDCL, CUT and CycleGAN
DCLGAN is a more robust unsupervised image-to-image translation model compared to previous models. Our performance is usually better than CUT & CycleGAN.

SIMDCL is a different version, it was designed to solve mode collpase. We recommend using it for small-scale, unbalanced dataset.

### [Datasets](./docs/datasets.md)
Download CUT/CycleGAN/pix2pix datasets and learn how to create your own datasets.

Or download it here: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/.

When preparing the CityScape dataset, please use Pillow=5.0.0 to run prepare_dataset.py for consistency. 

### Apply a pre-trained DCL model and evaluate
We provide our pre-trained DCLGAN models for:

Cat <-> Dog : https://drive.google.com/file/d/1-0SICLeoySDG0q2k1yeJEI2QJvEL-DRG/view?usp=sharing

Horse <-> Zebra: https://drive.google.com/file/d/16oPsXaP3RgGargJS0JO1K-vWBz42n5lf/view?usp=sharing

CityScapes: https://drive.google.com/file/d/1ZiLAhYG647ipaVXyZdBCsGeiHgBmME6X/view?usp=sharing

Download the pre-tained model, unzip it and put it inside ./checkpoints (You may need to create checkpoints folder by yourself if you didn't run the training code).

Example usage: Download the dataset of Horse2Zebra and test the model using:

```bash
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_dcl
```

For FID score, use [pytorch-fid](https://github.com/mseitzer/pytorch-fid).

Test the FID for Horse-> Zebra:
```bash
python -m pytorch_fid ./results/horse2zebra_dcl/test_latest/images/fake_B ./results/horse2zebra_dcl/test_latest/images/real_B
```

and Zorse-> Hebra:
```bash
python -m pytorch_fid ./results/horse2zebra_dcl/test_latest/images/fake_A ./results/horse2zebra_dcl/test_latest/images/real_A
```

### Citation
If you use our code or our results, please consider citing our paper. Thanks in advance!
```
@inproceedings{han2021dcl,
  title={Dual Contrastive Learning for Unsupervised Image-to-Image Translation},
  author={Junlin Han and Mehrdad Shoeiby and Lars Petersson and Mohammad Ali Armin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2021}
}
```
If you use something included in CUT, you may also [CUT](https://arxiv.org/pdf/2007.15651).
```
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

### Contact
junlinhcv@gmail.com

### Acknowledgments
Our code is developed based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CUT](http://taesung.me/ContrastiveUnpairedTranslation/). We thank the awesome work provided by CycleGAN and CUT.
We thank [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID computation.
Great thanks to the anonymous reviewers, from both the main CVPR conference and NTIRE. They provided invaluable feedbacks and suggestions.
