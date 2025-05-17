# HyCoEM for Fine-grained Emotion Classification

## Data
All datasets used in this work are publicly available. The original versions can be accessed at: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues). We use the same dataset version as in both [HypEmo](https://github.com/dinobby/HypEmo/tree/main) and [LCL](https://github.com/varsha33/LCL_loss), with identical preprocessing. The files are provided in the `data` folder. This includes the full GoEmotions and ED datasets, as well as the four challenging subsets of ED proposed in [LCL](https://github.com/varsha33/LCL_loss): `ED_easy_4`, `ED_hard_a`, `ED_hard_b`, `ED_hard_c`, and `ED_hard_d`.

## Install dependencies
The code is tested with Python 3.7.11. First, install the required dependencies:
```bash
pip install -r requirements.txt
```
## Training and Inference
To train and evaluate the model, run the following command: </br>
`python train.py --name Checkpoint1  --dataset ED --cl_loss 1 --cl_temp 0.07  --neg_sample 8  --enc_type roberta-base  --batch_size 64` 

### Some Important Arguments:
- `--name` Name of the checkpoint for the run.
- `--data` Name of the dataset directory containing the data and related files. Possible choices: `go_emotion`, `ED`, and the four challenging ED variants: `ED_easy_4`, `ED_hard_a`, `ED_hard_b`, `ED_hard_c`, `ED_hard_d`.
- `--cl_loss` Set to `1` to enable contrastive loss in Lorentz hyperbolic space, which is then used to weight the cross-entropy loss. The loss is defined in the script `criterion.py`.
- `--cl_temp` Temperature parameter for the contrastive loss. We use a fixed value of `0.07` for all datasets.
- `--neg_sample` Number of negative samples in the negative label set. We use `8` for ED and `6` for GoEmotions.
- `--enc_type` Type of text encoder. Our model HyCoEM uses `roberta-base` as its text encoder.  Supported choices: `bert-base-uncased`, `roberta-base`, `google/electra-base-discriminator`.
- `--batch_size` Batch size for training. We use `64` for all datasets.

### Hyperbolic Curvature Arguments
The following arguments are not explicitly specified in the above example command since their default values are used:
- `--curv_init` Initial curvature \( -k \) for the Lorentz model (with \( k > 0 \)). Default: `1`.
- `--learn_curv` Whether to make the curvature learnable. Default: `True`.
.

## For Exponential map transformation and Geodesic distance calculation in Lorentz hyeprbolic space
- The code for operations in lorentz geometry is in the script `lorentz.py` which is obtained from [meru](https://github.com/facebookresearch/meru/blob/main/meru/lorentz.py) repository.
- Specifically we use the functions  `exp_map0()` and `pairwise_dist()` provided in  the script `lorentz.py`

## For Contrastive loss in Lorentz hyperbolic space
- The code for the contrastive loss and is in the script `criterion.py`, where the loss is defined as the PyTorch class `CLLoss`.
