# HyCoEM for Fine-grained Emotion Classification

## Data
All datasets used in this work are publicly available. The original versions can be accessed at: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues). We use the same dataset version as in both [HypEmo](https://github.com/dinobby/HypEmo/tree/main) and [LCL](https://github.com/varsha33/LCL_loss), with identical preprocessing. The files are provided in the `data` folder.

## Install dependencies
The code is tested with Python 3.7.11. First, install the required dependencies:
```bash
pip install -r requirements.txt
```
## Training



### For Exponential map transformation and Geodesic distance calculation in Lorentz hyeprbolic space
- The code for operations in lorentz geometry is in the script `lorentz.py` which is obtained from [meru](https://github.com/facebookresearch/meru/blob/main/meru/lorentz.py) repository.
- Specifically we use the functions  `exp_map0()` and `pairwise_dist()` provided in  the script `lorentz.py`

### For Contrastive loss in Lorentz hyperbolic space
- The code for the contrastive loss and is in the script `criterion.py`, where the loss is defined as the PyTorch class `CLLoss`.
