# HyCoEM for Fine-grained Emotion Classification

## Data
All datasets used in this work are publicly available. The original versions can be accessed at: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues). We use the same dataset version as in both [HypEmo](https://github.com/dinobby/HypEmo/tree/main) and [LCL](https://github.com/varsha33/LCL_loss), with identical preprocessing. The files are provided in the `data` folder.

## Install dependencies
The code is tested with Python 3.7.11. First, install the required dependencies:
```bash
pip install -r requirements.txt
```
## Training and Inference
To train and evaluate the model, run the following command: </br>
`python train.py --name Checkpoint1  --dataset ED --cl_loss 1  --neg_sample 8  --enc_type roberta-base  --batch_size 64` 

Some Important arguments: </br>
- `--name` A name of the checkpoint for each run
- `--data` name of dataset directory which contains your data and related files. Possible choices are: `go_emotion`, `ED`, and four challenging varaints of ED namely `ED_easy_4`, `ED_hard_a,` `ED_hard_b`, `ED_hard_c`, `ED_hard_d`
- 
- `--batch_size` batch_size for training. We set it to 64 for all datasets.
- `--` name of dataset directory which contains your data and related files. Possible choices are `wos`, `rcv`, `bgc`  and `nyt`.
- `--cl_loss` Set to 1 for using contrastive loss in Lorentz hyperbolic space
- `--cl_temp` Temperature for the contarstive loss. We use a value of 0.07 for all datasets
- `--cl_wt` weight for contrastive loss. We use the following weights for the contrastive loss across datasets: `WOS:0.3` , `RCV1-V2:0.4`, `BGC:0.4`,  and `NYT:0.6`


### For Exponential map transformation and Geodesic distance calculation in Lorentz hyeprbolic space
- The code for operations in lorentz geometry is in the script `lorentz.py` which is obtained from [meru](https://github.com/facebookresearch/meru/blob/main/meru/lorentz.py) repository.
- Specifically we use the functions  `exp_map0()` and `pairwise_dist()` provided in  the script `lorentz.py`

### For Contrastive loss in Lorentz hyperbolic space
- The code for the contrastive loss and is in the script `criterion.py`, where the loss is defined as the PyTorch class `CLLoss`.
