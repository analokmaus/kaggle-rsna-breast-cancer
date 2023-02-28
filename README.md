# RSNA Screening Mammography Breast Cancer Detection
**6th place solution: Team Chiral Mistrals**

RabotniKuma (Hiroshi Yoshihara) part

## Environmnet
We recommend you to use [Kaggle GPU docker v128](https://console.cloud.google.com/gcr/images/kaggle-gpu-images/GLOBAL/python).

Conda environment yaml file can be found at `./environment.yaml`.


## Data preparation
1. Download competition dataset and place them at `./input/rsna-breast-cancer-detection/`.
2. Run image conversion script: `python convert_image.py`


## Experiments
Expriment configs are stored in `./configs.py`. 
### Expriment lists
| Config name | Description                       | CV    | Public LB | Private LB |
|-------------|-----------------------------------|-------|-----------|------------|
| Aug07lr0    | Multi-view model, 1024x512        | 0.493 | 0.64      | 0.46       |
| Res02lr0    | Multi-view model, 1536x768        | 0.488 | 0.59      | 0.46       |
| Res02mod2   | Multi-view fusion model, 1536x768 | 0.516 | -         | -          |
| Res02mod3   | Multi-view fusion model, 1536x768 | 0.525 | 0.63      | 0.48       |

### Run experiments
Make your hardware has at least a total of 64 GB of GPU RAM and run the following: 
```bash
python train.py --config {config name} --num_works {number of cpu cores to be used}
```
Please modify batch size and learning rate in config file(`./configs.py` ) if your hardware has less GPU RAM.

Results (weights, predictions, training logs) will be export to `./results/{config name}/`.
