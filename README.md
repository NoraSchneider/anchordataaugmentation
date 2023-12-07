# Anchor Data Augmentation
Authors: Nora Schneider, Shirin Goshtasbpour, Fernando Perez Cruz.

This is the Python implementation for [Anchor Data Augmentation](https://arxiv.org/abs/2311.06965) published at NeurIPS 2023. ADA is a simple yet effective data augmentation algorithm that is specifically suitable for regression problems.

Note that the folder `baselines/CMixup` is cloned from the [C-Mixup Repository](https://github.com/huaxiuyao/C-Mixup) (Yao et. al. 2022) and was extended with `__init__.py` files to run experiments. Similarly, most of `experiments/PovertyMap` originates from the same repository and was extended for ADA. For the indistribution robustness and outofdistribution generalization experiments, we also closely followed their setup. 


## Installation

To setup the corresponding [Conda](https://docs.conda.io/en/latest/miniconda.html) environment run:

```bash
git clone $REPOSITORY_URL
cd anchordataaugmentation
conda create -n ada python==3.10
pip install -r requirements.txt
```

To install `ada` run:
```bash
pip install -e .
```


## Example Usage
As explained in our paper, ADA works well for augmentating samples for training neural networks via stochastic-gradient based methods. As part of stochastic gradient descent methods (applying per minibatch), you can use the following example:

```python
from ada import ADA
# assume a dataloader is initialized before that returns training (X, y) pairs and the anchor matrix for the corresponding samples
for i, (X, y, anchor) in enumerate(trainloader):
  gamma = np.random.uniform(low=0.25, high=4) #equivalent to alpha = 4
  X_aug, y_aug = Augmentor.transform_pytorch(X=X, y=y, gamma=gamma, anchorMatrix=anchor)
```


Yet, to obtain a full  "augmented" dataset, you can also use the following code example

```python
from ada import ADA
from ada.data.linearData import LinearData

data = LinearData(random_state_train=314, size=100)
X_train, y_train = data.X_train , data.y_train

gammas = [0.5, 0.75, 1.5, 2] # values to use for ADA augmentation
number_anchors = 10

X_aug, y_aug = ADA(X=X_train, y=y_train, generate_anchor_args={"anchor_levels": number_anchors}).augment(gamma=gammas, return_original_data=True)

# for given anchor matrix
anchor = ...
X_aug, y_aug = ADA(X=X_train, y=y_train, anchor=anchor).augment(gamma=gammas, return_original_data=True)

# for given gamma value and anchor matrix you can also use
X_aug, y_aug = ADA.transform_numpy(X=X_train, y=y_train, gamma=gammas[0], anchorMatrix=anchor)

```



## Experiments
The code for all of our experiments can be found in the `experiments` folder. So to run the experiments change into the folder via `cd experiments` and run the respective code. 

### Linear synthetic data

For running linear synthetic data, run the following commands 
```
python syntheticdataexperiments.py --model ridgeregression --ada --erm --cmixup --vanilla

python syntheticdataexperiments.py --model mlp --ada --erm --cmixup --vanilla 
```

### Real-world housing data
The datasets are being downloaded with running the code. Run the following commands

```
python realdataexperiments.py --model mlp --ada
python realdataexperiments --model mlp --ada
```


### In-distribution and out-of-distribution experiments
First, download the datasets from [CMixup](https://github.com/huaxiuyao/C-Mixup) and according to their README-file. Note that our setup closely follows their approach to provide a fair comparison. Place the datafolders into `experiments/data/` and run the following commands:

```bash
python cmixupexperiments.py --dataset Airfoil --method ada  --seed 1
```

```bash
python cmixupexperiments.py --dataset NO2 --method ada  --seed 1
```

```bash
python cmixupexperiments.py --dataset TimeSeries-exchange_rate --method ada  --seed 1
```

```bash
python cmixupexperiments.py --dataset TimeSeries-electricity --method ada  --seed 1
```

```bash
python cmixupexperiments.py --dataset RCF_MNIST --method ada --seed 1
```

```bash
python cmixupexperiments.py --dataset CommunitiesAndCrime --method ada --seed 1
```

```bash
python cmixupexperiments.py --dataset SkillCraft --method ada --seed 1
```

```bash
python cmixupexperiments.py --dataset Dti_dg --method ada --seed 1
```

For the "PovertyMap" dataset, we reused the code from the CMixup repository and made the corresponding adjustments to augment the training with ADA. So first, navigate into the corresponding folder `cd PovertyMap`. The dataset is being downloaded when running the code. To run the code use the following: 
```
python main.py --algorithm ada --seed 0
```

## Reference
```
@inproceedings{schneider2023anchor,
  title={Anchor Data Augmentation},
  author={Schneider, Nora and Goshtasbpour, Shirin and Perez-Cruz, Fernando},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
