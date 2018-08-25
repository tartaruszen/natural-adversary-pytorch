# natural-adversary-pytorch
Pytorch implementation of [Generating Natural Adversarial Examples](https://arxiv.org/abs/1710.11342).


## Usage

#### 1. Clone the repository
```bash
$ git clone https://github.com/hjbahng/natural-adversary-pytorch.git
$ cd natural-adversary-pytorch/
```

#### 2. Train
```bash
$ python main.py --mode train
```

#### 3. Generate natural adversary examples
```bash
$ python main.py --mode generate --resume_iters 1000
```

## Results

<p align="center"><img width="100%" src="PNG/results.png" /></p>