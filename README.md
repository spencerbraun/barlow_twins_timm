# Barlow Twins with timm

This is an implementation of [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230).

I borrow from the original repo and mostly refactor to make configurable (via [hydra](https://github.com/facebookresearch/hydra)), loggable via [wandb](https://wandb.ai/), and integrate the [timm](https://github.com/rwightman/pytorch-image-models) library for fast experimentation.

## Usage

### Requirements

Install dependencies into a virtualenv:

```bash
$ python -m venv env
$ source env/bin/activate
(env) $ pip install -r requirements.txt
```

Written with python version `3.8.11`

### Data and Configuration

Custom datasets can be placed in the `data/` dir. Edits should be made to the `conf/data/default.yaml` file to reflect the correct properties of the data. All other configuration hyperparameters can be set in the hydra configs.

### Train

Once properly configured, a model can be trained via `python train.py`.

## Citations

```bibtex
@misc{zbontar2021barlow,
      title={Barlow Twins: Self-Supervised Learning via Redundancy Reduction},
      author={Jure Zbontar and Li Jing and Ishan Misra and Yann LeCun and Stéphane Deny},
      year={2021},
      eprint={2103.03230},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
