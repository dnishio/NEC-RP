# Random Projection in Neural Episodic Control

An implementation of **Random Projection in Neural Episodic Control** (NEC-RP) based on [*Coach*](https://github.com/NervanaSystems/coach).

[https://arxiv.org/abs/1904.01790](https://arxiv.org/abs/1904.01790)



## dependency

The dependencies are almost same as *Coach*'s one.

In addition, we use [*faiss*](https://github.com/facebookresearch/faiss) for approximate nearset neighbor algorithm.

Therefore, we recommend that you use conda environment.  



## Installation

Please see the [*Coach*'s installation](https://github.com/NervanaSystems/coach#installation).

After you clone our repository, you can slimply run the below. 

`pip install -e .`

The installation of *faiss* is [here](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md).



## Usage

Please refer to the [*Coach*'s usage](https://github.com/NervanaSystems/coach#usage).


For example,

* Pong using the Random Projection in Neural Episodic Control (NEC-RP) algorithm (no render):

```
coach -p Atari_NEC_RP -lvl pong
```

* If you want to try to change some parameters, please change the [presets](rl_coach/presets/Atari_NEC_RP.py) or [fc_middleware.py](rl_coach/architectures/tensorflow_components/middlewares/fc_middleware.py).

## licence

This software includes the work that is distributed in the Apache License 2.0

