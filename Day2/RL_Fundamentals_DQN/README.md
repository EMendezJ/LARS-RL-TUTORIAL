# RL_Fundamentals_DQN

This example is a constructive modification of the DQN example provided by pytorch. To run the notebook, we suggest creating a python virtual environment (.venv), then running the following commands in the terminal:

```
python -m pip install matplotlib
python -m pip install torch
python -m pip install numpy
python -m pip install gymnasium
python -m pip install pygame
```

Then, a simple way to run the jupyter notebook is through vs code's Jupyter extension. Set the jupyter kernel to use this virtual environment and select run all. To run the simulation faster, rendering can be turned off by changing the following line:

```
env = gym.make("CartPole-v1", render_mode='human')
```

To:
```
env = gym.make("CartPole-v1")
```

This notebook can also be ran in Google Colab, by installing the dependencies using the following code block:

```
%pip install matplotlib
%pip install torch
%pip install numpy
%pip install gymnasium
%pip install pygame
```


The code for this section originates from:

_Reinforcement Learning (DQN) Tutorial_ (2017). Pytorch. https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 