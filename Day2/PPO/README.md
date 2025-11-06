# PPO

This example is a constructive modification of the PPO example provided by pytorch. To run the notebook, we suggest creating a python virtual environment (.venv), then running the following commands:

```
python -m pip install matplotlib
python -m pip install torch
python -m pip install torchrl
python -m pip install gymnasium[mujoco]
python -m pip install tqdm
```

Then, a simple way to run the jupyter notebook is through vs code's Jupyter extension. Set the jupyter kernel to use this virtual environment and select run all. 

This example also includes a comparison with DQN discretization, which is a simple python file which can be ran in the same virtual environment.

The notebook can also be ran in Google Colab. However, the gymnasium simulation cannot be easily rendered. Instead, we suggest removing the rendering by changing the line:

```
base_env = GymEnv("InvertedPendulum-v5", device=device, render_mode="human")
```

To the following:

```
base_env = GymEnv("InvertedPendulum-v5", device=device)
```
This disables the environment rendering, which removes the visualization but the training statistics can still be viewed. With this, the notebook can be ran successfully in Google Colab by installing the dependencies using the following code block:

```
%pip install matplotlib
%pip install torch
%pip install torchrl
%pip install gymnasium[mujoco]
%pip install tqdm
```


The code for this section was modified from:

_Reinforcement Learning (PPO) with TorchRL Tutorial_ (2017). Pytorch. https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html#reinforcement-learning-ppo-with-torchrl-tutorial