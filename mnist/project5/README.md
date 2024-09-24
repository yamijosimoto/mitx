## Text-Based-Game-with-Markov-Decision-Process

# MITx - MicroMasters Program on Statistics and Data Science - Machine Learning with Python

Fifth Project - Text-Based Game with Markov Decision Processes

The fifth and final project of the MIT MicroMasters Program course on Machine Learning with Python was to create
a game based on Reinforcement Learning (RL), Neural Networks (NN) and Markov Decision Process (MDP).

The aim was to create a RL based model where the agent of the game learns by conducting certain actions in an environment that mimics that of a typical house, consisting of four rooms - a living room, a bed room, a kitchen and a garden with connecting path ways. Inside each room there are different tasks the agent can do (e.g. in the kitchen you can eat an apple), and the goal of the game is to navigate through the different rooms and complete each given quest in order to finish the game. For example, if the agent is told that it is hungry, it needs to first navigate to the correct room - the kitchen - and complete the task there - eat an apple - in order to complete the quest and receive the reward.

In a MDP we have a random process with the Markov property that says that if we know the value taken by the process at a given time, we won't get any additional information about the future behaviour of the process by gathering more knowledge about the past. Thus, the conditional distribution of the future states of the process given present and past states depends only on the present state and not at all on the past states. In the game, the MDP policy comes from an agent that earns rewards or points by taking the right action in order to complete a quest (+1), and looses them by taking the wrong one (-0.1) (e.g. trying to eat a TV in the living room). Each navigation step from a room to another consume points (-0.01). The agent thus gets feedback from its actions from which it learns the right policy (i.e. set of steps and actions) in order to complete the game with the maximum amount of rewards.

The project first required to implement a tabular Q-learning algorithm for a simple setting where each text description is associated with a unique index. The next task was to implement the Q-learning algorithm with linear approximation architecture by using bag-of-words representation for textual state description. Finally, the last task was to implement a deep Q-work based on NN and use the Q-learning algorithms inside the game.

## Dataset

      - framework.py: contains the framework to run the game.

## Access and requirements

The different models are run each separetely from their files with the help of the two modules (utils.py and framework.py) that contain helper and utility functions. The files agent_tabular_ql.py and agent_linear.py contain models run on basic libraries while the file deep_q_learning.py uses neural networks from PyTorch. 

The dependencies and requirements can be seen from requirements.txt that can be installed in shell with the command:

      pip install -r requirements.txt

