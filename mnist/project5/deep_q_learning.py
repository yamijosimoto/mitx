# ---------------------------------------------------------------------------------------------#
#                                                                                              #
#                              Q-Learning Algorithm (Neural Network)                           #
#                                                                                              #
#     The agent first trains each episode following an epsilon-greedy policy and by updating   #
#     the Q-values through an approximated target value and a gradient step in a deep          #
#     Neural Network. Textual state space is mapped into vector representation instead of      #
#     unique indexing. After training, for each testing phase of each epoch, the cumulative    #
#     discounted reward and the average reward performance for each episode is calculated.     #
#                                                                                              #
# ---------------------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

np.random.seed(42)        # Setting random seed 

DEBUG = False

GAMMA = 0.5               # Discounted factor
TRAINING_EP = 0.5         # Epsilon-greedy parameter for training
TESTING_EP = 0.05         # Epsilon-greedy parameter for testing
NUM_RUNS = 10             # Number of runs
NUM_EPOCHS = 300          # Number of epochs
NUM_EPIS_TRAIN = 25       # Number of episodes for training 
NUM_EPIS_TEST = 50        # Number of episodes for testing
ALPHA = 0.1               # Learning rate 

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)

model = None
optimizer = None


def epsilon_greedy(state_vector, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy
    
    Args:
        state_vector (torch.FloatTensor): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command
        
    Returns:
        (int, int): the indices describing the action/object to take
    """
    # Coin toss to decide whether to take a random or the best action
    if np.random.binomial(1, epsilon):
        action_index, object_index = np.random.randint(NUM_ACTIONS, size=1), np.random.randint(NUM_OBJECTS, size=1)   # Random action and object
    else:                                                                                                             # Choosing best action and object
        act_arr, obj_arr = model(state_vector)
        action_index, object_index = torch.argmax(act_arr), torch.argmax(obj_arr)

    return (int(action_index), int(object_index))                                                                     # Returning a tuple with integers


class DQN(nn.Module):
    """A simple deep Q network implementation.
    Computes Q-values for each (action, object) tuple given an input state vector
    """

    def __init__(self, state_dim, action_dim, object_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)
        self.state2object = nn.Linear(hidden_size, object_dim)

    def forward(self, x):
        state = F.relu(self.state_encoder(x))
        return self.state2action(state), self.state2object(state)


def deep_q_learning(current_state_vector, action_index, object_index, reward, next_state_vector, terminal):
    """Updates the weights of the DQN for a given transition
    
    Args:
        current_state_vector (torch.FloatTensor): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (torch.FloatTensor): vector representation of next state
        terminal (bool): True if this epsiode is over
        
    Returns:
        None
    """
    
    with torch.no_grad():                                                                                         # Disabling gradient calculation to save memory
        q_values_action_next, q_values_object_next = model(next_state_vector)
    maxq_next = 1/2 * (q_values_action_next.max() + q_values_object_next.max())

    q_value_current_state = model(current_state_vector)
    Q_value_current = 1/2 * (q_value_current_state[0][action_index] + q_value_current_state[1][object_index])     # Current Q-value

    maxQ = 0.0 if terminal else maxq_next                                                                         # Checking if episode is over
    y = reward + GAMMA * maxQ                                                                                     # y = R(s, c) + gamma[Max(Q)(s',c')

    loss = 1/2 * (y - Q_value_current)**2                                                                         # L(theta) = 1/2(y-Q(s,c,theta))^2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run_episode(for_training):
    """
        Runs one episode
        If for training, update Q function
        If for testing, computes and return cumulative discounted reward
    """
    
    epsilon = TRAINING_EP if for_training else TESTING_EP
    episode_reward = 0.0                                                                      # Initializing rewards

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()

    while not terminal:                                                                               
        current_state = current_room_desc + current_quest_desc
        current_state_vector = torch.FloatTensor(utils.extract_bow_feature_vector(current_state, dictionary))

        next_action_index, next_object_index = epsilon_greedy(current_state_vector, epsilon)  # Choosing next action and executing from epsilon-greedy policy

        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(              # Taking a step
            current_room_desc,
            current_quest_desc,
            next_action_index,
            next_object_index)  

        next_state = next_room_desc + next_quest_desc                                         # Mapping next state vector
        next_state_vector = torch.FloatTensor(utils.extract_bow_feature_vector(next_state, dictionary))

        if for_training:                                                                      # Updating Q-function
            
            deep_q_learning(current_state_vector, next_action_index, next_object_index, reward, next_state_vector, terminal)  

        if not for_training:
            
            episode_reward += (GAMMA**(framework.STEP_COUNT - 1)) * reward                    # Updating reward

        current_room_desc, current_quest_desc = next_room_desc, next_quest_desc               # Preparing next step

    if not for_training:
        return episode_reward


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global model
    global optimizer
    model = DQN(state_dim, NUM_ACTIONS, NUM_OBJECTS)
    optimizer = optim.SGD(model.parameters(), lr=ALPHA)

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    state_texts = utils.load_data('game.tsv')
    dictionary = utils.bag_of_words(state_texts)                                                    # Creating dictionary with vectors of states
    state_dim = len(dictionary)

    framework.load_game_data()                                                                      # Loading data

    epoch_rewards_test = []                                                                         # (NUM_RUNS * NUM_EPOCHS)

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test, axis=0))                                               # Plotting reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
