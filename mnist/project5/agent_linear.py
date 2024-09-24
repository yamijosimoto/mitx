# ---------------------------------------------------------------------------------------------#
#                                                                                              #
#                              Q-Learning Algorithm (linear)                                   #
#                                                                                              #
#     The agent first trains each episode following an epsilon-greedy policy and by            #
#     updating the Q-values through an approximated target value and a gradient step.          #
#     Textual state space is mapped into vector representation instead of unique indexing.     #
#     After training, for each testing phase of each epoch, the cumulative discounted reward   #
#     and the average reward performance for each episode is calculated.                       #
#                                                                                              #
# ---------------------------------------------------------------------------------------------#


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
NUM_RUNS = 5              # Number of runs
NUM_EPOCHS = 600          # Number of epochs
NUM_EPIS_TRAIN = 25       # Number of episodes for training 
NUM_EPIS_TEST = 50        # Number of episodes for testing
ALPHA = 0.01              # Learning rate 

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


def tuple2index(action_index, object_index):
    """Converts a tuple (a,b) to an index c"""
    
    return action_index * NUM_OBJECTS + object_index


def index2tuple(index):
    """Converts an index c to a tuple (a,b)"""
    
    return index // NUM_OBJECTS, index % NUM_OBJECTS


def epsilon_greedy(state_vector, theta, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy
    
    Args:
        state_vector (np.ndarray): extracted vector representation
        theta (np.ndarray): current weight matrix (which is shared across state-action pairs)
        epsilon (float): the probability of choosing a random command
        
    Returns:
        (int, int): the indices describing the action/object to take
    """
    
    # Coin toss to decide whether to take a random or the best action
    if np.random.binomial(1, epsilon):
        action_index, object_index = np.random.randint(NUM_ACTIONS, size=1), np.random.randint(NUM_OBJECTS, size=1)       # Random action and object
    else:                                                                                                                 # Choosing best action and object
        q_values = theta @ state_vector
        action_index, object_index = np.unravel_index(np.argmax(q_values), (NUM_ACTIONS, NUM_OBJECTS))                    # From index to tuple

    return (int(action_index), int(object_index))                                                                         # Returning a tuple with integers


def linear_q_learning(theta, current_state_vector, action_index, object_index, reward, next_state_vector, terminal):
    """Update theta for a given transition
    Theta is updated by taking a gradient step with respect to the squared loss
    theta <- theta + alpha[R(s,c) + gamma[Max(Q)(s',c',theta) - Q(s,c,theta)]phi(s,c)
    Phi(s,c) maps the textual state space into vector representation
    Args:
        theta (np.ndarray): current weight matrix
        current_state_vector (np.ndarray): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (np.ndarray): vector representation of next state
        terminal (bool): True if this epsiode is over
        
    Returns:
        None
    """
    
    if terminal:
        maxQ = 0.0                                                                                   # Checking if episode is over
    else:                                                                              
        maxQ = np.max(theta @ next_state_vector)                                                     # If not: Q-value approximation update through y and theta     
                                                                                       
                                                                                      
    current_indices = tuple2index(action_index, object_index)                                        # Accessing Q(s,c,theta) with tuple2index()
    Q_val = (theta @ current_state_vector)[current_indices]                            

    y = reward + GAMMA * maxQ                                                                        # y = R(s, c) + gamma[Max(Q)(s',c')
    
    theta[current_indices] = theta[current_indices] + ALPHA * (y - Q_val) * current_state_vector     # Updating theta

    
def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward
    
    Args:
        for_training (bool): True if for training
    
    Returns:
        None
    """
    
    epsilon = TRAINING_EP if for_training else TESTING_EP

    episode_reward = 0.0                                                                             # Initializing rewards
    
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()                          # Descriptions of the current room and current quest state

    while not terminal:                                                                              # Choosing next action and executing from epsilon-greedy policy 
        
        current_state = current_room_desc + current_quest_desc
        current_state_vector = utils.extract_bow_feature_vector(current_state, dictionary)           # Mapping room and quest descriptions into vectors

        next_action_index, next_object_index = epsilon_greedy(current_state_vector, theta, epsilon)  

        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(                     # Taking a step
            current_room_desc,
            current_quest_desc,
            next_action_index,
            next_object_index)  

        next_state = next_room_desc + next_quest_desc                                                # Mapping next state vector
        next_state_vector = utils.extract_bow_feature_vector(next_state, dictionary)

        if for_training:                                                                             # Updating Q-function
            
            linear_q_learning(theta, current_state_vector, next_action_index, next_object_index, reward, next_state_vector, terminal)  

        if not for_training:
            
            episode_reward += (GAMMA**(framework.STEP_COUNT - 1)) * reward                            # Updating reward

        current_room_desc, current_quest_desc = next_room_desc, next_quest_desc                       # Preparing next step

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
    
    global theta
    theta = np.zeros([action_dim, state_dim])

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
    dictionary = utils.bag_of_words(state_texts)                                                        # Creating dictionary with vectors of states
    state_dim = len(dictionary)
    action_dim = NUM_ACTIONS * NUM_OBJECTS

    framework.load_game_data()                                                                          # Loading data

    epoch_rewards_test = []                                                                             # (NUM_RUNS * NUM_EPOCHS)

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test, axis=0))                                                   # Plotting reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
