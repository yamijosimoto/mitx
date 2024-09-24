# ---------------------------------------------------------------------------------------------#
#                                                                                              #
#                              Q-Learning Algorithm (tabular)                                  #
#                                                                                              #
#     The agent first trains each episode following an epsilon-greedy policy and by updating   #
#     the Q-values. After this, for each testing phase of each epoch, the cumulative           #
#     discounted reward and the average reward performance for each episode is calculated      #
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
NUM_RUNS = 10             # Number of runs
NUM_EPOCHS = 200          # Number of epochs
NUM_EPIS_TRAIN = 25       # Number of episodes for training
NUM_EPIS_TEST = 50        # Number of episodes for testing
ALPHA = 0.1               # Learning rate 

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


def epsilon_greedy(state_1, state_2, q_func, epsilon):
    """Returns an action selected by an epsilon-Greedy exploration policy
    (i.e. the agent chooses its action depending on an eploration parameter.)
    
    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command
    
    Returns:
        (int, int): the indices describing the action/object to take
    """
    
    # Coin toss to decide whether to take a random or the best action
    if np.random.binomial(1, epsilon):
        action_index, object_index = np.random.randint(NUM_ACTIONS, size=1), np.random.randint(NUM_OBJECTS, size=1)   # Random action and object
    else:
        q_values = q_func[state_1, state_2, :, :]                                                                     # (actions, objects)
        (action_index, object_index) = np.unravel_index(np.argmax(q_values, axis=None), q_values.shape)               # Choosing best action and object
    
    return (int(action_index), int(object_index))                                                                     # Returning a tuple with integers


def tabular_q_learning(q_func, current_state_1, current_state_2, action_index, object_index, reward, next_state_1, next_state_2, terminal):
    """Update q_func for a given transition state (s, c, R(s,c), s').
    The agent takes an action c(a,b) at state s, getting a reward R(s,c) and observing the next state s'
    
    Args:
        q_func (np.ndarray): current Q-function
        current_state_1, current_state_2 (int, int): two indices describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_1, next_state_2 (int, int): two indices describing the next state
        terminal (bool): True if this eposode is over or the number of steps reaches the maximum number of steps for each episode
    
    Returns:
        None
    """
    
    if terminal:
        maxQ = 0                                                  # Checking if episode is over
    else:
        maxQ = np.max(q_func[next_state_1, next_state_2, :, :])   # If not: Q-value update: Q(s,c) <- (1-alpha)Q(s,c) + alpha[R(s,c) + gamma[Max(Q)(s',c')]

    q_value = q_func[current_state_1, current_state_2, action_index, object_index]
    q_func[current_state_1, current_state_2, action_index, object_index] = (1 - ALPHA) * q_value + ALPHA * (reward + GAMMA * maxQ)

    return None  

  
def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward
    The observable state to the agent is described in text. Therefore each description is mapped
    into vector representations from two dictionaries that contain the room and the quest descriptions.
    
    Args:
        for_training (bool): True if for training
    
    Returns:
        None
    """
    
    epsilon = TRAINING_EP if for_training else TESTING_EP
  
    episode_reward = 0.0                                                              # Initializing rewards
    
    (current_room_desc, current_quest_desc, terminal) = framework.newGame()           # Descriptions of the current room and current quest state

    while not terminal:                                                               # Choosing next action and executing from epsilon-greedy policy 
        
        current_room_desc_index = dict_room_desc[current_room_desc]                   # Current room description indices
        current_quest_desc_index = dict_quest_desc[current_quest_desc]                # Current quest description indices

        (next_action_index, next_object_index) = epsilon_greedy(current_room_desc_index, current_quest_desc_index, q_func, epsilon)

        (next_room_desc, next_quest_desc, reward, terminal) = framework.step_game(    # Taking a step 
            current_room_desc,
            current_quest_desc,
            next_action_index,
            next_object_index)


        if for_training:                                                              # Updating Q-function
            
            next_room_desc_index = dict_room_desc[next_room_desc]                     # Updating room description index 
            next_quest_desc_index = dict_quest_desc[next_quest_desc]                  # Updating quest description index 
            
            tabular_q_learning(
              q_func,
              current_room_desc_index,
              current_quest_desc_index,
              next_action_index,
              next_object_index,
              reward,                                                                 # A real valued number representing the one-step reward obtained at this step
              next_room_desc_index,
              next_quest_desc_index,
              terminal)

        if not for_training:
            
            episode_reward += (GAMMA**(framework.STEP_COUNT - 1)) * reward            # Updating reward

        current_room_desc, current_quest_desc = next_room_desc, next_quest_desc       # Preparing next step

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
    
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

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
    
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()                 # Building dictionaries that use unique index for each state
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

    framework.load_game_data()                                                            # Loading data

    epoch_rewards_test = []                                                               # (NUM_RUNS * NUM_EPOCHS)

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test, axis=0))                                     # Plotting reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()

