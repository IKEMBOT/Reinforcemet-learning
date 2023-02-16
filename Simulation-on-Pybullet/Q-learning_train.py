# Slide Example for Q-LEarning (RL-Course NTNU, Saeedvand)
import time
import pybullet as p
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import json
from simple_driving_env import SimpleDrivingEnv
render = 'human'
env = SimpleDrivingEnv()

def plot(rewards):
    plt.figure(2)
    plt.title('Aveage Reward Q-Learning')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.plot(rewards, color='green', label='Reward)')
    plt.grid(axis='x', color='0.80')
    plt.legend(title='Parameter where:')
    plt.show()

def Q_value_initialize(state, action, type = 0):
    if type == 1:
        return np.ones((state, action))
    elif type == 0:
        return np.zeros((state, action))
    elif type == -1:
        return np.random.random((state, action))
   

def epsilon_greedy(Q, epsilon, s):
    if np.random.rand() < epsilon:
        action = np.argmax(Q[s, :]).item()
    else:
        action = env.action_space.sample() 
    return (action,epsilon)

def normalize(list):
    xmin = min(list) 
    xmax=max(list)
    for i, x in enumerate(list):
        list[i] = (x-xmin) / (xmax-xmin)
    return list 

def Qlearning(alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, EPS_DECAY, n_tests):
    episodic_rewards = []
    avg_episodic_rewards = []
    stdev_episodic_rewards = []
    best_avg_episodic_reward = -np.inf
    episodes_passed = 0
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = Q_value_initialize(n_states, n_actions, type = 0)
  
    timestep_reward = []
    for episode in range(episodes):
        #print(f"Episode: {episode}")
        s,_ = env.reset() # read also state
        EPS_START = EPS_START
        EPS_END = EPS_END
        EPS_DECAY = EPS_DECAY 
        epsilon_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)
        t = 0
        total_reward = 0
        while t < max_steps:
            t += 1
            a ,eps_val = epsilon_greedy(Q, epsilon_threshold, s)
            #start = time.time()
            s_, reward,done,info = env.step(a)
            # end = time.time()
            # print(end - start)            
            #s_, reward, done,  = env.step(a)
            total_reward += reward
            a_next = np.argmax(Q[s_, :]).item()
            if done:
                Q[s, a] += alpha * (reward - Q[s, a])
            else:
                Q[s, a] += alpha * (reward + (gamma * Q[s_, a_next]) - Q[s, a])
            s, a = s_, a_next
            if  done:
                s,_ = env.reset()
                print("Finish !! hore !!")
            # print(total_reward)

        timestep_reward.append(total_reward/t)
        print(f"Episode: {episode}, steps: {t}, reward: {total_reward}")
        episodic_rewards.append(total_reward/t)
        episodes_passed += 1
        
        Q_val = [{"Q_value": Q.tolist()}]
        if episode % 100 :
            jsonFile = open("Qvalue.json", "w")
            jsonFile.write(json.dumps(Q_val))
            jsonFile.close()

        # Compute average reward and variance (standard deviation)
        if len(episodic_rewards) <= 10:
            avg_episodic_rewards.append(np.mean(np.array(episodic_rewards)))
            if len(episodic_rewards) >= 2:
                stdev_episodic_rewards.append(np.std(np.array(episodic_rewards)))

        else:
            avg_episodic_rewards.append(np.mean(np.array(episodic_rewards[-10:])))
            stdev_episodic_rewards.append(np.std(np.array(episodic_rewards[-10:])))

        if avg_episodic_rewards[-1] > best_avg_episodic_reward:
            best_avg_episodic_reward = avg_episodic_rewards[-1]
    
        if episodes_passed % 100 == 0:
            plot_rewards(np.array(episodic_rewards), np.array(avg_episodic_rewards),
                        np.array(stdev_episodic_rewards))
            print('Episode {}\tAvg. Reward: {}\tEpsilon: {}\t'.format(episodes_passed, avg_episodic_rewards[-1],eps_val))
            print('Best avg. episodic reward:', best_avg_episodic_reward)

    env.close()
    if n_tests > 0:
        test_agent(Q, n_tests)
    
    plot(normalize(timestep_reward))
    return timestep_reward

#----------------------------------------------------
def test_agent(Q, n_tests = 0, delay=1):
    env = SimpleDrivingEnv("human",track=1)
    #env.render()
    for testing in range(n_tests):
        print(f"Test #{testing}")
        s,_  = env.reset()
        while True:
            time.sleep(delay)
            a = np.argmax(Q[s, :]).item()
            print(f"Chose action {a} for state {s}")
            s, reward , done = env.step(a)
            #time.sleep(1)
            if  done:
                print("Finished!", reward)
                time.sleep(5)
                break
    env.close()

def plot_rewards(reward_arr, avg_reward_arr, stdev_reward_arr, save=True):

    fig1 = plt.figure(1)
    # rewards + average rewards
    plt.plot(reward_arr, color='b', alpha=0.3)
    plt.plot(avg_reward_arr, color='b')
    plt.xlabel('# episodes')
    plt.ylabel('Acc. episodic reward')
    plt.title('Accumulated episodic reward vs. num. of episodes')
    plt.legend(['Acc. episodic reward', 'Avg. acc. episodic reward'])
    plt.tight_layout()
    fig1.savefig("Reward + Average Reward")

    # average rewards + stdevs
    fig2 = plt.figure(2)
    plt.plot(avg_reward_arr, color='b')
    plt.fill_between(range(1, len(avg_reward_arr)), avg_reward_arr[1:] - stdev_reward_arr,
                     avg_reward_arr[1:] + stdev_reward_arr, color='b', alpha=0.2)
    plt.xlabel('# episodes')
    plt.ylabel('Acc. episodic reward')
    plt.title('Accumulated episodic reward vs. num. of episodes')
    plt.legend(['Avg. acc. episodic reward', 'Stdev envelope of acc. episodic reward'])
    plt.tight_layout()
    fig2.savefig("Average Rewards + Stdevs")
    plt.pause(0.1)

if __name__ == "__main__":
    alpha = 0.01 # learning rate
    gamma = 0.99 # discount factor
    epsilon = 0.01 # epsilon greedy exploration-explotation (smaller more random)
    episodes = 10000
    
    EPS_START = 0.1
    EPS_END = 1
    EPS_DECAY = 50

    #when you are goint to use Advance Reward Functionv1
    #plese change max_step = 300
    #otherwise max_step = 1000

    # max_steps = 500 # to make it infinite make sure reach objective
    max_steps = 500 # to make it infinite make sure reach objective
    timestep_reward = Qlearning(alpha, gamma, epsilon, episodes, max_steps, EPS_START, EPS_END, EPS_DECAY, n_tests = 2)
    label = ["Q_learning"]
    plot(timestep_reward)
    plt.show()
  
    
  