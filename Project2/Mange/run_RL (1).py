# For intro to AI gym, see https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym

# For blackjack and reinforcement learning, see Example 5.1 p. 76 in Sutton & Barto
# In that example, an infinite deck is used

import blackjack_extended as bjk #The extended Black-Jack environment
import blackjack_base as bjk_base #The standard sum-based Black-Jack environment
from math import inf
import RL as rl
import sys
import os
import numpy as np
import plotting as pl
import time
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
matplotlib.style.use('ggplot')

if __name__ == "__main__":
    directory = "{}/data".format(sys.path[0])
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_fun = lambda x: "{}/{}_{}.txt".format(directory,x, decks)
    # init constants
    omega = 0.77        # power decay of the learning rate
    n_sims = 10 ** 3    # Number of episodes generated
    epsilon = 0.05      # Probability in epsilon-soft strategy
    init_val = 0.0
    warmup = n_sims//10
    # Directory to save plots in
    plot_dir = "Documents⁩/GitHub⁩/SF2957⁩/Project2/figures⁩"
    plot_dir = "{}/figures/".format(sys.path[0])
    

    for decks in [1,2,6,8,inf]:
        print("----- deck number equal to {} -----".format(decks))
        # set seed
        seed = 31233
        # init envs.
        env = bjk.BlackjackEnvExtend(decks=decks, seed=seed)
        sum_env = bjk_base.BlackjackEnvBase(decks=decks, seed=seed)

        print("----- Starting MC training on expanded state space -----")
        # MC-learning wit expanded state representation
        start_time_MC = time.time()
        Q_MC, MC_avg_reward, state_action_count = rl.learn_MC(
            env, n_sims, gamma = 1, epsilon = epsilon, init_val = init_val,
            episode_file=path_fun("hand_MC_state"), warmup=warmup)
        print("Number of explored states: " + str(len(Q_MC)))
        print("Cumulative avg. reward = " + str(MC_avg_reward))
        time_to_completion_MC = time.time() - start_time_MC

        print("----- Starting Q-learning on expanded state space -----")
        # Q-learning with expanded state representation
        start_time_expanded = time.time()
        Q, avg_reward, state_action_count = rl.learn_Q(
            env, n_sims, gamma = 1, omega = omega, epsilon = epsilon, init_val = init_val,
            episode_file=path_fun("hand_state"), warmup=warmup)
        print(Q)
        print("Number of explored states: " + str(len(Q)))
        print("Cumulative avg. reward = " + str(avg_reward))
        time_to_completion_expanded = time.time() - start_time_expanded
        
        print("----- Starting Q-learning for sum-based state space -----")
        # Q-learning with player sum state representation
        start_time_sum = time.time()
        sumQ, sum_avg_reward, sum_state_action_count = rl.learn_Q(
            sum_env, n_sims, omega = omega, epsilon = epsilon, init_val = init_val,
            episode_file=path_fun("sum_state"), warmup=warmup)
        time_to_completion_sum = time.time() - start_time_sum
        print("Number of explored states (sum states): " + str(len(sumQ)))
        print("Cumulative avg. reward = " + str(sum_avg_reward))

        print("Training time: \n " +
              "Expanded state space MC: {} \n Expanded state space: {} \n Sum state space: {}".format(
                 time_to_completion_MC, time_to_completion_expanded, time_to_completion_sum))

        

        
        actionsTrue=np.zeros((33,27))

        actionsFalse=np.zeros((33,27))
        
        for key in sumQ.keys():
            
            if key[2]==False:
                playersum=key[0]
                dealersum=key[1]
                decision=np.argmax(sumQ[key])
                actionsFalse[playersum,dealersum]=decision
                
            else: # Key is true
                playersum=key[0]
                dealersum=key[1]
                decision=np.argmax(sumQ[key])
                actionsTrue[playersum,dealersum]=decision
        
        actionsTrue=np.delete(actionsTrue,[22,23,24,25,26,27,28,29,30,31,32,21],0)
        sns.heatmap(actionsTrue)
        plt.figure()
        actionsFalse=np.delete(actionsFalse,[22,23,24,25,26,27,28,29,30,31,32,21],0)
        sns.heatmap(actionsFalse)
        
        
        # Convert Q (extended state) to sum state representation and make 3D plots
        # Extended state MC-learning
        Q_conv_MC = rl.convert_to_sum_states(Q_MC, env)
        V_conv_MC = rl.convert_to_value_function(Q_conv_MC)
        V_conv_filt_MC = rl.fill_missing_sum_states(rl.filter_states(V_conv_MC))
        print(plot_dir)
        pl.plot_value_function(V_conv_filt_MC,
                               title = "Expanded state MC, " + str(decks) + " decks",
                               directory=plot_dir,
                               file_name = "3D_exp_MC_" + str(decks) + "_decks.png")
        
        # Extended state Q-learning
        Q_conv = rl.convert_to_sum_states(Q, env)
        V_conv = rl.convert_to_value_function(Q_conv)
        V_conv_filt = rl.fill_missing_sum_states(rl.filter_states(V_conv))
        pl.plot_value_function(V_conv_filt,
                               title = "Expanded state, " + str(decks) + " decks",
                               directory = plot_dir,
                               file_name = "3D_exp_" + str(decks) + "_decks.png")

        # Likewise make 3D plots for sumQ
        V_sum = rl.convert_to_value_function(sumQ)
        V_sum_filt = rl.fill_missing_sum_states(rl.filter_states(V_sum))
        pl.plot_value_function(V_sum_filt,
                               title = "Sum state, " + str(decks) + " decks",
                               directory = plot_dir,
                               file_name = "3D_sum_" + str(decks) + "_decks.png")
        # create line plots
        env_types = ["hand_MC", "hand", "sum"]
        fig, lgd = pl.plot_avg_reward_episode(directory, env_types, [str(decks)])
        fig.savefig("{}/avgReturnEp_ndeck{}.png".format(plot_dir, decks),
                                bbox_extra_artists=(lgd,), bbox_inches='tight')
        matplotlib.pyplot.close(fig)
