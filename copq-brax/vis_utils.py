import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Patch
import matplotlib.ticker as ticker


y_labels ={0: "Return (k)",
           1: "Cost (k)",
           2: "TrainingEpCost",
           }

velocity_limit = {'hopper':740.2,
                  'ant':2622.2,
                  'humanoid':1414.9,
                  'walker2d':2341.5}

def visualize(
        model_list,
        label_list,
        colors,
        figsize = (10,2),
        anchor = (0, 0),
):
    env_list = ['hopper', 'walker2d', 'ant', 'humanoid']
    # env_list = ['hopper', 'walker2d']
    fig, ax = plt.subplots(2, len(env_list), figsize=figsize)
    ax = np.atleast_2d(ax)

    for k in range(2):
        for i, env in enumerate(env_list):
            for j, model in enumerate(model_list):
                lw = 3
                zorder = j

                data = np.load("./results/" + env + "/" + model +".npz", allow_pickle=True)
                x = data["x"][0]/1000000
                y = data["y"]

                cost = y[...,0]/1000.
                reward = y[...,1]/1000.
                assert np.nan not in reward

                if k == 0:
                    ax[k,i].set_title(env, fontsize=20)
                    ax[k,i].plot(x, reward.mean(0), color=colors[j], label=label_list[j], lw= lw, zorder=zorder)
                    te = reward.std(0)
                    ax[k,i].fill_between(x, reward.mean(0)-te, reward.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)
                    # ax[k,i].axhline(y=velocity_limit[env], xmin=0., xmax=3.3, lw = 3, linestyle='--', color='black', zorder = 200)
                
                    if env == 'humanoid':
                        ax[k,i].set_xlim(0, 3.072)
                    elif env == 'hopper':
                        ax[k,i].set_xlim(0, 1.024)
                    else:
                        ax[k,i].set_xlim(0, 1.024)
                    
                elif k == 1: 
                    ax[k,i].plot(x, cost.mean(0), color=colors[j], label=label_list[j], lw= lw, zorder=zorder)
                    te = cost.std(0)
                    ax[k,i].fill_between(x, cost.mean(0)-te, cost.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)
                    ax[k,i].axhline(y=velocity_limit[env]/1000, xmin=0., xmax=3.3, lw=3, alpha=0.3, linestyle='--', color="black", label = "cost constraint")
                    # ax[k,i].axhline(y=1.0, xmin=0., xmax=3.3, lw = 3, linestyle='--', color='black', zorder = 200)
                
                    if env == 'humanoid':
                        ax[k,i].set_xlim(0, 3.072)
                    elif env == 'hopper':
                        ax[k,i].set_xlim(0, 1.024)
                    else:
                        ax[k,i].set_xlim(0, 1.024)
                    # ax[k,i].set_ylim(0, 1.1)
                

            if k == 1:
                ax[k,i].set_xlabel("env_step (M)", fontsize=20)
            if i == 0:
                ax[k,i].set_ylabel(y_labels[k], fontsize=20)
            
            ax[k,i].tick_params(axis='both', labelsize=18)
            if i<3:
                ax[k,i].xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            else:
                ax[k,i].xaxis.set_major_locator(ticker.MultipleLocator(1.))
            ax[k,i].grid(True, zorder=-100)
            ax[k,i].set_facecolor('#F5F5F5')
            
    #ax[3,0].set_ylim(-100, 800)
    handles, labels = ax[0][0].get_legend_handles_labels()
    handles = [Patch(facecolor=colors[i], label=label_list[i]) for i in range(len(model_list))]


    fig.legend(handles,labels, loc='lower center', bbox_to_anchor=anchor, fontsize=18, ncol=5)
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.show()
    return fig


def ratio_visualize(
        model_list,
        label_list,
        colors,
        figsize = (10,2),
        anchor = (0, 0),
):
    env_list = ['hopper', 'walker2d', 'ant']

    fig, ax = plt.subplots(1, len(env_list), figsize=figsize)
    ax = np.atleast_2d(ax)

    for k in range(1):
        for i, env in enumerate(env_list):
            for j, model in enumerate(model_list):
                lw = 3
                zorder = j

                data = np.load("./boundary/" + env + "/" + model +".npz", allow_pickle=True)
                x = data["x"][0]/1000000
                y = data["y"]
                test_reward = y[...,1]
                test_cost = y[...,0]
                if env == 'humanoid':
                    train_cost = -np.cumsum(data["rewards"][:,16384:,0], -1)[:, :1007576*3][:,::9976*3]
                else:
                    train_cost = -np.cumsum(data["rewards"][:,16384:,0], -1)[:, :1007576][:,::9976]


                train_cost = train_cost.mean(0)/1000

                # print(train_cost.shape)
                

                if k == 0:
                    ax[k,i].set_title(env, fontsize=20)
                    ax[k,i].plot(train_cost, test_reward.mean(0)/1000, color=colors[j], label=label_list[j], lw= lw, zorder=zorder)
                    #te = test_reward.std(0)
                    # ax[k,i].fill_between(x, test_reward.mean(0)-te, test_reward.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)
                    # ax[k,i].axhline(y=velocity_limit[env], xmin=0., xmax=3.3, lw = 3, linestyle='--', color='black', zorder = 200)
                
                    if env == 'humanoid':
                        ax[k,i].set_xlim(0, 40.8)
                    else:
                        ax[k,i].set_xlim(0, 25.6)

                if k == 1:
                    ax[k,i].set_title(env, fontsize=20)
                    ax[k,i].plot(train_cost, test_cost.mean(0), color=colors[j], label=label_list[j], lw= lw, zorder=zorder)
                    #te = test_reward.std(0)
                    #ax[k,i].fill_between(x, test_reward.mean(0)-te, test_reward.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)
                    # ax[k,i].axhline(y=velocity_limit[env], xmin=0., xmax=3.3, lw = 3, linestyle='--', color='black', zorder = 200)
                
                    if env == 'humanoid':
                        ax[k,i].set_xlim(0, 40.8)
                    else:
                        ax[k,i].set_xlim(0, 25.6)

            if k == 0:
                ax[k,i].set_xlabel("training cost (k)", fontsize=20)
            if i == 0:
                ax[k,i].set_ylabel(y_labels[k], fontsize=20)
            
            ax[k,i].tick_params(axis='both', labelsize=18)
            ax[k,i].grid(True, zorder=-100)
            ax[k,i].set_facecolor('#F5F5F5')
            
    #ax[3,0].set_ylim(-100, 800)
    handles, labels = ax[0][0].get_legend_handles_labels()
    handles = [Patch(facecolor=colors[i], label=label_list[i]) for i in range(len(model_list))]


    fig.legend(handles,labels, loc='lower center', bbox_to_anchor=anchor, fontsize=18, ncol=5)
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.show()
    return fig


def compare_onpolicy(
        target_model,
        colors,
        figsize = (10,2),
        anchor = (0, 0),
):
    env_list = ['hopper', 'walker2d', 'ant', 'humanoid']
    fig, ax = plt.subplots(2, 4, figsize=figsize)
    ax = np.atleast_2d(ax)

    for k in range(2):
        for i, env in enumerate(env_list):
            lw = 3

            data = np.load("./boundary/"+ env + "/" + target_model +".npz", allow_pickle=True)
            x = data["x"][0]/1000000
            y = data["y"]
            test_reward = y[...,1]/1000
            test_cost = y[...,0]

            loaded = np.load(env + '_on_policy.npz', allow_pickle=True)
            # onpolicy_models = list(loaded.keys())
            # onpolicy_models = ["RCPO", "PPOSaute", "CUP", "PPOSimmerPID", "CPPOPID"]
            onpolicy_models = ["CUP", "RCPO", "PPOSimmerPID", "CPPOPID"]

            if k == 0:
                # for j, model in enumerate(onpolicy_models):
                for model in onpolicy_models:
                    j = onpolicy_models.index(model)
                    rewards = loaded[model].item()['return']/1000
                    if env == 'humanoid':
                        rewards = np.concatenate([rewards[:,::15], rewards[:,-1:]], -1)
                        ax[k,i].plot(x, rewards.mean(0), color=colors[j], label=model, lw= lw, zorder=j)
                        te = rewards.std(0)
                        ax[k,i].fill_between(x, rewards.mean(0)-te, rewards.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)

                    else:
                        rewards = np.concatenate([rewards[:,::5], rewards[:,-1:]], -1)
                        ax[k,i].plot(x, rewards.mean(0), color=colors[j], label=model, lw= lw, zorder=j)
                        te = rewards.std(0)
                        ax[k,i].fill_between(x, rewards.mean(0)-te, rewards.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)
                ax[k,i].set_title(env, fontsize=20)
                ax[k,i].plot(x, test_reward.mean(0), color=colors[-1], label="COP-Q (ours)", lw= 3, zorder=100)
                te = test_reward.std(0)
                ax[k,i].fill_between(x, test_reward.mean(0)-te, test_reward.mean(0)+te, alpha=0.2,lw=0, color=colors[-1], zorder=-100)
                #if k == 1:
                #ax[k,i].axhline(y=25, xmin=0., xmax=3.3, lw = 3, linestyle='--', color='black', label = "cost constraint", zorder = 200)
                
                if env == 'humanoid':
                    ax[k,i].set_xlim(0, 3.072)
                else:
                    ax[k,i].set_xlim(0, 1.024)

                # if env == 'ant':
                #     ax[k,i].set_ylim(-400, 4000)
            elif k == 1: 
                # for j, model in enumerate(onpolicy_models):
                for model in onpolicy_models:
                    j = onpolicy_models.index(model)
                    rewards = loaded[model].item()['cost']
                    if env == 'humanoid':
                        rewards = np.concatenate([rewards[:,::15], rewards[:,-1:]], -1)
                        ax[k,i].plot(x, rewards.mean(0), color=colors[j], label=model, lw= lw, zorder=j)
                        te = rewards.std(0)
                        ax[k,i].fill_between(x, rewards.mean(0)-te, rewards.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)

                    else:
                        rewards = np.concatenate([rewards[:,::5], rewards[:,-1:]], -1)
                        ax[k,i].plot(x, rewards.mean(0), color=colors[j], label=model, lw= lw, zorder=j)
                        te = rewards.std(0)
                        ax[k,i].fill_between(x, rewards.mean(0)-te, rewards.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)
                ax[k,i].plot(x, test_cost.mean(0), color=colors[-1], label="COP-Q (ours)", lw= 3, zorder=100)
                te = test_cost.std(0)
                ax[k,i].fill_between(x, test_cost.mean(0)-te, test_cost.mean(0)+te, alpha=0.2,lw=0, color=colors[-1], zorder=-100)
                ax[k,i].axhline(y=25, xmin=0., xmax=3.3, lw = 3, linestyle='--', color='black', label = "cost constraint", zorder = 200)
                
                if env == 'humanoid':
                    ax[k,i].set_xlim(0, 3.072)
                else:
                    ax[k,i].set_xlim(0, 1.024)
                ax[k,i].set_ylim(-20, 100.)
            
            if k == 1:
                ax[k,i].set_xlabel("env_step (million)", fontsize=20)
            if i == 0:
                ax[k,i].set_ylabel(y_labels[k], fontsize=20)
            
            ax[k,i].tick_params(axis='both', labelsize=18)
            ax[k,i].grid(True, zorder=-100)
            ax[k,i].set_facecolor('#F5F5F5')
            
    #ax[3,0].set_ylim(-100, 800)
    label_list = onpolicy_models + ["COP-Q (ours)"]
    handles, labels = ax[0][0].get_legend_handles_labels()
    handles = [Patch(facecolor=colors[i], label=label_list[i]) for i in range(len(label_list))]


    fig.legend(handles,labels, loc='lower center', bbox_to_anchor=anchor, fontsize=18, ncol=6)
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.show()
    return fig



def safenav_compare_onpolicy(
        colors,
        figsize = (10,2),
        anchor = (0, 0),
):
    env_list = ['SafetyPointButton2', 
                'SafetyPointGoal2',
                'SafetyCarButton2',
                'SafetyPointPush1'
                ]
    fig, ax = plt.subplots(2, len(env_list), figsize=figsize)
    ax = np.atleast_2d(ax)

    for k in range(2):
        for i, env in enumerate(env_list):
            lw = 2

            x = np.arange(0, 1, 0.002)
            loaded = np.load("./safe_navigation/" + env + '_onpolicy.npz', allow_pickle=True)
            onpolicy_models = ["PPOSimmerPID", "PPOSaute", "RCPO", "CUP", "CPPOPID"] # list(loaded.keys())

            if k == 0:
                for model in onpolicy_models:
                    j= onpolicy_models.index(model)
                    rewards = loaded[model].item()['return']
                    
                    #rewards = np.concatenate([rewards[:,::5], rewards[:,-1:]], -1)
                    ax[k,i].plot(x, rewards.mean(0), color=colors[j], label=model, lw= lw, zorder=j)
                    te = rewards.std(0)
                    print(env, model, "return: ", rewards.mean(0)[-1], rewards.std(0)[-1])
                    ax[k,i].fill_between(x, rewards.mean(0)-te, rewards.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)
                ax[k,i].set_title(env, fontsize=20)
                
                ax[k,i].set_xlim(0, 1.0)

                #ax[k,i].set_ylim(-400, 4000)
            elif k == 1: 
                for model in onpolicy_models:
                    j= onpolicy_models.index(model)
                    rewards = loaded[model].item()['cost']
                    ax[k,i].plot(x, rewards.mean(0), color=colors[j], label=model, lw= lw, zorder=j)
                    te = rewards.std(0)
                    print(env, model, "cost: ", rewards.mean(0)[-1], rewards.std(0)[-1])
                    ax[k,i].fill_between(x, rewards.mean(0)-te, rewards.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)
                    ax[k,i].axhline(y=10., xmin=0., xmax=1.02, lw = 2, alpha=0.5, linestyle='--', color='black', label = "cost constraint", zorder = 200)
                ax[k,i].set_xlim(0, 1.0)
                ax[k,i].set_ylim(-5, 50.)
            
            if k == 1:
                ax[k,i].set_xlabel("env_step (million)", fontsize=20)
            if i == 0:
                ax[k,i].set_ylabel(y_labels[k], fontsize=20)
            
            ax[k,i].tick_params(axis='both', labelsize=18)
            ax[k,i].grid(True, zorder=-100)
            ax[k,i].set_facecolor('#F5F5F5')
            
    #ax[3,0].set_ylim(-100, 800)
    label_list = onpolicy_models + ["COP-Q (ours)"]
    handles, labels = ax[0][0].get_legend_handles_labels()
    handles = [Patch(facecolor=colors[i], label=label_list[i]) for i in range(len(label_list))]


    fig.legend(handles,labels, loc='lower center', bbox_to_anchor=anchor, fontsize=20, ncol=5)
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.show()
    return fig



def safenav_compare_offpolicy(
        colors,
        figsize = (10,2),
        anchor = (0, 0),
):
    env_list = ['SafetyPointButton2', 
                'SafetyPointGoal2',
                'SafetyCarButton2',
                'SafetyPointPush1'
                ]
    
    model_list = ["SACPID", "CAL", "ORAC", "COP-Q"]
    fig, ax = plt.subplots(3, len(env_list), figsize=figsize)
    ax = np.atleast_2d(ax)

    x = np.arange(0, 1, 0.002)

    for k in range(3):

        for i, env in enumerate(env_list):
            lw = 2
            for j, model in enumerate(model_list):
                loaded = np.load("./safe_navigation/" + model + '.npz', allow_pickle=True)

                if k == 0:
                    rewards = loaded[env].item()['return']
                        
                    ax[k, i].plot(x, rewards.mean(0), color=colors[j], label=model, lw= lw, zorder=j)
                    te = rewards.std(0)
                    print(env, model, "return: ", rewards.mean(0)[-1], rewards.std(0)[-1])
                    ax[k, i].fill_between(x, rewards.mean(0)-te, rewards.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-j)
                    ax[k, i].set_title(env, fontsize=20)
                    
                    ax[k, i].set_xlim(0, 1.0)
                    ax[k, i].tick_params(axis='both', labelsize=18)
                    ax[k, i].grid(True, zorder=-100)
                    ax[k, i].set_facecolor('#F5F5F5')
                

                    #ax[k,i].set_ylim(-400, 4000)
                elif k == 1: 
                    rewards = loaded[env].item()['cost']
                    ax[k, i].plot(x, rewards.mean(0), color=colors[j], label=model, lw= lw, zorder=i)
                    te = rewards.std(0)
                    print(env, model, "cost: ", rewards.mean(0)[-1], rewards.std(0)[-1])
                    ax[k, i].fill_between(x, rewards.mean(0)-te, rewards.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-i)
                    ax[k, i].axhline(y=10., xmin=0., xmax=1.02, lw = 2, alpha=0.5, linestyle='--', color='black', label = "cost constraint", zorder = 200)
                    ax[k, i].set_xlim(0, 1.0)
                    ax[k, i].set_ylim(-5, 50.)
                    ax[k, i].tick_params(axis='both', labelsize=18)
                    ax[k, i].grid(True, zorder=-100)
                    ax[k, i].set_facecolor('#F5F5F5')
                

                elif k == 2: 
                    rewards = loaded[env].item()['train_cost']
                    ax[k, i].plot(x, rewards.mean(0), color=colors[j], label=model, lw= lw, zorder=i)
                    te = rewards.std(0)
                    print(env, model, "cost: ", rewards.mean(0)[-1], rewards.std(0)[-1])
                    ax[k, i].fill_between(x, rewards.mean(0)-te, rewards.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-i)
                    ax[k, i].axhline(y=10., xmin=0., xmax=1.02, lw = 2, alpha=0.5, linestyle='--', color='black', label = "cost constraint", zorder = 200)
                    ax[k, i].set_xlim(0, 1.0)
                    ax[k,i].set_ylim(-5, 50.)
                    ax[k, i].tick_params(axis='both', labelsize=18)
                    ax[k, i].grid(True, zorder=-100)
                    ax[k, i].set_facecolor('#F5F5F5')

                
            
                
                if k == 2:
                    ax[k, i].set_xlabel("env_step (million)", fontsize=20)
                if i == 0:
                    ax[k, i].set_ylabel(y_labels[k], fontsize=20)
            
            
    #ax[3,0].set_ylim(-100, 800)
    label_list = model_list[:-1] + ["COP-Q (ours)"]
    handles, labels = ax[0][0].get_legend_handles_labels()
    handles = [Patch(facecolor=colors[j], label=label_list[j]) for j in range(len(label_list))]


    fig.legend(handles,labels, loc='lower center', bbox_to_anchor=anchor, fontsize=20, ncol=5)
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.show()
    return fig


def safenav_bias_offpolicy(
        colors,
        figsize = (10,2),
        anchor = (0, 0),
):
    env_list = ['SafetyPointButton2', 
                'SafetyPointGoal2',
                'SafetyCarButton2',
                'SafetyPointPush1'
                ]
    
    model_list = ["SACPID", "CAL", "ORAC", "COP-Q"]
    fig, ax = plt.subplots(1, len(env_list), figsize=figsize)
    ax = np.atleast_2d(ax)

    x = np.arange(0, 1, 0.002)

    for k in range(1):

        for i, env in enumerate(env_list):
            lw = 2
            for j, model in enumerate(model_list):
                loaded = np.load("./safe_navigation/" + model + '.npz', allow_pickle=True)

                if k == 0: 
                    rewards = loaded[env].item()['estimate_cost'] - loaded[env].item()['cost']/10.
                    ax[k, i].plot(x, rewards.mean(0), color=colors[j], label=model, lw= lw, zorder=i)
                    te = rewards.std(0)
                    ax[k, i].set_title(env, fontsize=20)
                    # print(env, model, "cost: ", rewards.mean(0)[-1], rewards.std(0)[-1])
                    ax[k, i].fill_between(x, rewards.mean(0)-te, rewards.mean(0)+te, alpha=0.2,lw=0, color=colors[j], zorder=-i)
                    ax[k, i].axhline(y=10., xmin=0., xmax=1.02, lw = 2, alpha=0.5, linestyle='--', color='black', zorder = 200)
                    ax[k, i].set_xlim(0, 1.0)
                    ax[k,i].set_ylim(-2., 1.)
                    ax[k, i].tick_params(axis='both', labelsize=18)
                    ax[k, i].grid(True, zorder=-100)
                    ax[k, i].set_facecolor('#F5F5F5')

                
                ax[k, i].set_xlabel("env_step (million)", fontsize=20)
                if i == 0:
                    ax[k, i].set_ylabel('ValueBias', fontsize=20)
            
            
    #ax[3,0].set_ylim(-100, 800)
    label_list = model_list[:-1] + ["COP-Q (ours)"]
    handles, labels = ax[0][0].get_legend_handles_labels()
    handles = [Patch(facecolor=colors[j], label=label_list[j]) for j in range(len(label_list))]


    fig.legend(handles,labels, loc='lower center', bbox_to_anchor=anchor, fontsize=20, ncol=5)
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.show()
    return fig
