import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common import logger
from stable_baselines3.common.utils import set_random_seed
set_random_seed(0)
import os

class StockTradingEnv(gym.Env):
    def __init__(self, 
                df, 
                stock_dim,
                state_space,
                action_space,
                tech_indicator_list,
                initial_amount=1,
                reward_scaling=1,
                make_plots = True, 
                print_verbosity = 10,
                day = 0, 
                initial=True,
                previous_state=[],
                model_name = '',
                mode='',
                iteration=''):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
        self.data = self.df.loc[self.day,:]
        self.terminal = False
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name=model_name
        self.mode=mode 
        self.iteration=iteration
        # initalize state
        self.state = self._initiate_state()
        
        # initialize reward
        self.reward = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        # self.reset()
        self._seed(1)

    def _make_plot(self):
        import os
        plt.plot(self.asset_memory,'r')
        plt.savefig('../../models/reinforcement_learning/results/account_value_trade_{}.png'.format(self.episode))
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1 # on last day
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()            
            # end_total_asset = self.state[0]+ \
            #     sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_return = self.state[0]+sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))- 1 
            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory
            df_total_value['daily_return']=df_total_value['account_value'].pct_change(1)
            if df_total_value['daily_return'].std() !=0:
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ df_total_value['daily_return'].std()
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ['account_rewards']
            df_rewards['date'] = self.date_memory[:-1]
            
            # Save Sharpe and returns for epoch
            sharpes_dir = 'sharpes_by_epoch.csv'
            if((self.episode <= 2) & (not os.path.isfile(sharpes_dir))):
                pd.DataFrame(columns=['epoch', 'Sharpe']).to_csv(sharpes_dir, index=False)
            sharpes = pd.read_csv(sharpes_dir)
            if((len(sharpes.index) == 0) | (self.episode > sharpes['epoch'].max())):
                sharpes = sharpes.append(pd.DataFrame([[self.episode, sharpe]], columns=sharpes.columns), ignore_index=True)
                sharpes.to_csv(sharpes_dir, index=False)
            
            avg_return = df_total_value['daily_return'].mean()
            returns_dir = 'returns_by_epoch.csv'
            if((self.episode <= 2) & (not os.path.isfile(returns_dir))):
                pd.DataFrame(columns=['epoch', 'return']).to_csv(returns_dir, index=False)
            returns = pd.read_csv(returns_dir)
            if((len(returns.index) == 0) | (self.episode > returns['epoch'].max())):
                returns = returns.append(pd.DataFrame([[self.episode, avg_return]], columns=returns.columns), ignore_index=True)
                returns.to_csv(returns_dir, index=False)
            
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"total_reward: {tot_return:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value['daily_return'].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name!='') and (self.mode!=''):
                df_actions = self.save_action_memory()
                df_actions.to_csv('../../models/reinforcement_learning/results/actions_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration))
                df_total_value.to_csv('../../models/reinforcement_learning/results/account_value_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                df_rewards.to_csv('../../models/reinforcement_learning/results/account_rewards_{}_{}_{}.csv'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.plot(self.asset_memory,'r')
                plt.savefig('../../models/reinforcement_learning/results/account_value_{}_{}_{}.png'.format(self.mode,self.model_name, self.iteration),index=False)
                plt.close()

            # Add outputs to logger interface
            #logger.record("environment/portfolio_value", end_total_asset)
            #logger.record("environment/total_reward", tot_reward)
            #logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            #logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, {}

        else: # not on the last day

            actions = np.round(actions) #actions initially is scaled between -1 to 1
            actions = (actions.astype(int)) #convert into integer because we can't by fraction of shares
            
            # Reset positions before buying/selling again
            self.state[(self.stock_dim+1):(self.stock_dim*2+1)] = [0] * self.stock_dim
            
            argsort_actions = np.argsort(actions)
                
            for index in argsort_actions:
                self.state[index+self.stock_dim+1] += actions[index]
                self.trades+=1
                
            self.actions_memory.append(actions)
            
            #state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day,:]    
            self.state =  self._update_state()
            
            # Calculates how trades performed
            returns = np.array(self.state[1:(self.stock_dim+1)])
            actions = np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)])
            end_total_asset = self.state[0]+ sum(returns * actions)
            self.state[0] = end_total_asset
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            df_total_value = pd.DataFrame(self.asset_memory)
            
            df_total_value.columns = ['account_value']
            df_total_value['date'] = self.date_memory
            df_total_value['daily_return']=df_total_value['account_value'].pct_change(1)
            if(self.day != 1):
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ df_total_value['daily_return'].std()
            else:
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()
                
            correct_minus_wrong = 0
            for i in range(len(returns)):
                if((actions[i] == 1) & (returns[i] > 0)):
                    correct_minus_wrong = correct_minus_wrong + 1
                elif((actions[i] == 1) & (returns[i] <= 0)):
                    correct_minus_wrong = correct_minus_wrong - 1
                elif((actions[i] == -1) & (returns[i] > 0)):
                    correct_minus_wrong = correct_minus_wrong - 1
                elif((actions[i] == -1) & (returns[i] <= 0)):
                    correct_minus_wrong = correct_minus_wrong + 1
                elif(actions[i] == 0):
                    pass
                else:
                    raise ValueError(str(actions[i]) + ' is not a valid action')
            
            # self.reward = end_total_asset - self.state[0]
            self.reward = correct_minus_wrong
            self.rewards_memory.append(self.reward)
            self.reward = self.reward*self.reward_scaling
        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        #initiate state
        self.state = self._initiate_state()
        
        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0]+ \
                sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory = [previous_total_asset]

        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.trades = 0
        self.terminal = False 
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        
        self.episode+=1

        return self.state
    
    # Returns the state
    def render(self, mode='human',close=False):
        return self.state

    def _initiate_state(self):
        days_fwd=1
        if self.initial:
            # For Initial State
            if len(self.df['Ticker'].unique())>1:
                # for multiple stock
                state = [self.initial_amount] + \
                         self.data['Chg_'+str(days_fwd)+'D_Fwd'].values.tolist() + \
                        [0]*self.stock_dim  + \
                         sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            else:
                # for single stock
                state = [self.initial_amount] + \
                        [self.data['Chg_'+str(days_fwd)+'D_Fwd']] + \
                        [0]*self.stock_dim  + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        else:
            #Using Previous State
            if len(self.df['Ticker'].unique())>1:
                # for multiple stock
                state = [self.previous_state[0]] + \
                         self.data['Chg_'+str(days_fwd)+'D_Fwd'].values.tolist() + \
                         self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]  + \
                         sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])
            else:
                # for single stock
                state = [self.previous_state[0]] + \
                        [self.data['Chg_'+str(days_fwd)+'D_Fwd']] + \
                        self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]  + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        return state

    def _update_state(self):
        days_fwd=1
        if len(self.df['Ticker'].unique())>1:
            # for multiple stock
            state =  [self.state[0]] + \
                      self.data['Chg_'+str(days_fwd)+'D_Fwd'].values.tolist() + \
                      list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                      sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list ], [])

        else:
            # for single stock
            state =  [self.state[0]] + \
                     [self.data['Chg_'+str(days_fwd)+'D_Fwd']] + \
                     list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
                     sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
                          
        return state

    def _get_date(self):
        if len(self.df['Ticker'].unique())>1:
            date = self.data['Date'].unique()[0]
        else:
            date = self.data['Date']
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        if len(self.df['Ticker'].unique())>1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']
            
            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data['Ticker'].values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs