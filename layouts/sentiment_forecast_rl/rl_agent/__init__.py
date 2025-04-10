import sys
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.spaces import Box
import pandas as pd
import numpy as np


__original_stdout = sys.stdout
class __DualWriter:
    def __init__(self, file_name):
        self.file = open(file_name, 'w')
    
    def write(self, msg):
        __original_stdout.write(msg)  # This prints to the console
        self.file.write(msg)       # This writes to the file
    
    def flush(self):
        __original_stdout.flush()  # Ensure it flushes the console output
        self.file.flush()       # Ensure it flushes the file output

def __start_dual_write(path):
    sys.stdout = __DualWriter(path)

def __end_dual_write():
    sys.stdout = __original_stdout


def __set_first_na(df, value, column=None):
    if column is None:
        first_na_index = df[df.isna().any(axis=1)].index[0]
        df.loc[first_na_index] = value
        return df
    else:
        first_na_index = df[df[column].isna()].index[0]
        df.loc[first_na_index, column] = value
        return df
    
def __get_last_row(df, column=None):
    if column is None:
        last_valid_index = df.dropna().index[-1]
        return df.loc[last_valid_index]
    else:
        last_valid_index = df[df[column].notna()].index[-1]
        return df.loc[last_valid_index]



"""
FLAVOR: Single Asset (Plain)
Provide one row of the df per time-step
Reward is cumulative return


Requires columns:
unnormalized: Target (=Open), SPLIT
(various normalized, financial indicators)
"""
def _singleAsset_env_functions(df, lmbda=0.9, episodeLength=100):
    split_returns = {SPLIT: df[df["SPLIT"] == SPLIT]["Target"] for SPLIT in ["train", "test"]}
    split_rows = {SPLIT: df[df["SPLIT"] == SPLIT].drop(columns=["Target", "SPLIT"]).values for SPLIT in ["train", "test"]}

    number_of_indicators = df.shape[1] - 2

    def gen_gen(SPLIT):
        returns = split_returns[SPLIT]
        rows = split_rows[SPLIT]

        def state_gen():
            reward.actionNotebook = pd.DataFrame([float('nan')] * episodeLength, columns=["action"])
            reward.total = 0
            index = np.random.randint(0, len(returns) - episodeLength)
            episode_returns = returns.iloc[index:index + episodeLength]
            episode_rows = rows.iloc[index:index + episodeLength]

            reward.actionNotebook['returns'] = episode_returns
            for row in rows:
                yield row

        def reward(action):
            __set_first_na(reward.actionNotebook, action, column="action")
            last_row = __get_last_row(reward.actionNotebook)
            reward.total = last_row["action"] * last_row["returns"] + reward.total * lmbda
            return reward.total

        return {'state_gen': state_gen, 'reward': reward}


    action_space = Box(low=0, high=1, shape=(1,), dtype=np.float64)
    observation_space = Box(low=0, high=1, shape=(number_of_indicators,), dtype=np.float64)

    return {
        'train_gen_gen': lambda : gen_gen("train"),
        'test_gen_gen': lambda: gen_gen("test"),
        'action_space': action_space,
        'observation_space': observation_space,
    }


def get_env(env_functions, seed=None):
    class PortfolioEnv(Env):
        def __init__(self, seed=None, testing=False):
            super(PortfolioEnv, self).__init__()
            self.action_space = env_functions['action_space']
            self.observation_space = env_functions['observation_space']
            
            self.testing = testing
            self.train_gen_gen = env_functions.get('train_gen_gen', None)
            self.test_gen_gen = env_functions.get('test_gen_gen', None)
            self.gen_next()

        def gen_next(self):
            if self.testing:
                gen = self.train_gen_gen()
            else:
                gen = self.test_gen_gen()
            self.state_gen = gen['state_gen']
            self.log_and_reward = gen['reward']
            self.state = self.state_gen.__next__()

        def step(self, action):
            reward = self.log_and_reward(action)
            
            self.state = self.state_gen.__next__()
            terminated = False
            truncated = False if self.state is not None else True
            return self.state, reward, terminated, truncated, {}

        def reset(self, seed=None):
            super().reset(seed=seed)
            self.gen_next()
            return self.state, {}
        
    return {
        'train_env': DummyVecEnv([lambda: PortfolioEnv(seed=seed)]),
        'test_env': DummyVecEnv([lambda: PortfolioEnv(testing=True)])
    }



def train_model(model, timestamps, log_path=""):
    try:
        if log_path:
            __start_dual_write(log_path)
        model.learn(total_timesteps=timestamps, reset_num_timesteps=False)
    except Exception as e:
        print(f"Error in train_model: {e}")
        raise
    finally:
        __end_dual_write()



def forecast_agent(model, env_functions):
    test_env = get_env(env_functions)['test_env']
    obs = test_env.reset()
    rewards = []
    actions = []
    done = False
    i = 0
    while not done:
        i += 1
        if i % 100 == 0:
            print(f"Step {i}")
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, _ = test_env.step(action)
        rewards.append(reward)
    return { 'actions': actions, 'rewards': rewards }