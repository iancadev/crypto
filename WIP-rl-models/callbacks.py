from stable_baselines3.common.callbacks import BaseCallback

class ExplainedVarianceCallback(BaseCallback):
    def __init__(self, model, verbose=1):
        super(ExplainedVarianceCallback, self).__init__(verbose)
        self.explained_variances = []  # List to store historical values
        self.model = model

    def _on_step(self):
        print(self.model.get_metrics())
        return True  # Continue training

    def _on_rollout_end(self):
        # Called after each rollout (n_steps collected)
        pass

    def _on_training_end(self):
        # Write list to callbacks/cblk_explained_variance.csv
        with open("callbacks/cblk_explained_variance.csv", "w") as f:
            for ev in self.explained_variances:
                f.write(f"{ev}\n")

    def _on_training_step(self):
        # Called after each policy update
        # Access explained_variance from locals
        if "explained_variance" in self.locals:
            ev = self.locals["explained_variance"]
            self.explained_variances.append(ev)
        return True
    
    def plot(self, from_csv=False):
        if from_csv:
            with open("callbacks/cblk_explained_variance.csv", "r") as f:
                explained_variances = [float(line.strip()) for line in f]
        else:
            explained_variances = self.explained_variances
        import matplotlib.pyplot as plt
        plt.plot(explained_variances)
        plt.xlabel("Training step")
        plt.ylabel("Explained variance")
        plt.title("Explained variance over time")
        plt.show()
    

# explained_variance = ExplainedVarianceCallback()

class PPOExplainedVarianceCallback():
    def __init__(self, model):
        self.model = model

    def _on_step(self, **kwargs):
        metrics = self.model.get_metrics()
        explained_variance = metrics.get("explained_variance")
        if explained_variance is not None:
            print(f"Model {self.model.name} at step {self._step} has explained_variance: {explained_variance:.2f}")


class TensorboardCallback(BaseCallback):
    """ Logs the net change in cash between the beginning and end of each epoch/run. """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = self.training_env.envs[0]

    def _on_step(self) -> bool:
        print(self.env)
        # if self.env.done:
        #     net_change = self.env.list_networth[-1] - self.env.list_networth[0]
        #     self.logger.record("net_change", net_change)

        return True

from stable_baselines3.common.monitor import Monitor

class CustomMonitor(Monitor):
    def __init__(self, env, filename):
        super().__init__(env, filename)
        self.custom_metrics = {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Hypothetically add training metrics here (not directly available)
        if "explained_variance" in info:  # This won't work without extra work
            self.custom_metrics["explained_variance"] = info["explained_variance"]
        return obs, reward, terminated, truncated, info


class CustomLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Access training metrics like explained_variance from the model
        print(self.locals.keys())
        if "explained_variance" in self.locals:
            explained_variance = self.locals["explained_variance"]
            self.logger.record("train/explained_variance", explained_variance)
        return True

    def _on_rollout_end(self) -> None:
        # Optionally log something at the end of a rollout
        pass


import sys

original_value = sys.stdout
class DualWriter:
    def __init__(self, file_name):
        self.file = open(file_name, 'w')
    
    def write(self, msg):
        sys.__stdout__.write(msg)  # This prints to the console
        self.file.write(msg)       # This writes to the file
    
    def flush(self):
        sys.__stdout__.flush()  # Ensure it flushes the console output
        self.file.flush()       # Ensure it flushes the file output

# Create an instance of DualWriter and assign it to sys.stdout
def startDualWrite(fname):
    sys.stdout = DualWriter(fname)

# Reset sys.stdout to the original value
def stopDualWrite():
    sys.stdout = original_value


def plotExplainedVariance():
    with open("callbacks/monitor.txt", "r") as f:
        lines = f.readlines()
        timestep_lines = [line for line in lines if "total_timesteps" in line][1:]
        explained_variance_lines = [line for line in lines if "explained_variance" in line]
    timesteps = [int(line.split("|")[2].strip()) for line in timestep_lines]
    explained_variances = [float(line.split("|")[2].strip()) for line in explained_variance_lines]
    print(timesteps, explained_variances)
    import matplotlib.pyplot as plt
    plt.plot(timesteps, explained_variances)
    plt.xlabel("Timesteps")
    plt.ylabel("Explained variance")
    plt.title("Explained variance over time")
    plt.show()