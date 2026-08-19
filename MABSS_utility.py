import numpy as np
import pandas as pd
import yfinance as yf
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle as pkl
import keras
import json
from pathlib import Path
from typing import Dict, Any
import multiprocessing as mp
from functools import partial
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import os
from datetime import datetime
import re
import gc
import keras.backend as K
from scipy.linalg import cholesky, solve_triangular



# Data preparation

def window_time_series(ts, window_size):
  """
  Creates a sliding window view of a time series.

  Args:
    ts: A pandas Series representing the time series.
    window_size: The size of the sliding window.

  Returns:
    A list of windows, where each window is a pandas Series.
  """

  windows = []
  targets = []
  for i in range(len(ts) - window_size):
    window = ts.iloc[i:i + window_size]
    windows.append(window)
    targets.append(ts.iloc[i + window_size])
  return np.array(windows), np.array(targets)

def create_dense_model(random_seed: int, input_dim: int):
  model = keras.Sequential(
    [
      keras.Input(shape=(input_dim,)),
      keras.layers.Dense(64,
      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed)),
      keras.layers.Dense(64,
      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed)),
      keras.layers.Dense(1,
      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed))
    ]
  )
  model.compile(optimizer='adam', loss='mse', metrics=['mse'])
  return model

def create_model(arch: str, random_seed: int, input_dim: int):
    """Unified MLP/RNN/CNN factory."""
    if arch == 'mlp':
        model = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            keras.layers.Dense(64, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed)),
            keras.layers.Dense(64, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed)),
            keras.layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed))
        ])
    elif arch == 'rnn':
        model = keras.Sequential([
            keras.Input(shape=(input_dim, 1)),
            keras.layers.SimpleRNN(64, return_sequences=True, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed)),
            keras.layers.SimpleRNN(64, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed)),
            keras.layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed))
        ])
    elif arch == 'cnn':
        model = keras.Sequential([
            keras.Input(shape=(input_dim, 1)),
            keras.layers.Conv1D(64, 3, activation='relu', kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed)),
            keras.layers.Conv1D(64, 3, activation='relu', kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed)),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(stddev=0.01, seed=random_seed))
        ])
    else:
        raise ValueError(f"Unknown arch: {arch}")
    
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model


def load_price_series(ticker: str, start: str) -> pd.Series:
  data = yf.Ticker(ticker).history(start=start)
  return data['Close'].dropna()

def compute_returns_from_preds(preds: pd.DataFrame) -> pd.DataFrame:
    actual_returns = preds['target'].pct_change()
    
    pct = pd.DataFrame(index=preds.index)
    pct['target'] = actual_returns
    
    model_cols = [c for c in preds.columns if c != 'target']
    for col in model_cols:
        # Expected return = (Predicted Price at t - Actual Price at t-1) / Actual Price at t-1
        pct[col] = (preds[col] - preds['target'].shift(1)) / preds['target'].shift(1)
    
    return pct.dropna()

def walk_forward_training(data, embedding_len, window_size, config, epochs=10):
    """
    Enhanced: Multi-arch support via CONFIG flags.
    Returns preds_df with N_ARMS = sum(enabled * N_SEEDS_PER_ARCH) models.
    """
    # Build arm list dynamically
    arms_list = []
    if config.get('MLP', False):
        arms_list += [('mlp', m) for m in range(config['N_SEEDS_PER_ARCH'])]
    if config.get('RNN', False):
        arms_list += [('rnn', m) for m in range(config['N_SEEDS_PER_ARCH'])]
    if config.get('CNN', False):
        arms_list += [('cnn', m) for m in range(config['N_SEEDS_PER_ARCH'])]

    n_models = len(arms_list)
    print(f"Generating {n_models} arms")

    Xdata, Ydata = window_time_series(data, embedding_len)
    trainlen = 4 * window_size
    n_samples = len(Xdata)

    predictions = {f"model_{i}": [] for i in range(n_models)}
    predictionstarget = []

    for window in tqdm(range(trainlen - 1, n_samples - window_size, window_size)):
        train_start = window - trainlen + 1
        train_end = window + 1
        test_start = window + 1
        test_end = window + 1 + window_size

        traindata = Xdata[train_start:train_end]
        traintarget = Ydata[train_start:train_end].reshape(-1, 1)
        testdata = Xdata[test_start:test_end]
        testtarget = Ydata[test_start:test_end].reshape(-1, 1)

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        traindata_scaled = x_scaler.fit_transform(traindata)
        testdata_scaled = x_scaler.transform(testdata)
        traintarget_scaled = y_scaler.fit_transform(traintarget)
        testtarget_scaled = y_scaler.transform(testtarget)

        predictionstarget.append(testtarget.reshape(1, -1))

        # Reshape for seq models: (batch, timesteps, 1)
        traindata_seq = traindata_scaled.reshape(traindata_scaled.shape + (1,))
        testdata_seq = testdata_scaled.reshape(testdata_scaled.shape + (1,))

        for i, (arch, seed) in enumerate(arms_list):
            model = create_model(arch, seed, embedding_len)

            if arch == 'mlp':
                model.fit(traindata_scaled, traintarget_scaled, epochs=epochs, verbose=0)
                pred_scaled = model.predict(testdata_scaled, verbose=0)
            else:  # RNN/CNN
                model.fit(traindata_seq, traintarget_scaled, epochs=epochs, verbose=0)
                pred_scaled = model.predict(testdata_seq, verbose=0)

            pred = y_scaler.inverse_transform(pred_scaled)
            predictions[f"model_{i}"].append(pred.reshape(1, -1))
            del model
        K.clear_session()
        gc.collect()

    # Concat
    for i in range(n_models):
        predictions[f"model_{i}"] = np.concatenate(predictions[f"model_{i}"], axis=1)
    predictionstarget = np.concatenate(predictionstarget, axis=1)

    preds_df = pd.DataFrame(
    np.hstack([predictionstarget.T] + [predictions[f"model_{i}"].T for i in range(n_models)]),
    columns=["target"] + [f"model_{i}" for i in range(n_models)],
    index=data.iloc[-predictionstarget.shape[1]:].index # Fix: Use predictionstarget.shape[1] for index length
    )
    preds_df['mean'] = preds_df[[f"model_{i}" for i in range(n_models)]].mean(axis=1)

    return preds_df


 # Version handling

def make_experiment_id(config: dict) -> str:
  """Create a short, stable ID from key hyperparameters."""
  key_fields_pred = [
      "TICKER",
      "START_DATE",
      "WINDOW_SIZE",
      "MLP",
      "RNN",
      "CNN",
      "MODEL_EMBEDDING",
      "EPOCHS",
      "N_SEEDS_PER_ARCH",
  ]

  key_fields_bandit = [
    "TICKER",
    "START_DATE",
    "N_ARMS",
    "WINDOW_SIZE",
    "BANDIT_EMBEDDING",
    "METRIC",
    "ALPHA",
    "GAMMA",
    "EPISODES",
    "POLICY",
  ]
  # Only keep keys that exist in CONFIG
  compact_pred = {k: config[k] for k in key_fields_pred if k in config}
  compact_bandit = {k: config[k] for k in key_fields_bandit if k in config}
  # JSON → stable string, then replace non-filename chars
  raw_pred = json.dumps(compact_pred, sort_keys=True)
  raw_bandit = json.dumps(compact_bandit, sort_keys=True)
  safe_pred = (
      raw_pred.replace(" ", "")
          .replace("{", "")
          .replace("}", "")
          .replace('"', "")
          .replace(":", "-")
          .replace(",", "_")
          .replace(".", "p")
  )
  safe_bandit = (
      raw_bandit.replace(" ", "")
          .replace("{", "")
          .replace("}", "")
          .replace('"', "")
          .replace(":", "-")
          .replace(",", "_")
          .replace(".", "p")
  )
  return safe_pred, safe_bandit

def get_experiment_paths(config: dict, base_dir: str):
  pred_id, bandit_id = make_experiment_id(config)
  pred_dir = base_dir / pred_id
  bandit_dir = base_dir / bandit_id

  paths = {
      "pred_dir": pred_dir,
      "bandit_dir": bandit_dir,
      "preds_csv": pred_dir / "preds.csv",
      "pctchange_csv": pred_dir / "pctchange_preds.csv",
      "rewards_npy": bandit_dir / "rewards_matrix.npy",
      "contexts_npy": bandit_dir / "contexts_matrix.npy",
      "results_pkl": bandit_dir / "bandit_results.pkl",
      "multirun_results_pkl": bandit_dir / "multirun_results.pkl",
  }
  return paths  

def experiment_in_memory(paths: dict) -> bool:
    """Return True if all required artifacts for this experiment are saved."""
    required_files = [
        paths["preds_csv"],
        paths["pctchange_csv"],
        paths["rewards_npy"],
        paths["contexts_npy"],
        paths["results_pkl"],  # include if you only want to skip everything when bandit is also done
        paths["multirun_results_pkl"],
    ]
    return all(p.exists() for p in required_files) 

class ExperimentLoader:
    def __init__(self, config: dict, base_dir: Path):
        self.config = config
        self.pred_id, self.bandit_id = make_experiment_id(config)
        self.pred_dir = base_dir / self.pred_id
        self.bandit_dir = base_dir / self.bandit_id
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.bandit_dir.mkdir(parents=True, exist_ok=True)
        
        self.paths = {
            "preds": self.pred_dir / "preds.csv",
            "pctchange_preds": self.pred_dir / "pctchange_preds.csv",
            "contexts": self.bandit_dir / "contexts.npy",
            "rewards": self.bandit_dir / "rewards.npy",
            "rewards_df": self.bandit_dir / "rewards_df.csv",
            "results": self.bandit_dir / "results.pkl",
            "multirun_results": self.bandit_dir / "multirun_results.pkl"
        }
        
        self._cache = {}  # loaded objects

    def exists(self, key: str) -> bool:
        """Check if specific artifact exists."""
        return self.paths[key].exists()

    def is_fully_cached(self) -> bool:
        """Check if ALL artifacts exist."""
        return all(self.exists(k) for k in self.paths)

    def load(self, key: str, force_reload: bool = False) -> Any:
        """Load specific artifact. Returns cached object if available."""
        if not force_reload and key in self._cache:
            return self._cache[key]
        
        path = self.paths[key]
        if not path.exists():
            raise FileNotFoundError(f"{key} not found: {path}")
        
        if key == "preds" or key == "rewards_df":
            obj = pd.read_csv(path, index_col=0)
        elif key == "pctchange_preds":
            obj = pd.read_csv(path, index_col=0)
            obj.index = pd.to_datetime(obj.index, utc=True)
        elif key == "contexts" or key == "rewards":
            obj = np.load(path)
        elif key == "results" or key == "multirun_results":
            with open(path, "rb") as ff:
                obj = pkl.load(ff)
        else:
            raise ValueError(f"Unknown artifact: {key}")
            
        self._cache[key] = obj
        return obj

    def load_all(self) -> Dict[str, Any]:
        """Load everything that exists, return dict."""
        available = {}
        for key in self.paths:
            if self.exists(key):
                available[key] = self.load(key)
        return available

    def save(self, data: Dict[str, Any]):
        """Save dict of {key: object} to disk."""
        for key, obj in data.items():
            path = self.paths[key]
            if key == "preds" or key == "rewards_df":
                obj.to_csv(path)
            elif key == "pctchange_preds":
                obj.to_csv(path)
            elif isinstance(obj, np.ndarray):
                np.save(path, obj)
            elif key == "results" or key == "multirun_results":
                with open(path, "wb") as ff:
                    pkl.dump(obj, ff)
            else:
                raise ValueError(f"Cannot save {key}: {type(obj)}")

# Bandit's auxiliary functions

def reward_function(predictions, targets, metric="l1", prob_scale=50.0, eps=1e-6):
    """
    predictions: array-like, shape (N, K) or (N,)   (predicted returns, one column per model/arm)
    targets:     array-like, shape (N,) or (N, 1)  (true returns)
    metric:      'l1', 'l2', or 'crossentropy'
    prob_scale:  scale factor for mapping returns -> probabilities (crossentropy)
    """

    # Convert to NumPy arrays
    pred_arr = predictions.values if hasattr(predictions, "values") else np.asarray(predictions)
    targ_arr = targets.values if hasattr(targets, "values") else np.asarray(targets)
    
    # Handle scalars → 0D → make 1D
    if pred_arr.ndim == 0:
        pred_arr = pred_arr[None]
    if targ_arr.ndim == 0:
        targ_arr = targ_arr[None]

    # Ensure 2D shapes:
    #   pred_arr: (N, K)
    #   targ_arr: (N, 1) so it broadcasts across columns
    if pred_arr.ndim == 1:
        pred_arr = pred_arr[:, None]      # (N,) -> (N, 1)

    if targ_arr.ndim == 1:
        targ_arr = targ_arr[:, None]      # (N,) -> (N, 1)

    # Safety check
    if pred_arr.shape[0] != targ_arr.shape[0]:
        raise ValueError(f"Incompatible first dimensions: {pred_arr.shape} vs {targ_arr.shape}")

    if metric == "l1":
        # (N, K) - (N, 1) -> (N, K)
        return -np.abs(pred_arr - targ_arr)

    elif metric == "l2":
        # (N, K) - (N, 1) -> (N, K)
        return -np.square(pred_arr - targ_arr)

    elif metric == "crossentropy":
        # Direction classification: up (1) vs down (0), broadcast along columns
        true_classes = (targ_arr > 0).astype(float)    # (N, 1)

        # Map signed prediction to probability via sigmoid per model
        logits = prob_scale * pred_arr                 # (N, K)
        probs = 1.0 / (1.0 + np.exp(-logits))          # (N, K)

        # Clip to avoid log(0)
        probs = np.clip(probs, eps, 1.0 - eps)

        # Binary cross-entropy per (N, K)
        ce_loss = -(true_classes * np.log(probs) +
                    (1.0 - true_classes) * np.log(1.0 - probs))
        return -ce_loss   # reward = -loss, shape (N, K)

    else:
        raise ValueError(f"Invalid metric: {metric}")

def build_rewards_and_contexts(pct_change_preds: pd.DataFrame,
                               metric: str,
                               bandit_embedding: int,
                               n_arms: int):
    """
    Returns:
        rewards_df: DataFrame with columns ['target', 'model_0', ..., 'model_{K-1}', 'mean']
        contexts_matrix: np.ndarray, the context windows per time step
        rewards_matrix: np.ndarray, aligned rewards per arm per time step
    """
    target = pct_change_preds["target"].values  # shape (T,)
    model_cols = [f"model_{n}" for n in range(n_arms)]
    preds_mat = pct_change_preds[model_cols].values  # shape (T, K)

    rewards_mat = reward_function(preds_mat, target, metric=metric)  # (T, K)

    rewards_df = pd.DataFrame(
        np.column_stack([target, rewards_mat]),
        index=pct_change_preds.index,
        columns=["target"] + model_cols,
    )
    rewards_df["mean"] = reward_function(pct_change_preds["mean"],target,metric=metric)


    # Build contexts and rewards matrices using window_time_series
    context_list = []
    reward_list = []

    for col in model_cols:
        arm_contexts, arm_rewards = window_time_series(
            rewards_df[col], bandit_embedding
        )
        context_list.append(arm_contexts)  # shape (T_ctx, bandit_embedding)
        reward_list.append(arm_rewards)    # shape (T_ctx,)

    # Stack into shape (T_ctx, K, bandit_embedding) and (T_ctx, K)
    contexts_matrix = np.stack(context_list, axis=1)
    rewards_matrix = np.stack(reward_list, axis=1)

    return rewards_df, contexts_matrix, rewards_matrix

# Bandit classes 

class SoftmaxPolicyCMAB():
    """
    A class implementing the Softmax policy for Contextual Multi-Armed Bandits (CMAB).
    """
    def __init__(self,K_arms,context_dim,alpha,gamma):
        """
        Initializes the SoftmaxPolicyCMAB class.

        Args:
            K_arms: The number of arms (actions) available.
            context_dim: The dimension of the context vector.
            alpha: The learning rate.
            gamma: The discount factor.
        """
        self.K_arms=K_arms # Number of arms
        self.theta=np.zeros([K_arms,context_dim]) # Parameters for each arm (initialized to 0)
        self.a=alpha # Learning rate
        self.g=gamma # Discount factor
        self.avarage_reward=0 # Average reward (initialized to 0)
        self.num_selected=0 # Number of times an arm has been selected (initialized to 0)
        self.context_dim = context_dim
        
    def select_arm(self, xarrays):
        """
        xarrays: list/array of context vectors for each arm, shape (K, d).
        Returns: np.array of probabilities over arms (shape (K,)).
        """
        x = np.asarray(xarrays)              # (K, d)
        logits = (self.theta * x).sum(axis=1)  # theta_k · x_k

        # Numerical stability: subtract max
        logits = logits - np.max(logits)

        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        return probs

    def reward_update(self, xarrays, probs, arm_selected, reward):
        x = np.asarray(xarrays)
        probs = np.asarray(probs)

        self.num_selected += 1
        if self.g is None:
            g = 1.0 / self.num_selected
        else:
            g = self.g

        baselined_reward = reward - self.avarage_reward

        for k in range(self.K_arms):
            indicator = 1.0 if k == arm_selected else 0.0
            grad_logpi = (indicator - probs[k]) * x[k]
            self.theta[k] += self.a * baselined_reward * grad_logpi

        self.avarage_reward += g * (reward - self.avarage_reward)

    def reset(self):
        self.theta = np.zeros((self.K_arms, self.context_dim))
        self.average_reward = 0
        self.num_selected = 0

class UCBPolicyCMAB():
    def __init__(self, K_arms, d, alpha=1.0, gamma = None):
        self.K_arms = K_arms
        self.d = d
        self.alpha = alpha
        
        # Track the INVERSE of the covariance matrix directly
        self.A_inv = [np.eye(d) for _ in range(K_arms)]
        self.b = [np.zeros((d, 1)) for _ in range(K_arms)]

    def select_arm(self, x_arrays):
        x = np.asarray(x_arrays)
        ucb_values = np.zeros(self.K_arms)
        
        for k in range(self.K_arms):
            x_k = x[k].reshape(-1, 1)
            
            # Directly use the tracked inverse (No np.linalg.inv needed!)
            A_inv_k = self.A_inv[k]
            
            # Calculate expected reward and confidence bound
            theta_k = A_inv_k @ self.b[k]
            p_k = theta_k.T @ x_k
            V_k = x_k.T @ A_inv_k @ x_k
            
            ucb_k = p_k + self.alpha * np.sqrt(V_k)
            ucb_values[k] = ucb_k.item()
            
        arm = int(np.argmax(ucb_values))
        probs = np.zeros(self.K_arms)
        probs[arm] = 1.0
        return probs

    def reward_update(self, x_arrays, probs, arm_selected, reward):
        # Extract the context vector for the chosen arm
        x_k = np.asarray(x_arrays)[arm_selected].reshape(-1, 1)
        
        # 1. Update b
        self.b[arm_selected] += reward * x_k
        
        # 2. Sherman-Morrison update for A_inv
        A_inv = self.A_inv[arm_selected]
        A_inv_x = A_inv @ x_k
        numerator = A_inv_x @ A_inv_x.T
        denominator = 1.0 + (x_k.T @ A_inv_x).item()
        
        # Apply the update rule
        self.A_inv[arm_selected] = A_inv - (numerator / denominator)
        
    def reset(self):
        # Reset the tracked inverse covariance matrix back to the identity matrix
        self.A_inv = [np.eye(self.d) for _ in range(self.K_arms)]
        
        # Reset the target vector b back to zeros
        self.b = [np.zeros((self.d, 1)) for _ in range(self.K_arms)]


class ThompsonSamplingPolicyCMAB():
    def __init__(self, K_arms, context_dim, alpha, gamma=None, lam=1.0):
        self.K_arms = K_arms
        self.d = context_dim
        self.v = alpha  # Variance multiplier (exploration parameter)
        self.lam = lam  # Regularization parameter (lambda)
        self.g = gamma

        # Track the precision matrix A and target vector b directly
        # A is initialized as lambda * I
        self.A = [self.lam * np.eye(self.d) for _ in range(self.K_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(self.K_arms)]

    def select_arm(self, x_arrays):
        x = np.asarray(x_arrays)  # (K, d)
        sampled_values = np.zeros(self.K_arms)

        for k in range(self.K_arms):
            # 1. Compute Cholesky decomposition: A = L L^T
            # L is lower-triangular. This is highly optimized and numerically stable.
            L = cholesky(self.A[k], lower=True)
            
            # 2. Fast posterior mean: mu = L^{-T} (L^{-1} b_k)
            # Step A: Forward substitution -> y = L^{-1} b
            y = solve_triangular(L, self.b[k], lower=True)  
            # Step B: Backward substitution -> mu = L^{-T} y
            mu = solve_triangular(L.T, y, lower=False)      

            # 3. Sample from standard normal distribution
            z = np.random.standard_normal(size=(self.d, 1))
            
            # Step C: Transform sample to have covariance A^{-1}
            # w = L^{-T} z (Backward substitution)
            w = solve_triangular(L.T, z, lower=False)       
            sample = mu + self.v * w

            # 4. Calculate payoff estimate for the arm
            x_k = x[k].reshape(-1, 1)
            sampled_values[k] = float(sample.T @ x_k)

        # Select the arm with the highest sampled value
        arm = int(np.argmax(sampled_values))
        probs = np.zeros(self.K_arms)
        probs[arm] = 1.0
        return probs

    def reward_update(self, x_arrays, probs, arm_selected, reward):
        # Extract the context vector for the chosen arm
        x_k = np.asarray(x_arrays)[arm_selected].reshape(-1, 1)
        
        # Standard ridge regression update for A and b
        self.A[arm_selected] += x_k @ x_k.T
        self.b[arm_selected] += reward * x_k

    def reset(self):
        # Reset the precision matrix A back to lambda * I
        self.A = [self.lam * np.eye(self.d) for _ in range(self.K_arms)]
        
        # Reset the target vector b back to zeros
        self.b = [np.zeros((self.d, 1)) for _ in range(self.K_arms)]
        
        # Reset tracking variables (if you are utilizing these elsewhere)
        self.average_reward = 0.0
        self.num_selected = 0

# Outdated
# class SoftmaxWalkForwardEnvCMAB():
#     """
#     A class that simulates a contextual multi-armed bandit environment using a walk-forward approach.

#     This class uses a Softmax policy to select arms based on context vectors and rewards.
#     It simulates a walk-forward training and testing process, where the policy is updated
#     incrementally over time using a sliding window.

#     Args:
#         context (np.ndarray): The context vectors for each time step.
#         X_data (np.ndarray): The predictions made by each model for each time step.
#         Y_data (np.ndarray): The target values for each time step.
#         rewards (np.ndarray): The rewards for each arm at each time step.
#         reward_metric (string): The metric used to calculate the reward.
#         k (int): The number of arms (models) available.
#         d (int): The dimension of the context vectors.
#         alpha (float): The learning rate for the Softmax policy.
#         gamma (float, optional): The discount factor for the average reward. Defaults to None.
#         window_size (int, optional): The size of the sliding window for walk-forward training. Defaults to 252.
#         incremental (bool, optional): Whether to use incremental training. Defaults to False.
#         shuffle (bool, optional): Whether to shuffle the training data. Defaults to False.

#     Attributes:
#         policy (SoftmaxPolicyCMAB): The Softmax policy used to select arms.
#         x (np.ndarray): The context vectors for each time step.
#         r (np.ndarray): The rewards for each arm at each time step.
#         X_data (np.ndarray): The predictions made by each model for each time step.
#         Y_data (np.ndarray): The target values for each time step.
#         w (int): The size of the sliding window for walk-forward training.
#         incremental (bool): Whether to use incremental training.
#         shuffle (bool): Whether to shuffle the training data.

#     Methods:
#         calc_reward(prediction, target): Calculates the reward for a given prediction and target.
#         run(episodes): Runs the walk-forward simulation for a specified number of episodes.
#     """
#     def __init__(self,context,X_data,Y_data,rewards,reward_metric,
#                  k,d,alpha,gamma=None,window_size=252,incremental=False,shuffle=False,reset=False):
#         """Initializes the SoftmaxWalkForwardEnvCMAB class."""
#         self.policy=SoftmaxPolicyCMAB(K_arms=k,context_dim=d,alpha=alpha,gamma=gamma) # Initialize the Softmax policy
#         self.x=context # Context vectors
#         self.r=rewards # Rewards for each arm
#         self.metric=reward_metric # Reward function
#         self.X_data=X_data # Predictions made by each model
#         self.Y_data=Y_data # Target values
#         self.w=window_size # Window size for walk-forward training
#         self.incremental=incremental # Whether to use incremental training
#         self.shuffle=shuffle # Whether to shuffle the training data
#         self.reset=reset # Whether to reset the policy weights

#     def calc_reward(self,predictions,targets):
#         """Calculates the reward for a given prediction and target."""
#         return reward_function(predictions, targets, metric=self.metric)

#     def reset_policy_weight(self):
#         """Resets the policy weights to zero."""
#         self.policy.theta=np.zeros_like(self.policy.theta) # Reset policy weights to zero

#     def run(self,episodes):
#         """Runs the walk-forward simulation for a specified number of episodes."""
#         total_reward=[] # List to store the total reward obtained in each episode
#         total_reward_greedy=[] # List to store the total reward obtained using greedy arm selection in each episode
#         total_reward_average=[] # List to store the total reward obtained using average arm selection in each episode
#         mean_pred_reward = [] # List to store the total reward obtained using the mean of predictions
#         arm_selected=np.zeros(self.policy.K_arms) # Array to store the number of times each arm was selected
#         arm_selected_greedy=np.zeros_like(arm_selected) # Array to store the number of times each arm was selected using greedy selection
#         best_arms=[] # List to store the index of the best arm selected in each time step
#         best_arms_greedy=[] # List to store the index of the best arm selected using greedy selection in each time step
#         weighted_predictions=[] # List to store the weighted predictions made by the policy
#         control=[]
#         len_window=(len(self.x)%self.w)+4*self.w # Length of the training window
#         for i in tqdm(range(len_window,len(self.x),self.w)):
#             if self.incremental:
#                 x_train=self.x[:i]
#                 r_train=self.r[:i]
#             else:
#                 x_train=self.x[(i-len_window):i]
#                 r_train=self.r[(i-len_window):i]
#             x_test=self.x[i:(i+self.w)]
#             r_test=self.r[i:(i+self.w)]
#             X_data_test=self.X_data[i:(i+self.w)]
#             Y_data_test=self.Y_data[i:(i+self.w)]
#             control+=list(Y_data_test)

#             if self.reset:
#                 self.reset_policy_weight()

#             for _ in range(episodes):
#                 index = [kk for kk in range(len(x_train))]
#                 if self.shuffle:
#                     np.random.shuffle(index)
#                 for t in index:
#                     arm_probs=self.policy.select_arm(x_train[t])
#                     arm_index=np.random.choice(range(self.policy.K_arms),p=arm_probs)
#                     self.policy.reward_update(x_train[t],probs=arm_probs,arm_selected=arm_index,reward=r_train[t][arm_index])
#             for t in range(len(x_test)):
#                 probs=self.policy.select_arm(x_test[t])
#                 arm_index=np.random.choice(range(self.policy.K_arms),p=arm_probs)
#                 arm_index_greedy=np.argmax(probs)
#                 best_arms.append(arm_index)
#                 arm_selected[arm_index]+=1
#                 total_reward.append(r_test[t][arm_index])
#                 best_arms_greedy.append(arm_index_greedy)
#                 arm_selected_greedy[arm_index_greedy]+=1
#                 total_reward_greedy.append(r_test[t][arm_index_greedy])
#                 prediction=np.dot(probs,X_data_test[t])
#                 weighted_predictions.append(prediction)
#                 total_reward_average.append(self.calc_reward(prediction,Y_data_test[t]))
#                 mean_pred_reward.append(self.calc_reward(np.mean(X_data_test[t]),Y_data_test[t]))
#         return {'total return':np.array(total_reward),'arm_selected':arm_selected,
#                 'best_arms':best_arms,'total reward greedy':np.array(total_reward_greedy),'arm selected greedy':arm_selected_greedy,
#                 'best arms greedy':best_arms_greedy,'total reward average':np.array(total_reward_average),
#                 'weighted predictions':np.array(weighted_predictions), 'mean predictions reward': mean_pred_reward, 'control':control}

class WalkForwardEnvCMAB():
    """
    A class that simulates a contextual multi-armed bandit environment using a walk-forward approach.

    This class uses the given policy to select arms based on context vectors and rewards.
    It simulates a walk-forward training and testing process, where the policy is updated
    incrementally over time using a sliding window.

    Args:
        context (np.ndarray): The context vectors for each time step.
        X_data (np.ndarray): The predictions made by each model for each time step.
        Y_data (np.ndarray): The target values for each time step.
        rewards (np.ndarray): The rewards for each arm at each time step.
        reward_metric (string): The metric used to calculate the reward.
        k (int): The number of arms (models) available.
        d (int): The dimension of the context vectors.
        alpha (float): The learning rate for the Softmax policy.
        gamma (float, optional): The discount factor for the average reward. Defaults to None.
        window_size (int, optional): The size of the sliding window for walk-forward training. Defaults to 252.
        incremental (bool, optional): Whether to use incremental training. Defaults to False.
        shuffle (bool, optional): Whether to shuffle the training data. Defaults to False.

    Attributes:
        policy: The policy used to select arms.
        x (np.ndarray): The context vectors for each time step.
        r (np.ndarray): The rewards for each arm at each time step.
        X_data (np.ndarray): The predictions made by each model for each time step.
        Y_data (np.ndarray): The target values for each time step.
        w (int): The size of the sliding window for walk-forward training.
        incremental (bool): Whether to use incremental training.
        shuffle (bool): Whether to shuffle the training data.

    Methods:
        calc_reward(prediction, target): Calculates the reward for a given prediction and target.
        run(episodes): Runs the walk-forward simulation for a specified number of episodes.
    """
    def __init__(self,context,X_data,Y_data,rewards,reward_metric,
                 k,d,config,window_size=252,policy_type = 'softmax', 
                 incremental=False,shuffle=False,reset=False):
        """Initializes the WalkForwardEnvCMAB class."""
        self.policy_type = policy_type.lower()
        if self.policy_type == 'softmax':
            self.policy = SoftmaxPolicyCMAB(k, d, config.get("ALPHA", 0.01), config.get("GAMMA", None))
        elif self.policy_type == 'ucb':
            self.policy = UCBPolicyCMAB(k, d, config.get("ALPHA", 1), config.get("GAMMA", None))
        elif policy_type == 'thompson':
            self.policy = ThompsonSamplingPolicyCMAB(K_arms=k,
                                                     context_dim=d,
                                                     alpha=config.get("ALPHA", 0.1),
                                                     gamma=config.get("GAMMA", None)
                                                     )
        else:
            raise ValueError(f"Unsupported policy_type: {policy_type}")
        self.x=context # Context vectors
        self.r=rewards # Rewards for each arm
        self.metric=reward_metric # Reward function
        self.X_data=X_data # Predictions made by each model
        self.Y_data=Y_data # Target values
        self.w=window_size # Window size for walk-forward training
        self.incremental=incremental # Whether to use incremental training
        self.shuffle=shuffle # Whether to shuffle the training data
        self.reset=reset # Whether to reset the policy weights

    def calc_reward(self,predictions,targets):
        """Calculates the reward for a given prediction and target."""
        return reward_function(predictions, targets, metric=self.metric)

    def run(self,episodes):
        """Runs the walk-forward simulation for a specified number of episodes."""
        total_reward=[] # List to store the total reward obtained in each episode
        total_reward_greedy=[] # List to store the total reward obtained using greedy arm selection in each episode
        total_reward_average=[] # List to store the total reward obtained using average arm selection in each episode
        mean_pred_reward = [] # List to store the total reward obtained using the mean of predictions
        arm_selected=np.zeros(self.policy.K_arms) # Array to store the number of times each arm was selected
        arm_selected_greedy=np.zeros_like(arm_selected) # Array to store the number of times each arm was selected using greedy selection
        best_arms=[] # List to store the index of the best arm selected in each time step
        best_arms_greedy=[] # List to store the index of the best arm selected using greedy selection in each time step
        weighted_predictions=[] # List to store the weighted predictions made by the policy
        control=[]
        len_window=(len(self.x)%self.w)+4*self.w # Length of the training window
        for i in tqdm(range(len_window,len(self.x),self.w)):
            if self.incremental:
                x_train=self.x[:i]
                r_train=self.r[:i]
            else:
                x_train=self.x[(i-len_window):i]
                r_train=self.r[(i-len_window):i]
            x_test=self.x[i:(i+self.w)]
            r_test=self.r[i:(i+self.w)]
            X_data_test=self.X_data[i:(i+self.w)]
            Y_data_test=self.Y_data[i:(i+self.w)]
            control+=list(Y_data_test)

            if self.reset:
                self.policy.reset()

            for _ in range(episodes):
                index = [kk for kk in range(len(x_train))]
                if self.shuffle:
                    np.random.shuffle(index)
                for t in index:
                    arm_probs=self.policy.select_arm(x_train[t])
                    arm_index=np.random.choice(range(self.policy.K_arms),p=arm_probs)
                    self.policy.reward_update(x_train[t],probs=arm_probs,arm_selected=arm_index,reward=r_train[t][arm_index])
            for t in range(len(x_test)):
                probs=self.policy.select_arm(x_test[t])
                arm_index=np.random.choice(range(self.policy.K_arms),p=arm_probs)
                arm_index_greedy=np.argmax(probs)
                best_arms.append(arm_index)
                arm_selected[arm_index]+=1
                total_reward.append(r_test[t][arm_index])
                best_arms_greedy.append(arm_index_greedy)
                arm_selected_greedy[arm_index_greedy]+=1
                total_reward_greedy.append(r_test[t][arm_index_greedy])
                prediction=np.dot(probs,X_data_test[t])
                weighted_predictions.append(prediction)
                total_reward_average.append(self.calc_reward(prediction,Y_data_test[t]))
                mean_pred_reward.append(self.calc_reward(np.mean(X_data_test[t]),Y_data_test[t]))
        return {'total return':np.array(total_reward),'arm_selected':arm_selected,
                'best_arms':best_arms,'total reward greedy':np.array(total_reward_greedy),'arm selected greedy':arm_selected_greedy,
                'best arms greedy':best_arms_greedy,'total reward average':np.array(total_reward_average),
                'weighted predictions':np.array(weighted_predictions), 'mean predictions reward': mean_pred_reward, 'control':control}

# Run bandit

def run_bandit_seed(seed: int, contexts: np.ndarray, X_data: np.ndarray, 
                   Y_data: np.ndarray, rewards: np.ndarray, config: dict) -> dict:
    """Single bandit run for one seed."""
    np.random.seed(seed)
    
    # FIX (plan Phase 2): WalkForwardEnvCMAB.__init__ takes a required `config`
    # positional (used internally via config.get("ALPHA", ...)/config.get("GAMMA",
    # ...)), not `alpha=`/`gamma=` keyword arguments -- the original call here
    # passed the latter, so run_bandit_parallel raised unconditionally (masked
    # because the notebook loads cached multirun_results.pkl instead of calling
    # this function on a cache miss).
    env = WalkForwardEnvCMAB(
        context=contexts,
        X_data=X_data,
        Y_data=Y_data,
        rewards=rewards,
        reward_metric=config["METRIC"],
        k=config["N_ARMS"],
        d=config["BANDIT_EMBEDDING"],
        config=config,
        window_size=config["WINDOW_SIZE"],
        policy_type=config["POLICY"],
        shuffle=True,
        reset=True,
    )
    return env.run(episodes=config["EPISODES"])

def run_bandit_parallel(contexts_matrix, X_data, Y_data, rewards_matrix, config, 
                       n_seeds: int = 20, n_jobs: int = None):
    """
    Run multiple bandit episodes in parallel across seeds.
    """
    if n_jobs is None:
        n_jobs = min(mp.cpu_count(), n_seeds)
    
    run_func = partial(run_bandit_seed, contexts=contexts_matrix, 
                      X_data=X_data, Y_data=Y_data, rewards=rewards_matrix, 
                      config=config)
    
    with mp.Pool(n_jobs) as pool:
        all_results = pool.map(run_func, range(n_seeds))
    
    # Dict of lists: {seed0: results, seed1: results, ...}
    results_dict = {f"seed_{i}": res for i, res in enumerate(all_results)}
    return results_dict

# Plots and visualization

def save_current_plot(save_dir=None, title_fallback="plot"):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    fig = plt.gcf()
    if fig.axes == []:  # Skip empty figures
        print("Skipping save: empty figure")
        return
    
    # Priority 1: Main axes title
    ax = fig.gca()
    title = ax.get_title().strip()
    
    # Priority 2: Suptitle (safe check)
    if not title:
        suptitle = fig.get_suptitle() or ""
        title = suptitle.strip()
    
    # Priority 3: Fallback
    if not title:
        title = title_fallback
    # Sanitize: remove invalid chars, limit length
    title = re.sub(r'[<>:"/\\|?*]', '_', title)[:50]  # Safe filename chars
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title}.png"
    filepath = os.path.join(save_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")

def plot_prediction_analysis(pct_change_preds, save_dir = None):
    """Section 0.3 + related NAV plots based on pct_change_preds."""
    # Main NAV plot: target vs models/mean
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.cumprod(1 + np.array(pct_change_preds['target'])), label='target')
    for col in pct_change_preds.columns[1:]:
        predicted_nav = pct_change_preds['target'].copy()
        predicted_nav[pct_change_preds[col] < 0] = 0
        color = "orange" if col == "mean" else "gray"
        label = col if col == "mean" else None
        ax.plot(np.cumprod(1 + np.array(predicted_nav)), label=label, c=color)
    ax.legend()
    ax.set_title('Cumulative return of predictions')
    ax.set_ylabel('Cumulative return')
    ax.set_xlabel('Time')
    save_current_plot(save_dir, "prediction_analysis_1")
    plt.show()


    # Reward signals cumulative return (cell [9] in the notebook)
    fig, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    for col in pct_change_preds.columns[1:]:
        predicted_nav = pct_change_preds['target'].copy()
        predicted_nav[pct_change_preds[col] < 0] = 0
        color = "orange" if col == "mean" else "gray"
        label = col if col == "mean" else None
        alpha = 1 if col == "mean" else 0.5
        ax3.plot(np.cumprod(1 + predicted_nav), label=label, c=color, alpha=alpha)
    ax3.legend()
    ax3.set_title('Cumulative return of reward signals')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulative return')
    save_current_plot(save_dir, "prediction_analysis_2")
    plt.show()


def plot_rewards_stats(rewards_df, pct_change_preds, CONFIG, save_dir = None):
    """
    Section 0.4 stats analysis plots:
    - ACF of reward signals (mean ± std across models)
    - Std dev of rewards vs S&P500 index
    - Cumulative return of reward signals (mean ± std)
    
    `rewards_df`: DataFrame with columns 'target', 'model_0'..., 'mean'
    """
    
    # Ensure datetime index
    rewards_df.index = pd.to_datetime(rewards_df.index, utc=True)
    
    # ACF computation
    acfs_1 = []
    for i in range(len(np.array(rewards_df[[f'model_{n}' for n in range(CONFIG['N_ARMS'])]].T))):
        acfs_1.append(acf(np.array(rewards_df).T[i, 0:int(0.8 * len(np.array(rewards_df)))], nlags=36))
    mean_acfs_1 = np.mean(acfs_1, axis=0)
    std_acfs_1 = np.std(acfs_1, axis=0)
    
    # 1) ACF plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_acf(np.array(rewards_df['mean']), lags=36, ax=ax, auto_ylims=True)
    acf_values = acf(np.array(rewards_df['mean']), nlags=36)
    lags = np.arange(len(acf_values))
    ax.scatter(lags, acf_values, label='Average ACF', c='tab:blue', marker='.')
    ax.fill_between(
        range(len(mean_acfs_1)),
        mean_acfs_1 - std_acfs_1,
        mean_acfs_1 + std_acfs_1,
        alpha=0.2,
        label='Standard deviation',
        color='tab:blue'
    )
    ax.set_title('ACF of reward signals')
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.legend()
    save_current_plot(save_dir, "reward_stats_1")
    plt.show()

    
    # 2) Std dev scatter vs S&P500
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    ax2.scatter(
        rewards_df.index,
        np.std(np.array(rewards_df[[f'model_{n}' for n in range(CONFIG['N_ARMS'])]]), axis=1),
        marker='.',
        label="Reward's stddev"
    )
    ax2.plot(
        np.cumprod(1 + rewards_df['target']) / 1000,
        c='gray',
        alpha=0.5,
        label='S&P500 Index'
    )
    ax2.set_title('Standard deviation of reward signal across random seeds')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Standard deviation')
    ax2.legend()
    save_current_plot(save_dir, "reward_stats_2")
    plt.show()


def plot_single_run_bandit(results, pct_change_preds, save_dir = None):
    """
    Plots for a single run of the bandit (0.5.2 first part).

    Expects `results` = env.run(episodes=...) with keys:
    'mean predictions reward', 'total return',
    'total reward greedy', 'total reward average',
    'best arms greedy', 'weighted predictions'.
    """
    # Cumulative rewards (full)
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(results['mean predictions reward']), label='average')
    plt.plot(np.cumsum(results['total return']), label='agent sm random')
    plt.plot(np.cumsum(results['total reward greedy']), label='agent sm greedy')
    plt.plot(np.cumsum(results['total reward average']), label='agent sm weighted')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.title('Score Softmax agent vs Average - Walk Forward Training')
    save_current_plot(save_dir, "single_run_1")
    plt.show()


    # Last 100 steps
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(results['mean predictions reward'])[-100:], label='average')
    plt.plot(np.cumsum(results['total return'])[-100:], label='agent sm random')
    plt.plot(np.cumsum(results['total reward greedy'])[-100:], label='agent sm greedy')
    plt.plot(np.cumsum(results['total reward average'])[-100:], label='agent sm weighted')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.title('Cumulative reward - last 100 steps')
    save_current_plot(save_dir, "single_run_2")
    plt.show()


    # CMAB-derived strategies (greedy vs mean vs weighted)
    mab_preds_greedy = [
        pct_change_preds.iloc[
            len(pct_change_preds) - len(results['best arms greedy']) + ii, arm + 1
        ]
        for ii, arm in enumerate(results['best arms greedy'])
    ]
    mab_preds_weighted = results['weighted predictions']
    strategies = pct_change_preds.iloc[-len(results['best arms greedy']):].copy()
    strategies['greedy'] = mab_preds_greedy
    strategies['weighted'] = mab_preds_weighted

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ['greedy', 'mean', 'weighted']:
        predicted_nav = strategies['target'].copy()
        predicted_nav[strategies[col] < 0] = 0
        ax.plot(np.cumprod(1 + np.array(predicted_nav)), label=col)
    ax.set_title('Cumulative return of CMAB-derived strategies')
    ax.set_ylabel('Cumulative Return')
    ax.set_xlabel('Time steps')
    ax.legend()
    save_current_plot(save_dir, "single_run_3")
    plt.show()

def plot_multi_run_bandit(results_multi, pct_change_preds, CONFIG, save_dir=None):
    """
    Plots using multiple runs (20 seeds) from results_multi,
    plus average/STD bands, best-seed diagnostics, and colored-arm plots.
    
    `results_multi` should be a dict: seed -> results dict from env.run().
    """
    if not results_multi:
        print("No results to plot.")
        return None, None
    
    # Consistent horizon from best arms greedy (all seeds should have same length)
    first_seed = list(results_multi.keys())[0]
    T = len(results_multi[first_seed]['best arms greedy'])
    print(f"Plotting over horizon T={T}")
    
    # Slice data to matching horizon
    data_slice = pct_change_preds.iloc[-T:].copy()
    target_slice = data_slice['target'].values  # For trading signals
    
    strategies = pd.DataFrame({'target': target_slice}, index=data_slice.index)
    
    # 0) Build greedy predictions, returns, NAVs per seed
    tot_returns = []
    for seed in results_multi:
        arms_greedy = results_multi[seed]['best arms greedy']
        if len(arms_greedy) != T:
            print(f"Warning: seed {seed} has mismatched length {len(arms_greedy)} vs {T}")
            continue
            
        # Greedy predictions: selected model's prediction at each step
        greedy_preds = [data_slice.iloc[ii, arm + 1] for ii, arm in enumerate(arms_greedy)]
        strategies[f'greedy_{seed}'] = greedy_preds
        
        # Strategy returns: long target if signal > 0, else cash (0)
        strat_ret = target_slice.copy()
        strat_ret[np.array(greedy_preds) < 0] = 0.0
        strategies[f'returns_greedy_{seed}'] = strat_ret
        
        # Cumulative NAV
        strategies[f'nav_greedy_{seed}'] = np.cumprod(1 + strat_ret)
        
        # Total return for seed ranking
        tot_returns.append(np.prod(1 + strat_ret) - 1)
    
    if not tot_returns:
        print("No valid seeds.")
        return None, None
    
    # Best seed by total return
    seed_list = [seed for seed in results_multi if len(results_multi[seed]['best arms greedy']) == T]
    best_seed = seed_list[np.argmax(tot_returns)]
    print(f"Best seed: {best_seed} (total return: {max(tot_returns):.2%})")
    
    # Mean/STD across valid greedy NAVs
    nav_cols = [f'nav_greedy_{s}' for s in seed_list if f'nav_greedy_{s}' in strategies.columns]
    mean_nav_greedy = strategies[nav_cols].mean(axis=1)
    stddev_nav_greedy = strategies[nav_cols].std(axis=1)
    
    # Average ensemble baseline (mean prediction signal)
    if 'mean' in data_slice.columns:
        mean_signal = data_slice['mean'].values
    else:
        mean_signal = data_slice.iloc[:, 1:].mean(axis=1).values
    ensemble_ret = target_slice.copy()
    ensemble_ret[mean_signal < 0] = 0.0
    strategies['returns_average_ensemble'] = ensemble_ret
    strategies['nav_average_ensemble'] = np.cumprod(1 + ensemble_ret)
    
    # === PLOT 1: Mean ± STD vs ensemble ===
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mean_nav_greedy, c='tab:blue', label='greedy CMAB (mean)')
    ax.fill_between(
        mean_nav_greedy.index,
        mean_nav_greedy - stddev_nav_greedy,
        mean_nav_greedy + stddev_nav_greedy,
        alpha=0.2, color='tab:blue', label='±1 STD'
    )
    ax.plot(strategies['nav_average_ensemble'], c='tab:orange', label='average ensemble')
    ax.set_title('CMAB greedy strategy (multi-seed mean ± STD)')
    ax.set_ylabel('Cumulative Return')
    ax.set_xlabel('Date')
    ax.legend()
    if save_dir:
        save_current_plot(save_dir, "multi_run_1")
    plt.show()
    
    # === PLOT 2: Best seed colored by arm ===
    from matplotlib.collections import LineCollection
    x = np.arange(T)
    y = strategies[f'nav_greedy_{best_seed}'].values
    categories = results_multi[best_seed]['best arms greedy']
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    n_arms = CONFIG.get('NARMS', len(set(categories)) + 1)
    cmap = plt.get_cmap('rainbow', n_arms)
    lc = LineCollection(segments, cmap=cmap, linewidths=3)
    lc.set_array(categories)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    map_lc = ax.add_collection(lc)
    ax.autoscale()
    ax.set_title(f'Best seed {best_seed}: Cumulative return by arm')
    ax.set_ylabel('Cumulative Return')
    ax.set_xlabel('Steps')
    fig.colorbar(map_lc, ax=ax, label='Arm/Model')
    if save_dir:
        save_current_plot(save_dir, "multi_run_2")
    plt.show()
    
    # === PLOT 3: Best predictor signal ===
    returns_of_predictors = []
    model_cols = [col for col in data_slice.columns if col.startswith('model_')]
    for col in model_cols:
        signal = data_slice[col].values
        strat_ret = target_slice.copy()
        strat_ret[signal < 0] = 0.0
        total_ret = np.prod(1 + strat_ret) - 1
        returns_of_predictors.append(total_ret)
    
    best_pred_idx = np.argmax(returns_of_predictors)
    best_pred_name = model_cols[best_pred_idx]
    
    # Build its NAV
    signal_best = data_slice[best_pred_name].values
    best_ret = target_slice.copy()
    best_ret[signal_best < 0] = 0.0
    best_nav = pd.Series(np.cumprod(1 + best_ret), index=strategies.index)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategies[f'nav_greedy_{best_seed}'], c='tab:blue', label='greedy CMAB (best seed)')
    ax.plot(strategies['nav_average_ensemble'], c='tab:orange', label='average ensemble')
    ax.plot(best_nav, c='gray', label=f'best signal ({best_pred_name})')
    ax.set_title('CMAB vs baselines')
    ax.set_ylabel('Cumulative Return')
    ax.set_xlabel('Date')
    ax.legend()
    if save_dir:
        save_current_plot(save_dir, "multi_run_3")
    plt.show()
    
    return best_seed, strategies


def compute_and_highlight_stats(strategies, strategies_greedy_cols=None, save_dir=None):
    """
    Computes strategy statistics with CORRECT DataFrame structure.
    """
    
    def strategy_stats(daily_returns):
        returns = pd.Series(daily_returns)
        cumulative_returns = (1 + returns).cumprod()
        stats = {}
        stats['total_return'] = cumulative_returns.iloc[-1] - 1
        stats['annualized_return'] = (stats['total_return'] + 1)**(252 / len(returns)) - 1
        stats['hit_ratio'] = (returns > 0).sum() / len(returns)
        roll_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - roll_max) / roll_max
        stats['max_drawdown'] = drawdown.min()
        stats['volatility'] = returns.std() * np.sqrt(252)
        stats['sharpe_ratio'] = stats['annualized_return'] / stats['volatility']
        if stats['max_drawdown'] != 0:
            stats['calmar_ratio'] = stats['annualized_return'] / abs(stats['max_drawdown'])
        else:
            stats['calmar_ratio'] = float('inf')
        return stats

    # 1. Compute individual stats
    if strategies_greedy_cols is None:
        strategies_greedy_cols = [c for c in strategies.columns if 'returns_greedy_' in c]
    
    stats_greedy = [strategy_stats(strategies[col]) for col in strategies_greedy_cols]
    stats_avg = strategy_stats(strategies['returns_average_ensemble'])
    
    stats_greedy_mean = {k: np.mean([s[k] for s in stats_greedy]) for k in stats_greedy[0]}
    stats_greedy_std = {k: np.std([s[k] for s in stats_greedy]) for k in stats_greedy[0]}
    
    # 2. FIXED: Create DataFrame with STRATEGY NAMES as COLUMNS
    metric_names = list(stats_avg.keys())
    df_data = {
        'average_ensemble': [stats_avg[m] for m in metric_names],
        'greedy (mean)': [stats_greedy_mean[m] for m in metric_names],
        'greedy (stddev)': [stats_greedy_std[m] for m in metric_names]
    }
    df = pd.DataFrame(df_data, index=metric_names)
    if save_dir:
      df.to_csv(os.path.join(save_dir, "stats_df.csv"))
    
    def highlight_max(s):
        """Highlight BEST strategy per metric."""
        is_max = (s == s.max())
        return ['font-weight: bold' if v else '' for v in is_max]
    
    pd.set_option('display.precision', 2)
    styled_df = df.style.apply(highlight_max, axis=1).format({
        'total_return': '{:.1%}',
        'annualized_return': '{:.1%}',
        'hit_ratio': '{:.1%}',
        'max_drawdown': '{:.1%}',
        'volatility': '{:.1%}',
        'sharpe_ratio': '{:.3f}',
        'calmar_ratio': '{:.3f}'
    })
    display(styled_df)
    if save_dir:
      styled_df.to_latex(os.path.join(save_dir, "stats_df.tex"), convert_css=True)
    
    return {'stats_df': df, 'stats_list': [stats_avg, stats_greedy_mean, stats_greedy_std]}

# Plots for the Paper

def plot_multiticker_reward_std(rewards_dfs_dict, tickers_to_plot=['SPY', 'BTC-USD', 'TLT', 'EURUSD=X'], save_dir=None):
    """
    Figure 1: 2x2 Grid showing model disagreement (reward std dev) vs. Asset Price.
    Adapted from `plot_rewards_stats` (scatter plot logic).

    Args:
        rewards_dfs_dict: dict of `rewards_df` DataFrames keyed by ticker.
                          (Each df must contain 'target' and model columns).
        tickers_to_plot: list of 4 tickers to plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    axes = axes.flatten()

    for idx, ticker in enumerate(tickers_to_plot):
        if ticker not in rewards_dfs_dict:
            continue

        ax = axes[idx]
        ax2 = ax.twinx()  # Secondary axis for the price
        rewards_df = rewards_dfs_dict[ticker]

        # Ensure datetime index
        if not isinstance(rewards_df.index, pd.DatetimeIndex):
            rewards_df.index = pd.to_datetime(rewards_df.index, utc=True)

        # Extract model columns dynamically (excluding 'target' and 'mean')
        model_cols = [col for col in rewards_df.columns if col not in ['target', 'mean']]

        # Calculate standard deviation across models (like original script)
        reward_std = np.std(np.array(rewards_df[model_cols]), axis=1)

        # Plot Reward Std Dev
        ax.scatter(rewards_df.index, reward_std, marker='.', color='tab:blue', alpha=0.5, label="Reward's stddev")
        ax.set_ylabel('Reward Std Dev', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')

        # Plot Underlying Asset Price Proxy (using cumulative target returns scaled)
        # We don't divide by 1000 rigidly anymore to adapt to different assets natively
        price_proxy = np.cumprod(1 + rewards_df['target'])
        ax2.plot(rewards_df.index, price_proxy, c='gray', alpha=0.7, linewidth=1.5, label=f'{ticker} Index')
        ax2.set_ylabel(f'{ticker} Price Proxy', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        ax.set_title(f'{ticker}: Std Dev of rewards vs Price')

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_dir:
        # Assuming save_current_plot logic is available or just use savefig
        plt.savefig(os.path.join(save_dir, "reward_stddev_grid.png"), dpi=300, bbox_inches='tight')
    plt.show()


def plot_global_reward_acf(rewards_dfs_dict, nlags=36, save_dir=None):
    """
    Figure 2: Mean ACF of reward signals with standard deviation shading.
    Adapted from `plot_rewards_stats` (ACF computation logic) but aggregates across all assets.
    """
    all_acfs = []

    for ticker, rewards_df in rewards_dfs_dict.items():
        # Extract model columns dynamically
        model_cols = [col for col in rewards_df.columns if col not in ['target', 'mean']]

        # Use first 80% of data to match original logic
        cutoff = int(0.8 * len(rewards_df))

        for col in model_cols:
            model_rewards = np.array(rewards_df[col])[0:cutoff]
            # Handle potential NaNs
            model_rewards = model_rewards[~np.isnan(model_rewards)]
            if len(model_rewards) > nlags:
                acf_vals = acf(model_rewards, nlags=nlags, fft=True)
                all_acfs.append(acf_vals)

    # Calculate global mean and std across all models and datasets
    mean_acfs = np.mean(all_acfs, axis=0)
    std_acfs = np.std(all_acfs, axis=0)
    lags = np.arange(len(mean_acfs))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(lags, mean_acfs, label='Average ACF (All Models & Tickers)', c='tab:blue', marker='o')

    # Fill between using original script's shading style
    ax.fill_between(
        lags,
        mean_acfs - std_acfs,
        mean_acfs + std_acfs,
        alpha=0.2,
        label='Standard deviation',
        color='tab:blue'
    )

    ax.set_title('Global ACF of Reward Signals Across All Predictors')
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.legend()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "reward_acf_global.png"), dpi=300, bbox_inches='tight')
    plt.show()


def plot_representative_cum_return(pct_change_preds, ticker='QQQ', save_dir=None):
    """
    Figure 3: Cumulative returns of individual models vs Ensemble for a representative asset.
    Adapted entirely from `plot_prediction_analysis`.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot baseline target
    ax.plot(np.cumprod(1 + np.array(pct_change_preds['target'])), label='Underlying Asset', c='black', linestyle='--')

    # Plot individual models and mean
    for col in pct_change_preds.columns[1:]:
        if col == 'target':
            continue

        # Replicating original investment strategy logic: 0 return if prediction is < 0
        predicted_nav = pct_change_preds['target'].copy()
        predicted_nav[pct_change_preds[col] < 0] = 0

        color = "orange" if col == "mean" else "gray"
        label = 'Ensemble Mean' if col == "mean" else None
        alpha = 1.0 if col == "mean" else 0.2 # Lower alpha for gray lines

        ax.plot(np.cumprod(1 + np.array(predicted_nav)), label=label, c=color, alpha=alpha)

    ax.legend()
    ax.set_title(f'Cumulative Return of Predictions: Individual vs Ensemble ({ticker})')
    ax.set_ylabel('Cumulative return')
    ax.set_xlabel('Time (Test Window)')

    if save_dir:
        plt.savefig(os.path.join(save_dir, "mean_cum_return_rep.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_variance_comparison(rewards_df_homo, rewards_df_hetero, ticker='QQQ', window=21, save_dir=None):
    """
    Figure 4: Compares the prediction variance (reward standard deviation) of a
    homogeneous model pool (MLP only) vs a heterogeneous pool (MLP+CNN+RNN).

    Args:
        rewards_df_homo: DataFrame of rewards for the MLP-only pool.
        rewards_df_hetero: DataFrame of rewards for the MLP+CNN+RNN pool.
        ticker: String name of the asset being plotted.
        window: Integer for the rolling average window to smooth the lines.
        save_dir: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Ensure datetime indices
    if not isinstance(rewards_df_homo.index, pd.DatetimeIndex):
        rewards_df_homo.index = pd.to_datetime(rewards_df_homo.index, utc=True)
    if not isinstance(rewards_df_hetero.index, pd.DatetimeIndex):
        rewards_df_hetero.index = pd.to_datetime(rewards_df_hetero.index, utc=True)

    # Extract model columns dynamically
    cols_homo = [col for col in rewards_df_homo.columns if col not in ['target', 'mean']]
    cols_hetero = [col for col in rewards_df_hetero.columns if col not in ['target', 'mean']]

    # Calculate daily standard deviations
    std_homo = np.std(np.array(rewards_df_homo[cols_homo]), axis=1)
    std_hetero = np.std(np.array(rewards_df_hetero[cols_hetero]), axis=1)

    # Convert to pandas series for rolling calculations
    std_homo_series = pd.Series(std_homo, index=rewards_df_homo.index)
    std_hetero_series = pd.Series(std_hetero, index=rewards_df_hetero.index)

    # Calculate rolling means to smooth the noise
    smooth_homo = std_homo_series.rolling(window=window).mean()
    smooth_hetero = std_hetero_series.rolling(window=window).mean()

    # Plot smoothed lines
    ax.plot(smooth_homo.index, smooth_homo, color='tab:blue', linewidth=2.5,
            label=f'Homogeneous Pool (MLPs only) - {window}d Moving Avg')
    ax.plot(smooth_hetero.index, smooth_hetero, color='tab:red', linewidth=2.5,
            label=f'Heterogeneous Pool (MLP+CNN+RNN) - {window}d Moving Avg')

    # Optional: Plot light scatter points in the background for raw data
    ax.scatter(std_homo_series.index, std_homo_series, color='tab:blue', alpha=0.1, s=10)
    ax.scatter(std_hetero_series.index, std_hetero_series, color='tab:red', alpha=0.1, s=10)

    ax.set_title(f'Prediction Disagreement: Homogeneous vs Heterogeneous Pools ({ticker})')
    ax.set_ylabel('Reward Standard Deviation (Disagreement)')
    ax.set_xlabel('Time')
    ax.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"variance_comparison_{ticker}.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_multiticker_grid_plot1(ticker_data_dict, save_dir=None):
    """
    Creates a 2x2 grid recreating PLOT 1 from plot_multi_run_bandit.

    Args:
        ticker_data_dict: A dictionary where keys are tickers (e.g., 'SPY') and values
                          are tuples of (results_multi, pct_change_preds).
                          Example: {'SPY': (results_multi_spy, pct_preds_spy), ...}
        save_dir: Directory to save the output plot.
    """
    grid_tickers = ['SPY', 'BTC-USD', 'EURUSD=X', 'GC=F']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, ticker in enumerate(grid_tickers):
        ax = axes[idx]

        if ticker not in ticker_data_dict:
            print(f"Skipping {ticker} - Data not provided.")
            ax.set_visible(False)
            continue

        results_multi, pct_change_preds = ticker_data_dict[ticker]

        if not results_multi:
            print(f"Skipping {ticker} - results_multi is empty.")
            continue

        # 1. Consistent horizon
        first_seed = list(results_multi.keys())[0]
        T = len(results_multi[first_seed]['best arms greedy'])

        # 2. Slice data
        data_slice = pct_change_preds.iloc[-T:].copy()
        target_slice = data_slice['target'].values

        # 3. Build greedy predictions per seed
        strategies = pd.DataFrame({'target': target_slice}, index=data_slice.index)
        seed_list = []

        for seed in results_multi:
            arms_greedy = results_multi[seed]['best arms greedy']
            if len(arms_greedy) != T:
                continue

            seed_list.append(seed)
            greedy_preds = [data_slice.iloc[ii, arm + 1] for ii, arm in enumerate(arms_greedy)]

            strat_ret = target_slice.copy()
            strat_ret[np.array(greedy_preds) < 0] = 0.0

            strategies[f'nav_greedy_{seed}'] = np.cumprod(1 + strat_ret)

        # 4. Calculate Mean and STD across seeds
        nav_cols = [f'nav_greedy_{s}' for s in seed_list]
        mean_nav_greedy = strategies[nav_cols].mean(axis=1)
        stddev_nav_greedy = strategies[nav_cols].std(axis=1)

        # 5. Calculate Average Ensemble Baseline
        if 'mean' in data_slice.columns:
            mean_signal = data_slice['mean'].values
        else:
            mean_signal = data_slice.iloc[:, 1:].mean(axis=1).values

        ensemble_ret = target_slice.copy()
        ensemble_ret[mean_signal < 0] = 0.0
        nav_average_ensemble = np.cumprod(1 + ensemble_ret)

        # 6. Plotting logic (Exact match to PLOT 1)
        # We will assume you are evaluating the UCB policy here for the main visual
        ax.plot(mean_nav_greedy.index, mean_nav_greedy, c='tab:blue', label='greedy CMAB (mean)')
        ax.fill_between(
            mean_nav_greedy.index,
            mean_nav_greedy - stddev_nav_greedy,
            mean_nav_greedy + stddev_nav_greedy,
            alpha=0.2, color='tab:blue', label='±1 STD'
        )
        ax.plot(mean_nav_greedy.index, nav_average_ensemble, c='tab:orange', label='average ensemble')

        # Formatting
        ax.set_title(f'{ticker}: CMAB vs Ensemble', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Return')
        ax.set_xlabel('Date')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Only add legend to the first plot to keep the grid clean
        if idx == 0:
            ax.legend(loc='upper left')

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "multi_run_grid_plot1.png"), dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_dir}/multi_run_grid_plot1.png")

    plt.show()

def plot_bandit_strategy_comparison(results, pct_change_preds, save_dir=None):
    """
    Plots a streamlined comparison of the CMAB Greedy strategy vs the Ensemble Mean baseline.
    """

    # Align data slices
    strategies = pct_change_preds.iloc[-len(results['best arms greedy']):].copy()
    strategies['greedy_mab'] = results['weighted predictions']

    plt.figure(figsize=(12, 6))

    # Define the strategies to plot
    plot_configs = [
        ('greedy_mab', 'CMAB Greedy Strategy', '#1f77b4', '-'),
        ('mean', 'Ensemble Mean Baseline', '#ff7f0e', '--')
    ]

    for col, label, color, style in plot_configs:
        # Calculate NAV: assume 0 return if the signal is negative
        nav = strategies['target'].copy()
        nav[strategies[col] < 0] = 0
        cumulative_return = np.cumprod(1 + np.array(nav))

        plt.plot(cumulative_return, label=label, color=color, linestyle=style, linewidth=2.5)

    plt.title('Performance Comparison: CMAB Selection vs. Passive Ensemble Mean', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Growth of $1 Investment (NAV)', fontsize=12)
    plt.xlabel('Time Steps (Out-of-Sample)', fontsize=12)
    plt.legend(frameon=True, fontsize=11, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    if save_dir:
        save_current_plot(save_dir, "bandit_vs_mean_comparison")

    plt.show()


# =============================================================================
# Phase 1 migration: compatibility overrides
# =============================================================================
# The functions/classes below have been ported into the `mabss` package (see
# mabss/rewards.py, mabss/policies/, mabss/env.py, mabss/models.py,
# mabss/data.py, mabss/training.py, mabss/experiments/) with real bug fixes and
# a real test suite (see tests/unit, tests/regression). Re-binding these names
# HERE, at module scope, means every function ABOVE this point in the file --
# including the plotting/stats functions that have NOT been migrated yet
# (plot_prediction_analysis, plot_rewards_stats, plot_single_run_bandit,
# plot_multi_run_bandit, compute_and_highlight_stats, the "Plots for the Paper"
# section, run_bandit_parallel) -- picks up the fixed implementations too: a
# Python function looks up module-level globals (like `WalkForwardEnvCMAB` or
# `reward_function`) at CALL time, not at def time, so rebinding the name here
# retroactively "patches" every earlier call site in this same module.
#
# This is a deliberate mid-point, not a finished migration: the plotting/stats
# section above still lives here rather than in mabss/plots.py, because
# migrating and testing ~700 lines of plotting code (each with its own
# hardcoded-filename, column-order, and CONFIG-typo bugs -- see the plan) was
# out of scope for this pass. What's below is safe to depend on; what's above
# this line (other than the fixes already applied in place, e.g.
# run_bandit_seed's constructor call) is the original, not-yet-hardened code.
from mabss.data import (  # noqa: E402
    compute_returns_from_preds,
    load_price_series,
    window_time_series,
)
from mabss.env import WalkForwardEnv  # noqa: E402
from mabss.experiments import ExperimentStore as ExperimentLoader  # noqa: E402, F811
from mabss.experiments import make_experiment_id  # noqa: E402, F811
from mabss.models import create_model  # noqa: E402, F811
from mabss.policies import LinUCBPolicy as UCBPolicyCMAB  # noqa: E402, F811
from mabss.policies import SoftmaxPolicy as SoftmaxPolicyCMAB  # noqa: E402, F811
from mabss.policies import ThompsonSamplingPolicy as ThompsonSamplingPolicyCMAB  # noqa: E402, F811
from mabss.rewards import build_rewards_and_contexts, reward_function  # noqa: E402, F811
from mabss.training import walk_forward_training  # noqa: E402, F811


class WalkForwardEnvCMAB(WalkForwardEnv):  # noqa: F811
    """
    Compat wrapper: `mabss.env.WalkForwardEnv.run()` returns a `BanditRunResult`
    dataclass (cleaner for new code -- see mabss/env.py), but every
    not-yet-migrated plotting/stats function in this module (plot_single_run_bandit,
    plot_multi_run_bandit, compute_and_highlight_stats, run_bandit_seed's
    callers, ...) still expects `.run()` to return the original 10-key dict
    (`results['best arms greedy']`, `results['total return']`, etc.) -- exactly
    the shape every cached results.pkl/multirun_results.pkl on disk already has.
    This subclass exists purely to bridge that gap without touching
    mabss.env.WalkForwardEnv's own return type. New code should use
    `mabss.env.WalkForwardEnv` directly and consume the dataclass.
    """

    def run(self, episodes):
        return super().run(episodes).to_legacy_dict()


def create_dense_model(random_seed: int, input_dim: int):  # noqa: F811
    """DEPRECATED: identical to create_model('mlp', ...) since MABSS_utility.py's
    original create_dense_model was byte-for-byte the same as create_model's
    'mlp' branch. Kept only so old call sites don't break; prefer create_model."""
    import warnings

    warnings.warn(
        "create_dense_model is deprecated and identical to "
        "create_model('mlp', random_seed, input_dim); call that instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_model("mlp", random_seed, input_dim)


def get_experiment_paths(config: dict, base_dir):
    """DEPRECATED and DEAD in the original: declared filenames
    (rewards_matrix.npy, contexts_matrix.npy, bandit_results.pkl) that exist
    NOWHERE in experiments/ or experiments_cluster/ -- ExperimentLoader/
    ExperimentStore's actual filenames (rewards.npy, contexts.npy, results.pkl)
    are what every real artifact on disk uses. Reviving this function's old
    behavior would report every one of the 63 committed experiment directories
    as uncached. Raising here rather than "fixing" it silently, since anyone
    calling it almost certainly wants ExperimentStore/ExperimentLoader instead."""
    raise NotImplementedError(
        "get_experiment_paths is dead code from the original module: it declares "
        "artifact filenames that don't match anything ExperimentLoader/"
        "ExperimentStore actually writes or that exist on disk. Use "
        "ExperimentLoader(config, base_dir).paths instead."
    )


def experiment_in_memory(paths: dict) -> bool:
    """DEPRECATED and DEAD in the original -- see get_experiment_paths."""
    raise NotImplementedError(
        "experiment_in_memory is dead code paired with get_experiment_paths "
        "(see that function's docstring). Use "
        "ExperimentLoader(config, base_dir).is_fully_cached() instead."
    )

