"""
Shared utilities for reward component analysis (Task 20 & 21).
Copied/adapted from Training/A2C-mod.ipynb and Training/DQN.ipynb.
No dummy/demo code — uses real TFRecords data and actual training logic.
"""

import os
import numpy as np
import tensorflow as tf
from scipy import stats

# ============================================================
# CONSTANTS (match Training notebooks exactly)
# ============================================================
NUM_PRODUCTS = 220
NUM_FEATURES = 3
WASTE_RATE = 0.025
ZERO_INVENTORY = 1e-5
GAMMA = 0.99

ACTION_SPACE = np.array([
    0, 0.005, 0.01, 0.0125, 0.015, 0.0175,
    0.02, 0.03, 0.04, 0.08, 0.12, 0.2, 0.5, 1.0
], dtype=np.float32)

# ============================================================
# TFRecord Parsers (exact copy from Training/A2C-mod.ipynb)
# ============================================================
def sales_parser(serialized_example):
    example = tf.io.parse_single_example(
        serialized_example,
        features={"sales": tf.io.FixedLenFeature([NUM_PRODUCTS], tf.float32)})
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.float32)
            example[name] = t
    return example


def capacity_parser(serialized_example):
    example = tf.io.parse_single_example(
        serialized_example,
        features={"capacity": tf.io.FixedLenFeature([NUM_PRODUCTS], tf.float32)})
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.float32)
            example[name] = t
    return example


def stock_parser(serialized_example):
    example = tf.io.parse_single_example(
        serialized_example,
        features={"stock": tf.io.FixedLenFeature([NUM_PRODUCTS], tf.float32)})
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.float32)
            example[name] = t
    return example


def waste_fn(x):
    """Waste fraction q = waste_rate * inventory."""
    return WASTE_RATE * x


def quantile_fn(x, q):
    return np.quantile(x, q)


# ============================================================
# Reward Calculation (exact copy from Training/DQN.ipynb calc_reward)
# ============================================================
def calc_reward(x_old, overstock, waste_rate=WASTE_RATE, zero_inventory=ZERO_INVENTORY):
    """
    Reward per product — identical to A2C_mod in training1.py (lines 399-404).
    All penalty terms computed from x_old (inventory BEFORE action).

    Parameters
    ----------
    x_old     : np.ndarray [P]  — inventory BEFORE action
    overstock : np.ndarray [P]  — max(0, x_old + u - 1)
    waste_rate: float           — 0.025
    zero_inventory: float       — 1e-5

    Returns
    -------
    r       : np.ndarray [P]  — reward per product
    z       : np.ndarray [P]  — stockout indicator
    quan    : float           — quantile spread (scalar, broadcast to all products)
    """
    # z: stockout indicator on OLD inventory (before action)
    z = (x_old < zero_inventory).astype(np.float32)

    # q: waste on OLD inventory (before action)
    q = waste_rate * x_old

    # quan: quantile spread on OLD inventory, broadcast to all products
    quan_scalar = float(np.quantile(x_old, 0.95) - np.quantile(x_old, 0.05))
    quan_vec = np.full_like(x_old, quan_scalar, dtype=np.float32)

    r = (1.0 - z - overstock - q - quan_vec).astype(np.float32)
    return r, z, quan_scalar


def calc_reward_with_weights(x_old, overstock, weights=None):
    """
    Extended reward calculation with configurable weights.
    weights: dict with keys 'base', 'stockout', 'overstock', 'waste', 'quantile'
    Default all = 1.0 (matches original).
    """
    if weights is None:
        weights = {'base': 1.0, 'stockout': 1.0, 'overstock': 1.0, 'waste': 1.0, 'quantile': 1.0}

    z = (x_old < ZERO_INVENTORY).astype(np.float32)
    q = WASTE_RATE * x_old
    quan_scalar = float(np.quantile(x_old, 0.95) - np.quantile(x_old, 0.05))
    quan_vec = np.full_like(x_old, quan_scalar, dtype=np.float32)

    r = (weights['base'] * 1.0
         - weights['stockout'] * z
         - weights['overstock'] * overstock
         - weights['waste'] * q
         - weights['quantile'] * quan_vec).astype(np.float32)
    return r, z, quan_scalar


# ============================================================
# Data Loading Utilities
# ============================================================
def load_tfrecord_data(data_dir='data'):
    """Load all raw data from TFRecords."""
    train_file = os.path.join(data_dir, 'train.tfrecords')
    test_file = os.path.join(data_dir, 'test.tfrecords')
    capacity_file = os.path.join(data_dir, 'capacity.tfrecords')
    stock_file = os.path.join(data_dir, 'stock.tfrecords')

    # Capacity (single record)
    capacity_ds = tf.data.TFRecordDataset(capacity_file).map(capacity_parser)
    capacity = next(iter(capacity_ds))['capacity'].numpy()  # [P]

    # Initial stock (single record)
    stock_ds = tf.data.TFRecordDataset(stock_file).map(stock_parser)
    x_init = next(iter(stock_ds))['stock'].numpy()  # [P]

    # Sales data
    train_sales_raw = []
    for rec in tf.data.TFRecordDataset(train_file).map(sales_parser):
        train_sales_raw.append(rec['sales'].numpy())
    train_sales_raw = np.array(train_sales_raw, dtype=np.float32)  # [T_train, P]

    test_sales_raw = []
    for rec in tf.data.TFRecordDataset(test_file).map(sales_parser):
        test_sales_raw.append(rec['sales'].numpy())
    test_sales_raw = np.array(test_sales_raw, dtype=np.float32)  # [T_test, P]

    return {
        'capacity': capacity,
        'x_init': x_init,
        'train_sales_raw': train_sales_raw,
        'test_sales_raw': test_sales_raw,
    }


def normalize_sales(sales_raw, capacity):
    """Normalize sales by capacity (same as training)."""
    return sales_raw / capacity[np.newaxis, :]  # [T, P]


# ============================================================
# Environment Step Simulation (matches training loop)
# ============================================================
def env_step(x, u, sales, capacity):
    """
    Single environment step.
    Returns: x_next, overstock, x_clip, stockout_amount
    """
    x_rep = x + u
    overstock = np.maximum(0.0, x_rep - 1.0)
    x_clip = np.minimum(1.0, x_rep)
    x_next = np.maximum(0.0, x_clip - sales)
    stockout_amount = np.maximum(0.0, sales - x_clip)
    return x_next, overstock, x_clip, stockout_amount


def simulate_episode(sales_norm, x_init, policy_fn, capacity, waste_rate=WASTE_RATE):
    """
    Simulate one episode using given policy function.
    policy_fn(state_flat) -> action_indices [P]
    Returns lists of per-timestep components.
    """
    P = NUM_PRODUCTS
    x = x_init.copy()
    q = waste_rate * x

    components = {
        'z': [], 'overstock': [], 'q': [], 'quan': [],
        'reward': [], 'stockout_amount': [], 'x': [], 'x_clip': []
    }

    for t in range(sales_norm.shape[0]):
        sales_now = sales_norm[t]

        # Build state (flat 660)
        state_flat = np.concatenate([x, sales_now, q]).astype(np.float32)

        # Get action from policy
        action_idx = policy_fn(state_flat)
        u = ACTION_SPACE[action_idx]

        # Environment step
        x_next, overstock, x_clip, stockout_amount = env_step(x, u, sales_now, capacity)

        # Compute reward components (using x_old = x before action)
        r, z, quan = calc_reward(x, overstock, waste_rate)

        # Store
        components['z'].append(z)
        components['overstock'].append(overstock)
        components['q'].append(waste_rate * x)  # q on x_old
        components['quan'].append(np.full(P, quan, dtype=np.float32))
        components['reward'].append(r)
        components['stockout_amount'].append(stockout_amount)
        components['x'].append(x.copy())
        components['x_clip'].append(x_clip)

        # Next state
        x = x_next
        q = waste_rate * x

    # Stack to arrays [T, P]
    for k in components:
        components[k] = np.array(components[k], dtype=np.float32)

    return components


# ============================================================
# Statistics & Plotting
# ============================================================
def descriptive_stats(arr, name=''):
    """Compute comprehensive descriptive statistics."""
    flat = arr.flatten()
    return {
        'component': name,
        'count': int(flat.size),
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'min': float(np.min(flat)),
        'p25': float(np.percentile(flat, 25)),
        'p50': float(np.percentile(flat, 50)),
        'p75': float(np.percentile(flat, 75)),
        'max': float(np.max(flat)),
        'skew': float(stats.skew(flat)),
        'kurtosis': float(stats.kurtosis(flat)),
        'zero_frac': float(np.mean(flat == 0)) if name == 'z' else None,
    }


def compute_all_stats(components_dict, prefix=''):
    """Compute stats for all components in dict."""
    stats_list = []
    for name, arr in components_dict.items():
        if name in ['z', 'overstock', 'q', 'quan', 'reward', 'stockout_amount']:
            stats_list.append(descriptive_stats(arr, f'{prefix}{name}'))
    return stats_list


def plot_component_distribution(components_before, components_after, component_name, split_name, save_path):
    """Plot histogram + boxplot before/after normalization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{component_name.upper()} Distribution — {split_name}', fontsize=14, fontweight='bold')

    # Before normalization
    data_before = components_before[component_name].flatten()
    data_after = components_after[component_name].flatten()

    # Histogram before
    axes[0, 0].hist(data_before, bins=50, density=True, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].set_title(f'Before Norm — Histogram (N={len(data_before):,})')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)

    # Boxplot before
    axes[0, 1].boxplot(data_before, notch=True, showfliers=True)
    axes[0, 1].set_title('Before Norm — Boxplot')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram after
    axes[1, 0].hist(data_after, bins=50, density=True, alpha=0.7, edgecolor='black', color='lightcoral')
    axes[1, 0].set_title(f'After Norm — Histogram (N={len(data_after):,})')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)

    # Boxplot after
    axes[1, 1].boxplot(data_after, notch=True, showfliers=True)
    axes[1, 1].set_title('After Norm — Boxplot')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


# ============================================================
# Covariate Shift Detection
# ============================================================
def wasserstein_distance(arr1, arr2):
    """Wasserstein distance between two 1D arrays."""
    from scipy.stats import wasserstein_distance as wd
    return float(wd(arr1.flatten(), arr2.flatten()))


def kl_divergence(arr1, arr2, bins=50):
    """KL divergence using histogram binning."""
    hist1, bin_edges = np.histogram(arr1.flatten(), bins=bins, density=True)
    hist2, _ = np.histogram(arr2.flatten(), bins=bin_edges, density=True)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist1 = hist1 + eps
    hist2 = hist2 + eps
    return float(np.sum(hist1 * np.log(hist1 / hist2)))


def check_covariate_shift(train_components, test_components):
    """Check shift for each component."""
    results = {}
    for name in ['z', 'overstock', 'q', 'quan']:
        wd = wasserstein_distance(train_components[name], test_components[name])
        kl = kl_divergence(train_components[name], test_components[name])
        results[name] = {'wasserstein': wd, 'kl_divergence': kl}
    return results
