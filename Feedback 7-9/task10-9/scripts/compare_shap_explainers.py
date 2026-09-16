import argparse, os, time, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import tensorflow as tf
import shap

# === Config from topk_shap_analysis.ipynb ===
NUM_PRODUCTS = 220
NUM_FEATURES_PP = 3
NUM_FEATURES = NUM_PRODUCTS * NUM_FEATURES_PP  # 660
NUM_ACTIONS = 14
WASTE_RATE = 0.025
ZERO_INVENTORY = 1e-5
DQN_HIDDEN = 32
A2C_HIDDEN = 32
DROPOUT = 0.1

def get_args():
    p = argparse.ArgumentParser(description='Compare Partition vs Kernel SHAP on 660-dim')
    p.add_argument('--n_states', type=int, default=10, help='test states per scenario')
    p.add_argument('--nsamples', type=int, default=500, help='Kernel SHAP nsamples (2000 is heavy for 660-dim)')
    p.add_argument('--num_bg', type=int, default=100, help='background samples for Partition')
    p.add_argument('--data_dir', type=str, default=r'C:\GitHub\Q-learning-for-Inventory-Management\data')
    p.add_argument('--dqn_ckpt', type=str, default=r'C:\GitHub\Q-learning-for-Inventory-Management\output Training\checkpointDQN')
    p.add_argument('--a2c_ckpt', type=str, default=r'C:\GitHub\Q-learning-for-Inventory-Management\output Training\outputA2Cmod\checkpoints_a2cmod')
    p.add_argument('--out', type=str, default=r'C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task10-9\compare_kernel_vs_partition_result.csv')
    p.add_argument('--agents', type=str, default='both', choices=['dqn','a2c','both'])
    p.add_argument('--scenarios', type=str, default='EASY,MEDIUM,HARD')
    return p.parse_args()

# === Model classes from topk_shap_analysis.ipynb ===
class Dense(tf.Module):
    def __init__(self, input_dim, output_size, activation=None, stddev=1.0):
        super(Dense, self).__init__()
        self.w = tf.Variable(tf.random.truncated_normal([input_dim, output_size], stddev=stddev), name='w')
        self.b = tf.Variable(tf.zeros([output_size]), name='b')
        self.activation = activation
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        if self.activation:
            y = self.activation(y)
        return y

class Actor(tf.Module):
    def __init__(self, num_features, num_actions, hidden_size, activation=tf.nn.relu, dropout_prob=0.1):
        super(Actor, self).__init__()
        self.layer1 = Dense(num_features, hidden_size, activation=None)
        self.layer2 = Dense(hidden_size, hidden_size, activation=None)
        self.layer3 = Dense(hidden_size, hidden_size, activation=None)
        self.layer4 = Dense(hidden_size, num_actions, activation=None)
        self.activation = activation
        self.dropout_prob = dropout_prob
    def __call__(self, state):
        x = self.activation(self.layer1(state))
        x = tf.nn.dropout(x, self.dropout_prob)
        x = self.activation(self.layer2(x))
        x = tf.nn.dropout(x, self.dropout_prob)
        x = self.activation(self.layer3(x))
        x = tf.nn.dropout(x, self.dropout_prob)
        x = self.layer4(x)
        return tf.nn.softmax(x)

class Critic(tf.Module):
    def __init__(self, num_features, hidden_size, activation=tf.nn.relu, dropout_prob=0.1):
        super(Critic, self).__init__()
        self.layer1 = Dense(num_features, hidden_size, activation=None)
        self.layer2 = Dense(hidden_size, 1, activation=None)
        self.activation = activation
        self.dropout_prob = dropout_prob
        self.group_norm = tf.keras.layers.GroupNormalization(groups=1)
    def __call__(self, state):
        x = self.layer1(state)
        x = self.group_norm(x)
        x = self.activation(x)
        x = tf.nn.dropout(x, self.dropout_prob)
        x = self.layer2(x)
        return tf.squeeze(x, axis=-1, name='factor_squeeze')

class MultiProductQNetwork(tf.keras.Model):
    def __init__(self, num_features, num_products, num_actions, hidden_size, dropout_prob=0.1, use_group_norm=True, name=None):
        super().__init__(name=name)
        self.num_products = num_products
        self.num_actions = num_actions
        self.features_per_prod = num_features // num_products
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation=None, name='dense1')
        self.dense2 = tf.keras.layers.Dense(hidden_size, activation=None, name='dense2')
        self.dense3 = tf.keras.layers.Dense(hidden_size, activation=None, name='dense3')
        self.out = tf.keras.layers.Dense(num_actions, activation=None, name='output')
        self._use_gn = use_group_norm
        if use_group_norm:
            self.gn1 = tf.keras.layers.GroupNormalization(groups=1, name='gn1')
            self.gn2 = tf.keras.layers.GroupNormalization(groups=1, name='gn2')
            self.gn3 = tf.keras.layers.GroupNormalization(groups=1, name='gn3')
        self.drop1 = tf.keras.layers.Dropout(dropout_prob)
        self.drop2 = tf.keras.layers.Dropout(dropout_prob)
        self.drop3 = tf.keras.layers.Dropout(dropout_prob)
    def call(self, state, training=False):
        B = tf.shape(state)[0]
        P, F = self.num_products, self.features_per_prod
        s3d = tf.transpose(tf.reshape(state, [B, F, P]), [0, 2, 1])
        x = tf.reshape(s3d, [B * P, F])
        x = self.dense1(x)
        if self._use_gn: x = self.gn1(x, training=training)
        x = tf.nn.relu(x); x = self.drop1(x, training=training)
        x = self.dense2(x)
        if self._use_gn: x = self.gn2(x, training=training)
        x = tf.nn.relu(x); x = self.drop2(x, training=training)
        x = self.dense3(x)
        if self._use_gn: x = self.gn3(x, training=training)
        x = tf.nn.relu(x); x = self.drop3(x, training=training)
        return tf.reshape(self.out(x), [B, P, self.num_actions])

def get_feature_info(idx):
    if idx < NUM_PRODUCTS:
        return ('inventory', idx)
    elif idx < 2 * NUM_PRODUCTS:
        return ('sales', idx - NUM_PRODUCTS)
    else:
        return ('waste_feat', idx - 2 * NUM_PRODUCTS)

FEATURE_NAMES_660 = [f'{get_feature_info(i)[0]}_SKU{get_feature_info(i)[1]}' for i in range(NUM_FEATURES)]

def load_trained_agents(dqn_ckpt_dir, a2c_ckpt_dir):
    # A2C_mod
    actor = Actor(NUM_FEATURES_PP, NUM_ACTIONS, A2C_HIDDEN, activation=tf.nn.relu, dropout_prob=DROPOUT)
    critic = Critic(NUM_FEATURES_PP, A2C_HIDDEN, activation=tf.nn.relu, dropout_prob=DROPOUT)
    _d = tf.zeros([1, NUM_FEATURES_PP])
    _ = actor(_d); _ = critic(_d)
    a2c_ckpt = tf.train.Checkpoint(
        critic_optimizer=tf.optimizers.Adam(0.0005),
        actor_optimizer=tf.optimizers.Adam(0.0001),
        critic=critic, actor=actor, step=tf.Variable(0))
    a2c_latest = tf.train.latest_checkpoint(a2c_ckpt_dir)
    if a2c_latest is None:
        raise FileNotFoundError(f'A2C checkpoint not found in {a2c_ckpt_dir}')
    a2c_ckpt.restore(a2c_latest).expect_partial()
    print(f'OK A2C_mod restored: {a2c_latest}')

    # DQN
    q_net = MultiProductQNetwork(NUM_FEATURES, NUM_PRODUCTS, NUM_ACTIONS, DQN_HIDDEN, DROPOUT, use_group_norm=True, name='q_network')
    t_net = MultiProductQNetwork(NUM_FEATURES, NUM_PRODUCTS, NUM_ACTIONS, DQN_HIDDEN, DROPOUT, use_group_norm=True, name='target_network')
    _d = tf.zeros([1, NUM_FEATURES], dtype=tf.float32)
    _ = q_net(_d, training=False); _ = t_net(_d, training=False)
    # Note: checkpoint step dtype is int32 (not int64) in output Training/checkpointDQN
    dqn_ckpt = tf.train.Checkpoint(
        optimizer=tf.optimizers.Adam(0.001),
        q_network=q_net, target_network=t_net,
        step=tf.Variable(0))
    dqn_latest = tf.train.latest_checkpoint(dqn_ckpt_dir)
    if dqn_latest is None:
        raise FileNotFoundError(f'DQN checkpoint not found in {dqn_ckpt_dir}')
    dqn_ckpt.restore(dqn_latest).expect_partial()
    print(f'OK DQN restored: {dqn_latest}')

    return {'actor': actor, 'critic': critic, 'q_network': q_net}

def main():
    args = get_args()
    scenarios_list = [s.strip() for s in args.scenarios.split(',')]
    SCENARIOS = {
        'EASY':   {'sales_scale': 0.5,  'waste_rate': 0.010},
        'MEDIUM': {'sales_scale': 1.0,  'waste_rate': 0.025},
        'HARD':   {'sales_scale': 1.5,  'waste_rate': 0.050},
    }
    # Filter to requested
    scenarios_list = [s for s in scenarios_list if s in SCENARIOS]

    print(f'Args: n_states={args.n_states} nsamples={args.nsamples} num_bg={args.num_bg}')
    print(f'Data dir: {args.data_dir}')
    print(f'DQN ckpt: {args.dqn_ckpt} -> {tf.train.latest_checkpoint(args.dqn_ckpt)}')
    print(f'A2C ckpt: {args.a2c_ckpt} -> {tf.train.latest_checkpoint(args.a2c_ckpt)}')
    print(f'Agents: {args.agents} Scenarios: {scenarios_list}')

    # Load agents
    agents = load_trained_agents(args.dqn_ckpt, args.a2c_ckpt)

    # Load test data TFRecords
    def _parse(serialized, key, n):
        return tf.io.parse_single_example(serialized, {key: tf.io.FixedLenFeature([n], tf.float32)})[key]

    cap_file = os.path.join(args.data_dir, 'capacity.tfrecords')
    stock_file = os.path.join(args.data_dir, 'stock.tfrecords')
    test_file = os.path.join(args.data_dir, 'test.tfrecords')

    if not os.path.exists(cap_file):
        raise FileNotFoundError(cap_file)
    capacity = next(iter(tf.data.TFRecordDataset(cap_file).map(lambda s: _parse(s, 'capacity', NUM_PRODUCTS)))).numpy()
    x_init = next(iter(tf.data.TFRecordDataset(stock_file).map(lambda s: _parse(s, 'stock', NUM_PRODUCTS)))).numpy()
    all_sales = []
    for rec in tf.data.TFRecordDataset(test_file).map(lambda s: _parse(s, 'sales', NUM_PRODUCTS)):
        all_sales.append(rec.numpy())
    all_sales = np.array(all_sales, dtype=np.float32) / capacity[None, :]
    T_MAX = len(all_sales)
    print(f'OK Test data loaded: {T_MAX} timesteps x {NUM_PRODUCTS} products')
    print(f'   capacity mean {capacity.mean():.2f} x_init mean {x_init.mean():.3f}')

    # Generate background
    def generate_background_660(num_samples=args.num_bg):
        bg = np.zeros((num_samples, NUM_FEATURES), dtype=np.float32)
        bg[:, :NUM_PRODUCTS] = np.random.uniform(0.0, 1.0, size=(num_samples, NUM_PRODUCTS))
        bg[:, NUM_PRODUCTS:2*NUM_PRODUCTS] = np.random.uniform(0.0, 1.0, size=(num_samples, NUM_PRODUCTS))
        waste_noise = np.random.normal(0, 0.005, size=(num_samples, NUM_PRODUCTS))
        bg[:, 2*NUM_PRODUCTS:] = np.clip(WASTE_RATE * bg[:, :NUM_PRODUCTS] + waste_noise, 0, 0.1)
        return bg

    def create_test_states_660(scenario_name, num_states=args.n_states):
        config = SCENARIOS[scenario_name]
        sales_scale = config['sales_scale']
        waste_rate = config['waste_rate']
        test_sales = all_sales[:num_states, :] * sales_scale
        state_660 = np.zeros((num_states, NUM_FEATURES), dtype=np.float32)
        state_660[:, :NUM_PRODUCTS] = x_init[None, :]
        state_660[:, NUM_PRODUCTS:2*NUM_PRODUCTS] = test_sales
        state_660[:, 2*NUM_PRODUCTS:] = x_init[None, :] * waste_rate
        return state_660

    background_660 = generate_background_660(args.num_bg)
    print(f'OK Background 660-dim: {background_660.shape} range [{background_660.min():.4f},{background_660.max():.4f}]')

    test_states_660 = {}
    for sc in scenarios_list:
        s660 = create_test_states_660(sc, args.n_states)
        test_states_660[sc] = s660
        print(f'OK {sc:6s}: {s660.shape} inv [{s660[:,:NUM_PRODUCTS].min():.3f},{s660[:,:NUM_PRODUCTS].max():.3f}] sales [{s660[:,NUM_PRODUCTS:2*NUM_PRODUCTS].min():.3f},{s660[:,NUM_PRODUCTS:2*NUM_PRODUCTS].max():.3f}]')

    # Wrappers
    def dqn_predict_660(X):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        q_values = agents['q_network'](X, training=False)
        q_mean = tf.reduce_mean(q_values, axis=1)
        return q_mean.numpy()

    def a2c_predict_660(X):
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)
        B = X.shape[0]
        P, F = NUM_PRODUCTS, NUM_FEATURES_PP
        s3d = tf.transpose(tf.reshape(X, [B, F, P]), [0, 2, 1])
        s_pp = tf.reshape(s3d, [B * P, F])
        probs = agents['actor'](s_pp)
        probs_3d = tf.reshape(probs, [B, P, -1])
        probs_mean = tf.reduce_mean(probs_3d, axis=1)
        return probs_mean.numpy()

    # Test wrappers
    test_in = np.random.randn(2, NUM_FEATURES).astype(np.float32)
    print(f'OK DQN wrapper {test_in.shape} -> {dqn_predict_660(test_in).shape}')
    print(f'OK A2C wrapper {test_in.shape} -> {a2c_predict_660(test_in).shape}')

    # Determine agents to run
    agent_list = []
    if args.agents in ('dqn','both'):
        agent_list.append(('DQN', dqn_predict_660))
    if args.agents in ('a2c','both'):
        agent_list.append(('A2C_mod', a2c_predict_660))

    results = []

    for agent_name, predict_fn in agent_list:
        print(f"\n{'='*80}\n Agent {agent_name}\n{'='*80}")
        # Partition explainer init (once per agent)
        print(f'Initializing PartitionExplainer for {agent_name}...')
        masker = shap.maskers.Partition(background_660, max_samples=args.num_bg)
        explainer_p = shap.PartitionExplainer(predict_fn, masker)
        print(f'OK PartitionExplainer ready')
        sampled_bg = shap.sample(background_660, min(50, args.num_bg))
        print(f'Init KernelExplainer background {sampled_bg.shape} (nsamples={args.nsamples}) - heavy for 660-dim')

        for sc in scenarios_list:
            print(f"\n--- {agent_name} {sc} ({args.n_states} states) ---")
            X_test = test_states_660[sc]

            # Partition
            t0 = time.time()
            sv_result = explainer_p(X_test)
            if hasattr(sv_result, 'values'):
                sv_array_p = sv_result.values  # (n_states, 660, 14)
            elif isinstance(sv_result, list):
                sv_array_p = np.stack(sv_result, axis=-1)
            else:
                sv_array_p = np.array(sv_result)
            # shap sometimes returns (n_states, n_features) if single output? but we have 14
            # Ensure shape is (n_states, 660, 14)
            if sv_array_p.ndim == 2:
                # unlikely, expand
                sv_array_p = np.expand_dims(sv_array_p, -1)
            importance_p = np.mean(np.abs(sv_array_p), axis=(0, 2)) if sv_array_p.ndim==3 else np.mean(np.abs(sv_array_p), axis=0)
            t_p = time.time() - t0
            distinct_p = len(np.unique(np.round(importance_p, 10)))
            tied_p = int(np.sum(importance_p == np.min(importance_p)) if False else (NUM_FEATURES - distinct_p + 1))  # approx
            # more accurate tied count: count of min value
            unique_vals, counts = np.unique(np.round(importance_p, 10), return_counts=True)
            tied_count_p = int(counts[np.argmin(unique_vals)] if len(unique_vals)>0 else 0)
            print(f'Partition done {t_p:.1f}s |SHAP| [{importance_p.min():.6f},{importance_p.max():.6f}] distinct {distinct_p} tied_min {tied_count_p} shape {sv_array_p.shape}')

            # Kernel - expensive, run with reduced nsamples for 660-dim
            t1 = time.time()
            try:
                explainer_k = shap.KernelExplainer(predict_fn, sampled_bg)
                # shap.KernelExplainer.shap_values expects nsamples; for multi-output it returns list
                sv_k = explainer_k.shap_values(X_test, nsamples=args.nsamples, silent=True)
                # sv_k is list of 14 arrays each (n_states, 660) or array (n_states, 660, 14)
                if isinstance(sv_k, list):
                    sv_array_k = np.stack(sv_k, axis=-1)  # (n_states, 660, 14)
                elif isinstance(sv_k, np.ndarray) and sv_k.ndim==3:
                    sv_array_k = sv_k
                else:
                    sv_array_k = np.array(sv_k)
                importance_k = np.mean(np.abs(sv_array_k), axis=(0, 2)) if sv_array_k.ndim==3 else np.mean(np.abs(sv_array_k), axis=0)
                t_k = time.time() - t1
                distinct_k = len(np.unique(np.round(importance_k, 10)))
                unique_vals_k, counts_k = np.unique(np.round(importance_k, 10), return_counts=True)
                tied_count_k = int(counts_k[np.argmin(unique_vals_k)] if len(unique_vals_k)>0 else 0)
                print(f'Kernel done {t_k:.1f}s |SHAP| [{importance_k.min():.6f},{importance_k.max():.6f}] distinct {distinct_k} tied_min {tied_count_k} shape {sv_array_k.shape}')
            except Exception as e:
                print(f'Kernel failed for {agent_name} {sc}: {e}')
                import traceback; traceback.print_exc()
                importance_k = np.full_like(importance_p, np.nan)
                t_k = -1
                distinct_k = -1
                tied_count_k = -1
                sv_array_k = None

            # Metrics between explainers
            if not np.isnan(importance_k).all():
                # Top-20 overlap
                top20_p_idx = np.argsort(importance_p)[-20:][::-1]
                top20_k_idx = np.argsort(importance_k)[-20:][::-1]
                top20_p_set = set(top20_p_idx)
                top20_k_set = set(top20_k_idx)
                jaccard20 = len(top20_p_set & top20_k_set) / len(top20_p_set | top20_k_set) if len(top20_p_set | top20_k_set)>0 else 0
                # Spearman
                try:
                    from scipy.stats import spearmanr
                    rho, _ = spearmanr(importance_p, importance_k)
                except Exception:
                    rho = float(np.corrcoef(np.argsort(importance_p), np.argsort(importance_k))[0,1])
                # RBO
                def rbo(list1, list2, p=0.9):
                    # simple RBO
                    overlap=0
                    score=0
                    for k in range(1, len(list1)+1):
                        overlap = len(set(list1[:k]) & set(list2[:k]))
                        score += (p**(k-1)) * (overlap/k)
                    return (1-p)*score
                rbo20 = rbo(list(top20_p_idx), list(top20_k_idx), p=0.9) if len(top20_p_idx)==20 else 0
                # Top-5 names
                top5_p_names = [FEATURE_NAMES_660[i] for i in np.argsort(importance_p)[-5:][::-1]]
                top5_k_names = [FEATURE_NAMES_660[i] for i in np.argsort(importance_k)[-5:][::-1]]
            else:
                jaccard20 = rho = rbo20 = np.nan
                top5_p_names = []; top5_k_names = []

            # Record detailed per-feature rows for this config
            for idx in range(NUM_FEATURES):
                results.append({
                    'Agent': agent_name,
                    'Scenario': sc,
                    'FeatureIdx': idx,
                    'FeatureName': FEATURE_NAMES_660[idx],
                    'Partition_MeanAbsSHAP': float(importance_p[idx]),
                    'Kernel_MeanAbsSHAP': float(importance_k[idx]) if not np.isnan(importance_k[idx]) else '',
                    'Partition_rank': int(np.argsort(np.argsort(-importance_p))[idx])+1,
                    'Kernel_rank': int(np.argsort(np.argsort(-importance_k))[idx])+1 if not np.isnan(importance_k[idx]) else '',
                })

            # Summary row
            summary = {
                'Agent': agent_name,
                'Scenario': sc,
                'n_states': args.n_states,
                'nsamples': args.nsamples,
                'partition_time_s': round(t_p,1),
                'kernel_time_s': round(t_k,1) if t_k!=-1 else '',
                'partition_distinct': distinct_p,
                'kernel_distinct': distinct_k,
                'partition_tied_min': tied_count_p,
                'kernel_tied_min': tied_count_k,
                'partition_top5': '|'.join(top5_p_names) if 'top5_p_names' in locals() else '',
                'kernel_top5': '|'.join(top5_k_names) if 'top5_k_names' in locals() else '',
                'jaccard20': round(jaccard20,3) if not np.isnan(jaccard20) else '',
                'spearman_rho': round(float(rho),3) if not np.isnan(rho) else '',
                'rbo_p09': round(rbo20,3) if not np.isnan(rbo20) else '',
            }
            print(f"Summary {agent_name} {sc}: distinct P {distinct_p} vs K {distinct_k}, Jaccard20 {summary['jaccard20']}, rho {summary['spearman_rho']}, RBO {summary['rbo_p09']}, time P {t_p:.1f}s K {t_k:.1f}s")
            print(f"  Top5 P: {summary['partition_top5']}")
            print(f"  Top5 K: {summary['kernel_top5']}")

    # Save detailed csv
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved detailed comparison to {args.out} rows {len(df)}")

    # Also save summary csv
    # Recompute summary from already printed? Instead build from results aggregation
    # For simplicity, summarize file is the same out, user can group. Also create summary file
    summary_path = args.out.replace('.csv', '_summary.csv')
    # Build summary df from distinct per config (deduplicate)
    # We have stored summary dicts per loop, reconstruct by grouping
    summary_rows = []
    for (ag, sc), g in df.groupby(['Agent','Scenario']):
        # Need to recover partition/kernel importance from first occurrence per feature? Already have per feature
        # Compute metrics again for summary file
        imp_p = g['Partition_MeanAbsSHAP'].values
        # kernel may have '' for failed
        imp_k_vals = pd.to_numeric(g['Kernel_MeanAbsSHAP'], errors='coerce').values
        if np.isnan(imp_k_vals).all():
            continue
        distinct_p = len(np.unique(np.round(imp_p, 10)))
        distinct_k = len(np.unique(np.round(imp_k_vals, 10)))
        top20_p = np.argsort(imp_p)[-20:][::-1]
        top20_k = np.argsort(imp_k_vals)[-20:][::-1]
        jacc = len(set(top20_p)&set(top20_k))/len(set(top20_p)|set(top20_k))
        try:
            from scipy.stats import spearmanr
            rho,_ = spearmanr(imp_p, imp_k_vals)
        except:
            rho = np.nan
        summary_rows.append({
            'Agent': ag, 'Scenario': sc,
            'partition_distinct': distinct_p,
            'kernel_distinct': distinct_k,
            'jaccard20': round(jacc,3),
            'spearman': round(float(rho),3) if not np.isnan(rho) else '',
            'partition_top5': '|'.join([FEATURE_NAMES_660[i] for i in np.argsort(imp_p)[-5:][::-1]]),
            'kernel_top5': '|'.join([FEATURE_NAMES_660[i] for i in np.argsort(imp_k_vals)[-5:][::-1]])
        })
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f'Saved summary to {summary_path}')

if __name__ == '__main__':
    main()
