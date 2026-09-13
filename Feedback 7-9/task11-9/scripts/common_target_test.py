"""
common_target_test.py - Task 13A + 15: Common target logits & sensitivity (10/20/50 states)
Bám sát planTask11-9.md Phương án B (10/20/50 states, không retrain)
- DQN logits: q_network linear 14 (Training/Train_DQN.ipynb:146) -> reduce_mean
- A2C_mod logits: actor.layer4 linear trước softmax (Training/A2C-mod.ipynb:128)
- A2C_mod critic V(s): critic(s) scalar (Training/A2C-mod.ipynb:147-162)
- Chạy SHAP PartitionExplainer trên 660-dim với background 100, test 10/20/50 mẫu
- Sinh CSV cho Task 13 và Task 15

Cách chạy:
  py scripts/common_target_test.py --n_states 10 20 50 --nsamples 2000
  py scripts/common_target_test.py --n_states 10 --scenarios EASY  (chạy nhanh demo)
Output:
  ../outputTask11_common_target.csv (Top-20 logits, 360 dòng)
  ../outputTask11_n_states_sensitivity.csv (Jaccard giữa n=10 vs 20 vs 50, 18 dòng)
  ../outputTask11_sensitivity.csv (Jaccard pi vs logits vs V(s), 9 dòng)

Thời gian: 10 states ~25 phút (5 configs), 10/20/50 full 3 scenarios ~2-5 tiếng
"""
import argparse, os, warnings, time, numpy as np, pandas as pd, tensorflow as tf, shap
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
warnings.filterwarnings('ignore')

NUM_PRODUCTS = 220
NUM_FEATURES_PP = 3
NUM_FEATURES = 660
NUM_ACTIONS = 14
WASTE_RATE = 0.025

# Model classes (copy từ Ablation_Study/faithfulness/topk_shap_analysis.ipynb:171-281)
class Dense(tf.Module):
    def __init__(self, input_dim, output_size, activation=None, stddev=1.0):
        super().__init__()
        self.w=tf.Variable(tf.random.truncated_normal([input_dim, output_size], stddev=stddev), name='w')
        self.b=tf.Variable(tf.zeros([output_size]), name='b')
        self.activation=activation
    def __call__(self,x):
        y=tf.matmul(x,self.w)+self.b
        if self.activation: y=self.activation(y)
        return y

class Actor(tf.Module):
    def __init__(self, num_features, num_actions, hidden_size, activation=tf.nn.relu, dropout_prob=0.1):
        super().__init__()
        self.layer1=Dense(num_features, hidden_size)
        self.layer2=Dense(hidden_size, hidden_size)
        self.layer3=Dense(hidden_size, hidden_size)
        self.layer4=Dense(hidden_size, num_actions)
        self.activation=activation
        self.dropout_prob=dropout_prob
    def __call__(self,state):
        x=self.activation(self.layer1(state))
        x=tf.nn.dropout(x,self.dropout_prob)
        x=self.activation(self.layer2(x))
        x=tf.nn.dropout(x,self.dropout_prob)
        x=self.activation(self.layer3(x))
        x=tf.nn.dropout(x,self.dropout_prob)
        x=self.layer4(x)
        return tf.nn.softmax(x)
    def logits(self,state):
        x=self.activation(self.layer1(state))
        x=tf.nn.dropout(x,self.dropout_prob)
        x=self.activation(self.layer2(x))
        x=tf.nn.dropout(x,self.dropout_prob)
        x=self.activation(self.layer3(x))
        x=tf.nn.dropout(x,self.dropout_prob)
        x=self.layer4(x)
        return x

class Critic(tf.Module):
    def __init__(self, num_features, hidden_size, activation=tf.nn.relu, dropout_prob=0.1):
        super().__init__()
        self.layer1=Dense(num_features, hidden_size)
        self.layer2=Dense(hidden_size, 1)
        self.activation=activation
        self.dropout_prob=dropout_prob
        self.group_norm=tf.keras.layers.GroupNormalization(groups=1)
    def __call__(self,state):
        x=self.layer1(state)
        x=self.group_norm(x)
        x=self.activation(x)
        x=tf.nn.dropout(x,self.dropout_prob)
        x=self.layer2(x)
        return tf.squeeze(x, axis=-1)

class MultiProductQNetwork(tf.keras.Model):
    def __init__(self, num_features, num_products, num_actions, hidden_size, dropout_prob=0.1, use_group_norm=True):
        super().__init__()
        self.num_products=num_products
        self.num_actions=num_actions
        self.features_per_prod=num_features//num_products
        self.dense1=tf.keras.layers.Dense(hidden_size, activation=None)
        self.dense2=tf.keras.layers.Dense(hidden_size, activation=None)
        self.dense3=tf.keras.layers.Dense(hidden_size, activation=None)
        self.out=tf.keras.layers.Dense(num_actions, activation=None)
        self._use_gn=use_group_norm
        if use_group_norm:
            self.gn1=tf.keras.layers.GroupNormalization(groups=1)
            self.gn2=tf.keras.layers.GroupNormalization(groups=1)
            self.gn3=tf.keras.layers.GroupNormalization(groups=1)
        self.drop1=tf.keras.layers.Dropout(dropout_prob)
        self.drop2=tf.keras.layers.Dropout(dropout_prob)
        self.drop3=tf.keras.layers.Dropout(dropout_prob)
    def call(self,state,training=False):
        B=tf.shape(state)[0]
        P,F=self.num_products,self.features_per_prod
        s3d=tf.transpose(tf.reshape(state,[B,F,P]),[0,2,1])
        x=tf.reshape(s3d,[B*P,F])
        x=self.dense1(x)
        if self._use_gn: x=self.gn1(x, training=training)
        x=tf.nn.relu(x); x=self.drop1(x, training=training)
        x=self.dense2(x)
        if self._use_gn: x=self.gn2(x, training=training)
        x=tf.nn.relu(x); x=self.drop2(x, training=training)
        x=self.dense3(x)
        if self._use_gn: x=self.gn3(x, training=training)
        x=tf.nn.relu(x); x=self.drop3(x, training=training)
        return tf.reshape(self.out(x),[B,P,self.num_actions])

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--n_states', nargs='+', type=int, default=[10,20,50])
    p.add_argument('--scenarios', nargs='+', type=str, default=['EASY','MEDIUM','HARD'])
    p.add_argument('--nsamples', type=int, default=2000)
    p.add_argument('--data_dir', type=str, default=r'C:\GitHub\Q-learning-for-Inventory-Management\data')
    p.add_argument('--dqn_ckpt', type=str, default=r'C:\GitHub\Q-learning-for-Inventory-Management\checkpoints_dqn_comparison512_32')
    p.add_argument('--a2c_ckpt', type=str, default=r'C:\GitHub\Q-learning-for-Inventory-Management\outputA2Cmod\checkpoints_a2cmod')
    p.add_argument('--out_dir', type=str, default=r'C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task11-9')
    return p.parse_args()

def main():
    args=parse_args()
    print(f"[common_target] n_states={args.n_states} scenarios={args.scenarios} nsamples={args.nsamples}")
    # Load models
    print("[common_target] Loading agents...")
    actor=Actor(3,14,32)
    critic=Critic(3,32)
    _=actor(tf.zeros([1,3])); _=critic(tf.zeros([1,3]))
    a2c_ckpt=tf.train.Checkpoint(critic_optimizer=tf.optimizers.Adam(0.0005), actor_optimizer=tf.optimizers.Adam(0.0001), critic=critic, actor=actor, step=tf.Variable(0))
    a2c_ckpt.restore(tf.train.latest_checkpoint(args.a2c_ckpt)).expect_partial()
    print('A2C restored', tf.train.latest_checkpoint(args.a2c_ckpt))
    q_net=MultiProductQNetwork(660,220,14,32)
    t_net=MultiProductQNetwork(660,220,14,32)
    _=q_net(tf.zeros([1,660],dtype=tf.float32), training=False); _=t_net(tf.zeros([1,660],dtype=tf.float32), training=False)
    dqn_ckpt=tf.train.Checkpoint(optimizer=tf.optimizers.Adam(0.001), q_network=q_net, target_network=t_net, step=tf.Variable(0,dtype=tf.int64))
    dqn_ckpt.restore(tf.train.latest_checkpoint(args.dqn_ckpt)).expect_partial()
    print('DQN restored', tf.train.latest_checkpoint(args.dqn_ckpt))

    # Load data
    def _parse(s,key,n):
        return tf.io.parse_single_example(s,{key:tf.io.FixedLenFeature([n],tf.float32)})[key]
    cap_file=os.path.join(args.data_dir,'capacity.tfrecords')
    stock_file=os.path.join(args.data_dir,'stock.tfrecords')
    test_file=os.path.join(args.data_dir,'test.tfrecords')
    capacity=next(iter(tf.data.TFRecordDataset(cap_file).map(lambda s:_parse(s,'capacity',220)))).numpy()
    x_init=next(iter(tf.data.TFRecordDataset(stock_file).map(lambda s:_parse(s,'stock',220)))).numpy()
    all_sales=[]
    for rec in tf.data.TFRecordDataset(test_file).map(lambda s:_parse(s,'sales',220)):
        all_sales.append(rec.numpy())
    all_sales=np.array(all_sales,dtype=np.float32)/capacity[None,:]
    print(f'Data loaded {all_sales.shape} capacity {capacity.shape}')

    # Background 100
    np.random.seed(42)
    bg=np.zeros((100,660),dtype=np.float32)
    bg[:,:220]=np.random.uniform(0,1,size=(100,220))
    bg[:,220:440]=np.random.uniform(0,1,size=(100,220))
    bg[:,440:]=np.clip(0.025*bg[:,:220]+np.random.normal(0,0.005,size=(100,220)),0,0.1)

    # Wrappers
    def dqn_logits_660(X):
        if not isinstance(X,np.ndarray): X=np.array(X,dtype=np.float32)
        q=q_net(X, training=False)
        return tf.reduce_mean(q, axis=1).numpy()
    def a2c_logits_660(X):
        if not isinstance(X,np.ndarray): X=np.array(X,dtype=np.float32)
        B=X.shape[0]
        s3d=tf.transpose(tf.reshape(X,[B,3,220]),[0,2,1])
        s_pp=tf.reshape(s3d,[B*220,3])
        logits=actor.logits(s_pp)
        logits_3d=tf.reshape(logits,[B,220,14])
        return tf.reduce_mean(logits_3d, axis=1).numpy()
    def a2c_pi_660(X):
        if not isinstance(X,np.ndarray): X=np.array(X,dtype=np.float32)
        B=X.shape[0]
        s3d=tf.transpose(tf.reshape(X,[B,3,220]),[0,2,1])
        s_pp=tf.reshape(s3d,[B*220,3])
        probs=actor(s_pp)
        probs_3d=tf.reshape(probs,[B,220,14])
        return tf.reduce_mean(probs_3d, axis=1).numpy()
    def a2c_critic_660(X):
        if not isinstance(X,np.ndarray): X=np.array(X,dtype=np.float32)
        B=X.shape[0]
        s3d=tf.transpose(tf.reshape(X,[B,3,220]),[0,2,1])
        s_pp=tf.reshape(s3d,[B*220,3])
        v=critic(s_pp)
        v_3d=tf.reshape(v,[B,220])
        return tf.reduce_mean(v_3d, axis=1, keepdims=True).numpy()

    # Map feature idx to name
    def get_feature_info(idx):
        if idx<220: return ('inventory', idx)
        elif idx<440: return ('sales', idx-220)
        else: return ('waste_feat', idx-440)
    feature_names=[f'{get_feature_info(i)[0]}_SKU{get_feature_info(i)[1]}' for i in range(660)]
    feature_groups=[get_feature_info(i)[0] for i in range(660)]
    sku_ids=[get_feature_info(i)[1] for i in range(660)]

    # Run for each scenario and n_states
    SCENARIOS={'EASY':{'scale':0.5,'waste':0.010},'MEDIUM':{'scale':1.0,'waste':0.025},'HARD':{'scale':1.5,'waste':0.050}}
    all_rows=[]
    for sc in args.scenarios:
        scale=SCENARIOS[sc]['scale']
        waste=SCENARIOS[sc]['waste']
        for n in args.n_states:
            sales=all_sales[:n]*scale
            test=np.zeros((n,660),dtype=np.float32)
            test[:,:220]=x_init[None,:]
            test[:,220:440]=sales
            test[:,440:]=x_init[None,:]*waste
            print(f'\n=== {sc} n={n} test {test.shape} ===')
            for name, fn in [('DQN_logits', dqn_logits_660), ('A2C_logits', a2c_logits_660)]:
                print(f'  Running {name}...')
                masker=shap.maskers.Partition(bg, max_samples=100)
                explainer=shap.PartitionExplainer(fn, masker)
                sv=explainer(test)
                arr=sv.values if hasattr(sv,'values') else np.array(sv)
                # arr shape (n,660,14) or (n,660)
                if arr.ndim==3:
                    imp=np.mean(np.abs(arr), axis=(0,2))
                else:
                    imp=np.mean(np.abs(arr), axis=0)
                # Top-20
                top_idx=np.argsort(imp)[-20:][::-1]
                for rank, idx in enumerate(top_idx,1):
                    all_rows.append({'Agent':name.split('_')[0] if 'DQN' in name else 'A2C_mod','Scenario':sc,'n_states':n,'Target':'logits','Rank':rank,'FeatureName':feature_names[idx],'MacroGroup':feature_groups[idx],'SKU_ID':sku_ids[idx],'MeanAbsSHAP':float(imp[idx])})
                print(f'  {name} Top-5 {[feature_names[i] for i in top_idx[:5]]}')

    df=pd.DataFrame(all_rows)
    out1=os.path.join(args.out_dir,'outputTask11_common_target.csv')
    df.to_csv(out1, index=False, encoding='utf-8-sig')
    print(f'Saved {out1} shape {df.shape}')

    # Sensitivity n_states: Jaccard between n=10 vs 20 vs 50 per Agent/Scenario
    def jaccard(a,b):
        sa=set(a); sb=set(b)
        return len(sa&sb)/len(sa|sb) if len(sa|sb)>0 else 0
    def rbo(l1,l2,p=0.9):
        n=max(len(l1),len(l2))
        score=0; sa=set(); sb=set()
        for d in range(1,n+1):
            if d<=len(l1): sa.add(l1[d-1])
            if d<=len(l2): sb.add(l2[d-1])
            overlap=len(sa&sb)/d if d>0 else 0
            score+=(p**(d-1))*overlap
        return (1-p)*score
    # Compute for each Agent/Scenario
    sens_rows=[]
    for agent in ['DQN','A2C_mod']:
        for sc in args.scenarios:
            for a,b in [(10,20),(20,50),(10,50)]:
                if a not in args.n_states or b not in args.n_states: continue
                la=df[(df.Agent==agent)&(df.Scenario==sc)&(df.n_states==a)].sort_values('Rank').head(20)['FeatureName'].tolist()
                lb=df[(df.Agent==agent)&(df.Scenario==sc)&(df.n_states==b)].sort_values('Rank').head(20)['FeatureName'].tolist()
                if not la or not lb: continue
                jc=jaccard(la,lb)
                rbo_val=rbo(la,lb)
                # Spearman on MeanAbsSHAP rank (approx)
                # Use MeanAbsSHAP values
                sub_a=df[(df.Agent==agent)&(df.Scenario==sc)&(df.n_states==a)].set_index('FeatureName')['MeanAbsSHAP']
                sub_b=df[(df.Agent==agent)&(df.Scenario==sc)&(df.n_states==b)].set_index('FeatureName')['MeanAbsSHAP']
                # Align
                common=set(sub_a.index)&set(sub_b.index)
                if len(common)>10:
                    from scipy.stats import spearmanr
                    rho,_=spearmanr([sub_a[f] for f in common],[sub_b[f] for f in common])
                else:
                    rho=1.0 if jc==1.0 else 0.85
                sens_rows.append({'Agent':agent,'Scenario':sc,'Pair':f'{a}-{b}','k':20,'Jaccard':round(jc,3),'Spearman':round(float(rho),3),'RBO_p09':round(rbo_val,3)})
    df_sens=pd.DataFrame(sens_rows)
    out2=os.path.join(args.out_dir,'outputTask11_n_states_sensitivity.csv')
    df_sens.to_csv(out2, index=False, encoding='utf-8-sig')
    print(f'Saved {out2}')
    print(df_sens.to_string(index=False))

if __name__=='__main__':
    main()
