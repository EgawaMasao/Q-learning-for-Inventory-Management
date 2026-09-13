"""
sensitivity_task15_full.py - Task 15: SHAP sensitivity actor vs critic (EASY, MEDIUM, HARD)
Bám sát planTask11-9.md:138-150 - Chạy SHAP trên 3 outputs của A2C_mod:
  - pi (softmax)
  - logits (pre-softmax)
  - V(s) critic scalar
Với 10 states cho mỗi scenario EASY/MEDIUM/HARD (30 states total, background 100, 660-dim)
So sánh Jaccard Top-20 giữa pi vs logits và pi vs V(s) cho mỗi scenario

Cách chạy:
  py scripts/sensitivity_task15_full.py
Output:
  ../outputTask11_sensitivity.csv (9 dòng: 3 scenarios x 3 comparisons)

Thời gian: ~25 phút cho 9 configs (3 scenarios x 3 outputs)
"""
import os, warnings, time, numpy as np, tensorflow as tf, pandas as pd, shap
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
warnings.filterwarnings('ignore')

BASE=r'C:\GitHub\Q-learning-for-Inventory-Management'
A2C_CKPT=os.path.join(BASE,'outputA2Cmod','checkpoints_a2cmod')
DQN_CKPT=os.path.join(BASE,'checkpoints_dqn_comparison512_32')
DATA_DIR=os.path.join(BASE,'data')
CAP_FILE=os.path.join(DATA_DIR,'capacity.tfrecords')
STOCK_FILE=os.path.join(DATA_DIR,'stock.tfrecords')
TEST_FILE=os.path.join(DATA_DIR,'test.tfrecords')

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
        x=self.activation(self.layer1(state)); x=tf.nn.dropout(x,self.dropout_prob)
        x=self.activation(self.layer2(x)); x=tf.nn.dropout(x,self.dropout_prob)
        x=self.activation(self.layer3(x)); x=tf.nn.dropout(x,self.dropout_prob)
        x=self.layer4(x); return tf.nn.softmax(x)
    def logits(self,state):
        x=self.activation(self.layer1(state)); x=tf.nn.dropout(x,self.dropout_prob)
        x=self.activation(self.layer2(x)); x=tf.nn.dropout(x,self.dropout_prob)
        x=self.activation(self.layer3(x)); x=tf.nn.dropout(x,self.dropout_prob)
        x=self.layer4(x); return x

class Critic(tf.Module):
    def __init__(self, num_features, hidden_size, activation=tf.nn.relu, dropout_prob=0.1):
        super().__init__()
        self.layer1=Dense(num_features, hidden_size)
        self.layer2=Dense(hidden_size, 1)
        self.activation=activation
        self.dropout_prob=dropout_prob
        self.group_norm=tf.keras.layers.GroupNormalization(groups=1)
    def __call__(self,state):
        x=self.layer1(state); x=self.group_norm(x); x=self.activation(x); x=tf.nn.dropout(x,self.dropout_prob)
        x=self.layer2(x); return tf.squeeze(x, axis=-1)

class MultiProductQNetwork(tf.keras.Model):
    def __init__(self, num_features, num_products, num_actions, hidden_size, dropout_prob=0.1, use_group_norm=True):
        super().__init__()
        self.num_products=num_products; self.num_actions=num_actions; self.features_per_prod=num_features//num_products
        self.dense1=tf.keras.layers.Dense(hidden_size, activation=None)
        self.dense2=tf.keras.layers.Dense(hidden_size, activation=None)
        self.dense3=tf.keras.layers.Dense(hidden_size, activation=None)
        self.out=tf.keras.layers.Dense(num_actions, activation=None)
        self._use_gn=use_group_norm
        if use_group_norm:
            self.gn1=tf.keras.layers.GroupNormalization(groups=1)
            self.gn2=tf.keras.layers.GroupNormalization(groups=1)
            self.gn3=tf.keras.layers.GroupNormalization(groups=1)
        self.drop1=tf.keras.layers.Dropout(dropout_prob); self.drop2=tf.keras.layers.Dropout(dropout_prob); self.drop3=tf.keras.layers.Dropout(dropout_prob)
    def call(self,state,training=False):
        B=tf.shape(state)[0]; P,F=self.num_products,self.features_per_prod
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

print('Loading A2C and DQN...')
actor=Actor(3,14,32); critic=Critic(3,32)
_=actor(tf.zeros([1,3])); _=critic(tf.zeros([1,3]))
a2c_ckpt=tf.train.Checkpoint(critic_optimizer=tf.optimizers.Adam(0.0005), actor_optimizer=tf.optimizers.Adam(0.0001), critic=critic, actor=actor, step=tf.Variable(0))
a2c_ckpt.restore(tf.train.latest_checkpoint(A2C_CKPT)).expect_partial()
print('A2C restored', tf.train.latest_checkpoint(A2C_CKPT))
q_net=MultiProductQNetwork(660,220,14,32); t_net=MultiProductQNetwork(660,220,14,32)
_=q_net(tf.zeros([1,660],dtype=tf.float32), training=False); _=t_net(tf.zeros([1,660],dtype=tf.float32), training=False)
dqn_ckpt=tf.train.Checkpoint(optimizer=tf.optimizers.Adam(0.001), q_network=q_net, target_network=t_net, step=tf.Variable(0,dtype=tf.int64))
dqn_ckpt.restore(tf.train.latest_checkpoint(DQN_CKPT)).expect_partial()
print('DQN restored', tf.train.latest_checkpoint(DQN_CKPT))

def _parse(s,key,n):
    return tf.io.parse_single_example(s,{key:tf.io.FixedLenFeature([n],tf.float32)})[key]
capacity=next(iter(tf.data.TFRecordDataset(CAP_FILE).map(lambda s:_parse(s,'capacity',220)))).numpy()
x_init=next(iter(tf.data.TFRecordDataset(STOCK_FILE).map(lambda s:_parse(s,'stock',220)))).numpy()
all_sales=[]
for rec in tf.data.TFRecordDataset(TEST_FILE).map(lambda s:_parse(s,'sales',220)):
    all_sales.append(rec.numpy())
all_sales=np.array(all_sales,dtype=np.float32)/capacity[None,:]
print('Data', all_sales.shape)

np.random.seed(42)
bg=np.zeros((100,660),dtype=np.float32)
bg[:,:220]=np.random.uniform(0,1,size=(100,220))
bg[:,220:440]=np.random.uniform(0,1,size=(100,220))
bg[:,440:]=np.clip(0.025*bg[:,:220]+np.random.normal(0,0.005,size=(100,220)),0,0.1)

def make_test(scenario, n=10):
    scale={'EASY':0.5,'MEDIUM':1.0,'HARD':1.5}[scenario]
    waste={'EASY':0.010,'MEDIUM':0.025,'HARD':0.050}[scenario]
    sales=all_sales[:n]*scale
    s=np.zeros((n,660),dtype=np.float32)
    s[:,:220]=x_init[None,:]
    s[:,220:440]=sales
    s[:,440:]=x_init[None,:]*waste
    return s

def dqn_logits_660(X):
    if not isinstance(X,np.ndarray): X=np.array(X,dtype=np.float32)
    q=q_net(X, training=False)
    return tf.reduce_mean(q, axis=1).numpy()
def dqn_softmax_660(X):
    if not isinstance(X,np.ndarray): X=np.array(X,dtype=np.float32)
    q=q_net(X, training=False)
    return tf.nn.softmax(tf.reduce_mean(q, axis=1)).numpy()
def a2c_pi_660(X):
    if not isinstance(X,np.ndarray): X=np.array(X,dtype=np.float32)
    B=X.shape[0]; s3d=tf.transpose(tf.reshape(X,[B,3,220]),[0,2,1]); s_pp=tf.reshape(s3d,[B*220,3])
    probs=actor(s_pp); probs_3d=tf.reshape(probs,[B,220,14]); return tf.reduce_mean(probs_3d, axis=1).numpy()
def a2c_logits_660(X):
    if not isinstance(X,np.ndarray): X=np.array(X,dtype=np.float32)
    B=X.shape[0]; s3d=tf.transpose(tf.reshape(X,[B,3,220]),[0,2,1]); s_pp=tf.reshape(s3d,[B*220,3])
    logits=actor.logits(s_pp); logits_3d=tf.reshape(logits,[B,220,14]); return tf.reduce_mean(logits_3d, axis=1).numpy()
def a2c_critic_660(X):
    if not isinstance(X,np.ndarray): X=np.array(X,dtype=np.float32)
    B=X.shape[0]; s3d=tf.transpose(tf.reshape(X,[B,3,220]),[0,2,1]); s_pp=tf.reshape(s3d,[B*220,3])
    v=critic(s_pp); v_3d=tf.reshape(v,[B,220]); return tf.reduce_mean(v_3d, axis=1, keepdims=True).numpy()

def get_imp(fn, test):
    masker=shap.maskers.Partition(bg, max_samples=100)
    explainer=shap.PartitionExplainer(fn, masker)
    sv=explainer(test)
    arr=sv.values if hasattr(sv,'values') else np.array(sv)
    if arr.ndim==3:
        imp=np.mean(np.abs(arr), axis=(0,2))
    elif arr.ndim==2:
        imp=np.mean(np.abs(arr), axis=0)
    else:
        imp=np.mean(np.abs(arr), axis=0)
    return imp

# Run for EASY, MEDIUM, HARD 10-state for Task 15
# Task 15: A2C pi vs logits and pi vs V(s) for each scenario, plus DQN softmax vs logits
results=[]
for scenario in ['EASY','MEDIUM','HARD']:
    test=make_test(scenario,10)
    print(f'\n=== {scenario} 10-state Task 15 ===')
    imps={}
    for name, fn in [('DQN_softmax', dqn_softmax_660), ('DQN_logits', dqn_logits_660), ('A2C_pi', a2c_pi_660), ('A2C_logits', a2c_logits_660), ('A2C_critic', a2c_critic_660)]:
        imp=get_imp(fn, test)
        imps[name]=imp
        top5=np.argsort(imp)[-5:][::-1]
        def n(i):
            if i<220: return f'inventory_SKU{i}'
            elif i<440: return f'sales_SKU{i-220}'
            else: return f'waste_feat_SKU{i-440}'
        print(f'{name} Top-5 {[n(i) for i in top5]}')

    def top20_set(imp): return set(np.argsort(imp)[-20:][::-1])
    def jaccard(a,b):
        sa=top20_set(imps[a]); sb=top20_set(imps[b])
        return len(sa&sb)/len(sa|sb)
    def rbo_score(a,b,p=0.9):
        # RBO for Top-20
        l1=list(np.argsort(imps[a])[-20:][::-1])
        l2=list(np.argsort(imps[b])[-20:][::-1])
        n=max(len(l1),len(l2))
        score=0; sa=set(); sb=set()
        for d in range(1,n+1):
            if d<=len(l1): sa.add(l1[d-1])
            if d<=len(l2): sb.add(l2[d-1])
            overlap=len(sa&sb)/d if d>0 else 0
            score+=(p**(d-1))*overlap
        return (1-p)*score

    # DQN softmax vs logits
    results.append({'Agent':'DQN','Scenario':scenario,'Comparison':'softmax(Q) vs logits','k':20,'Jaccard':round(jaccard('DQN_softmax','DQN_logits'),3),'RBO_p09':round(rbo_score('DQN_softmax','DQN_logits'),3),'Note':'Same Top-8 sales' if jaccard('DQN_softmax','DQN_logits')>0.5 else 'Different'})
    # A2C pi vs logits
    results.append({'Agent':'A2C_mod','Scenario':scenario,'Comparison':'pi vs logits','k':20,'Jaccard':round(jaccard('A2C_pi','A2C_logits'),3),'RBO_p09':round(rbo_score('A2C_pi','A2C_logits'),3),'Note':'Sales dominant both' if jaccard('A2C_pi','A2C_logits')>0.5 else 'Ranking different'})
    # A2C pi vs V(s)
    results.append({'Agent':'A2C_mod','Scenario':scenario,'Comparison':'pi vs V(s)','k':20,'Jaccard':round(jaccard('A2C_pi','A2C_critic'),3),'RBO_p09':round(rbo_score('A2C_pi','A2C_critic'),3),'Note':'Actor vs Critic'})

out_csv=r'C:\GitHub\Q-learning-for-Inventory-Management\Feedback 7-9\task11-9\outputTask11_sensitivity.csv'
pd.DataFrame(results).to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f'\nSaved {out_csv}')
print(pd.DataFrame(results).to_string(index=False))
