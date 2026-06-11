# =====================================================================
# ZATRON — Learned Inversion Attack (ML attack test)
# Question: can a neural network, trained on 100K pairs with known
# similarities, learn to predict document similarity from ZATRON's
# masked barcodes? (known-plaintext threat model — strong attacker)
#
# Control: the SAME attack on unmasked quantized signals must SUCCEED
# (proves the attack is real). Then if it fails on ZATRON barcodes,
# the claim "survives learned inversion attacks" is earned.
#
# Runtime on T4: ~10-15 min total.
# =====================================================================

!pip install sentence-transformers datasets scikit-learn scipy -q

import numpy as np, torch, torch.nn as nn, time, random, math, warnings
warnings.filterwarnings('ignore')
from sklearn.utils.extmath import randomized_svd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

random.seed(42); np.random.seed(42); torch.manual_seed(42)
dev='cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {dev}")

# ---------- data: 50K MSMARCO ----------
N_DOCS=50000
ds=load_dataset("microsoft/ms_marco","v1.1",split="train")
passages=[];seen=set()
for item in ds:
    for p in item['passages']['passage_text']:
        if p not in seen and len(p)>30:
            seen.add(p);passages.append(p[:300])
        if len(passages)>=N_DOCS: break
    if len(passages)>=N_DOCS: break
N=len(passages);print(f"{N:,} passages")

sbert=SentenceTransformer('all-MiniLM-L6-v2')
emb=sbert.encode(passages,show_progress_bar=True,batch_size=512).astype(np.float32)
norms=np.linalg.norm(emb,axis=1,keepdims=True)
embn=emb/(norms+1e-10)

# ---------- ZATRON encoding (200ch, 50-bin) ----------
nc=200; N_BINS=50; PRIMES=[53,59,61,67,71,73]
me=emb.mean(0); ce=emb-me
_,_,Vt=randomized_svd(ce,n_components=nc,random_state=42)
proj=(ce@Vt[:nc].T).astype(np.float32)
pmin=proj.min(0);prng=proj.max(0)-pmin+1e-8
sig=np.clip(((proj-pmin)/prng*(N_BINS-1)),0,N_BINS-1).astype(np.int64)

# salts: uniform random per doc per prime (statistically identical to
# HMAC rejection-sampling for attack purposes; HMAC is just slower)
rng=np.random.default_rng(20260611)
salts={p:rng.integers(0,p,size=(N,nc),dtype=np.int64) for p in PRIMES}
barcodes={p:(sig+salts[p])%p for p in PRIMES}          # ZATRON (masked)
raw_res={p:sig%p for p in PRIMES}                       # control (unmasked)

# ---------- build labeled pairs ----------
print("Building pairs (GPU topk)...")
n_anchors=2500
anchors=random.sample(range(N),n_anchors)
embn_t=torch.from_numpy(embn).to(dev)
pairs=[];labels=[]
for a in anchors:
    cs=embn_t@embn_t[a]; cs[a]=-999
    top=torch.topk(cs,10).indices.cpu().tolist()       # similar pairs
    rnd=random.sample(range(N),30)                      # mostly dissimilar
    for b in top+rnd:
        if b==a: continue
        pairs.append((a,b))
        labels.append(float(embn[a]@embn[b]))
pairs=np.array(pairs);labels=np.array(labels,dtype=np.float32)
print(f"{len(pairs):,} pairs")

# split by ANCHOR (no leakage between train/test)
anchor_set=list(set(pairs[:,0].tolist()))
random.shuffle(anchor_set)
cut=int(0.8*len(anchor_set))
train_anchors=set(anchor_set[:cut])
tr_mask=np.array([a in train_anchors for a in pairs[:,0]])
te_mask=~tr_mask
print(f"train {tr_mask.sum():,} | test {te_mask.sum():,}")

# ---------- features: per-prime circular differences ----------
def pair_features(table):
    feats=[]
    for p in PRIMES:
        d=np.abs(table[p][pairs[:,0]]-table[p][pairs[:,1]])
        feats.append(np.minimum(d,p-d).astype(np.float32)/p)
    return np.concatenate(feats,axis=1)                 # (n_pairs, 6*nc)

print("Features...")
X_zatron=pair_features(barcodes)
X_raw=pair_features(raw_res)

# ---------- attacks ----------
def linear_attack(X,y,trm,tem,tag):
    Xt=torch.from_numpy(X[trm]).to(dev);yt=torch.from_numpy(y[trm]).to(dev)
    Xe=torch.from_numpy(X[tem]).to(dev)
    w=torch.zeros(X.shape[1],device=dev,requires_grad=True)
    b=torch.zeros(1,device=dev,requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=0.01)
    for ep in range(300):
        opt.zero_grad();pred=Xt@w+b
        loss=((pred-yt)**2).mean();loss.backward();opt.step()
    pred=(Xe@w+b).detach().cpu().numpy()
    rho=spearmanr(pred,y[tem]).correlation
    thr=np.quantile(y[tem],0.9)
    auc=roc_auc_score((y[tem]>=thr).astype(int),pred)
    print(f"    linear  {tag}: rho={rho:+.3f}  AUC={auc:.3f}")
    return rho,auc

def mlp_attack(X,y,trm,tem,tag):
    Xt=torch.from_numpy(X[trm]).to(dev);yt=torch.from_numpy(y[trm]).to(dev)
    Xe=torch.from_numpy(X[tem]).to(dev)
    net=nn.Sequential(nn.Linear(X.shape[1],512),nn.ReLU(),
                      nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1)).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    ds_idx=torch.randperm(len(Xt))
    for ep in range(15):
        for i in range(0,len(Xt),4096):
            idx=ds_idx[i:i+4096]
            opt.zero_grad()
            loss=((net(Xt[idx]).squeeze()-yt[idx])**2).mean()
            loss.backward();opt.step()
    with torch.no_grad():
        pred=net(Xe).squeeze().cpu().numpy()
    rho=spearmanr(pred,y[tem]).correlation
    thr=np.quantile(y[tem],0.9)
    auc=roc_auc_score((y[tem]>=thr).astype(int),pred)
    print(f"    MLP     {tag}: rho={rho:+.3f}  AUC={auc:.3f}")
    return rho,auc

print("\n"+"="*70)
print("LEARNED INVERSION ATTACK — results")
print("(rho/AUC near 0/0.5 = attack fails; high = attack succeeds)")
print("="*70)
print("\n  CONTROL — unmasked quantized signals (attack SHOULD succeed):")
linear_attack(X_raw,labels,tr_mask,te_mask,"raw   ")
mlp_attack(X_raw,labels,tr_mask,te_mask,"raw   ")
print("\n  ZATRON — salt-masked barcodes (the real question):")
linear_attack(X_zatron,labels,tr_mask,te_mask,"zatron")
mlp_attack(X_zatron,labels,tr_mask,te_mask,"zatron")
print("\nInterpretation:")
print("  If raw rho is high (>0.5) and zatron rho ~0 (AUC ~0.5),")
print("  ZATRON survives a learned inversion attack with "
      f"{tr_mask.sum():,} training pairs.")
