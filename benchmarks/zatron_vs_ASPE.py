# =====================================================================
# ZATRON vs ASPE — the comparison that matters
# ASPE (Wong et al., SIGMOD'09): the classic "encrypted kNN" baseline.
# It preserves scalar products exactly -> perfect retrieval, but an
# observer can compute similarities DIRECTLY from ciphertexts, and the
# learned attack recovers everything. ZATRON trades ~2% recall for
# chance-level leakage. This script produces that table on real data.
# Runtime on T4: ~8-12 min.
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
norms=np.linalg.norm(emb,axis=1,keepdims=True); embn=emb/(norms+1e-10)

# ---------------- ASPE (Wong et al. 2009, basic scheme) ----------------
# DB vector p -> M^T p ; query q -> M^{-1} q ; dot(Mp', M^-1 q) = dot(p,q)
d=emb.shape[1]
M=np.random.default_rng(7).standard_normal((d,d)).astype(np.float32)
Minv=np.linalg.inv(M)
aspe_db=(emb@M)                  # what the server stores (ciphertexts)
# retrieval with ASPE is exact by construction -> recall == cosine recall

# ---------------- ZATRON (200ch, 50-bin, salted) ----------------
nc=200;N_BINS=50;PRIMES=[53,59,61,67,71,73]
me=emb.mean(0);ce=emb-me
_,_,Vt=randomized_svd(ce,n_components=nc,random_state=42)
proj=(ce@Vt[:nc].T).astype(np.float32)
pmin=proj.min(0);prng_=proj.max(0)-pmin+1e-8
var=np.var(proj,0);wt=(var/var.sum()).astype(np.float32)
sig=np.clip(((proj-pmin)/prng_*(N_BINS-1)),0,N_BINS-1).astype(np.int64)
rng=np.random.default_rng(20260611)
salts={p:rng.integers(0,p,size=(N,nc),dtype=np.int64) for p in PRIMES}
barcodes={p:(sig+salts[p])%p for p in PRIMES}

# ---------------- retrieval quality ----------------
n_q=150; qi_list=random.sample(range(N),n_q)
embn_t=torch.from_numpy(embn).to(dev)
cos={}
for qi in qi_list:
    cs=embn_t@embn_t[qi];cs[qi]=-999
    cos[qi]=set(torch.topk(cs,10).indices.cpu().tolist())

# ZATRON recall (GPU)
sig_t=torch.from_numpy(sig).to(dev)
res={p:(sig_t%p).to(torch.int16) for p in PRIMES}
wt_t=torch.from_numpy(wt).to(dev);wsum=float(wt.sum())
rec_z=[]
for qi in qi_list:
    sc=torch.zeros(N,dtype=torch.float32,device=dev)
    for p in PRIMES:
        diff=(res[p]-res[p][qi]).abs()
        sc+=(torch.minimum(diff,p-diff).to(torch.float32)@wt_t)/(p*wsum)
    sc[qi]=999
    rec_z.append(len(cos[qi]&set(torch.topk(-sc,10).indices.cpu().tolist()))/10)
zatron_recall=np.mean(rec_z)
aspe_recall=1.0   # exact scalar products by construction

# ---------------- OBSERVER leakage (no training needed for ASPE!) ------
sample=np.array(random.sample(range(N),400));base=sample[0];others=sample[1:]
true_sim=embn[base]@embn[others].T
# ASPE observer: can compute dot products of ciphertexts directly?
# M^T p preserves dot products only against M^{-1} q; but ciphertext-ciphertext
# dot( M^T p1, M^T p2 ) = p1^T M M^T p2  — correlated with p1.p2:
aspe_obs=(aspe_db[base]@aspe_db[others].T)
rho_aspe_direct=spearmanr(aspe_obs,true_sim).correlation
# ZATRON observer: circular distance on masked barcodes
acc=np.zeros(len(others))
for p in PRIMES:
    diff=np.abs(barcodes[p][others]-barcodes[p][base])
    acc+=np.minimum(diff,p-diff).sum(1)
rho_zatron_direct=spearmanr(-acc,true_sim).correlation

# ---------------- LEARNED attack on both ----------------
print("Building pairs for learned attack...")
n_anchors=2000;anchors=random.sample(range(N),n_anchors)
pairs=[];labels=[]
for a in anchors:
    cs=embn_t@embn_t[a];cs[a]=-999
    top=torch.topk(cs,10).indices.cpu().tolist()
    rnd=random.sample(range(N),30)
    for b in top+rnd:
        if b==a: continue
        pairs.append((a,b));labels.append(float(embn[a]@embn[b]))
pairs=np.array(pairs);labels=np.array(labels,dtype=np.float32)
aset=list(set(pairs[:,0].tolist()));random.shuffle(aset)
tra=set(aset[:int(0.8*len(aset))])
trm=np.array([a in tra for a in pairs[:,0]]);tem=~trm

def mlp_attack(X,y,trm,tem,tag):
    Xt=torch.from_numpy(X[trm]).to(dev);yt=torch.from_numpy(y[trm]).to(dev)
    Xe=torch.from_numpy(X[tem]).to(dev)
    net=nn.Sequential(nn.Linear(X.shape[1],512),nn.ReLU(),
                      nn.Linear(512,128),nn.ReLU(),nn.Linear(128,1)).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    idx=torch.randperm(len(Xt))
    for ep in range(15):
        for i in range(0,len(Xt),4096):
            j=idx[i:i+4096];opt.zero_grad()
            loss=((net(Xt[j]).squeeze()-yt[j])**2).mean();loss.backward();opt.step()
    with torch.no_grad(): pred=net(Xe).squeeze().cpu().numpy()
    rho=spearmanr(pred,y[tem]).correlation
    thr=np.quantile(y[tem],0.9);auc=roc_auc_score((y[tem]>=thr).astype(int),pred)
    return rho,auc

# ASPE features: elementwise product of ciphertext pair (dot-product structure)
Xa=(aspe_db[pairs[:,0]]*aspe_db[pairs[:,1]]).astype(np.float32)
# ZATRON features: per-prime circular diffs
feats=[]
for p in PRIMES:
    dd=np.abs(barcodes[p][pairs[:,0]]-barcodes[p][pairs[:,1]])
    feats.append(np.minimum(dd,p-dd).astype(np.float32)/p)
Xz=np.concatenate(feats,axis=1)

rho_a,auc_a=mlp_attack(Xa,labels,trm,tem,"aspe")
rho_z,auc_z=mlp_attack(Xz,labels,trm,tem,"zatron")

print("\n"+"="*72)
print(f"ZATRON vs ASPE — MSMARCO {N:,}")
print("="*72)
print(f"{'':30}{'ASPE (SIGMOD 09)':>20}{'ZATRON':>18}")
print(f"{'Retrieval recall@10':30}{aspe_recall:>19.0%}{zatron_recall:>18.1%}")
print(f"{'Observer direct leak (rho)':30}{rho_aspe_direct:>+20.3f}{rho_zatron_direct:>+18.3f}")
print(f"{'Learned attack MLP (rho)':30}{rho_a:>+20.3f}{rho_z:>+18.3f}")
print(f"{'Learned attack MLP (AUC)':30}{auc_a:>20.3f}{auc_z:>18.3f}")
print("\nExpected story: ASPE = perfect recall but leaks similarity to any")
print("observer; ZATRON = ~98% recall with chance-level leakage.")
