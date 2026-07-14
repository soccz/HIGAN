import json, numpy as np

def load(path, adkey, fdkey, rowkey):
    d=json.load(open(path)); rows=d[rowkey]
    names=[r["name"] for r in rows]
    ad=np.array([r[adkey] for r in rows])
    fd={s:np.array([r[fdkey][s] for r in rows]) for s in ["3.0","1.0","0.3","0.1"]}
    return names,ad,fd

def irreducible(names,ad,fd):
    out={}
    for s in ["3.0","1.0","0.3","0.1"]:
        k=np.exp(np.median(np.log(fd[s])-np.log(ad))); rsc=fd[s]/k
        lo=min(ad.min(),rsc.min()); hi=max(ad.max(),rsc.max())
        taus=np.linspace(lo,hi,3000)
        best=(99,None)
        for tau in taus:
            dis=int(((ad>tau)!=(rsc>tau)).sum())
            if dis<best[0]: best=(dis,tau)
        flips=[names[i] for i in range(len(names)) if (ad>best[1])[i]!=(rsc>best[1])[i]]
        out[s]=(best[0],flips,k)
    return out

for tag,path,ak,fk,rk in [
    ("D1 decision_gt","/home/soccz/22tb/study/HIGAN/note/submission/evidence/curvature_decision_gt_metrics.json","c_AD","c_FD","rows"),
    ("D2 ordering_flip","/home/soccz/22tb/study/HIGAN/note/submission/evidence/curvature_ordering_flip_metrics.json","c_exact","c_fd2","pass_B_directions"),
]:
    names,ad,fd=load(path,ak,fk,rk)
    res=irreducible(names,ad,fd)
    print(f"=== {tag}: irreducible flips after best global rescale (real reorder near cut) ===")
    for s,(n,fl,k) in res.items():
        print(f"    FD@{s}: scale k={k:.2f}; {n} irreducible flips {fl}")
    ai=int(np.argmax(ad))
    print(f"    top dir={names[ai]} exact={ad[ai]:.3f} -> always GO under both (no flip on the curved end)\n")

# Summary contrast: raw absolute-budget flips (median-tau) vs irreducible
print("=== CONTRAST: raw absolute-budget flips vs irreducible (D1, tau=median exact) ===")
names,ad,fd=load("/home/soccz/22tb/study/HIGAN/note/submission/evidence/curvature_decision_gt_metrics.json","c_AD","c_FD","rows")
tau=np.median(ad)
for s in ["3.0","1.0","0.3","0.1"]:
    raw=int(((ad>tau)!=(fd[s]>tau)).sum())
    print(f"  FD@{s}: raw absolute-tau flips={raw}/24  (vs irreducible {res})" if False else f"  FD@{s}: raw absolute-tau flips={raw}/24")
