import json, time, numpy as np, torch, os
from chronos import ChronosPipeline
from synthetic_surrogates import make_surrogate_set
from load_strata import load_stratum
from recall_harness import evaluate_series_recall

pipe = ChronosPipeline.from_pretrained('amazon/chronos-t5-small', device_map='cuda', torch_dtype=torch.float16)
def run(items):
    keys=['chronos_mase','dlinear_mase','chronos_slope','dlinear_slope','recall_index']
    acc={k:[] for k in keys}
    for it in items:
        r=evaluate_series_recall(pipe, it['series'], context=256, horizon=24, season=24, stride=24, max_windows=8)
        if r:
            for k in keys: acc[k].append(r[k])
    return {k: float(np.median(v)) for k,v in acc.items()} | {'n':len(acc['chronos_mase'])}

t0=time.time()
res={}
res['in_domain']=run(load_stratum('in_domain', max_series_per_config=15))
print('in_domain done', f'{time.time()-t0:.0f}s')
res['zero_shot']=run(load_stratum('zero_shot', max_series_per_config=15))
print('zero_shot done', f'{time.time()-t0:.0f}s')
res['synthetic']=run([{'series':d['series']} for d in make_surrogate_set(n=20,length=1024,seed=2027) if d['period']==24])
print('synthetic done', f'{time.time()-t0:.0f}s')
os.makedirs('../results',exist_ok=True); json.dump(res, open('../results/run_recall.json','w'), indent=2)
print()
print('='*78)
print('WITHIN-SERIES: Chronos vs DLinear (strong baseline) + RECALL INDEX')
print('='*78)
print(f"{'stratum':12s} {'Ch MASE':>8s} {'DL MASE':>8s} {'Ch slope':>9s} {'DL slope':>9s} {'recall_idx':>11s} {'n':>4s}")
for k in ['in_domain','zero_shot','synthetic']:
    s=res[k]
    print(f"{k:12s} {s['chronos_mase']:>8.3f} {s['dlinear_mase']:>8.3f} {s['chronos_slope']:>+9.3f} {s['dlinear_slope']:>+9.3f} {s['recall_index']:>+11.3f} {s['n']:>4d}")
print()
print('  recall_index = DLslope - Chslope. >0 = Chronos error-curve flatter than the')
print('  non-memorizing reference. Memorization => large on in_domain, ~0 on synthetic.')
