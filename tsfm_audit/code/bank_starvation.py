"""Banked finding: TSFM zero-shot advantage over a simple trained baseline is
inflated by BASELINE STARVATION. As the simple baseline (DLinear) is given more
training data, the Chronos-minus-DLinear MASE gap shrinks toward ~0.
ETT (4 variants, long real series), Chronos-t5-small zero-shot fixed.
"""
import json, numpy as np, pandas as pd, torch, os
from chronos import ChronosPipeline
from eval_harness import make_windows, chronos_forecast, mase
from baselines import dlinear_forecast

pipe = ChronosPipeline.from_pretrained('amazon/chronos-t5-small', device_map='cuda', torch_dtype=torch.float16)
DATA='/mnt/20t/RegFiLM/data'
CTX, H, SEASON = 512, 24, 24
TRAIN_K = [300, 600, 1200, 2400, 5000]   # DLinear training-window budget (points)

def eval_dataset(name):
    s = pd.read_csv(f'{DATA}/{name}.csv')['OT'].to_numpy('float32')
    te = int(len(s)*0.7)
    wins = make_windows(s[te-CTX:], CTX, H, H, 20)
    if len(wins) < 5: return None
    # Chronos zero-shot (fixed)
    ch = [mase(chronos_forecast(pipe,c,H),t,c,SEASON) for c,t in wins]
    ch_med = float(np.median(ch))
    # DLinear with varying training budget (last K points before test)
    row = {'chronos': ch_med, 'dlinear_by_K': {}}
    for K in TRAIN_K:
        tp = s[max(0,te-K):te]
        dl = [mase(dlinear_forecast(c,H,tp,CTX),t,c,SEASON) for c,t in wins]
        row['dlinear_by_K'][K] = float(np.median(dl))
    return row

res={}
for name in ['ETTh1','ETTh2','ETTm1','ETTm2']:
    r=eval_dataset(name)
    if r: res[name]=r; print(f'{name} done')
os.makedirs('../results',exist_ok=True); json.dump({'config':{'ctx':CTX,'H':H,'train_K':TRAIN_K},'results':res}, open('../results/bank_starvation.json','w'), indent=2)

print()
print('='*72)
print('BASELINE-STARVATION: Chronos-DLinear MASE gap vs DLinear training budget')
print('='*72)
hdr='dataset   Chronos ' + ''.join(f'K={K:>5d} ' for K in TRAIN_K)
print(hdr)
gaps_small=[]; gaps_large=[]
for name,r in res.items():
    line=f'{name:9s} {r["chronos"]:>7.3f} '
    for K in TRAIN_K:
        line+=f'{r["dlinear_by_K"][K]:>7.3f} '
    print(line)
    gaps_small.append(r['chronos']-r['dlinear_by_K'][TRAIN_K[0]])
    gaps_large.append(r['chronos']-r['dlinear_by_K'][TRAIN_K[-1]])
print()
print(f'  mean gap (Chronos-DLinear) at K={TRAIN_K[0]} (starved): {np.mean(gaps_small):+.3f}')
print(f'  mean gap (Chronos-DLinear) at K={TRAIN_K[-1]} (fed)   : {np.mean(gaps_large):+.3f}')
print('  => if gap shrinks toward 0 as K grows, the TSFM advantage is starvation-inflated.')
