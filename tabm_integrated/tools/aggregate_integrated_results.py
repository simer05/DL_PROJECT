from __future__ import annotations
import csv,json,math
from pathlib import Path
from statistics import mean,stdev
DATASETS=['sberbank-housing','ecom-offers','homesite-insurance','cooking-time','delivery-eta']
VARIANTS=['baseline_plr','rla_only','esam_only','mfb_only','cf_fisd_only','all_four_combined','all_minus_rla','all_minus_esam','all_minus_mfb','all_minus_cf_fisd']
PRIMARY={'sberbank-housing':'rmse','cooking-time':'rmse','delivery-eta':'rmse','ecom-offers':'roc-auc','homesite-insurance':'roc-auc'}
LOWER={'rmse'}
ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT/'paper'/'exp'/'integrated'
def load_report(dataset,variant,seed):
 path=EXP/dataset/f'{variant}-evaluation'/str(seed)/'report.json'; done=path.with_name('DONE')
 if not path.exists() or not done.exists(): return None,path
 r=json.loads(path.read_text()); return r,path
def delt(metric,v,b): return b-v if metric in LOWER else v-b
def fmt(x): return '' if x is None or (isinstance(x,float) and math.isnan(x)) else x
def main():
 summary=[]; audit=[]
 for dataset in DATASETS:
  metric=PRIMARY[dataset]; values={}
  for variant in VARIANTS:
   vals=[]
   for seed in [0,1,2]:
    r,path=load_report(dataset,variant,seed); ok=bool(r) and not r.get('failure')
    audit.append({'dataset':dataset,'variant':variant,'seed':seed,'result_path':str(path.parent),'report_json_present':path.exists(),'DONE_present':(path.parent/'DONE').exists(),'failure_block_absent':ok,'gpu_name':(r or {}).get('gpu_name'),'amp_dtype':(r or {}).get('amp_dtype'),'inference_mode':(r or {}).get('inference_mode','mean'),'metric':metric,'test_metric':((r or {}).get('metrics',{}).get('test',{}).get(metric)),'failure':(r or {}).get('failure')})
    if ok: vals.append(float(r['metrics']['test'][metric]))
   values[variant]=vals
  base=values['baseline_plr']
  for variant in VARIANTS:
   vals=values[variant]
   if len(vals)==3 and len(base)==3:
    m=mean(vals); s=stdev(vals); bm=mean(base); d=delt(metric,m,bm); pct=100*d/abs(bm); claim='baseline' if variant=='baseline_plr' else ('win' if d>0 else 'loss' if d<0 else 'tie')
   elif vals:
    m=mean(vals); s=stdev(vals) if len(vals)>1 else 0.0; bm=mean(base) if base else float('nan'); d=float('nan'); pct=float('nan'); claim='incomplete'
   else:
    m=s=bm=d=pct=float('nan'); claim='missing'
   summary.append({'dataset':dataset,'variant':variant,'metric':metric,'metric_direction':'lower' if metric in LOWER else 'higher','inference_mode':'mean','precision':'bfloat16','n_seeds':len(vals),'mean':m,'std':s,'baseline_mean':bm,'absolute_delta':d,'percent_delta':pct,'safe_claim_status':claim,'config_path':str(EXP/dataset/f'{variant}-evaluation/0.toml'),'result_path':str(EXP/dataset/f'{variant}-evaluation')})
 for name,rows in [('final_integrated_summary.csv',summary),('final_integrated_audit.csv',audit)]:
  path=ROOT/'paper'/'exp'/name; path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys()),lineterminator='\n'); w.writeheader(); w.writerows(rows)
 lines=['# Final Integrated TabM Experiment Report','','Mean ± std over available seeds. Safe claims require all 3 seeds.','','| dataset | variant | metric | direction | inference | precision | mean ± std | delta | percent delta | n | status | config path | result path |','|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|']
 for r in summary:
  lines.append(f"| {r['dataset']} | {r['variant']} | {r['metric']} | {r['metric_direction']} | {r['inference_mode']} | {r['precision']} | {float(r['mean']):.6g} ± {float(r['std']):.3g} | {float(r['absolute_delta']):.6g} | {float(r['percent_delta']):.3f}% | {r['n_seeds']} | {r['safe_claim_status']} | {r['config_path']} | {r['result_path']} |")
 (ROOT/'FINAL_EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(ROOT/'paper'/'exp'/'final_integrated_summary.csv'); print(ROOT/'FINAL_EXPERIMENT_REPORT.md')
if __name__=='__main__': main()
