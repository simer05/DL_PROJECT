from __future__ import annotations
from pathlib import Path
import tomli_w
DATASETS=['sberbank-housing','ecom-offers','homesite-insurance','cooking-time','delivery-eta']
VARIANTS={
 'baseline_plr':{},
 'rla_only':{'rla_enabled':True,'rla_rank':4,'rla_scale':0.05},
 'esam_only':{'esam_enabled':True,'esam_rho':0.0025},
 'mfb_only':{'mfb_enabled':True,'mfb_keep':0.8},
 'cf_fisd_only':{'cf_fisd_enabled':True},
 'all_four_combined':{'rla_enabled':True,'rla_rank':4,'rla_scale':0.05,'esam_enabled':True,'esam_rho':0.0025,'mfb_enabled':True,'mfb_keep':0.8,'cf_fisd_enabled':True},
 'all_minus_rla':{'esam_enabled':True,'esam_rho':0.0025,'mfb_enabled':True,'mfb_keep':0.8,'cf_fisd_enabled':True},
 'all_minus_esam':{'rla_enabled':True,'rla_rank':4,'rla_scale':0.05,'mfb_enabled':True,'mfb_keep':0.8,'cf_fisd_enabled':True},
 'all_minus_mfb':{'rla_enabled':True,'rla_rank':4,'rla_scale':0.05,'esam_enabled':True,'esam_rho':0.0025,'cf_fisd_enabled':True},
 'all_minus_cf_fisd':{'rla_enabled':True,'rla_rank':4,'rla_scale':0.05,'esam_enabled':True,'esam_rho':0.0025,'mfb_enabled':True,'mfb_keep':0.8},
}
BATCH={'sberbank-housing':1024,'ecom-offers':2048,'homesite-insurance':2048,'cooking-time':2048,'delivery-eta':2048}
def cfg(dataset,variant,seed):
 return {'dataset':dataset,'variant':variant,'seed':seed,'batch_size':BATCH[dataset],'patience':6,'n_epochs':20,'gradient_clipping_norm':1.0,'amp':True,'data':{'path':f'data/{dataset}','num_policy':'standard','cache':False},'optimizer':{'type':'AdamW','lr':3e-4,'weight_decay':1e-5},'model':{'arch_type':'tabm','k':16,'n_blocks':2,'d_block':384,'dropout':0.1,'d_embedding':16,'n_bins':48,**VARIANTS[variant]}}
def main():
 root=Path(__file__).resolve().parents[1]/'paper'/'exp'/'integrated'
 for seed in [0,1,2]:
  for dataset in DATASETS:
   for variant in VARIANTS:
    d=root/dataset/f'{variant}-evaluation'; d.mkdir(parents=True,exist_ok=True); (d/f'{seed}.toml').write_text(tomli_w.dumps(cfg(dataset,variant,seed)))
 print(root)
if __name__=='__main__': main()
