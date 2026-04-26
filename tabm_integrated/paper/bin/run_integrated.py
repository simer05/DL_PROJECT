from __future__ import annotations
import argparse, json, math, random, time, tomllib
from pathlib import Path
from typing import Any
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tabm
import rtdl_num_embeddings

LOWER = {'rmse'}

def load_toml(path: Path) -> dict[str, Any]:
    with path.open('rb') as f:
        return tomllib.load(f)

def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def read_arrays(data_path: Path):
    info = json.loads((data_path / 'info.json').read_text())
    x_num, x_cat, y = {}, {}, {}
    for part in ['train', 'val', 'test']:
        nums = []
        for name in ['X_num', 'X_bin']:
            p = data_path / f'{name}_{part}.npy'
            if p.exists():
                nums.append(np.load(p, allow_pickle=True).astype('float32'))
        y[part] = np.load(data_path / f'Y_{part}.npy', allow_pickle=True)
        x_num[part] = np.concatenate(nums, axis=1).astype('float32') if nums else np.zeros((len(y[part]), 0), dtype='float32')
        pcat = data_path / f'X_cat_{part}.npy'
        if pcat.exists():
            x_cat[part] = np.load(pcat, allow_pickle=True).astype('int64')
    return x_num, (x_cat if x_cat else None), y, info

def standardize_num(x):
    if x['train'].shape[1] == 0:
        return x
    mean = np.nanmean(x['train'], axis=0, keepdims=True); mean = np.nan_to_num(mean, nan=0.0)
    std = np.nanstd(x['train'], axis=0, keepdims=True); std = np.nan_to_num(std, nan=0.0)
    span = np.nanmax(x['train'], axis=0) - np.nanmin(x['train'], axis=0); span = np.nan_to_num(span, nan=0.0)
    keep = (std.reshape(-1) >= 1e-6) & (span >= 1e-6)
    if not keep.any():
        return {k: np.zeros((v.shape[0], 0), dtype='float32') for k, v in x.items()}
    mean = mean[:, keep]; std = std[:, keep]
    return {k: np.nan_to_num((v[:, keep] - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype('float32') for k, v in x.items()}

def cat_cardinalities(x_cat):
    if not x_cat:
        return None
    return [max(int(x_cat[p][:, j].max()) for p in x_cat) + 1 for j in range(x_cat['train'].shape[1])]

def feature_importance(x_num, y):
    if x_num.shape[1] == 0:
        return np.zeros((0,), dtype='float32')
    yy = y.astype('float32'); yy = (yy - yy.mean()) / (yy.std() + 1e-6)
    scores = np.nan_to_num(np.abs((x_num.astype('float32') * yy[:, None]).mean(0)), nan=0.0)
    if float(scores.max(initial=0.0)) <= 0:
        scores = np.ones_like(scores)
    return np.clip(scores / (scores.mean() + 1e-6), 0.25, 4.0).astype('float32')

class IntegratedTabM(nn.Module):
    def __init__(self, cfg, n_num_features, cards, d_out, bins, cf_weights):
        super().__init__()
        mc = cfg['model']
        num_embeddings = None
        if n_num_features and bins is not None:
            num_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(bins, int(mc.get('d_embedding', 16)), activation=True, version='B')
        self.model = tabm.TabM(
            n_num_features=n_num_features,
            cat_cardinalities=cards,
            d_out=d_out,
            num_embeddings=num_embeddings,
            arch_type=mc.get('arch_type', 'tabm'),
            k=int(mc.get('k', 16)),
            n_blocks=int(mc.get('n_blocks', 2)),
            d_block=int(mc.get('d_block', 384)),
            dropout=float(mc.get('dropout', 0.1)),
            start_scaling_init=mc.get('start_scaling_init', 'random-signs'),
        )
        self.rla_enabled = bool(mc.get('rla_enabled', False)) and n_num_features > 0
        self.rla_scale = float(mc.get('rla_scale', 0.05))
        if self.rla_enabled:
            rank = int(mc.get('rla_rank', 4))
            self.rla_a = nn.Parameter(torch.randn(n_num_features, rank) * 1e-3)
            self.rla_b = nn.Parameter(torch.zeros(rank, n_num_features))
        self.mfb_enabled = bool(mc.get('mfb_enabled', False)) and n_num_features > 0
        self.mfb_keep = float(mc.get('mfb_keep', 0.8))
        self.cf_enabled = bool(mc.get('cf_fisd_enabled', False)) and cf_weights is not None and n_num_features > 0
        self.register_buffer('cf_weights', cf_weights if cf_weights is not None else torch.ones(n_num_features))
    def forward(self, x_num, x_cat):
        if x_num is not None and x_num.shape[1] == 0:
            x_num = None
        if x_num is not None and self.cf_enabled:
            x_num = x_num * self.cf_weights[None, :]
        if x_num is not None and self.rla_enabled:
            x_num = x_num + self.rla_scale * (x_num @ self.rla_a @ self.rla_b)
        if self.training and x_num is not None and self.mfb_enabled:
            mask = (torch.rand((1, x_num.shape[1]), device=x_num.device) < self.mfb_keep).to(x_num.dtype) / max(self.mfb_keep, 1e-6)
            x_num = x_num * mask
        return self.model(x_num, x_cat)

def make_loaders(x_num, x_cat, y, task_type, batch_size):
    out = {}
    for part in ['train', 'val', 'test']:
        xc = torch.as_tensor(x_cat[part], dtype=torch.long) if x_cat is not None else torch.empty((len(y[part]), 0), dtype=torch.long)
        ds = TensorDataset(torch.as_tensor(x_num[part], dtype=torch.float32), xc, torch.as_tensor(y[part], dtype=torch.float32))
        out[part] = DataLoader(ds, batch_size=batch_size if part == 'train' else batch_size * 4, shuffle=(part == 'train'), num_workers=2, pin_memory=True)
    return out

def predict(model, loader, device, task_type, y_mean, y_std):
    model.eval(); preds=[]; targets=[]
    with torch.no_grad():
        for xb, xc, yb in loader:
            xb=xb.to(device, non_blocking=True); xc=xc.to(device, non_blocking=True)
            out = model(xb, xc if xc.shape[1] else None).mean(1).squeeze(-1)
            out = torch.sigmoid(out) if task_type == 'binclass' else out * y_std + y_mean
            preds.append(out.float().cpu().numpy()); targets.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(targets)

def metric(task_type, pred, target):
    if task_type == 'binclass':
        auc = float(roc_auc_score(target.astype(int), pred)); return {'roc-auc': auc, 'score': auc}
    rmse = float(math.sqrt(mean_squared_error(target.astype('float32'), pred.astype('float32')))); return {'rmse': rmse, 'score': -rmse}

def train_one(config_path: Path, output: Path, force=False):
    if output.exists() and not force and (output/'DONE').exists() and (output/'report.json').exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    cfg=load_toml(config_path); set_seed(int(cfg.get('seed',0)))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_num,x_cat,y_raw,info=read_arrays((Path.cwd()/cfg['data']['path']).resolve())
    x_num=standardize_num(x_num); task_type=info['task_type']; y={k:v.copy() for k,v in y_raw.items()}
    y_mean=0.0; y_std=1.0
    if task_type == 'regression':
        y_mean=float(y['train'].mean()); y_std=float(y['train'].std()+1e-6)
        y={k:((v.astype('float32')-y_mean)/y_std).astype('float32') for k,v in y.items()}
    cards=cat_cardinalities(x_cat); n_num=x_num['train'].shape[1]
    bins = rtdl_num_embeddings.compute_bins(torch.as_tensor(x_num['train'], dtype=torch.float32), n_bins=int(cfg['model'].get('n_bins',48))) if n_num else None
    cf = torch.as_tensor(feature_importance(x_num['train'], y_raw['train']), dtype=torch.float32) if cfg['model'].get('cf_fisd_enabled', False) else None
    model=IntegratedTabM(cfg,n_num,cards,1,bins,cf).to(device)
    loaders=make_loaders(x_num,x_cat,y,task_type,int(cfg.get('batch_size',1024)))
    opt=torch.optim.AdamW(model.parameters(), lr=float(cfg['optimizer'].get('lr',3e-4)), weight_decay=float(cfg['optimizer'].get('weight_decay',1e-5)))
    loss_fn=nn.BCEWithLogitsLoss() if task_type=='binclass' else nn.MSELoss(); amp=bool(cfg.get('amp',True)) and device.type=='cuda'
    rho=float(cfg['model'].get('esam_rho',0.0)) if cfg['model'].get('esam_enabled',False) else 0.0
    best=None; best_score=-1e100; bad=0; history=[]; start=time.time()
    for epoch in range(int(cfg.get('n_epochs',20))):
        model.train(); losses=[]
        for xb,xc,yb in loaders['train']:
            xb=xb.to(device,non_blocking=True); xc=xc.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp):
                logits=model(xb, xc if xc.shape[1] else None).squeeze(-1); target=yb[:,None].expand_as(logits); loss=loss_fn(logits,target)
            if not torch.isfinite(loss): raise RuntimeError(f'NaN/Inf loss at epoch={epoch}')
            loss.backward()
            if rho > 0:
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                norm=torch.norm(torch.stack([g.detach().norm() for g in grads])) if grads else torch.tensor(0.0,device=device); scale=rho/(norm+1e-12); eps=[]
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is None: eps.append(None)
                        else:
                            e=p.grad*scale; p.add_(e); eps.append(e)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp):
                    logits2=model(xb, xc if xc.shape[1] else None).squeeze(-1); loss2=loss_fn(logits2,target)
                if not torch.isfinite(loss2): raise RuntimeError(f'NaN/Inf ESAM loss at epoch={epoch}')
                loss2.backward()
                with torch.no_grad():
                    for p,e in zip(model.parameters(),eps):
                        if e is not None: p.sub_(e)
                losses.append(float(loss2.detach().cpu()))
            else:
                losses.append(float(loss.detach().cpu()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get('gradient_clipping_norm',1.0))); opt.step()
        vp,vy=predict(model,loaders['val'],device,task_type,y_mean,y_std); tp,ty=predict(model,loaders['test'],device,task_type,y_mean,y_std)
        metrics={'val':metric(task_type,vp,vy),'test':metric(task_type,tp,ty)}; score=metrics['val']['score']
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'val_score':score,'test_score':metrics['test']['score']})
        if score > best_score: best_score=score; best={'metrics':metrics,'best_epoch':epoch}; bad=0
        else:
            bad += 1
            if bad >= int(cfg.get('patience',6)): break
    report={'dataset':cfg['dataset'],'variant':cfg['variant'],'seed':cfg['seed'],'config_path':str(config_path),'result_path':str(output),'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','amp_enabled':amp,'amp_dtype':'bfloat16' if amp else 'fp32','inference_mode':'mean','task_type':task_type,'metric_direction':'lower' if task_type=='regression' else 'higher','config':cfg,'time_seconds':time.time()-start,'history':history,**best}
    dump_json(output/'report.json',report); (output/'DONE').write_text('done\n')

def main():
    p=argparse.ArgumentParser(); p.add_argument('config',type=Path); p.add_argument('output',type=Path); p.add_argument('--force',action='store_true'); a=p.parse_args()
    try: train_one(a.config,a.output,a.force)
    except Exception as e:
        a.output.mkdir(parents=True, exist_ok=True); dump_json(a.output/'report.json', {'failure':repr(e),'config_path':str(a.config),'result_path':str(a.output)}); raise
if __name__ == '__main__': main()
