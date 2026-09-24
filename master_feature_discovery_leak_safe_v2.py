#!/usr/bin/env python3
"""Master leak-safe feature discovery for the existing NSE model panel.

Research only: never modifies panel.parquet or the live model.
Daily features are treated as resolving at t close and usable at t+1 open.
All derived features use information available at t close or earlier.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform
from pathlib import Path
import numpy as np
import pandas as pd

VERSION='2026-09-24.master.v2'
SEED=1729; N_FOLDS=5; EMBARGO=5
TARGETS=['label_tp_before_sl','label_tp_before_sl_3p2','label_tp_before_sl_5p3']
BANNED={'ret_1d_close_pct','ret_3d_close_pct','ret_5d_close_pct','ret_1d_oc_pct','ret_3d_oc_pct','ret_5d_oc_pct','ret_5d_open_to_close_pct'}
TOKENS=('label','target','future','fwd','forward','next_','tp_hit','sl_hit','exit','outcome','ret_5d','ret_10d','lead')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def objhash(x): return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()

def atomic_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix('.tmp'); t.write_text(json.dumps(x,indent=2,default=str)); os.replace(t,p)

def atomic_pq(df,p):
    p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+'.tmp.parquet'); df.to_parquet(t,index=False); os.replace(t,p)

def cp(out,stage,status,**kw):
    atomic_json(out/'checkpoints'/f'{stage}.json',{'version':VERSION,'stage':stage,'status':status,'updated_at':pd.Timestamp.now(tz='Asia/Kolkata').isoformat(),**kw})

def done(out,stage,h):
    p=out/'checkpoints'/f'{stage}.json'
    if not p.exists(): return False
    try:
        x=json.loads(p.read_text()); return x.get('status')=='COMPLETE' and x.get('input_hash')==h
    except Exception:return False

def load(path):
    df=pd.read_parquet(path).copy()
    req={'symbol','timestamp','close'}-set(df.columns)
    if req: raise RuntimeError(f'Missing required panel columns: {sorted(req)}')
    df['symbol']=df['symbol'].astype(str); df['timestamp']=pd.to_datetime(df['timestamp'],errors='coerce'); df['_date']=df.timestamp.dt.normalize()
    df['_row_id']=np.arange(len(df),dtype=np.int64)
    df=df.dropna(subset=['symbol','timestamp','_date']).sort_values(['symbol','timestamp'],kind='mergesort').reset_index(drop=True)
    return df

def suspicious(df):
    out=[]
    for c in df.columns:
        lc=str(c).lower()
        if c in {'symbol','timestamp','_date','_row_id'}: continue
        if c in BANNED or any(t in lc for t in TOKENS): out.append(c)
    return sorted(set(out))

def safe_cols(df):
    bad=set(suspicious(df)); return [c for c in df.columns if c not in bad and not str(c).startswith('_') and c not in {'symbol','timestamp'} and pd.api.types.is_numeric_dtype(df[c])]

def sh(df,c,n=1): return df.groupby('symbol',sort=False)[c].shift(n)
def roll(df,c,w,kind='mean'):
    r=df.groupby('symbol',sort=False)[c].rolling(w,min_periods=max(2,w//2))
    x={'mean':r.mean(),'std':r.std(),'median':r.median(),'min':r.min(),'max':r.max()}[kind]
    return x.reset_index(level=0,drop=True)
def rz(df,c,w):
    m=roll(df,c,w); s=roll(df,c,w,'std'); return (pd.to_numeric(df[c],errors='coerce')-m)/s.replace(0,np.nan)

def add_derived(df):
    out=df.copy(); meta=[]
    def add(n,v,f,h,w=0):
        if n not in out:
            if isinstance(v,pd.Series): out[n]=v.reindex(out.index).to_numpy()
            else: out[n]=np.asarray(v)
            meta.append({'feature':n,'family':f,'hypothesis':h,'lookback_sessions':w,'resolved_at':'session t close 15:30 IST','usable_from':'session t+1 open 09:15 IST','uses_future':False})
    close=pd.to_numeric(out.close,errors='coerce'); op=pd.to_numeric(out['open'] if 'open' in out else pd.Series(np.nan,index=out.index),errors='coerce'); hi=pd.to_numeric(out['high'] if 'high' in out else pd.Series(np.nan,index=out.index),errors='coerce'); lo=pd.to_numeric(out['low'] if 'low' in out else pd.Series(np.nan,index=out.index),errors='coerce')
    ret=close/sh(out,'close',1)-1; intr=close/op.replace(0,np.nan)-1; gap=op/sh(out,'close',1)-1; rng=(hi-lo)/close.replace(0,np.nan); body=(close-op)/(hi-lo).replace(0,np.nan); pos=(close-lo)/(hi-lo).replace(0,np.nan)
    for n,v,h in [('R_ret_1',ret,'one-session momentum'),('R_intraday',intr,'open-to-close pressure'),('R_gap',gap,'overnight repricing'),('R_range',rng,'normalised daily range'),('R_body',body,'candle conviction'),('R_close_pos',pos,'close location in range')]: add(n,v,'price_structure',h,1)
    for w in (3,5,10,20,60):
        add(f'R_ret_{w}',close/sh(out,'close',w)-1,'momentum',f'{w}-session momentum',w)
    for w in (10,20,50,100,200):
        ma=roll(out,'close',w); add(f'T_dist_sma_{w}',close/ma.replace(0,np.nan)-1,'trend',f'distance from {w}-session mean',w)
    for w in (5,10,20,60):
        add(f'V_close_std_{w}',roll(out,'close',w,'std'),'volatility',f'close volatility {w}',w)
        rr=((hi-lo)/close.replace(0,np.nan)); tmp=pd.DataFrame({'symbol':out.symbol,'x':rr},index=out.index); rm=tmp.groupby('symbol',sort=False).x.rolling(w,min_periods=max(2,w//2)).mean().reset_index(level=0,drop=True); add(f'V_range_mean_{w}',rm,'volatility',f'average normalised range {w}',w)
    if 'volume' in out:
        for w in (5,20,60):
            add(f'VOL_z_{w}',rz(out,'volume',w),'volume',f'volume surprise {w}',w)
            add(f'VOL_ratio_{w}',pd.to_numeric(out.volume,errors='coerce')/roll(out,'volume',w).replace(0,np.nan),'volume',f'volume / mean volume {w}',w)
    for w in (5,10,20,60):
        r=close/sh(out,'close',w)-1; add(f'MR_negret_{w}',-r,'mean_reversion',f'contrarian {w}-session return',w); add(f'MR_price_z_{w}',rz(out,'close',w),'mean_reversion',f'price z-score {w}',w)
    # Controlled normalisation of existing panel features; never touch forward-looking columns.
    base=[c for c in safe_cols(out) if c not in {'open','high','low','close','volume'} and not any(str(c).startswith(x) for x in ('R_','T_','V_','VOL_','MR_','CS_','X_'))]
    for c in base[:80]:
        if out[c].notna().mean()>=.30 and out[c].nunique(dropna=True)>=20: add(f'X_z60__{c}',rz(out,c,60),'existing_normalised',f'60-session normalisation of {c}',60)
    for c in ('R_ret_1','R_ret_5','R_ret_20','R_range','R_body','VOL_z_20','T_dist_sma_20','T_dist_sma_50','MR_price_z_20'):
        if c in out: add(f'CS_rank__{c}',out.groupby('_date',sort=False)[c].rank(pct=True),'cross_sectional',f'date-t cross-sectional percentile rank of {c}',0)
    bad=[m['feature'] for m in meta if m['feature'] in BANNED or any(t in m['feature'].lower() for t in TOKENS)]
    if bad: raise RuntimeError(f'Derived leak gate failed: {bad[:20]}')
    if not np.array_equal(out['_row_id'].to_numpy(),df['_row_id'].to_numpy()): raise RuntimeError('ROW_ID_ALIGNMENT_FAILURE')
    return out,pd.DataFrame(meta)

def sector_relative(df,mapping):
    out=df.copy(); meta=[]; close=pd.to_numeric(out.close,errors='coerce')
    for sector,col in mapping.items():
        if col not in out: continue
        s=pd.to_numeric(out[col],errors='coerce')
        for w in (1,5,20):
            sr=s/s.groupby(out.symbol,sort=False).shift(w)-1; rr=close/sh(out,'close',w)-1; n=f'SEC_{sector}_relret_{w}'; out[n]=rr-sr
            meta.append({'feature':n,'family':'sector_relative','hypothesis':f'stock return relative to {sector} index','lookback_sessions':w,'resolved_at':'session t close 15:30 IST','usable_from':'session t+1 open 09:15 IST','uses_future':False,'sector_source_column':col})
    return out,pd.DataFrame(meta)

def quality(df,features):
    rows=[]
    for c in features:
        s=pd.to_numeric(df[c],errors='coerce'); v=s.dropna(); rows.append({'feature':c,'coverage':float(s.notna().mean()),'n_unique':int(s.nunique()),'modal_share':float(v.value_counts(normalize=True).iloc[0]) if len(v) else 1.0,'std':float(v.std()) if len(v) else np.nan})
    return pd.DataFrame(rows)

def folds(dates):
    d=np.array(sorted(pd.to_datetime(dates).unique())); blocks=np.array_split(d,N_FOLDS+1)[1:]; out=[]
    for i,test in enumerate(blocks,1):
        tr=d[d<test[0]]
        if len(tr)>EMBARGO+20: out.append((i,tr[:-EMBARGO],test))
    if len(out)<N_FOLDS: raise RuntimeError(f'Only {len(out)} usable folds')
    return out

def ic(df,features,target):
    if target not in df: return pd.DataFrame()
    y=pd.to_numeric(df[target],errors='coerce'); rows=[]
    for c in features:
        x=pd.to_numeric(df[c],errors='coerce'); z=pd.DataFrame({'date':df._date,'x':x,'y':y}).dropna()
        if z.empty: continue
        a=z.groupby('date',sort=True).apply(lambda g: pd.Series({'ic':g.x.corr(g.y), 'rank_ic':g.x.corr(g.y,method='spearman')}),include_groups=False)
        rows.append({'feature':c,'target':target,'dates':len(a),'mean_ic':a.ic.mean(),'mean_rank_ic':a.rank_ic.mean(),'rank_ic_std':a.rank_ic.std(),'positive_rank_ic_frac':(a.rank_ic>0).mean()})
    return pd.DataFrame(rows)

def oos(df,features,target,out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score,log_loss
    y=pd.to_numeric(df[target],errors='coerce').to_numpy(float); dates=df._date.to_numpy(); rows=[]
    for fi,trd,ted in folds(dates):
        tr=np.isin(dates,trd); te=np.isin(dates,ted)
        for c in features:
            x=pd.to_numeric(df[c],errors='coerce').to_numpy(float)
            # Define train/test populations by TARGET availability only. Missing
            # feature values are imputed using TRAIN statistics, so every feature
            # is evaluated on the same rows in a fold. This prevents coverage
            # differences from changing the economic population being compared.
            a=tr&np.isfinite(y); b=te&np.isfinite(y)
            if a.sum()<100 or b.sum()<30 or len(np.unique(y[a]))<2 or len(np.unique(y[b]))<2: continue
            xr=x[a]; zr=x[b]; med=np.nanmedian(xr); med=med if np.isfinite(med) else 0.0
            X=np.where(np.isfinite(xr),xr,med).reshape(-1,1); Z=np.where(np.isfinite(zr),zr,med).reshape(-1,1); mu=X.mean(); sd=X.std()
            if not np.isfinite(sd) or sd<1e-12: continue
            X=(X-mu)/sd; Z=(Z-mu)/sd; m=LogisticRegression(max_iter=300,random_state=SEED).fit(X,y[a].astype(int)); p=m.predict_proba(Z)[:,1]; ll=log_loss(y[b].astype(int),p,labels=[0,1]); base=log_loss(y[b].astype(int),np.full(b.sum(),y[a].mean()),labels=[0,1]); rows.append({'fold':fi,'feature':c,'target':target,'auc':roc_auc_score(y[b],p),'delta_logloss':base-ll})
        atomic_pq(pd.DataFrame(rows),out/'oos'/f'{target}_through_fold_{fi}.parquet')
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--panel',required=True); ap.add_argument('--out',required=True); ap.add_argument('--sector-columns-json'); ap.add_argument('--null-permutations',type=int,default=200); ap.add_argument('--top-oos',type=int,default=100); ap.add_argument('--resume',action='store_true'); a=ap.parse_args()
    panel=Path(a.panel); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); ih=sha(panel)
    manifest={'version':VERSION,'panel':str(panel),'panel_sha256':ih,'targets':TARGETS,'folds':N_FOLDS,'embargo':EMBARGO,'seed':SEED,'status':'RUNNING','started_at':pd.Timestamp.now(tz='Asia/Kolkata').isoformat(),'python':platform.python_version()}; atomic_json(out/'run_manifest.json',manifest)
    # 01 load / immutable row identity
    h=objhash([ih,VERSION]);
    if a.resume and done(out,'01_load',h): df=pd.read_parquet(out/'working_panel.parquet')
    else:
        cp(out,'01_load','STARTED',input_hash=h); df=load(panel); atomic_json(out/'panel_audit.json',{'rows':len(df),'symbols':df.symbol.nunique(),'dates':df._date.nunique(),'first':str(df._date.min()),'last':str(df._date.max()),'suspicious_columns':suspicious(df),'row_id_unique':bool(df._row_id.is_unique)}); atomic_pq(df,out/'working_panel.parquet'); cp(out,'01_load','COMPLETE',input_hash=h)
    # 02 derived
    h=objhash([ih,'derived',VERSION]);
    if a.resume and done(out,'02_derived',h): df=pd.read_parquet(out/'working_panel.parquet'); dm=pd.read_parquet(out/'derived_inventory.parquet')
    else:
        cp(out,'02_derived','STARTED',input_hash=h); df,dm=add_derived(df); atomic_pq(df,out/'working_panel.parquet'); atomic_pq(dm,out/'derived_inventory.parquet'); cp(out,'02_derived','COMPLETE',input_hash=h,features=len(dm))
    # 03 optional sector index columns already present in panel
    h=objhash([ih,a.sector_columns_json and Path(a.sector_columns_json).read_text() or None]);
    if a.resume and done(out,'03_sector',h): df=pd.read_parquet(out/'working_panel.parquet'); sm=pd.read_parquet(out/'sector_inventory.parquet')
    else:
        cp(out,'03_sector','STARTED',input_hash=h); sm=pd.DataFrame()
        if a.sector_columns_json:
            cfg=json.loads(Path(a.sector_columns_json).read_text()); df,sm=sector_relative(df,cfg); atomic_pq(df,out/'working_panel.parquet')
        atomic_pq(sm,out/'sector_inventory.parquet'); cp(out,'03_sector','COMPLETE',input_hash=h,features=len(sm))
    # 04 inventory / quality
    safe=[c for c in safe_cols(df) if c not in TARGETS and c not in BANNED and not any(t in c.lower() for t in TOKENS)]; inv=pd.DataFrame({'feature':pd.unique(safe)})
    dmmap={};
    for z in (dm,sm):
        for _,r in z.iterrows(): dmmap[r.feature]=r.to_dict()
    rows=[]
    for c in inv.feature:
        rows.append(dmmap.get(c,{'feature':c,'family':'existing_panel','hypothesis':'existing panel feature; see feature_contract.csv','uses_future':False}))
    inv=pd.DataFrame(rows).drop_duplicates('feature'); q=quality(df,inv.feature.tolist()); atomic_pq(inv,out/'feature_inventory.parquet'); atomic_pq(q,out/'feature_quality.parquet')
    if inv.feature.duplicated().any() or inv.uses_future.fillna(False).astype(bool).any(): raise RuntimeError('Inventory leak/duplicate gate failed')
    cp(out,'04_inventory','COMPLETE',input_hash=objhash([ih,len(inv)]),features=len(inv))
    # 05 redundancy
    s=inv.feature.tolist(); sample=df[s].sample(min(len(df),200000),random_state=SEED); c=sample.corr(method='spearman',min_periods=100); seen=set(); rr=[]
    for i,n in enumerate(c.columns):
        if n in seen: continue
        mem=[c.columns[j] for j in range(i+1,len(c)) if np.isfinite(c.iloc[i,j]) and abs(c.iloc[i,j])>=.97]; cluster=[n]+mem; seen.update(cluster); rr.append({'representative':n,'cluster_size':len(cluster),'members':'|'.join(cluster[:100])})
    atomic_pq(pd.DataFrame(rr),out/'redundancy_clusters.parquet'); cp(out,'05_redundancy','COMPLETE',input_hash=objhash([ih,len(s)]))
    # 06 IC + 07 OOS + 08 null
    oos_all=[]
    for target in TARGETS:
        if target not in df: continue
        ihash=objhash([ih,target,s])
        cp(out,f'06_ic_{target}','STARTED',input_hash=ihash); z=inv.feature.tolist(); qx=q.set_index('feature'); z=[x for x in z if qx.loc[x,'coverage']>=.30 and qx.loc[x,'n_unique']>=10]; icdf=ic(df,z,target); atomic_pq(icdf,out/'ic'/f'{target}.parquet'); cp(out,f'06_ic_{target}','COMPLETE',input_hash=ihash,evaluated=len(z))
        if icdf.empty: continue
        icdf['screen']=icdf.mean_rank_ic.abs()*np.sqrt(icdf.positive_rank_ic_frac.clip(lower=0)); chosen=icdf.sort_values('screen',ascending=False).head(a.top_oos).feature.tolist(); cp(out,f'07_oos_{target}','STARTED',input_hash=ihash); od=oos(df,chosen,target,out); atomic_pq(od,out/'oos'/f'{target}_all.parquet'); cp(out,f'07_oos_{target}','COMPLETE',input_hash=ihash,evaluated=len(chosen)); oos_all.append(od)
        cp(out,f'08_null_{target}','STARTED',input_hash=objhash([ihash,a.null_permutations])); rng=np.random.default_rng(SEED); nr=[]
        for k in range(a.null_permutations):
            f=chosen[k%len(chosen)]; x=pd.to_numeric(df[f],errors='coerce').to_numpy(float).copy(); rng.shuffle(x); y=pd.to_numeric(df[target],errors='coerce').to_numpy(float); ok=np.isfinite(x)&np.isfinite(y); nr.append({'perm':k,'feature':f,'target':target,'corr':pd.Series(x[ok]).corr(pd.Series(y[ok]),method='spearman')})
        atomic_pq(pd.DataFrame(nr),out/'null'/f'{target}_null.parquet'); cp(out,f'08_null_{target}','COMPLETE',input_hash=objhash([ihash,a.null_permutations]))
    # 09 research map
    maps=[]
    for od in oos_all:
        if od.empty: continue
        z=od.groupby('feature',as_index=False).agg(mean_auc=('auc','mean'),median_auc=('auc','median'),min_auc=('auc','min'),mean_delta_logloss=('delta_logloss','mean'),min_delta_logloss=('delta_logloss','min'),folds=('fold','nunique')); z['target']=od.target.iloc[0]; z['research_pass']=(z.folds>=N_FOLDS)&(z.mean_delta_logloss>0)&(z.min_delta_logloss>0)&(z.mean_auc>.50); maps.append(z)
    dep=pd.concat(maps,ignore_index=True) if maps else pd.DataFrame(); atomic_pq(dep,out/'deployment_candidates.parquet'); cp(out,'09_final_map','COMPLETE',input_hash=objhash([ih,len(dep)]))
    # final gate
    required=['panel_audit.json','feature_inventory.parquet','feature_quality.parquet','redundancy_clusters.parquet','deployment_candidates.parquet']; missing=[x for x in required if not(out/x).exists()];
    if missing: raise RuntimeError(f'FINAL INTEGRITY FAILURE: {missing}')
    incomplete=[]
    for p in (out/'checkpoints').glob('*.json'):
        x=json.loads(p.read_text());
        if x.get('status')!='COMPLETE': incomplete.append(p.name)
    if incomplete: raise RuntimeError(f'FINAL INTEGRITY FAILURE: incomplete checkpoints {incomplete}')
    inv=pd.read_parquet(out/'feature_inventory.parquet'); bad=[c for c in inv.feature if c in BANNED or any(t in c.lower() for t in TOKENS)];
    if bad: raise RuntimeError(f'FINAL INTEGRITY FAILURE: forward-looking candidates {bad[:20]}')
    manifest.update(status='DISCOVERY COMPLETE',completed_at=pd.Timestamp.now(tz='Asia/Kolkata').isoformat(),feature_count=len(inv)); atomic_json(out/'run_manifest.json',manifest)
    print('DISCOVERY COMPLETE'); print('Output:',out); print('Features:',len(inv)); print('No live model/panel file was modified.')

if __name__=='__main__': main()
