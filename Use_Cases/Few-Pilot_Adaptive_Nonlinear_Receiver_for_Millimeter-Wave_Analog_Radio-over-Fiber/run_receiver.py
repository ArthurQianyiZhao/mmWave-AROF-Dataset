"""Reproduce only the receiver methods and calibration budgets in the paper.

Run from any directory; dataset and output paths are relative to this use case.
Saved reference results are never overwritten (new runs go to reproduced/).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from fewshot_equalizer.data import discover_scenarios, prepare_scenarios
from fewshot_equalizer.hammerstein import (
    fit_hammerstein_teacher, calibrate_hammerstein_latent, complex_vector,
    decision_directed_adapt,
)
from fewshot_equalizer.teacher_calibration import pca_basis
from fewshot_equalizer.engine import write_csv
from receiver_helpers import load, prepare, mean_prior_fit, dd_many, score, aggregate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = HERE / 'model/source_prior.npz'
CACHE = HERE / 'cache_2sps'
REGIONS = {'train': [0,59994], 'val': [59994,69993], 'test': [69993,99990]}

def load_prior(path=PRIOR):
    with np.load(path, allow_pickle=False) as data:
        cp = {k:data[k].copy() for k in ('center','basis','source_coefficients')}
        cp.update(json.loads(str(data['metadata'])))
    return cp

def save_prior(path, vectors, source_ids, target_ids):
    center,basis,pca=pca_basis(vectors,15)
    metadata={'source_ids_used':source_ids,'protocol':{'target_ids':target_ids},
              'regions':REGIONS,'cache_dir':str((HERE/'cache_2sps').relative_to(ROOT)),
              'model_args':{'taps':11,'orders':[1,3],'latent_dim':15,'samples_per_symbol':2},
              'pca':pca}
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,center=center,basis=basis,source_coefficients=vectors,
                        metadata=json.dumps(metadata))

def source_condition(s):
    return s.fiber in ('1m','5km') and s.carrier in ('28','30') and s.power in ('3','7')

def train(records):
    sources=[s for s in records if source_condition(s)]
    assert len(sources)==32
    prepare_scenarios(sources,CACHE,samples_per_symbol=2)
    vectors=[]
    for s in sources:
        rx,tx,_=load(CACHE,s)
        vectors.append(fit_hammerstein_teacher(rx,tx,REGIONS['train'],11,(1,3),25000,1e-6).coefficient_vector)
        print('fit',s.key,flush=True)
    path=HERE/'reproduced/source_prior.npz'
    save_prior(path,np.stack(vectors),[s.scenario_id for s in sources],
               [s.scenario_id for s in records if not source_condition(s)])
    print(path)

def teacher_initials(pd,tx,candidates,vectors,ridge=1.):
    pp=np.unique(np.linspace(0,len(candidates)-1,8,dtype=np.int64))
    phi,x=pd[pp],tx[candidates[pp]].astype(np.complex128)
    teachers=np.stack([complex_vector(v) for v in vectors]).astype(np.complex128)
    teacher=teachers[np.argmin(np.mean(np.abs(phi@teachers.T-x[:,None])**2,axis=0))]
    return teacher,mean_prior_fit(phi,x,teacher,ridge)

def evaluate(records,cp,budgets=False,max_scenarios=None,ridge=1.):
    targets=[s for s in records if s.scenario_id in cp['protocol']['target_ids']]
    if max_scenarios:targets=targets[:max_scenarios]
    prepare_scenarios(targets,CACHE,samples_per_symbol=2)
    out=HERE/'reproduced';out.mkdir(exist_ok=True)
    rows=[];budget_rows=[]
    for n,s in enumerate(targets,1):
        arrays=load(CACHE,s);_,tx,bits=arrays
        _,cand,ci,ti,pd,ud,td=prepare(arrays,REGIONS)
        assert len(ci)==8965 and len(ti)==29987
        h=calibrate_hammerstein_latent(pd,tx,cand,cp['center'],cp['basis'],8,1e-3,'uniform')[0]
        _,adjusted=teacher_initials(pd,tx,cand,cp['source_coefficients'],ridge)
        # Keep the original single-vector solver for the headline receiver.
        refined=decision_directed_adapt(h,ud,s.order,iterations=12,pilot_weight=0,ridge=2e-4)
        comparison=dd_many(adjusted,ud,s.order)[:,0]
        variants={'subspace_no_dd':h,'subspace_dd':refined,
                  f'ridge_teacher:{ridge:g}':comparison}
        for method,w in variants.items():rows.append(score(s,bits,tx,ti,td@w,method))
        if s.fiber=='10km' and s.carrier=='29' and s.power=='5':
            np.savez_compressed(out/f'constellation_{s.modulation}.npz',
                                prediction=td@refined,target=tx[ti],indices=ti)
        if budgets:
            for pilots,count in [(4,8965),(8,8965),(16,8965),(8,4096)]:
                initial=calibrate_hammerstein_latent(pd,tx,cand,cp['center'],cp['basis'],pilots,1e-3,'uniform')[0]
                pos=np.unique(np.linspace(0,len(ud)-1,count,dtype=np.int64))
                w=dd_many(initial,ud[pos],s.order)[:,0]
                budget_rows.append(score(s,bits,tx,ti,td@w,'subspace_dd',pilots=pilots,unlabeled_symbols=count))
        print(f'{n}/{len(targets)} {s.key}',flush=True)
    suffix='_smoke' if max_scenarios else ''
    write_csv(rows,out/f'matched_baselines_per_scenario{suffix}.csv')
    write_csv(aggregate(rows),out/f'matched_baselines_summary{suffix}.csv')
    if budgets:
        write_csv(budget_rows,out/f'budget_per_scenario{suffix}.csv')
        write_csv(aggregate(budget_rows,('pilots','unlabeled_symbols')),out/f'budget_summary{suffix}.csv')
    print(json.dumps(aggregate(rows),indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train','evaluate','budgets'])
    parser.add_argument('--prior',type=Path,default=PRIOR)
    parser.add_argument('--max-scenarios',type=int,help='Smoke evaluation only; never a paper result')
    parser.add_argument('--teacher-ridge',type=float,default=1.,help='Source-validation choice; never tune on target test')
    args=parser.parse_args();records=discover_scenarios(ROOT)
    if len(records)!=108:raise ValueError(f'Expected 108 synchronized captures, found {len(records)} in {ROOT}')
    if args.action=='prepare':prepare_scenarios(records,CACHE,samples_per_symbol=2)
    elif args.action=='train':train(records)
    else:evaluate(records,load_prior(args.prior),args.action=='budgets',args.max_scenarios,args.teacher_ridge)

if __name__=='__main__':main()
