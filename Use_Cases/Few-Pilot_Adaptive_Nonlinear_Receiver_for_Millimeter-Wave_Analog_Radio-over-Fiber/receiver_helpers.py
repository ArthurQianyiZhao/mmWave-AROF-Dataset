from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from fewshot_equalizer.compat import bits_for_symbol_indices, calculate_ber, calculate_evm, qam_demod
from fewshot_equalizer.hammerstein import normalize_from_region, parallel_hammerstein_design, qam_decisions

def load(cache,s):
    with np.load(cache/f'{s.key}.npz',allow_pickle=False) as a:
        return a['rx'].astype(np.complex64),a['tx'].astype(np.complex64),a['bits'].astype(np.uint8)


def mean_prior_fit(phi,x,prior,lam):
    residual=x-phi@prior
    dual=phi@phi.conj().T
    dual.flat[::len(x)+1]+=lam
    return prior+phi.conj().T@np.linalg.solve(dual,residual)


def dd_many(initial,design,order,steps=12,ridge=2e-4):
    """Algebraically identical full-row, zero-pilot-weight DD; reuse Gram."""
    w=np.asarray(initial,dtype=np.complex128)
    if w.ndim==1:w=w[:,None]
    else:w=w.copy()
    penalty=np.full(len(w),ridge);penalty[-1]*=.01
    gram=design.conj().T@design/len(design)
    gram.flat[::len(w)+1]+=penalty
    factor=cho_factor(gram)
    dh=design.conj().T/len(design)
    for _ in range(steps):
        pred=design@w
        pseudo=np.column_stack([qam_decisions(pred[:,j],order) for j in range(w.shape[1])])
        w=cho_solve(factor,dh@pseudo+penalty[:,None]*w)
    return w


def prepare(arrays,regions):
    rx,tx,bits=arrays
    norm,_,_=normalize_from_region(rx,tuple(regions['val']))
    lo,hi=regions['val'][0]+1024+5,regions['val'][1]-5
    candidates=np.unique(np.linspace(lo,hi-1,512,dtype=np.int64))
    ci=np.arange(lo,hi,dtype=np.int64)
    ti=np.arange(regions['test'][0]+5,regions['test'][1]-5,dtype=np.int64)
    return norm,candidates,ci,ti,parallel_hammerstein_design(norm,candidates,11,(1,3)),parallel_hammerstein_design(norm,ci,11,(1,3)),parallel_hammerstein_design(norm,ti,11,(1,3))


def score(s,bits,tx,ti,pred,method,**extras):
    truth=bits_for_symbol_indices(bits,ti,s.order)
    errors,ber=calculate_ber(truth,qam_demod(pred,s.order))
    return {**s.metadata(),'method':method,'bit_errors':int(errors),'total_bits':int(truth.size),
            'ber':float(ber),'evm_rms_percent':calculate_evm(tx[ti],pred),**extras}


def aggregate(rows,keys=('method',)):
    totals={}
    for row in rows:
        key=tuple(row[k] for k in keys)
        item=totals.setdefault(key,{'bit_errors':0,'total_bits':0,'captures':0,'below_reference':0})
        item['bit_errors']+=int(row['bit_errors']);item['total_bits']+=int(row['total_bits'])
        item['captures']+=1;item['below_reference']+=float(row['ber'])<.0038
    return [{**dict(zip(keys,key)),**v,'ber':v['bit_errors']/v['total_bits']} for key,v in totals.items()]

