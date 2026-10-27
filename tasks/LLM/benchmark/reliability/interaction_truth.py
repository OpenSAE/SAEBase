"""Known-order coalition behaviors and independent orthogonal interaction energy."""
from __future__ import annotations
from itertools import combinations,product
from typing import Any
import numpy as np

CASES=("first_order","pairwise","third_order")

def behavior(mask,case:str):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or mask.shape[1]<3 or case not in CASES:
        raise ValueError("invalid interaction mask/degree")
    x=2*mask.astype(np.float64)-1
    value=5+.3*x[:,0]-.8*x[:,1]+1.3*x[:,2]
    if mask.shape[1]>3:
        value+=.5*x[:,3]
    if case in ("pairwise","third_order"):
        value+=2.5*x[:,0]*x[:,1]
    if case=="third_order":
        value+=3.5*x[:,0]*x[:,1]*x[:,2]
    return value

def exact_order_energy(case:str,K:int=4)->dict[int,float]:
    """Fourier expansion under uniform Boolean coalitions: known degree truth."""
    if case not in CASES or K<4 or K>8:
        raise ValueError("unsupported synthetic truth dimensions")
    mask=np.asarray(list(product((False,True),repeat=K)),dtype=bool)
    y=behavior(mask,case)
    x=2*mask.astype(float)-1
    result={}
    for degree in range(1,K+1):
        energy=0.
        for subset in combinations(range(K),degree):
            basis=np.prod(x[:,subset],axis=1)
            coef=float(np.mean(y*basis))
            energy+=coef*coef
        result[degree]=float(energy)
    return result

def verify_original_sar(paper,*,seed:int=7,M:int=512,n_fit:int=350)->dict:
    """Call original SAR estimator; do not redefine its numerical definition."""
    if not callable(getattr(paper,"coalitions",None)) or not callable(getattr(paper,"sar",None)):
        raise TypeError("trusted original SAR module must expose coalitions and sar")
    masks,fit,held=paper.coalitions(K=4,M=M,n_fit=n_fit,seed=seed)
    cases={}
    for case in CASES:
        g=behavior(masks,case)
        g_full=float(behavior(np.ones((1,4),dtype=bool),case)[0])
        upstream=paper.sar(g,masks,fit,held,g_full=g_full)
        energy=exact_order_energy(case)
        first=float(upstream["sar1"])
        second=float(upstream["sar2"])
        if case=="first_order":
            passed=first<1e-8 and second<1e-8 and energy[2]<1e-12 and energy[3]<1e-12
        elif case=="pairwise":
            passed=first>.02 and second<1e-8 and energy[2]>1. and energy[3]<1e-12
        else:
            passed=first>.02 and second>.02 and energy[3]>1.
        cases[case]={"status":"passed" if passed else "failed",
            "oracle_energy_by_order":{str(k):v for k,v in energy.items()},
            "native_sar1":first,"native_sar2":second,
            "n_fit":len(fit),"n_held_out":len(held),"seed":seed,
            "note":"No SAR3 estimator exists in the original SAR paper implementation"}
    return cases
