"""One deterministic geometry-only SE(3) fit, shared by all downstream metrics."""
import itertools
import numpy as np
from scipy.spatial import cKDTree

VERSION='geometry_se3_v2'

def validate(r,t):
    r=np.asarray(r,dtype=np.float64);t=np.asarray(t,dtype=np.float64)
    if r.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError('Invalid rigid transform')
    if not np.allclose(r.T@r,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(r),1,atol=1e-6):
        raise ValueError('Only proper rotation is allowed: no scale, shear or reflection')
    return r,t

def fit(points,target):
    p=np.asarray(points,dtype=np.float64);g=np.asarray(target,dtype=np.float64)
    if p.ndim!=2 or g.ndim!=2 or p.shape[1]!=3 or g.shape[1]!=3 or min(len(p),len(g))<4:
        raise ValueError('Need finite nondegenerate 3D surface samples')
    if not np.isfinite(p).all() or not np.isfinite(g).all():raise ValueError('Nonfinite points')
    # Fixed geometry samples, bounded CPU cost; no image/normal metric enters fitting.
    p=p[np.linspace(0,len(p)-1,min(len(p),10000),dtype=int)]
    g=g[np.linspace(0,len(g)-1,min(len(g),10000),dtype=int)]
    tree=cKDTree(g)
    def score(q):
        return float((np.mean(tree.query(q,workers=4)[0]**2)+np.mean(cKDTree(q).query(g,workers=4)[0]**2))/2)
    rotations=[]
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1,1),repeat=3):
            r=np.eye(3)[list(perm)]*np.asarray(signs)[:,None]
            if np.linalg.det(r)>0:rotations.append(r)
    starts=[]
    for i,r in enumerate(rotations):
        t=g.mean(0)-p.mean(0)@r;starts.append((score(p@r+t),i,t))
    results=[]
    for _,idx,t in sorted(starts,key=lambda x:(x[0],x[1]))[:4]:
        r=rotations[idx].copy()
        for _ in range(40):
            q=p@r+t;_,nn=tree.query(q,workers=4);dest=g[nn]
            qa,ga=q.mean(0),dest.mean(0)
            u,_,vt=np.linalg.svd((q-qa).T@(dest-ga));d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt)
            dr=u@d@vt;dt=ga-qa@dr;r,t=r@dr,t@dr+dt
            if np.linalg.norm(dr-np.eye(3))+np.linalg.norm(dt)<1e-7:break
        validate(r,t);results.append((score(p@r+t),r,t))
    best=min(x[0] for x in results)
    # Near-identical geometry optima are resolved deterministically, favouring
    # the smallest rotation rather than arbitrary 180-degree symmetric flips.
    tied=[x for x in results if x[0]<=best+max(1e-10,best*1e-4)]
    cost,r,t=min(tied,key=lambda x:float(np.arccos(np.clip((np.trace(x[1])-1)/2,-1,1))))
    return r,t,dict(version=VERSION,objective='symmetric squared geometry distance',
        objective_value=cost,scale=1.,determinant=float(np.linalg.det(r)),
        candidates=4,near_tied_candidates=len(tied),reflection_allowed=False,
        image_metrics_used=False,row_vector_convention='vertices @ rotation + translation')

def rigid_alignment(p,g):
    r,t,_=fit(p,g);return r,t

def transform_mesh(mesh,r,t):
    r,t=validate(r,t);out=mesh.copy();out.vertices=np.asarray(mesh.vertices)@r+t
    # Recompute face/vertex normals from transformed geometry. Translation never
    # enters a normal; proper rotation preserves winding and outward orientation.
    out._cache.clear()
    return out
