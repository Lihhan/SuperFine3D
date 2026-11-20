"""R3-v2: continuous edge decisions and conservative teacher tangent planes.

Only matched decoder coordinates carry edge labels. Unmatched coordinates are
unknown, not empty. Tangent planes require an exact teacher mesh vertex and
agreement of incident face normals; no nearest-surface attraction across gaps.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tf11.native_surface import correspondence


def balanced_edges(logits, labels):
    terms=[]
    for axis in range(3):
        parts=[]
        for positive in [False, True]:
            mask=(labels[:,axis]>.5)==positive
            if mask.any():
                parts.append(F.binary_cross_entropy_with_logits(logits[mask,axis],labels[mask,axis]))
        if parts:terms.append(torch.stack(parts).mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0


def teacher_planes(coords, feats, vertices, faces, lo, core_edge, margin):
    """Normals only at exact decoded teacher vertices, away from creases."""
    offset=(1+2*margin)*feats[:,:3].float().sigmoid()-margin
    # Same physical coordinate and local normalization as RefinementSurface.
    positions=((coords[:,1:].float()+offset)/16-(lo.float()+core_edge/2))/(core_edge/2)
    v=vertices.detach().float().cpu().numpy();f=faces.detach().long().cpu().numpy()
    out=np.zeros((len(coords),3),np.float32);valid=np.zeros(len(coords),bool)
    if len(f)==0 or len(v)==0:return torch.from_numpy(out).to(feats.device),torch.from_numpy(valid).to(feats.device)
    tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    length=np.linalg.norm(cross,axis=1);unit=cross/np.maximum(length[:,None],1e-12)
    # Dual-grid winding need not be globally oriented. Align signs only for
    # normal averaging; the squared normal displacement is orientation invariant.
    reference=np.zeros_like(v)
    for j in range(3):reference[f[:,j]]=unit
    normal=np.zeros_like(v);count=np.zeros(len(v))
    for j in range(3):
        oriented=unit*np.where((reference[f[:,j]]*unit).sum(1,keepdims=True)<0,-1.,1.)
        np.add.at(normal,f[:,j],oriented);np.add.at(count,f[:,j],length>1e-12)
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
    mincos=np.ones(len(v))
    for j in range(3):np.minimum.at(mincos,f[:,j],np.abs((normal[f[:,j]]*unit).sum(1)))
    distance,index=cKDTree(v).query(positions.detach().cpu().numpy(),workers=1)
    # .02 final-grid voxel: exact correspondence tolerance, not NN surface fitting.
    tolerance=.02/(core_edge*8)
    valid=(distance<tolerance)&(mincos[index]>.9)&(count[index]>=3)
    out[valid]=normal[index[valid]]
    return torch.from_numpy(out).to(feats.device),torch.from_numpy(valid).to(feats.device)


def raw_losses(pred, teacher, normals, plane_valid, lo, core_edge, inset, margin):
    core=((pred.coords[:,1:] >= (lo+inset)*16)&(pred.coords[:,1:]<(lo+core_edge-inset)*16)).all(1)
    pc=pred.coords[core];pf=pred.feats[core].float()
    if len(pc)==0:raise ValueError('Empty decoder supervision core')
    valid,ids=correspondence(pc,teacher.coords,1024)
    qf=teacher.feats[ids].detach().float()
    edge=balanced_edges(pf[valid,3:6],(qf[valid,3:6]>0).float())
    reliable=valid & plane_valid[ids]
    po=(1+2*margin)*pf[:,:3].sigmoid()-margin
    qo=(1+2*margin)*qf[:,:3].sigmoid()-margin
    # Offset is already in final HR geometry voxel units. No tangential penalty.
    displacement=((po-qo)*normals[ids]).sum(-1)
    plane=F.smooth_l1_loss(displacement[reliable],torch.zeros_like(displacement[reliable]),beta=.1) if reliable.any() else pf.sum()*0
    report=dict(core_tokens=len(pc),matched=int(valid.sum()),unknown=int((~valid).sum()),
                plane_tokens=int(reliable.sum()),teacher_positive_edges=int((qf[valid,3:6]>0).sum()),
                negative_edges=int((qf[valid,3:6]<=0).sum()))
    return dict(edge=edge,plane=plane),report
