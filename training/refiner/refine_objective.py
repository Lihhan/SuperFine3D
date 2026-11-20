"""Resolution-normalized, teacher-masked refinement objectives.

Depths must share a camera and physical frame. h_lr uses those same units.
Teacher visibility defines validity; missing prediction is reported separately.
This module does not differentiate discrete mesh topology.
"""
import torch
import torch.nn.functional as F


def refinement_objective(pred, high, low, teacher_valid, pred_valid, h_lr,
                         tolerance_voxels=.25, keep_scale_voxels=.5):
    assert h_lr > 0 and pred.shape == high.shape == low.shape
    high, low = high.detach(), low.detach()
    # Erode teacher validity so finite differences cannot cross a silhouette.
    mask = teacher_valid.detach().float()
    mask = -F.max_pool2d(-F.pad(mask, (1,1,1,1), value=0), 3, stride=1)
    visible = mask * pred_valid.detach().float()
    change = (high-low)/h_lr
    estimated = (pred-low)/h_lr
    keep = torch.exp(-change.abs()/keep_scale_voxels) * visible
    detail = (1-torch.exp(-change.abs()/keep_scale_voxels)) * visible
    def mean(x,w):
        return (x*w).sum()/w.sum().clamp_min(1)
    excess = (estimated.abs()-change.abs()-tolerance_voxels).clamp_min(0)
    keep_loss = mean(excess.square(),keep)
    # Signed target matching remains essential: tolerance alone permits wrong sign.
    target_loss = mean(F.smooth_l1_loss(estimated,change,reduction='none',beta=.25),detail)
    grads=[]
    for dim in (-2,-1):
        a=estimated.diff(dim=dim);b=change.diff(dim=dim)
        w=detail.narrow(dim,0,detail.shape[dim]-1)*visible.narrow(dim,1,visible.shape[dim]-1)
        grads.append(mean(F.smooth_l1_loss(a,b,reduction='none',beta=.25),w))
    return dict(keep=keep_loss,detail=target_loss+sum(grads)/2), dict(
        valid_pixels=float(mask.sum()), keep_mass=float(keep.sum()),
        detail_mass=float(detail.sum()),
        missing_fraction=float((mask*(1-pred_valid.detach().float())).sum()/mask.sum().clamp_min(1)))


def self_check():
    x=torch.zeros(1,1,16,16);mask=torch.ones_like(x)
    same,_=refinement_objective(x,x,x,mask,mask,1.)
    assert all(float(v)==0 for v in same.values())
    p=x.clone();p[:,:,5:10,5:10]=1;p.requires_grad_()
    losses,_=refinement_objective(p,x,x,mask,mask,1.)
    g=torch.autograd.grad(losses['keep'],p)[0]
    assert float(losses['keep'])>0 and g[:,:,6:9,6:9].min()>0
    target=p.detach();perfect,_=refinement_objective(target,target,x,mask,mask,1.)
    assert all(float(v)==0 for v in perfect.values())
    erased,_=refinement_objective(x,target,x,mask,mask,1.)
    assert float(erased['detail'])>0
    scaled,_=refinement_objective(p*2,x*2,x*2,mask,mask,2.)
    assert torch.allclose(losses['keep'],scaled['keep'])
    _,missing=refinement_objective(x,x,x,mask,mask*0,1.)
    assert missing['missing_fraction']==1
    return dict(passed=True, checks=['identity','extra_bump_gradient','real_detail_preserved',
        'erased_detail_penalty','physical_scale_invariance','missing_surface_reported'])

if __name__=='__main__':
    import json
    print(json.dumps(self_check()))
