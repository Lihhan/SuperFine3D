"""Isolated inference-only TF7 optimizations; never alter trained parameters.

Each Session belongs to one fixed support/LR/resolution and explicit CFG branches.
Destroy it before changing weights, resolution, input, dtype or device.
"""
import inspect
import textwrap
import types
import torch
import tf7.model as original
from tf5.routing import lr_attention, lr_plan
from tf6.near_attention import near_attention


def compile_attention(fn, name, inplace=False):
    source = textwrap.dedent(inspect.getsource(fn))
    source = source.replace('def '+fn.__name__+'(', 'def '+name+'(sorted_q, ')
    source = source.replace('    sorted_q = q[prep.perm]\n', '')
    if inplace:
        assert 'result=result.index_copy(0,qi,values)' in source
        source = source.replace('result=result.index_copy(0,qi,values)',
                                'result.index_copy_(0,qi,values)')
    ns = dict(fn.__globals__)
    exec(compile(source, '<tf7_speed_attention>', 'exec'), ns)
    return ns[name]


fast_near = compile_attention(near_attention, 'fast_near')
fast_lr = compile_attention(lr_attention, 'fast_lr', inplace=True)


def optimized_forward():
    source = textwrap.dedent(inspect.getsource(original.ElasticTF7SLatFlowModel.forward))
    start = source.index('        lr,lp=self._prepare_lr_input')
    end = source.index('        times=torch.cat', start)
    fixed = source[start:end]
    source = source[:start] + (
        '        if self._speed.fixed and self._speed.slot in self._speed.features:\n'
        '            lr,lp,evidence,coverage,anchor=self._speed.features[self._speed.slot]\n'
        '        else:\n' + textwrap.indent(fixed, '    ') +
        '            if self._speed.fixed:\n'
        '                self._speed.features[self._speed.slot]=(lr,lp,evidence,coverage,anchor)\n'
    ) + source[end:]
    source = source.replace('plans,self.routing_stats=lr_plan(',
                            'plans,self.routing_stats=self._speed.plan(')
    source = source.replace('next_lr,lk,lv=self._lr_step(i,l,lm_,c,lp_)',
                            'next_lr,lk,lv=self._speed.lr_step(i,l,lm_,c,lp_)')
    source = source.replace('local=near_attention(q,k,v,prep,near)',
        'sorted_q=q[prep.perm] if self._speed.pack else None\n'
        '            local=fast_near(sorted_q,q,k,v,prep,near) if self._speed.pack else near_attention(q,k,v,prep,near)')
    source = source.replace('read=lr_attention(q,lk,lv,prep,plans).flatten(1)',
        'read=(fast_lr(sorted_q,q,lk,lv,prep,plans) if self._speed.pack else lr_attention(q,lk,lv,prep,plans)).flatten(1)')
    ns = dict(original.__dict__, fast_near=fast_near, fast_lr=fast_lr)
    exec(compile(source, '<tf7_speed_forward>', 'exec'), ns)
    return ns['forward']


FORWARD = optimized_forward()


class Session(torch.nn.Module):
    def __init__(self, model, lr, coords, conditions, resolution, *, route=False,
                 fixed=False, kv=False, pack=False, baseline=False):
        super().__init__()
        assert not torch.is_grad_enabled() and not model.training
        self.model, self.lr, self.coords = model, lr, coords
        self.conditions, self.resolution = conditions, resolution
        self.route, self.fixed, self.kv, self.pack = route, fixed, kv, pack
        self.baseline = baseline
        self.features, self.kvs, self.route_value = {}, {}, None
        self.slot = None
        self.versions = (coords._version, lr.coords._version, lr.feats._version)
        self.condition_versions = [c._version for c in conditions]
        self.calls = dict(forward=0, route_builds=0, lr_steps=0)

    def plan(self, *args):
        if not self.route or self.route_value is None:
            value = lr_plan(*args)
            self.calls['route_builds'] += 1
            if self.route:
                # Prove disjoint query destinations before enabling in-place writes.
                ids = torch.cat([p[0] for p in value[0]])
                assert len(ids.unique()) == len(ids)
                self.route_value = value
            return value
        return self.route_value

    def lr_step(self, i, lr, lm, cond, positions):
        key = self.slot, i
        if self.kv and key in self.kvs:
            k, v = self.kvs[key]
            return lr, k, v
        next_lr, k, v = self.model._lr_step(i, lr, lm, cond, positions)
        self.calls['lr_steps'] += 1
        if self.kv:
            # k/v are views into a qkv allocation: clone releases unused q storage.
            self.kvs[key] = k.clone(), v.clone()
        return next_lr, k, v

    def forward(self, x, t, cond, **kwargs):
        assert not torch.is_grad_enabled() and not self.model.training
        assert x.coords is self.coords, 'Support changed: create a new session'
        assert (self.coords._version,self.lr.coords._version,self.lr.feats._version)==self.versions
        self.slot = next((i for i,c in enumerate(self.conditions) if cond is c), None)
        assert self.slot is not None, 'Unregistered CFG condition'
        assert cond._version == self.condition_versions[self.slot]
        self.calls['forward'] += 1
        if self.baseline:
            return self.model(x,t,cond,lr_cond=self.lr,hr_resolution=self.resolution,**kwargs)
        object.__setattr__(self.model, '_speed', self)
        try:
            return FORWARD(self.model,x,t,cond,lr_cond=self.lr,hr_resolution=self.resolution,**kwargs)
        finally:
            del self.model._speed

    def clear(self):
        self.features.clear(); self.kvs.clear(); self.route_value=None
