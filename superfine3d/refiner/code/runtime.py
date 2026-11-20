"""Isolated inference ablations. No checkpoint or production module edits."""
import dataclasses
import inspect
import textwrap
import types
import torch
from torch.nn import functional as F
from flash_attn import flash_attn_varlen_func as fa
import optimized as old
import tf7.model as original
from tf5.routing import lr_plan


def grouped_plans(plans, device):
    groups = {}
    count_before = 0
    for qi, ki, cq, ck, _, _ in plans:
        q, k, a, b = [v.cpu().tolist() for v in (qi, ki, cq, ck)]
        for j in range(len(a)-1):
            keys = tuple(k[b[j]:b[j+1]])
            groups.setdefault(keys, []).extend(q[a[j]:a[j+1]])
            count_before += len(keys)
    result = []
    entries = list(groups.items())
    for offset in range(0,len(entries),64):
        chunk=entries[offset:offset+64]
        qs=[];ks=[];qc=[0];kc=[0]
        for keys, queries in chunk:
            qs.extend(queries);ks.extend(keys)
            qc.append(len(qs));kc.append(len(ks))
        tensor=lambda x,d:torch.tensor(x,device=device,dtype=d)
        result.append((tensor(qs,torch.long),tensor(ks,torch.long),tensor(qc,torch.int32),tensor(kc,torch.int32),
                       max(qc[i+1]-qc[i] for i in range(len(qc)-1)),max(kc[i+1]-kc[i] for i in range(len(kc)-1))))
    assert len({q for _,qs in entries for q in qs}) == sum(len(qs) for _,qs in entries)
    return tuple(result), {'packed_lr_before':count_before,'packed_lr_after':sum(len(k) for k in groups)}


def new_forward():
    source=textwrap.dedent(inspect.getsource(original.ElasticTF7SLatFlowModel.forward))
    start=source.index('        lr,lp=self._prepare_lr_input')
    end=source.index('        times=torch.cat',start)
    fixed=source[start:end]
    source=source[:start]+('        if self._speed.slot in self._speed.features:\n'
        '            lr,lp,evidence,coverage,anchor=self._speed.features[self._speed.slot]\n'
        '        else:\n'+textwrap.indent(fixed,'    ')+
        '            self._speed.features[self._speed.slot]=(lr,lp,evidence,coverage,anchor)\n')+source[end:]
    replacements={
        'plans,self.routing_stats=lr_plan(':'plans,self.routing_stats=self._speed.plan(',
        'next_lr,lk,lv=self._lr_step(i,l,lm_,c,lp_)':'next_lr,lk,lv=self._speed.lr_step(i,l,lm_,c,lp_)',
        'q,k,v=self._qkv(block.self_attn,hidden,positions)':'q,k,v=self._speed.qkv(block.self_attn,hidden,positions)',
        'local=near_attention(q,k,v,prep,near)':'sorted_q=q if self._speed.sorted_layout else q[prep.perm]\n            local=self._speed.near(sorted_q,q,k,v,prep,near)',
        'read=lr_attention(q,lk,lv,prep,plans).flatten(1)':'read=self._speed.read(sorted_q,q,lk,lv,prep,plans).flatten(1)',
        '    for i,block in enumerate(self.blocks):':'    h,positions,prep,near,evidence,coverage,times,anchor=self._speed.layout(h,positions,prep,near,evidence,coverage,times,anchor)\n    for i,block in enumerate(self.blocks):',
        '    return self.out_layer(h.replace(F.layer_norm(h.feats,h.feats.shape[-1:])))':'    result=self.out_layer(h.replace(F.layer_norm(h.feats,h.feats.shape[-1:])))\n    return x.replace(result.feats[self._speed.inverse]) if self._speed.sorted_layout else result'
    }
    for a,b in replacements.items():
        assert a in source,a
        source=source.replace(a,b)
    ns=dict(original.__dict__)
    exec(compile(source,'<tf7-speed-v2>','exec'),ns)
    return ns['forward']


FORWARD=new_forward()


class Session(old.Session):
    def __init__(self,*args,variant='baseline',**kwargs):
        super().__init__(*args,route=True,fixed=True,kv=True,pack=True,**kwargs)
        self.variant=variant
        self.sorted_layout=variant in ('V1','combined')
        self.buffers=variant in ('V2','combined')
        self.merge=variant in ('V3','combined')
        self.cache_extra=variant in ('V4','combined')
        self.policy='none' if variant=='V5_none' else 'cpu' if variant=='V5_cpu' else 'gpu'
        self.pending={}; self.cpu_stream=torch.cuda.Stream() if self.policy=='cpu' else None
        self.workspace={};self.layout_value=None;self.inverse=None;self.extra={};self.undo=[];self.stats={}
        if variant.startswith('V7_'): self.model.near_chunk_cells=int(variant.split('_')[1])
        if self.cache_extra:
            for i,block in enumerate(self.model.blocks):
                module=block.cross_attn.to_kv
                prior=module.forward
                def cached(value,*a,_i=i,_prior=prior,**kw):
                    key=('image',self.slot,_i)
                    if key not in self.extra:self.extra[key]=_prior(value,*a,**kw)
                    return self.extra[key]
                self.undo.append((module,prior));module.forward=cached
        if variant=='V6_gate':
            self.elementwise=torch.compile(lambda g,h:torch.sigmoid(g)*h,fullgraph=True)
            self.activation=torch.compile(lambda low,anchor:F.silu(low+anchor),fullgraph=True)
            for adapter in self.model.lr_gated_residual:
                prior=adapter.forward
                def gated(read,evidence,coverage,time,anchor,hr,_a=adapter):
                    logits=_a.gate(torch.cat((evidence,coverage,time),-1))+_a.hr_gate(hr)
                    low=_a.down(read)
                    activated=self.activation(low,anchor) if _a.anchor_enabled else F.silu(low)
                    return self.elementwise(logits,_a.up(activated))
                self.undo.append((adapter,prior));adapter.forward=gated
        if variant=='V8_compile':
            for adapter in self.model.lr_gated_residual:
                prior=adapter.forward;self.undo.append((adapter,prior))
                adapter.forward=torch.compile(prior,fullgraph=True,mode='default')
        if variant=='V6_qk':
            self.norm_rotary=torch.compile(self._norm_rotary,fullgraph=True)

    def plan(self,*args):
        if self.route_value is None:
            value,stats=lr_plan(*args)
            self.calls['route_builds']+=1
            if self.merge:
                value,extra=grouped_plans(value,self.coords.device);self.stats.update(extra)
            self.route_value=value,stats
        return self.route_value

    def layout(self,h,pos,prep,near,*extras):
        if not self.sorted_layout:return (h,pos,prep,near,*extras)
        if self.layout_value is None:
            p=prep.perm;inv=prep.inv_perm;identity=torch.arange(len(p),device=p.device)
            newprep=dataclasses.replace(prep,perm=identity,inv_perm=identity)
            chunks=tuple((a,b,inv[idx],cq,ck,mq,mk) for a,b,idx,cq,ck,mq,mk in near.chunks)
            newnear=dataclasses.replace(near,chunks=chunks)
            self.layout_value=p,inv,newprep,newnear,h.coords[p],pos[p]
        p,self.inverse,newprep,newnear,coords,positions=self.layout_value
        h=original.sp.SparseTensor(feats=h.feats[p],coords=coords)
        return (h,positions,newprep,newnear,*(v[p] if v is not None else None for v in extras))

    def gather(self,name,source,idx):
        shape=(len(idx),*source.shape[1:]);key=name,source.dtype,source.device
        buf=self.workspace.get(key)
        if buf is None or buf.shape[0]<len(idx):
            buf=torch.empty(shape,device=source.device,dtype=source.dtype);self.workspace[key]=buf
        out=buf[:len(idx)]
        torch.index_select(source,0,idx,out=out)
        return out

    def near(self,sorted_q,q,k,v,prep,plan):
        if not self.buffers:return old.fast_near(sorted_q,q,k,v,prep,plan)
        result=torch.empty_like(sorted_q)
        for a,b,idx,cq,ck,mq,mk in plan.chunks:
            kk=self.gather('hr_k',k,idx);vv=self.gather('hr_v',v,idx)
            result[a:b].copy_(fa(sorted_q[a:b],kk,vv,cq,ck,mq,mk,dropout_p=0.,causal=False))
        return result if self.sorted_layout else result[prep.inv_perm]

    def read(self,sorted_q,q,k,v,prep,plans):
        if not self.buffers:return old.fast_lr(sorted_q,q,k,v,prep,plans)
        result=torch.zeros_like(sorted_q)
        for qi,ki,cq,ck,mq,mk in plans:
            qq=self.gather('lr_q',sorted_q,qi);kk=self.gather('lr_k',k,ki);vv=self.gather('lr_v',v,ki)
            result.index_copy_(0,qi,fa(qq,kk,vv,cq,ck,mq,mk,dropout_p=0.,causal=False))
        return result if self.sorted_layout else result[prep.inv_perm]

    def prefetch(self,key):
        if key not in self.kvs or key in self.pending:return
        with torch.cuda.stream(self.cpu_stream):
            self.pending[key]=tuple(v.to(self.coords.device,non_blocking=True) for v in self.kvs[key])

    def lr_step(self,i,lr,lm,cond,positions):
        key=self.slot,i
        if self.policy=='gpu':return super().lr_step(i,lr,lm,cond,positions)
        if self.policy=='none':
            self.calls['lr_steps']+=1
            return self.model._lr_step(i,lr,lm,cond,positions)
        if key in self.kvs:
            self.prefetch(key);torch.cuda.current_stream().wait_stream(self.cpu_stream)
            k,v=self.pending.pop(key)
            k.record_stream(torch.cuda.current_stream());v.record_stream(torch.cuda.current_stream())
            self.prefetch((self.slot,i+1))
            return lr,k,v
        nxt,k,v=self.model._lr_step(i,lr,lm,cond,positions);self.calls['lr_steps']+=1
        # Synchronous cold copy intentionally included in cold sampling timing.
        self.kvs[key]=tuple(z.detach().to('cpu').pin_memory() for z in (k,v))
        return nxt,k,v

    @staticmethod
    def _norm_rotary(q,k,qn,kn,rope,phase):
        return rope._rotary_embedding(qn(q),phase),rope._rotary_embedding(kn(k),phase)

    def qkv(self,attn,hidden,positions):
        if not self.cache_extra and self.variant!='V6_qk':return self.model._qkv(attn,hidden,positions)
        key=('phase',len(positions),self.sorted_layout)
        if key not in self.extra:
            angle=torch.outer(positions.reshape(-1).float(),self.model.physical_rope.freqs)
            phase=torch.polar(torch.ones_like(angle),angle).reshape(len(positions),-1)
            pad=self.model.model_channels//self.model.num_heads//2-phase.shape[-1]
            if pad:phase=torch.cat((phase,torch.ones(len(positions),pad,device=phase.device,dtype=phase.dtype)),1)
            self.extra[key]=phase
        q,k,v=attn.to_qkv(hidden.feats).reshape(-1,3,self.model.num_heads,self.model.model_channels//self.model.num_heads).unbind(1)
        phase=self.extra[key]
        if self.variant=='V6_qk':
            assert attn.qk_rms_norm
            q,k=self.norm_rotary(q,k,attn.q_rms_norm,attn.k_rms_norm,attn.rope,phase)
        else:
            if attn.qk_rms_norm:q,k=attn.q_rms_norm(q),attn.k_rms_norm(k)
            q,k=attn.rope._rotary_embedding(q,phase),attn.rope._rotary_embedding(k,phase)
        return q,k,v

    def forward(self,x,t,cond,**kwargs):
        assert not torch.is_grad_enabled() and not self.model.training
        assert x.coords is self.coords
        assert (self.coords._version,self.lr.coords._version,self.lr.feats._version)==self.versions
        self.slot=next((i for i,c in enumerate(self.conditions) if cond is c),None)
        assert self.slot is not None and cond._version==self.condition_versions[self.slot]
        self.calls['forward']+=1
        object.__setattr__(self.model,'_speed',self)
        try:return FORWARD(self.model,x,t,cond,lr_cond=self.lr,hr_resolution=self.resolution,**kwargs)
        finally:del self.model._speed

    def clear(self):
        if self.cpu_stream is not None:self.cpu_stream.synchronize()
        super().clear();self.workspace.clear();self.extra.clear();self.pending.clear();self.layout_value=None

    def close(self):
        self.clear()
        for module,prior in reversed(self.undo):module.forward=prior
        self.undo=[]
