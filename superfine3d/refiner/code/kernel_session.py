"""Isolated kernel experiments. Keep model weights, supports and routes unchanged."""
import torch
import runtime


def indexed(q,k,v,cq,ck,mq,mk,idx,with_lse=False):
    try:
        import flash_attn_3d_rope_fa2_C as ext
    except (ImportError,OSError):
        # The system headers do not match its CUDA 12.8 compiler. Preserve
        # index-only HR reads with the installed varlen kernel as a fallback.
        import flash_attn
        ids=idx.to(device=k.device,dtype=torch.long)
        kk=torch.index_select(k,0,ids);vv=torch.index_select(v,0,ids)
        out=flash_attn.flash_attn_varlen_func(q,kk,vv,cq,ck,mq,mk,
            dropout_p=0.,causal=False)
        return (out,None) if with_lse else out
    assert q.shape[-1]==128 and k.shape[-1]==128
    result=ext.varlen_fwd(q,k,v,None,cq,ck,None,None,None,None,mq,mk,
        0.,q.shape[-1]**-.5,False,False,-1,-1,0.,False,None,
        None,None,False,None,0,10000.,1.,None,idx)
    return result[:2] if with_lse else result[0]


class Session(runtime.Session):
    def __init__(self,*args,variant='baseline',runtime_variant='baseline',verify_kernel=True,**kw):
        super().__init__(*args,variant=runtime_variant,**kw)
        self.verify_kernel=verify_kernel
        self.kernel_variant=variant
        self.index_cache={}
        self.graphs={}
        self.graph_pool=None
        self.kernel_checks=[]
        self.stats['kernel_checks']=self.kernel_checks
        if variant=='K4_norm_rope':
            self.variant='V6_qk'
            from kernel_norm_rope import fused
            self.norm_rotary=fused
        if variant=='K5_adaln_residual':
            # Separate script patches only the arithmetic in _pre/_post;
            # all linear, attention and normalization modules are preserved.
            from kernel_arithmetic import install
            self.prior_arithmetic=install(self.model)
        if variant=='K7_ffn':
            from torch.nn import functional as F
            def dense(x,w1,b1,w2,b2):
                return F.linear(F.gelu(F.linear(x,w1,b1),approximate='tanh'),w2,b2)
            compiled=torch.compile(dense,fullgraph=True,options={'triton.cudagraphs':False})
            for block in self.model.blocks:
                prior=block.mlp.forward
                self.undo.append((block.mlp,prior))
                first,act,last=block.mlp.mlp
                assert act.approximate=='tanh'
                def forward(x,_first=first,_last=last):
                    return x.replace(compiled(x.feats,_first.weight,_first.bias,_last.weight,_last.bias))
                block.mlp.forward=forward

    def idx(self,idx):
        key=id(idx)
        if key not in self.index_cache:self.index_cache[key]=(idx,idx.to(torch.int32))
        return self.index_cache[key][1]

    def near(self,sq,q,k,v,prep,plan):
        if self.kernel_variant not in ('K1_hr_indexed','K2_hr_allcells','K8_combined','K9_splitkv'):
            return super().near(sq,q,k,v,prep,plan)
        if self.kernel_variant=='K9_splitkv':
            outputs=[]
            for a,b,idx,cq,ck,mq,mk in plan.chunks:
                key=('split',id(idx))
                if key not in self.index_cache:
                    offsets=ck.cpu().tolist();parts=[]
                    for half in range(2):
                        selected=[];lengths=[]
                        for start,end in zip(offsets[:-1],offsets[1:]):
                            middle=(start+end)//2
                            segment=idx[start:middle] if half==0 else idx[middle:end]
                            selected.append(segment);lengths.append(len(segment))
                        cu=torch.tensor([0]+lengths,device=k.device,dtype=torch.int32).cumsum(0).int()
                        parts.append((torch.cat(selected).int(),cu,max(lengths)))
                    ids=torch.cat([p[0] for p in parts])
                    cks=torch.cat((parts[0][1],parts[1][1][1:]+len(parts[0][0])))
                    cqs=torch.cat((cq,cq[1:]+b-a))
                    active=torch.cat([torch.repeat_interleave((p[1][1:]-p[1][:-1])>0,cq[1:]-cq[:-1],output_size=b-a) for p in parts]).reshape(2,1,b-a)
                    self.index_cache[key]=(idx,ids,cks,cqs,max(p[2] for p in parts),active)
                _,ids,cks,cqs,mks,active=self.index_cache[key]
                # Both K partitions are dispatched together, not two sequential launches.
                value,lse=indexed(sq[a:b].repeat(2,1,1),k,v,cqs,cks,mq,mks,ids,True)
                vals=value.reshape(2,b-a,*value.shape[1:])
                lse=lse.reshape(q.shape[1],2,b-a).permute(1,0,2)
                lse=torch.where(active,lse,float('-inf'))
                weights=torch.softmax(lse,dim=0).transpose(1,2).unsqueeze(-1)
                outputs.append((vals.float()*weights).sum(0).to(q.dtype))
            out=torch.cat(outputs)
        elif self.kernel_variant in ('K2_hr_allcells','K8_combined'):
            key=('all',id(plan))
            if key not in self.index_cache:
                chunks=plan.chunks
                idx=torch.cat([p[2] for p in chunks]).to(torch.int32)
                lengths=torch.cat([p[4][1:]-p[4][:-1] for p in chunks])
                ck=torch.cat([torch.zeros(1,device=k.device,dtype=torch.int32),lengths.cumsum(0).int()])
                self.index_cache[key]=(plan,idx,ck,max(p[6] for p in chunks))
            _,idx,ck,mk=self.index_cache[key]
            out=indexed(sq,k,v,prep.cu_seqlens,ck,prep.max_seqlen,mk,idx)
        else:
            outputs=[]
            for a,b,idx,cq,ck,mq,mk in plan.chunks:
                outputs.append(indexed(sq[a:b],k,v,cq,ck,mq,mk,self.idx(idx)))
            out=torch.cat(outputs)
        output=out[prep.inv_perm]
        if self.verify_kernel and not any(c['operator']=='HR attention' for c in self.kernel_checks):
            reference=super().near(sq,q,k,v,prep,plan)
            d=(output.float()-reference.float()).abs()
            self.kernel_checks.append(dict(operator='HR attention',exact=torch.equal(output,reference),
                finite=bool(torch.isfinite(output).all()),max_abs=float(d.max()),mean_abs=float(d.mean())))
        return output

    def read(self,sq,q,k,v,prep,plans):
        if self.kernel_variant not in ('K3_lr_indexed','K8_combined'):
            return super().read(sq,q,k,v,prep,plans)
        result=torch.zeros_like(sq)
        for qi,ki,cq,ck,mq,mk in plans:
            assert k.dtype==v.dtype==q.dtype
            out=indexed(sq[qi],k,v,cq,ck,mq,mk,self.idx(ki))
            result.index_copy_(0,qi,out)
        output=result[prep.inv_perm]
        if self.verify_kernel and not any(c['operator']=='LR attention' for c in self.kernel_checks):
            reference=super().read(sq,q,k,v,prep,plans)
            d=(output.float()-reference.float()).abs()
            self.kernel_checks.append(dict(operator='LR attention',exact=torch.equal(output,reference),
                finite=bool(torch.isfinite(output).all()),max_abs=float(d.max()),mean_abs=float(d.mean())))
        return output

    def forward(self,x,t,cond,**kw):
        if self.kernel_variant!='K6_cuda_graph':return super().forward(x,t,cond,**kw)
        slot=next(i for i,c in enumerate(self.conditions) if c is cond)
        if slot not in self.graphs:
            # First call populates route and per-CFG LR caches before capture.
            super().forward(x,t,cond,**kw)
            static_x=x.replace(x.feats.clone());static_t=t.clone()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):super().forward(static_x,static_t,cond,**kw)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            if self.graph_pool is None:self.graph_pool=torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph,pool=self.graph_pool):out=super().forward(static_x,static_t,cond,**kw)
            self.graphs[slot]=(graph,static_x,static_t,out)
        graph,sx,st,out=self.graphs[slot]
        sx.feats.copy_(x.feats);st.copy_(t);graph.replay()
        return x.replace(out.feats.clone())

    def clear(self):
        if hasattr(self,'graphs'):self.graphs.clear()
        self.graph_pool=None
        if hasattr(self,'index_cache'):self.index_cache.clear()
        super().clear()

    def close(self):
        if hasattr(self,'prior_arithmetic'):
            self.model._pre,self.model._post=self.prior_arithmetic
        super().close()
