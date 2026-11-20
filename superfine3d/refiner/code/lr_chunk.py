"""Bound the pointwise LR FFN workspace, preserving the LR attention stream."""
import inspect,types,torch

def ffn(block,result,scale,shift,first,act,last,delta_in,delta_out,chunk):
    assert len(result.layout)==1,'This bounded inference path expects one asset'
    output=torch.empty_like(result.feats)
    for start in range(0,len(output),chunk):
        end=min(start+chunk,len(output))
        local=type(result)(coords=result.coords[start:end],feats=result.feats[start:end])
        ff=local.replace(block.norm3(local.feats))*(1+scale)+shift
        z=first(ff)
        z=z.replace(z.feats+delta_in(ff.feats).to(z.feats.dtype))
        z=act(z);y=last(z)
        y=y.replace(y.feats+delta_out(z.feats).to(y.feats.dtype))
        output[start:end]=y.feats
    return result.replace(output)

def install(model,chunk=16384):
    original=model._lr_step.__func__
    source=inspect.getsource(original)
    # Dedent only the method, preserving the original attention and early return.
    import textwrap
    source=textwrap.dedent(source)
    start=source.index('    ff=result.replace(block.norm3(result.feats))')
    stop=source.index('    return result+y*mlp_gate,k,v',start)
    replacement='    first,act,last=block.mlp.mlp\n    y=bounded_lr_ffn(block,result,scale,shift,first,act,last,self.lr_ffn_in_delta[i],self.lr_ffn_out_delta[i],'+str(chunk)+')\n'
    source=source[:start]+replacement+source[stop:]
    namespace=dict(original.__globals__,bounded_lr_ffn=ffn)
    exec(compile(source,'<bounded-lr-pointwise-ffn>','exec'),namespace)
    model._lr_step=types.MethodType(namespace['_lr_step'],model)
    print('BOUNDED_LR_FFN',chunk,flush=True)
