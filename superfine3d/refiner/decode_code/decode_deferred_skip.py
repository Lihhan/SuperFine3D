"""Inference-only C2S residual gather, deferred until the main branch is ready."""
import inspect
import textwrap
import os
import torch


def add_skip(block, h, x, subdivision, chunk=65536):
    assert not torch.is_grad_enabled()
    factor = block.updown.factor
    dim = x.coords.shape[-1] - 1
    children = factor ** dim
    cache = x.get_spatial_cache(f'channel2spatial_{factor}')
    if cache is None:
        assert subdivision is not None
        # nonzero uses the same parent-major, child-minor ordering as C2S.
        pairs = subdivision.feats.nonzero()
        parent, child = pairs[:, 0], pairs[:, 1]
    else:
        coords, parent, child = cache
        assert torch.equal(coords, h.coords)
    assert len(parent) == len(h.feats)
    source = x.feats.reshape(len(x.feats) * children, -1)
    repeat = block.out_channels // (block.channels // children)
    assert repeat * source.shape[1] == h.feats.shape[1]
    for start in range(0, len(parent), chunk):
        stop = min(start + chunk, len(parent))
        index = parent[start:stop] * children + child[start:stop]
        if source.device.type == 'cpu':
            index = index.cpu()
        value = source[index]
        if repeat != 1:
            value = value.repeat_interleave(repeat, dim=1)
        h.feats[start:stop].add_(value.to(h.feats.device))
    if h.feats.device.type == 'cpu' and not (os.environ.get('DECODE_HOST_ALL_STAGES')=='1' or (os.environ.get('DECODE_CPU_FINAL_STAGE')=='1' and h.feats.shape[1]==64)):
        h = h.replace(h.feats.to(h.coords.device))
    return h


def prepare_install():
    """Capture the original source BEFORE the existing dynamic inference patches."""
    import trellis2.models.sc_vaes.sparse_unet_vae as vae
    source = textwrap.dedent(inspect.getsource(vae.SparseResBlockC2S3d._forward))
    final_source = textwrap.dedent(inspect.getsource(vae.SparseUnetVaeDecoder.forward))
    assert source.count('x = self.updown(x, subdiv_binarized)') == 1
    assert source.count('h = h + self.skip_connection(x)') == 1
    source = source.replace('x = self.updown(x, subdiv_binarized)', '# Residual stays in parent coordinates until addition.')
    source = source.replace('h = h + self.skip_connection(x)', 'h = deferred_add_skip(self, h, x, subdiv_binarized)')
    if os.environ.get('DECODE_SUBDIV_TAU') not in (None, '', '0'):   # opt-in: keep child voxels with subdivision logit > -TAU... i.e. > TAU
        assert source.count('subdiv.feats > 0') == 1                  # (default 0 = the trained decoder); a negative TAU keeps borderline
        source = source.replace('subdiv.feats > 0', f"subdiv.feats > {float(os.environ['DECODE_SUBDIV_TAU'])!r}")   # children (fewer cracks)
    if os.environ.get('DECODE_CPU_RESIDUAL') == '1':
        needle = 'h = self.conv1(h)'
        assert source.count(needle) == 1
        source = source.replace(needle, needle+'\n    retire_parent_maps(x)\n    x = x.replace(x.feats.cpu())')
    if os.environ.get('DECODE_INPLACE_C2S_NORM') == '1':
        needle_norm = 'h = h.replace(self.norm2(h.feats))'
        assert source.count(needle_norm) == 1
        source = source.replace(needle_norm, 'h = h.replace(inplace_c2s_norm(self.norm2, h.feats))')
        source = source.replace('F.silu(h.feats)', 'inplace_c2s_silu(h.feats)')
    if os.environ.get('DECODE_CPU_C2S_CONV2') == '1':
        assert source.count('h = self.conv2(h)') == 1
        source = source.replace('h = self.conv2(h)', 'h = cpu_c2s_conv2(self.conv2, h)')
    cpu_updown = None
    if os.environ.get('DECODE_CPU_EXPAND') == '1':
        from trellis2.modules.sparse.spatial import spatial2channel as spatial
        up_source = textwrap.dedent(inspect.getsource(spatial.SparseChannel2Spatial.forward))
        assert up_source.count('new_feats = x_feats[idx * self.factor ** DIM + subidx]') == 1
        up_source = up_source.replace('device=x.device', 'device=x.coords.device')
        up_source = up_source.replace('sub = subdivision.feats', 'sub = subdivision.feats.to(x.coords.device)')
        up_source = up_source.replace('new_feats = x_feats[idx * self.factor ** DIM + subidx]',
            'new_feats = cpu_gather(x_feats, idx, subidx, self.factor ** DIM, x.coords.device)')
        up_ns = dict(vars(spatial)); up_ns['cpu_gather'] = cpu_gather
        exec(compile(up_source, 'cpu_staged_c2s', 'exec'), up_ns)
        cpu_updown = up_ns['forward']
        source = source.replace('h = self.updown(h, subdiv_binarized)',
            'h = h.replace(h.feats.cpu())\n    h = cpu_updown(self.updown, h, subdiv_binarized)')
    def install():
        if os.environ.get('DECODE_TILE_CONV') == '1':
            from decode_tiled_conv import install as install_tiled
            install_tiled()
        ns = dict(vars(vae))
        ns['deferred_add_skip'] = add_skip
        ns['inplace_c2s_norm'] = inplace_c2s_norm
        ns['inplace_c2s_silu'] = inplace_c2s_silu
        ns['cpu_c2s_conv2'] = cpu_c2s_conv2
        ns['cpu_updown'] = cpu_updown
        if os.environ.get('DECODE_CPU_RESIDUAL') == '1':
            ns['retire_parent_maps'] = vae.SparseUnetVaeDecoder.forward.__globals__['retire_conv_maps']
        exec(compile(source, 'deferred_c2s_skip', 'exec'), ns)
        vae.SparseResBlockC2S3d._forward = ns['_forward']
        if os.environ.get('DECODE_RETIRE_MESH_MAPS') == '1':
            import trellis2.models.sc_vaes.fdg_vae as fdg
            original_mesh = fdg.flexible_dual_grid_to_mesh
            mesh_source = textwrap.dedent(inspect.getsource(original_mesh))
            if os.environ.get('DECODE_CHUNK_MESH_QUERY') == '1':
                first = mesh_source.index('    edge_neighbor_voxel =')
                last = mesh_source.index('    M = connected_voxel.shape[0]',first)
                mesh_source = mesh_source[:first]+(
                    '    edge_neighbor_voxel = None\n'
                    '    connected_voxel = chunk_connected(coords, intersected_flag, flexible_dual_grid_to_mesh.edge_neighbor_voxel_offset)\n'
                )+mesh_source[last:]
            needle = '    # Construct triangles'
            assert mesh_source.count(needle) == 1
            mesh_source = mesh_source.replace(needle,
                '    del edge_neighbor_voxel, connected_voxel, connected_voxel_hash_key\n'
                '    del connected_voxel_indices, connected_voxel_valid, hashmap\n'+needle)
            if os.environ.get('DECODE_CPU_MESH_ASSEMBLY') == '1':
                mesh_source = mesh_source.replace('    mesh_vertices = (coords.float() + dual_vertices) / (2 * N) - 0.5', '    mesh_vertices = None')
                needle_tri = '    # Construct triangles'
                mesh_source = mesh_source.replace(needle_tri,
                    '    if not train and split_weight is not None:\n'
                    '        return cpu_mesh_assembly(coords, dual_vertices, quad_indices, split_weight, voxel_size, aabb, flexible_dual_grid_to_mesh.quad_split_1, flexible_dual_grid_to_mesh.quad_split_2)\n'+needle_tri)
            if os.environ.get('DECODE_STREAM_MESH_QUADS') == '1':
                first=mesh_source.index('    edge_neighbor_voxel =')
                last=mesh_source.index('    if not train and split_weight is not None:',first)
                mesh_source=mesh_source[:first]+(
                    '    quad_indices = stream_mesh_quads(coords, intersected_flag, flexible_dual_grid_to_mesh.edge_neighbor_voxel_offset, hashmap, grid_size)\n'
                    '    L = len(quad_indices)\n'
                    '    del hashmap\n'
                )+mesh_source[last:]
            mesh_ns = dict(original_mesh.__globals__)
            mesh_ns['stream_mesh_quads'] = stream_mesh_quads
            mesh_ns['cpu_mesh_assembly'] = cpu_mesh_assembly
            mesh_ns['chunk_connected'] = chunk_connected
            exec(compile(mesh_source, 'retire_mesh_maps', 'exec'), mesh_ns)
            fn = mesh_ns['flexible_dual_grid_to_mesh']
            fn.__dict__.update(original_mesh.__dict__)
            fdg.flexible_dual_grid_to_mesh = fn
        if os.environ.get('DECODE_INPLACE_FINAL') == '1':
            patched = final_source.replace('h = h.type(x.dtype)', 'retire_conv_maps(h)\n    h = h.type(x.dtype)')
            if os.environ.get('DECODE_CPU_RESIDUAL') == '1':
                # Release the half-precision GPU buffer before allocating its
                # float32 counterpart. Conversion is exact for every fp16 value.
                patched = patched.replace('h = h.type(x.dtype)',
                    'h = h.replace(h.feats.cpu())\n    h = h.replace(h.feats.to(x.dtype).to(x.coords.device))')
            needle = 'F.layer_norm(h.feats, h.feats.shape[-1:])'
            assert patched.count(needle) == 1
            patched = patched.replace(needle, 'inplace_final_norm(h.feats, h.feats.shape[-1:])')
            if os.environ.get('DECODE_CPU_FINAL_STAGE')=='1':
                patched=patched.replace('h = h.replace(h.feats.to(x.dtype).to(x.coords.device))','h = h.replace(cpu_final_norm(h.feats, x.dtype, x.coords.device))')
                patched=patched.replace('inplace_final_norm(h.feats, h.feats.shape[-1:])','h.feats')
                patched=patched.replace('h = self.output_layer(h)','h = cpu_final_linear(self.output_layer, h)')
            final_ns = dict(vae.SparseUnetVaeDecoder.forward.__globals__)
            final_ns['cpu_final_norm']=cpu_final_norm
            final_ns['cpu_final_linear']=cpu_final_linear
            final_ns['inplace_final_norm'] = inplace_final_norm
            exec(compile(patched, 'inplace_final_norm_decoder', 'exec'), final_ns)
            vae.SparseUnetVaeDecoder.forward = final_ns['forward']
        print('DEFERRED_C2S_SKIP_INSTALLED', flush=True)
    return install


def inplace_final_norm(x, shape):
    assert not torch.is_grad_enabled() and x.ndim == 2 and len(shape) == 1
    for start in range(0, len(x), 65536):
        view = x[start:start+65536]
        view.copy_(torch.nn.functional.layer_norm(view, shape))
    return x


def cpu_gather(feats, idx, subidx, children, device, chunk=65536):
    assert feats.device.type == 'cpu'
    parent = idx.cpu(); child = subidx.cpu()
    host_final=os.environ.get('DECODE_HOST_ALL_STAGES')=='1' or (os.environ.get('DECODE_CPU_FINAL_STAGE')=='1' and os.environ.get('DECODE_CPU_FINAL_GATHER','1')=='1' and feats.shape[1]==64)
    result = torch.empty((len(idx), feats.shape[1]), device='cpu' if host_final else device, dtype=feats.dtype)
    for start in range(0, len(idx), chunk):
        stop = min(start+chunk, len(idx))
        value = feats[parent[start:stop]*children+child[start:stop]]
        result[start:stop].copy_(value)
    return result


def chunk_connected(coords, flags, offset, chunk=262144):
    count = int(flags.sum().item())
    output = torch.empty((count,4,3),device=coords.device,dtype=coords.dtype)
    cursor = 0
    for start in range(0,len(coords),chunk):
        stop=min(start+chunk,len(coords))
        value=(coords[start:stop].reshape(-1,1,1,3)+offset)[flags[start:stop]]
        output[cursor:cursor+len(value)].copy_(value)
        cursor+=len(value)
    assert cursor==count
    return output


def inplace_c2s_norm(norm, x):
    assert not torch.is_grad_enabled() and norm.weight is None and norm.bias is None
    for start in range(0,len(x),65536):
        view=x[start:start+65536]
        value=view.float().to('cuda') if view.device.type=='cpu' else view.float()
        view.copy_(torch.nn.functional.layer_norm(value,norm.normalized_shape,None,None,norm.eps).to(x.dtype))
    return x


def cpu_c2s_conv2(conv, h):
    import decode_tiled_conv as tiled
    assert not torch.is_grad_enabled()
    previous=tiled._force_host_output
    tiled._force_host_output=True
    try:
        return conv(h)
    finally:
        tiled._force_host_output=previous


def cpu_mesh_assembly(coords, dual, quads, weights, voxel_size, aabb, split1, split2):
    assert not torch.is_grad_enabled()
    chunk=262144
    vertices=torch.empty((len(coords),3),device='cpu',dtype=dual.dtype)
    for start in range(0,len(coords),chunk):
        stop=min(start+chunk,len(coords))
        vertices[start:stop].copy_((coords[start:stop].float()+dual[start:stop])*voxel_size+aabb[0].reshape(1,3))
    triangles=torch.empty((len(quads)*2,3),device='cpu',dtype=split1.dtype)
    for start in range(0,len(quads),chunk):
        stop=min(start+chunk,len(quads));q=quads[start:stop].to(weights.device)
        w=weights[q]
        choose=w[:,0]*w[:,2]>w[:,1]*w[:,3]
        value=torch.where(choose,q[:,split1],q[:,split2]).reshape(-1,3)
        triangles[2*start:2*stop].copy_(value)
        if start%(chunk*64)==0: print('CPU_MESH_ASSEMBLY',start,len(quads),flush=True)
    return vertices,triangles


def stream_mesh_quads(coords, flags, offset, hashmap, grid_size):
    from o_voxel import _C
    assert not torch.is_grad_enabled()
    chunks=[]
    for start in range(0,len(coords),262144):
        stop=min(start+262144,len(coords))
        connected=(coords[start:stop].reshape(-1,1,1,3)+offset)[flags[start:stop]]
        if not len(connected): continue
        flat=connected.reshape(-1,3)
        key=torch.cat([torch.zeros((len(flat),1),device=coords.device,dtype=torch.int32),flat],1)
        idx=_C.hashmap_lookup_3d_cuda(*hashmap,key,*grid_size.tolist()).reshape(-1,4).int()
        valid=(idx!=0xffffffff).all(1)
        chunks.append(idx[valid].cpu())
        if start%(262144*64)==0:print('STREAM_MESH_QUADS',start,len(coords),flush=True)
    return torch.cat(chunks,0) if chunks else torch.empty((0,4),dtype=torch.int32)


def cpu_final_norm(feats,dtype,device):
    assert not torch.is_grad_enabled() and feats.device.type=='cpu'
    out=torch.empty(feats.shape,dtype=dtype,device='cpu')
    for start in range(0,len(feats),65536):
        stop=min(start+65536,len(feats))
        x=feats[start:stop].to(dtype).to(device)
        out[start:stop].copy_(torch.nn.functional.layer_norm(x,x.shape[-1:]))
    return out

def cpu_final_linear(linear, h):
    assert not torch.is_grad_enabled() and h.feats.device.type=='cpu'
    original=linear.forward
    def forward(x):
        out=torch.empty((len(x.feats),linear.out_features),device=linear.weight.device,dtype=linear.weight.dtype)
        for start in range(0,len(x.feats),65536):
            stop=min(start+65536,len(x.feats))
            out[start:stop].copy_(torch.nn.functional.linear(x.feats[start:stop].to(linear.weight.device),linear.weight,linear.bias))
        return x.replace(out)
    linear.forward=forward
    try: return linear(h)
    finally: linear.forward=original


def inplace_c2s_silu(x):
    assert not torch.is_grad_enabled()
    if x.device.type!='cpu':
        return torch.nn.functional.silu(x,inplace=True)
    for start in range(0,len(x),65536):
        view=x[start:start+65536]
        value=torch.nn.functional.silu(view.to('cuda'))
        view.copy_(value)
    return x
