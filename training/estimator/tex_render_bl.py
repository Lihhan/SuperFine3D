"""Blender (4.2) side of the textured crop-view renderer. Renders TEXTURED (original GLB materials) crop-filling views
with the SAME cameras as the grey views (hires/<split>/<cid>/cam.json, dump frame), so the existing normal targets /
oracles stay pixel-aligned.
Frame: pairs.load_dump normalises the dumped world-space vertices by their bbox (centre c, k = 0.99999/max extent);
that only depends on the vertex bbox, so after importing + normalising here we recompute c,k from the evaluated mesh
vertices and map the dump-frame camera back: X_world = X_dump / k + c.
Per view two lightings:  l0 = TRELLIS render_cond.init_random_lighting (same distribution as the 4K test renders),
                         l1 = one random grazing sun + weak ambient (shading cues on geometry clearly visible).
usage: blender -b -P tex_render_bl.py -- JOB.json     JOB = {glb, views:[{cid, cam, out}], seed}"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,os,json,math
import numpy as np,bpy
from mathutils import Matrix,Vector
T=f'{_E["SF3D_ROOT"]}/training/trellis2_variants/data_toolkit/blender_script/render_cond.py'
src=open(T).read().split("if __name__ == '__main__':")[0];ns={'__name__':'rc','__file__':T};exec(compile(src,T,'exec'),ns)
job=json.load(open(sys.argv[sys.argv.index('--')+1]));np.random.seed(job.get('seed',0))
ns['init_scene']();ns['load_object'](job['glb']);ns['normalize_scene']()
def fix_materials():
    # 1) unlit / emission materials (many photo scans: EMISSION mixed with TRANSPARENT) -> Principled BSDF with the same
    #    image as base colour, so scene lights actually shade the geometry (as in a real photo);
    # 2) cut normal / bump map links: image shading must come from the mesh geometry the targets are rendered from.
    nfix=nnm=0
    for m in bpy.data.materials:
        if not m.use_nodes:continue
        nt=m.node_tree;N=nt.nodes;Lk=nt.links
        for n in list(N):
            if n.type=='BSDF_PRINCIPLED' and n.inputs['Normal'].links:
                for l in list(n.inputs['Normal'].links):Lk.remove(l);nnm+=1
        out=[n for n in N if n.type=='OUTPUT_MATERIAL']
        if not out or not out[0].inputs['Surface'].links:continue
        if any(n.type=='EMISSION' for n in N) and not any(n.type=='BSDF_PRINCIPLED' for n in N):
            tex=[n for n in N if n.type=='TEX_IMAGE'];em=[n for n in N if n.type=='EMISSION'][0]
            b=N.new('ShaderNodeBsdfPrincipled');b.inputs['Roughness'].default_value=0.8
            if tex:Lk.new(tex[0].outputs['Color'],b.inputs['Base Color'])
            elif em.inputs['Color'].links:Lk.new(em.inputs['Color'].links[0].from_socket,b.inputs['Base Color'])
            else:b.inputs['Base Color'].default_value=em.inputs['Color'].default_value
            if tex and 'Alpha' in tex[0].outputs:Lk.new(tex[0].outputs['Alpha'],b.inputs['Alpha'])
            for l in list(out[0].inputs['Surface'].links):Lk.remove(l)
            Lk.new(b.outputs['BSDF'],out[0].inputs['Surface']);nfix+=1
    print('[TEX] materials relit',nfix,'normal-map links cut',nnm,flush=True)
fix_materials()
ns['init_render'](engine='CYCLES',resolution=int(job.get('res',1024)))
bpy.context.scene.cycles.samples=int(job.get('samples',48))
# dump-frame normalisation from the evaluated, world-space mesh vertices (same as dump_mesh.py + pairs.load_dump)
dg=bpy.context.evaluated_depsgraph_get();lo=np.full(3,np.inf);hi=np.full(3,-np.inf)
for ob in bpy.context.scene.objects:
    if ob.type!='MESH':continue
    me=ob.evaluated_get(dg).to_mesh()
    if len(me.vertices)==0 or len(me.polygons)==0:ob.evaluated_get(dg).to_mesh_clear();continue
    co=np.empty(len(me.vertices)*3,np.float32);me.vertices.foreach_get('co',co);co=co.reshape(-1,3)
    M=np.array(ob.matrix_world);w=co@M[:3,:3].T+M[:3,3];lo=np.minimum(lo,w.min(0));hi=np.maximum(hi,w.max(0));ob.evaluated_get(dg).to_mesh_clear()
c=(lo+hi)/2;k=0.99999/(hi-lo).max()
cam=bpy.data.objects.new('CropCam',bpy.data.cameras.new('CropCam'));bpy.context.collection.objects.link(cam);bpy.context.scene.camera=cam
cam.data.sensor_fit='HORIZONTAL';cam.data.sensor_width=cam.data.sensor_height=32;cam.data.clip_start=1e-3;cam.data.clip_end=100
def light_grazing(fwd):
    for o in [o for o in bpy.context.scene.objects if o.type=='LIGHT']:bpy.data.objects.remove(o,do_unlink=True)
    w=bpy.context.scene.world;bg=[n for n in w.node_tree.nodes if n.type=='BACKGROUND'][0]
    bg.inputs['Color'].default_value=(1,1,1,1);bg.inputs['Strength'].default_value=float(np.random.uniform(.05,.3))
    V=-fwd
    while True:                                                       # sun from the camera hemisphere, 35-80 deg off the view axis
        L=np.random.randn(3);L/=np.linalg.norm(L);a=math.degrees(math.acos(np.clip(L@V,-1,1)))
        if 35<a<80:break
    sun=bpy.data.objects.new('Sun',bpy.data.lights.new('Sun',type='SUN'));bpy.context.collection.objects.link(sun)
    sun.data.energy=float(np.random.uniform(2.,5.));sun.data.angle=math.radians(np.random.uniform(1,8))
    z=Vector(L.tolist());sun.rotation_mode='QUATERNION';sun.rotation_quaternion=z.to_track_quat('Z','Y')
for v in job['views']:
    P=v['cam'];eye=np.array(P['eye'])/k+c;r=np.array(P['r']);u=np.array(P['u']);fwd=np.array(P['fwd'])
    M=np.eye(4);M[:3,0]=r;M[:3,1]=u;M[:3,2]=-fwd;M[:3,3]=eye;cam.matrix_world=Matrix(M.tolist())
    half=math.atan(0.5/P['f']);cam.data.lens=16/math.tan(half)
    os.makedirs(v['out'],exist_ok=True)
    for li,fn in enumerate((lambda: ns['init_random_lighting'](-fwd),lambda: light_grazing(fwd))):
        o=f"{v['out']}/view_l{li}.png"
        if os.path.exists(o):continue
        fn();bpy.context.scene.render.filepath=o;bpy.ops.render.render(write_still=True)
    print('[TEX] done',v['cid'],flush=True)
print('[TEX] ALL_DONE',flush=True)
