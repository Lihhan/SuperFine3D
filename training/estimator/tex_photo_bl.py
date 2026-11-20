"""Blender (4.2) photo-style render of a textured scan, used as an in-the-wild-looking image input with exact GT.
Original GLB materials (same material fixes as tex_render_bl.py), standing on a floor in front of a back wall
(procedural stone floor / plaster wall), HDRI ambient light + one large soft key light, perspective camera (fov ~38 deg)
with mild depth of field. No alpha: the image is used like a photograph (background removed by the standard BiRefNet
preprocessing before generation).
Camera convention = eval_gt.py / metrics.camera: in the dump frame (scan normalised by its vertex bbox, centre c,
k = 0.99999 / max extent) the camera sits at radius*dir(yaw, pitch), looks at the origin, roll 0, square image,
horizontal = vertical fov. The job's camera is written back to OUT/source_camera.json with that radius.
usage: blender -b -P tex_photo_bl.py -- JOB.json
JOB = {glb, out, yaw, pitch, fov, fill, hdri, hdri_rot, key_az, key_el, res, samples, seed}"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,os,json,math
import numpy as np,bpy
from mathutils import Matrix,Vector
T=f'{_E["SF3D_ROOT"]}/training/trellis2_variants/data_toolkit/blender_script/render_cond.py'
src=open(T).read().split("if __name__ == '__main__':")[0];ns={'__name__':'rc','__file__':T};exec(compile(src,T,'exec'),ns)
job=json.load(open(sys.argv[sys.argv.index('--')+1]));np.random.seed(job.get('seed',0));OUT=job['out'];os.makedirs(OUT,exist_ok=True)
ns['init_scene']();ns['load_object'](job['glb']);ns['normalize_scene']()
B=open(f'{_E["SF3D_ROOT"]}/training/estimator/tex_render_bl.py').read()
exec(B[B.index('def fix_materials'):B.index('fix_materials()\n')]);fix_materials()
ns['init_render'](engine='CYCLES',resolution=int(job.get('res',4096)))
S=bpy.context.scene;S.cycles.samples=int(job.get('samples',256));S.render.film_transparent=False
S.cycles.use_denoising=False                                            # a denoiser would smooth exactly the fine detail we test
try:S.view_settings.view_transform='AgX';S.view_settings.look=job.get('look','AgX - Medium High Contrast');S.view_settings.exposure=job.get('exposure',0.)
except Exception as e:print('[PHOTO] view transform',e)
for o in [o for o in S.objects if o.type=='LIGHT']:bpy.data.objects.remove(o,do_unlink=True)
# ---- dump frame (identical to tex_render_bl.py)
dg=bpy.context.evaluated_depsgraph_get();lo=np.full(3,np.inf);hi=np.full(3,-np.inf);objs=[]
for ob in S.objects:
    if ob.type!='MESH':continue
    me=ob.evaluated_get(dg).to_mesh()
    if len(me.vertices)==0 or len(me.polygons)==0:ob.evaluated_get(dg).to_mesh_clear();continue
    co=np.empty(len(me.vertices)*3,np.float32);me.vertices.foreach_get('co',co);co=co.reshape(-1,3)
    M=np.array(ob.matrix_world);w=co@M[:3,:3].T+M[:3,3];lo=np.minimum(lo,w.min(0));hi=np.maximum(hi,w.max(0));ob.evaluated_get(dg).to_mesh_clear();objs.append(ob)
c=(lo+hi)/2;k=0.99999/(hi-lo).max();D=lambda p:(np.asarray(p)-c)*k;Wd=lambda p:np.asarray(p)/k+c   # world <-> dump
# ---- camera in the dump frame; radius chosen so the bbox fills `fill` of the frame
yaw,pitch,fov=job['yaw'],job['pitch'],job.get('fov',38.);a,b=map(math.radians,(yaw,pitch))
z=np.array([math.cos(b)*math.cos(a),math.cos(b)*math.sin(a),math.sin(b)]);x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x);y=np.cross(z,x)
corners=np.array([[i,j,l] for i in (lo[0],hi[0]) for j in (lo[1],hi[1]) for l in (lo[2],hi[2])]);cd=D(corners);t=math.tan(math.radians(fov)/2)
def extent(R):
    e=R*z;q=cd-e;dep=-(q@z);return max(np.abs(q@x/dep).max(),np.abs(q@y/dep).max())/t
R=3.0
for _ in range(60):R*=extent(R)/job.get('fill',.6)
eye=R*z
cam=bpy.data.objects.new('PhotoCam',bpy.data.cameras.new('PhotoCam'));bpy.context.collection.objects.link(cam);S.camera=cam
cam.data.sensor_fit='HORIZONTAL';cam.data.sensor_width=cam.data.sensor_height=32;cam.data.lens=16/t;cam.data.clip_start=1e-3;cam.data.clip_end=1e4
M=np.eye(4);M[:3,0]=x;M[:3,1]=y;M[:3,2]=z;M[:3,3]=Wd(eye);cam.matrix_world=Matrix(M.tolist())
cam.data.dof.use_dof=True;cam.data.dof.focus_distance=float(R/k);cam.data.dof.aperture_fstop=float(job.get('fstop',8.))
# ---- set: floor under the scan, back wall behind it (relative to the camera's horizontal direction)
def mat(name,base,var,scale,rough):
    m=bpy.data.materials.new(name);m.use_nodes=True;N=m.node_tree.nodes;L=m.node_tree.links;bs=N['Principled BSDF']
    tc=N.new('ShaderNodeTexCoord');nz=N.new('ShaderNodeTexNoise');nz.inputs['Scale'].default_value=scale;nz.inputs['Detail'].default_value=8
    L.new(tc.outputs['Object'],nz.inputs['Vector'])
    cr=N.new('ShaderNodeValToRGB');cr.color_ramp.elements[0].color=(*[max(0,v-var) for v in base],1);cr.color_ramp.elements[1].color=(*[min(1,v+var) for v in base],1)
    L.new(nz.outputs['Fac'],cr.inputs['Fac']);L.new(cr.outputs['Color'],bs.inputs['Base Color']);bs.inputs['Roughness'].default_value=rough;return m
span=float((hi-lo).max())
bpy.ops.mesh.primitive_plane_add(size=job.get('floor_size',400)*span,location=(c[0],c[1],lo[2]-1e-4*span));fl=bpy.context.object;fl.name='Floor'
fl.data.materials.append(mat('FloorMat',job.get('floor_rgb',(.36,.34,.31)),job.get('floor_var',.05),job.get('floor_scale',3)/span,job.get('floor_rough',.6)))
h=np.array([z[0],z[1],0.]);h/=np.linalg.norm(h)                         # horizontal direction towards the camera
wall_d=job.get('wall_dist',1.6)*span
if job.get('wall',True):
    bpy.ops.mesh.primitive_plane_add(size=60*span,location=tuple(np.array([c[0],c[1],lo[2]])-h*wall_d));wl=bpy.context.object;wl.name='Wall'
    wl.rotation_mode='QUATERNION';wl.rotation_quaternion=Vector(h.tolist()).to_track_quat('Z','Y')
    wl.data.materials.append(mat('WallMat',job.get('wall_rgb',(.58,.55,.5)),job.get('wall_var',.03),1.5/span,.9))
# ---- light: HDRI ambient (also seen in reflections) + one large soft key light from the camera side
w=S.world;w.use_nodes=True;N=w.node_tree.nodes;L=w.node_tree.links
for n in list(N):
    if n.type!='OUTPUT_WORLD':N.remove(n)
env=N.new('ShaderNodeTexEnvironment');env.image=bpy.data.images.load(job.get('hdri',f'{_E["SF3D_TRELLIS2_REPO"]}/assets/hdri/interior.exr'))
mp=N.new('ShaderNodeMapping');tc=N.new('ShaderNodeTexCoord');mp.inputs['Rotation'].default_value[2]=math.radians(job.get('hdri_rot',0))
bg=N.new('ShaderNodeBackground');bg.inputs['Strength'].default_value=job.get('hdri_strength',.8)
L.new(tc.outputs['Generated'],mp.inputs['Vector']);L.new(mp.outputs['Vector'],env.inputs['Vector']);L.new(env.outputs['Color'],bg.inputs['Color'])
L.new(bg.outputs['Background'],[n for n in N if n.type=='OUTPUT_WORLD'][0].inputs['Surface'])
ka,ke=math.radians(yaw+job.get('key_az',40)),math.radians(job.get('key_el',45))
kd=np.array([math.cos(ke)*math.cos(ka),math.cos(ke)*math.sin(ka),math.sin(ke)]);kpos=c+kd*3*span
if job.get('key_type','AREA')=='SUN':
    key=bpy.data.objects.new('Key',bpy.data.lights.new('Key',type='SUN'));key.data.energy=float(job.get('key_energy',4));key.data.angle=math.radians(job.get('sun_angle',1.5))
else:
    key=bpy.data.objects.new('Key',bpy.data.lights.new('Key',type='AREA'));key.data.shape='DISK';key.data.size=job.get('key_size',1.5)*span;key.data.energy=float(job.get('key_energy',600))*span**2
bpy.context.collection.objects.link(key)
key.location=tuple(kpos);key.rotation_mode='QUATERNION';key.rotation_quaternion=Vector((kpos-c).tolist()).to_track_quat('Z','Y')
S.render.filepath=f'{OUT}/photo.png';S.render.image_settings.color_mode='RGB';bpy.ops.render.render(write_still=True)
sc=dict(uid=os.path.basename(job['glb'])[:-4],gt=job.get('gt'),yaw=yaw,pitch=pitch,fov=fov,radius=float(R),pixels=int(job.get('res',4096)),
        source='photo-style textured GT render (ft_nirne/tex_photo_bl.py): floor + wall set, HDRI %s + soft key light, perspective, f/%s DOF'%(os.path.basename(job.get('hdri','interior.exr')),job.get('fstop',8.)),job=job)
json.dump(sc,open(f'{OUT}/source_camera.json','w'),indent=1);print('[PHOTO] done radius %.3f'%R,flush=True)
