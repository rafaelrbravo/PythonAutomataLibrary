import atexit
import math
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Protocol
from .NativeCore import _ReserveWindowPosition,_ReleaseWindow,_RegisterWindowInit,_RegisterCrashWindow,_UnregisterCrashWindow,_UseFastMode




# Maximum number of primitives in one rendered frame.
_FRAME_CAPACITY=262144

# JIT-compatible primitive collector. Rendering remains Python/OpenGL-side.
try:
    import numpy as np
    from numba import int64, float64, uint32
    from numba.experimental import jitclass

    _DRAW_SPEC=[
        ('kind',int64[:]),
        ('data',float64[:,::1]),
        ('color',uint32[:]),
        ('count',int64),
    ]

    @jitclass(_DRAW_SPEC)
    class OpenGLDraw:
        """Fixed-capacity Numba-native primitive collector."""
        def __init__(self):
            self.kind=np.empty(_FRAME_CAPACITY,dtype=np.int64)
            self.data=np.empty((_FRAME_CAPACITY,7),dtype=np.float64)
            self.color=np.empty(_FRAME_CAPACITY,dtype=np.uint32)
            self.count=0

        def Clear(self):
            self.count=0

        def _Next(self):
            i=self.count
            if i>=_FRAME_CAPACITY: raise RuntimeError("OpenGL frame primitive capacity exceeded")
            self.count=i+1
            return i

        def Circle(self,rad,color,x,y,z=0.0):
            i=self._Next()
            self.kind[i]=1;self.color[i]=color
            self.data[i,0]=rad;self.data[i,1]=x;self.data[i,2]=y;self.data[i,3]=z

        def Box(self,xLen,color,x,y,z=0.0,yLen=-1.0,zLen=-1.0):
            if yLen<0: yLen=xLen
            if zLen<0: zLen=xLen
            i=self._Next()
            # A 3D Box with exactly one zero length is an axis-aligned plane.
            # Keep ordinary boxes on kind 2 so their existing render path is unchanged.
            if zLen==0.0 and xLen!=0.0 and yLen!=0.0: kind=4
            elif yLen==0.0 and xLen!=0.0 and zLen!=0.0: kind=5
            elif xLen==0.0 and yLen!=0.0 and zLen!=0.0: kind=6
            else: kind=2
            self.kind[i]=kind;self.color[i]=color
            self.data[i,0]=xLen;self.data[i,1]=yLen;self.data[i,2]=zLen
            self.data[i,3]=x;self.data[i,4]=y;self.data[i,5]=z

        def BoxSQ(self,color,x,y,z=0.0):
            self.Box(1.0,color,x+0.5,y+0.5,z+0.5,1.0,1.0)

        def Line(self,width,color,x1,y1,x2,y2,z1=0.0,z2=0.0):
            i=self._Next()
            self.kind[i]=3;self.color[i]=color
            self.data[i,0]=width
            self.data[i,1]=x1;self.data[i,2]=y1;self.data[i,3]=z1
            self.data[i,4]=x2;self.data[i,5]=y2;self.data[i,6]=z2

except ImportError:
    OpenGLDraw=None


def _DrawColor(color):
    color=int(color)
    return ((color>>16)&255,(color>>8)&255,color&255,255)





def _NewFrameShared():
    # One shared latest-frame slot. Frames are overwritten rather than queued,
    # so a slow renderer can skip stale frames without blocking the simulation.
    return (
        mp.RawArray('q',_FRAME_CAPACITY),
        mp.RawArray('d',_FRAME_CAPACITY*7),
        mp.RawArray('I',_FRAME_CAPACITY),
        mp.Value('q',0,lock=False),
        mp.Value('q',0,lock=False),
        mp.Lock(),
    )

def _FlushDraw(win):
    draw=win._draw
    if draw.count==0: return
    n=draw.count
    if n>_FRAME_CAPACITY:
        raise RuntimeError(f"OpenGL frame has {n} primitives; shared frame capacity is {_FRAME_CAPACITY}")
    kindShared,dataShared,colorShared,countShared,seqShared,frameLock=win._frameShared
    # Latest-frame-wins publication. Never queue render frames, but never keep
    # an older pending frame merely because the renderer is snapshotting it.
    # The renderer holds this lock only for its shared-memory copy; GPU work is
    # performed after releasing it.
    with frameLock:
        np.frombuffer(kindShared,dtype=np.int64,count=n)[:]=draw.kind[:n]
        np.frombuffer(dataShared,dtype=np.float64,count=n*7).reshape(n,7)[:,:]=draw.data[:n,:]
        np.frombuffer(colorShared,dtype=np.uint32,count=n)[:]=draw.color[:n]
        countShared.value=n
        seqShared.value+=1
    draw.Clear()


def _OutputFrameSnapshot(win):
    # Explicit output is exact, unlike display rendering. Publish any draw work,
    # then take an immutable snapshot so later Update() calls cannot change what
    # this Save()/AddGifFrame() request captures.
    _FlushDraw(win)
    kindShared,dataShared,colorShared,countShared,seqShared,frameLock=win._frameShared
    with frameLock:
        n=countShared.value
        kinds=np.frombuffer(kindShared,dtype=np.int64,count=n).copy()
        data=np.frombuffer(dataShared,dtype=np.float64,count=n*7).reshape(n,7).copy()
        colors=np.frombuffer(colorShared,dtype=np.uint32,count=n).copy()
    return kinds,data,colors


class OpenGLWindow(Protocol):
    def Update(self): ...
    def IsOpen(self) -> bool: ...
    def _WaitAck(self,op,token,timeout):
        end=time.time()+timeout
        while time.time()<end:
            try:
                msg=self._ackQ.get(timeout=min(.1,max(0,end-time.time())))
                if msg[0]==op and msg[1]==token: return self
            except queue.Empty:
                if not self.IsOpen(): raise RuntimeError(f"OpenGLWindow closed while waiting for {op}")
        raise TimeoutError(f"timed out waiting for OpenGLWindow {op}")

    def StartGif(self,path,delay=100):
        if not self.IsOpen(): raise Exception("cannot start GIF on a closed OpenGLWindow")
        if self._gifActive: raise Exception("an OpenGLWindow GIF is already active")
        if not isinstance(path,(str,Path)): raise Exception(f"GIF path must be a string or Path path:{path}")
        _Finite('delay',delay)
        if delay<=0: raise Exception(f"GIF delay must be positive delay:{delay}")
        self._token+=1;token=self._token
        self._cmdQ.put(('gif_start',str(Path(path)),int(delay),token))
        self._WaitAck('gif_start',token,30)
        self._gifActive=True
        return self

    def AddGifFrame(self,block=False,timeout=30):
        if not self.IsOpen(): raise Exception("cannot add GIF frame from a closed OpenGLWindow")
        if not self._gifActive: raise Exception("StartGif must be called before AddGifFrame")
        if not isinstance(block,bool): raise Exception(f"block must be bool block:{block}")
        _Finite('timeout',timeout)
        if timeout<=0: raise Exception(f"timeout must be positive timeout:{timeout}")
        kinds,data,colors=_OutputFrameSnapshot(self)
        self._token+=1;token=self._token
        self._cmdQ.put(('gif_frame',token,kinds,data,colors))
        if block: return self._WaitAck('gif_frame',token,timeout)
        return self

    def StopGif(self,timeout=30):
        if not self._gifActive: raise Exception("no OpenGLWindow GIF is active")
        _Finite('timeout',timeout)
        if timeout<=0: raise Exception(f"timeout must be positive timeout:{timeout}")
        self._token+=1;token=self._token
        self._cmdQ.put(('gif_stop',token))
        self._WaitAck('gif_stop',token,timeout)
        self._gifActive=False
        return self

    def Circle(self,rad,color,x,y,z=None): ...
    def Box(self,xLen,color,x,y,z=None,yLen=None,zLen=None): ...
    def BoxSQ(self,color,x,y,z=None): ...
    def Line(self,width,color,x1,y1,x2,y2,z1=None,z2=None): ...
    def Borders(self,width,color): ...
    def Camera(self,x,y,z,yaw=None,pitch=None): ...
    def Clear(self): ...
    def Background(self,color): ...
    def Save(self,path:str,block=False): ...
    def StartGif(self,path:str,delay=100): ...
    def AddGifFrame(self,block=False,timeout=30): ...
    def StopGif(self,timeout=30): ...
    def Close(self): ...

def _Color(color):
    if isinstance(color, int):
        if color < 0 or color > 0xFFFFFF: raise ValueError(f"integer color must be between 0x000000 and 0xFFFFFF color:{color}")
        return ((color >> 16) & 255, (color >> 8) & 255, color & 255, 255)
    if len(color) not in (3,4): raise ValueError("color must be an RGB/RGBA sequence or packed RGB integer")
    vals=tuple(color)+(255,) if len(color)==3 else tuple(color)
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or v>255 for v in vals): raise ValueError(f"color channels must be finite values from 0 to 255 color:{color}")
    return tuple(int(v) for v in vals)


def _Finite(name,*vals):
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in vals): raise ValueError(f"{name} must be finite")


def _MatMul(a,b):
    return [[sum(a[r][k]*b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def _Perspective(fov,aspect,near,far):
    f=1.0/math.tan(math.radians(fov)/2.0)
    return [[f/aspect,0,0,0],[0,f,0,0],[0,0,(far+near)/(near-far),(2*far*near)/(near-far)],[0,0,-1,0]]


def _LookAt(eye,target,up):
    def norm(v):
        n=math.sqrt(sum(x*x for x in v)); return [x/n for x in v]
    def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    f=norm([target[i]-eye[i] for i in range(3)]); s=norm(cross(f,up)); u=cross(s,f)
    return [[s[0],s[1],s[2],-sum(s[i]*eye[i] for i in range(3))],
            [u[0],u[1],u[2],-sum(u[i]*eye[i] for i in range(3))],
            [-f[0],-f[1],-f[2],sum(f[i]*eye[i] for i in range(3))],[0,0,0,1]]


def _FlattenColumnMajor(m):
    return tuple(m[r][c] for c in range(4) for r in range(4))


def _WindowProcess(cmdQ,ackQ,frameShared,xDim,yDim,zDim,width,height,title,screenX,screenY,readyEvent,visualClosed,headless=False):
    # Imports stay in the child process. Importing PAL does not require a graphics stack.
    import moderngl
    import numpy as np

    is3D=zDim is not None
    renderWidth=int(width)
    renderHeight=int(height)
    window=None
    target=None
    if headless:
        # Standalone ModernGL context: no Pyglet window or display server.
        try:
            ctx=moderngl.create_standalone_context(require=330)
        except Exception as firstError:
            # EGL is the common display-free backend on Linux/HPC systems.
            try: ctx=moderngl.create_standalone_context(require=330,backend='egl')
            except Exception as eglError:
                raise RuntimeError(f"could not create a headless OpenGL context; install/configure a standalone OpenGL backend such as EGL or OSMesa. default:{firstError} egl:{eglError}") from eglError
        colorBuffer=ctx.texture((renderWidth,renderHeight),4)
        depthBuffer=ctx.depth_renderbuffer((renderWidth,renderHeight)) if is3D else None
        target=ctx.framebuffer(color_attachments=[colorBuffer],depth_attachment=depthBuffer)
        target.use()
    else:
        import pyglet
        config=pyglet.gl.Config(double_buffer=True,depth_size=24,major_version=3,minor_version=3)
        window=pyglet.window.Window(width=renderWidth,height=renderHeight,caption=title,resizable=True,config=config)
        window.set_location(screenX,screenY)
        ctx=moderngl.create_context(require=330)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func=(moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
    if is3D: ctx.enable(moderngl.DEPTH_TEST)

    prog=ctx.program(
        vertex_shader='''#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec4 in_color;
out vec4 color;
void main(){gl_Position=mvp*vec4(in_pos,1.0);color=in_color;}''',
        fragment_shader='''#version 330
in vec4 color;
out vec4 f_color;
void main(){f_color=color;}''')

    circleProg=ctx.program(
        vertex_shader='''#version 330
uniform mat4 mvp;
uniform vec3 camera_pos;
uniform int billboard_3d;
in vec2 unit_pos;
in vec4 instance_pos_rad;
in vec4 instance_color;
out vec4 color;
void main(){
    vec3 center=instance_pos_rad.xyz;
    float rad=instance_pos_rad.w;
    vec3 p;
    if(billboard_3d==0){
        p=vec3(center.xy + unit_pos*rad,center.z);
    }else{
        vec3 f=normalize(center-camera_pos);
        vec3 right=cross(f,vec3(0.0,0.0,1.0));
        if(dot(right,right)<1e-12) right=vec3(1.0,0.0,0.0);
        else right=normalize(right);
        vec3 up=cross(right,f);
        p=center+rad*(unit_pos.x*right+unit_pos.y*up);
    }
    gl_Position=mvp*vec4(p,1.0);
    color=instance_color;
}''',
        fragment_shader='''#version 330
in vec4 color;
out vec4 f_color;
void main(){f_color=color;}''')

    boxProg=ctx.program(
        vertex_shader='''#version 330
uniform mat4 mvp;
in vec3 unit_pos;
in vec3 instance_pos;
in vec3 instance_size;
in vec4 instance_color;
out vec4 color;
void main(){
    vec3 p=instance_pos+unit_pos*instance_size;
    gl_Position=mvp*vec4(p,1.0);
    color=instance_color;
}''',
        fragment_shader='''#version 330
in vec4 color;
out vec4 f_color;
void main(){f_color=color;}''')

    objects=[]
    persistentObjects=[]
    bgf=(0.0,0.0,0.0,1.0)
    kindShared,dataShared,colorShared,countShared,seqShared,frameLock=frameShared
    seenFrame=0
    gifPath=None
    gifDelay=100
    gifFrames=[]

    # Frame primitives are batched. Circles and boxes use GPU instancing; lines
    # are packed into one vertex buffer per distinct width. This avoids creating
    # thousands of VBO/VAO objects for dense lattice/PDE visualizations.
    circleSegments=32
    circleMesh=np.empty((circleSegments+2,2),dtype=np.float32)
    circleMesh[0]=(0.0,0.0)
    angles=np.arange(circleSegments+1,dtype=np.float32)*(2*np.pi/circleSegments)
    circleMesh[1:,0]=np.cos(angles)
    circleMesh[1:,1]=np.sin(angles)
    circleMeshVbo=ctx.buffer(circleMesh.tobytes())
    circleInstanceVbo=None
    circleVao=None
    circleInstanceCapacity=0
    circleInstances=0

    def rebuild_circle_instances(kinds,data,colors):
        nonlocal circleInstanceVbo,circleVao,circleInstanceCapacity,circleInstances
        inds=np.flatnonzero(kinds==1)
        n=len(inds)
        circleInstances=n
        if n==0: return

        if n>circleInstanceCapacity:
            newCapacity=max(n,max(64,circleInstanceCapacity*2))
            if circleVao is not None: circleVao.release()
            if circleInstanceVbo is not None: circleInstanceVbo.release()
            # x,y,z,radius,r,g,b,a = 8 float32 values per instance.
            circleInstanceVbo=ctx.buffer(reserve=newCapacity*8*4)
            circleVao=ctx.vertex_array(circleProg,[
                (circleMeshVbo,'2f','unit_pos'),
                (circleInstanceVbo,'4f 4f /i','instance_pos_rad','instance_color'),
            ])
            circleInstanceCapacity=newCapacity

        inst=np.empty((n,8),dtype=np.float32)
        inst[:,0]=data[inds,1]
        inst[:,1]=data[inds,2]
        inst[:,2]=data[inds,3]
        inst[:,3]=data[inds,0]
        c=colors[inds].astype(np.uint32,copy=False)
        inst[:,4]=((c>>16)&255)*(1.0/255.0)
        inst[:,5]=((c>>8)&255)*(1.0/255.0)
        inst[:,6]=(c&255)*(1.0/255.0)
        inst[:,7]=1.0
        circleInstanceVbo.write(inst.tobytes())


    if is3D:
        boxMesh=np.array([
            (-.5,-.5,-.5),(-.5,.5,-.5),(.5,.5,-.5),(-.5,-.5,-.5),(.5,.5,-.5),(.5,-.5,-.5),
            (-.5,-.5,.5),(.5,-.5,.5),(.5,.5,.5),(-.5,-.5,.5),(.5,.5,.5),(-.5,.5,.5),
            (-.5,-.5,-.5),(.5,-.5,-.5),(.5,-.5,.5),(-.5,-.5,-.5),(.5,-.5,.5),(-.5,-.5,.5),
            (.5,-.5,-.5),(.5,.5,-.5),(.5,.5,.5),(.5,-.5,-.5),(.5,.5,.5),(.5,-.5,.5),
            (.5,.5,-.5),(-.5,.5,-.5),(-.5,.5,.5),(.5,.5,-.5),(-.5,.5,.5),(.5,.5,.5),
            (-.5,.5,-.5),(-.5,-.5,-.5),(-.5,-.5,.5),(-.5,.5,-.5),(-.5,-.5,.5),(-.5,.5,.5),
        ],dtype=np.float32)
    else:
        boxMesh=np.array([(-.5,-.5,0),(.5,-.5,0),(.5,.5,0),(-.5,-.5,0),(.5,.5,0),(-.5,.5,0)],dtype=np.float32)
    boxMeshVbo=ctx.buffer(boxMesh.tobytes())
    boxInstanceVbo=None
    boxVao=None
    boxInstanceCapacity=0
    boxInstances=0

    # Degenerate 3D boxes with exactly one zero dimension are rendered as true
    # two-triangle planes rather than flattened six-face cubes. Separate meshes
    # preserve the ordinary box batch exactly as-is.
    planeMeshes=(
        np.array([(-.5,-.5,0),(.5,-.5,0),(.5,.5,0),(-.5,-.5,0),(.5,.5,0),(-.5,.5,0)],dtype=np.float32), # XY
        np.array([(-.5,0,-.5),(.5,0,-.5),(.5,0,.5),(-.5,0,-.5),(.5,0,.5),(-.5,0,.5)],dtype=np.float32), # XZ
        np.array([(0,-.5,-.5),(0,.5,-.5),(0,.5,.5),(0,-.5,-.5),(0,.5,.5),(0,-.5,.5)],dtype=np.float32), # YZ
    ) if is3D else ()
    planeMeshVbos=[ctx.buffer(mesh.tobytes()) for mesh in planeMeshes]
    planeInstanceVbos=[None,None,None]
    planeVaos=[None,None,None]
    planeInstanceCapacities=[0,0,0]
    planeInstances=[0,0,0]

    def rebuild_box_instances(kinds,data,colors):
        nonlocal boxInstanceVbo,boxVao,boxInstanceCapacity,boxInstances
        inds=np.flatnonzero(kinds==2)
        n=len(inds)
        boxInstances=n
        if n==0: return
        if n>boxInstanceCapacity:
            newCapacity=max(n,max(64,boxInstanceCapacity*2))
            if boxVao is not None: boxVao.release()
            if boxInstanceVbo is not None: boxInstanceVbo.release()
            # x,y,z,sx,sy,sz,r,g,b,a = 10 float32 values per instance.
            boxInstanceVbo=ctx.buffer(reserve=newCapacity*10*4)
            boxVao=ctx.vertex_array(boxProg,[
                (boxMeshVbo,'3f','unit_pos'),
                (boxInstanceVbo,'3f 3f 4f /i','instance_pos','instance_size','instance_color'),
            ])
            boxInstanceCapacity=newCapacity
        inst=np.empty((n,10),dtype=np.float32)
        inst[:,0]=data[inds,3];inst[:,1]=data[inds,4];inst[:,2]=data[inds,5]
        inst[:,3]=data[inds,0];inst[:,4]=data[inds,1];inst[:,5]=data[inds,2]
        c=colors[inds].astype(np.uint32,copy=False)
        inst[:,6]=((c>>16)&255)*(1.0/255.0)
        inst[:,7]=((c>>8)&255)*(1.0/255.0)
        inst[:,8]=(c&255)*(1.0/255.0)
        inst[:,9]=1.0
        boxInstanceVbo.write(inst.tobytes())

    def rebuild_plane_instances(kinds,data,colors):
        if not is3D: return
        for planeIndex,kind in enumerate((4,5,6)):
            inds=np.flatnonzero(kinds==kind)
            n=len(inds)
            planeInstances[planeIndex]=n
            if n==0: continue
            if n>planeInstanceCapacities[planeIndex]:
                newCapacity=max(n,max(64,planeInstanceCapacities[planeIndex]*2))
                if planeVaos[planeIndex] is not None: planeVaos[planeIndex].release()
                if planeInstanceVbos[planeIndex] is not None: planeInstanceVbos[planeIndex].release()
                vbo=ctx.buffer(reserve=newCapacity*10*4)
                planeInstanceVbos[planeIndex]=vbo
                planeVaos[planeIndex]=ctx.vertex_array(boxProg,[
                    (planeMeshVbos[planeIndex],'3f','unit_pos'),
                    (vbo,'3f 3f 4f /i','instance_pos','instance_size','instance_color'),
                ])
                planeInstanceCapacities[planeIndex]=newCapacity
            inst=np.empty((n,10),dtype=np.float32)
            inst[:,0]=data[inds,3];inst[:,1]=data[inds,4];inst[:,2]=data[inds,5]
            inst[:,3]=data[inds,0];inst[:,4]=data[inds,1];inst[:,5]=data[inds,2]
            c=colors[inds].astype(np.uint32,copy=False)
            inst[:,6]=((c>>16)&255)*(1.0/255.0)
            inst[:,7]=((c>>8)&255)*(1.0/255.0)
            inst[:,8]=(c&255)*(1.0/255.0)
            inst[:,9]=1.0
            planeInstanceVbos[planeIndex].write(inst.tobytes())

    lineBatches=[]

    def rebuild_line_batches(kinds,data,colors):
        nonlocal lineBatches
        for vao,vbo,_,_ in lineBatches:
            vao.release();vbo.release()
        lineBatches=[]
        inds=np.flatnonzero(kinds==3)
        if len(inds)==0: return
        widths=np.unique(data[inds,0])
        for width in widths:
            group=inds[data[inds,0]==width]
            verts=np.empty((len(group)*2,7),dtype=np.float32)
            c=colors[group].astype(np.uint32,copy=False)
            rgba=np.empty((len(group),4),dtype=np.float32)
            rgba[:,0]=((c>>16)&255)*(1.0/255.0)
            rgba[:,1]=((c>>8)&255)*(1.0/255.0)
            rgba[:,2]=(c&255)*(1.0/255.0)
            rgba[:,3]=1.0
            verts[0::2,0]=data[group,1];verts[0::2,1]=data[group,2];verts[0::2,2]=data[group,3]
            verts[1::2,0]=data[group,4];verts[1::2,1]=data[group,5];verts[1::2,2]=data[group,6]
            verts[0::2,3:]=rgba;verts[1::2,3:]=rgba
            vbo=ctx.buffer(verts.tobytes())
            vao=ctx.vertex_array(prog,[(vbo,'3f 4f','in_pos','in_color')])
            lineBatches.append((vao,vbo,float(width),len(group)*2))

    # 3D camera. yaw/pitch=None means automatically look at the domain center.
    cx,cy,cz=xDim/2,yDim/2,(zDim/2 if is3D else 0.0)
    extent=max(xDim,yDim,zDim) if is3D else max(xDim,yDim)
    cameraPos=[cx+extent*1.35,cy-extent*1.55,cz+extent*1.15] if is3D else [0.0,0.0,0.0]
    cameraYaw=None
    cameraPitch=None
    keys=None
    if not headless:
        keys=pyglet.window.key.KeyStateHandler()
        window.push_handlers(keys)
    if headless:
        class _HeadlessEventWindow:
            def event(self,fn): return fn
        eventWindow=_HeadlessEventWindow()
    else:
        eventWindow=window
    mouseLook=False
    cameraDirty=False
    moveSpeed=max(extent*.75,1.0)
    mouseSensitivity=.25

    def camera_forward():
        if cameraYaw is None or cameraPitch is None:
            v=np.array((cx-cameraPos[0],cy-cameraPos[1],cz-cameraPos[2]),dtype=float)
            n=np.linalg.norm(v)
            return v/n if n>1e-12 else np.array((0.,1.,0.))
        yaw=math.radians(cameraYaw); pitch=math.radians(cameraPitch)
        return np.array((math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch)),dtype=float)

    def camera_angles_from_forward(f):
        return math.degrees(math.atan2(f[1],f[0])),math.degrees(math.asin(max(-1.0,min(1.0,f[2]))))

    @eventWindow.event
    def on_mouse_press(x,y,button,modifiers):
        nonlocal mouseLook,cameraYaw,cameraPitch
        if is3D and button==pyglet.window.mouse.RIGHT:
            f=camera_forward()
            if cameraYaw is None or cameraPitch is None: cameraYaw,cameraPitch=camera_angles_from_forward(f)
            mouseLook=True

    @eventWindow.event
    def on_mouse_release(x,y,button,modifiers):
        nonlocal mouseLook
        if button==pyglet.window.mouse.RIGHT: mouseLook=False

    @eventWindow.event
    def on_mouse_drag(x,y,dx,dy,buttons,modifiers):
        nonlocal cameraYaw,cameraPitch,cameraDirty
        if is3D and mouseLook:
            cameraYaw=(cameraYaw-dx*mouseSensitivity)%360.0
            cameraPitch=max(-89.9,min(89.9,cameraPitch+dy*mouseSensitivity))
            cameraDirty=True

    def mvp():
        aspect=renderWidth/renderHeight if headless else max(window.width,1)/max(window.height,1)
        if not is3D:
            # PAL coordinates: origin lower-left, one world unit per model unit.
            return [[2/xDim,0,0,-1],[0,2/yDim,0,-1],[0,0,-1,0],[0,0,0,1]]
        f=camera_forward()
        target=tuple(cameraPos[i]+f[i] for i in range(3))
        return _MatMul(_Perspective(45,aspect,max(extent*.01,.01),extent*20),_LookAt(tuple(cameraPos),target,(0,0,1)))

    def add_vertices(vertices,color,mode,kind='other'):
        c=tuple(v/255.0 for v in color)
        data=np.empty((len(vertices),7),dtype='f4')
        data[:,:3]=vertices; data[:,3:]=c
        vbo=ctx.buffer(data.tobytes())
        vao=ctx.vertex_array(prog,[(vbo,'3f 4f','in_pos','in_color')])
        center=tuple(sum(v[i] for v in vertices)/len(vertices) for i in range(3))
        objects.append((vao,vbo,mode,None,center,color[3],kind))

    def circle(rad,color,x,y,z):
        # Camera-facing billboard in 3D; ordinary XY disc in 2D.
        n=32
        if not is3D:
            verts=[(x,y,0)]
            verts += [(x+rad*math.cos(2*math.pi*i/n),y+rad*math.sin(2*math.pi*i/n),0) for i in range(n+1)]
        else:
            f=np.array((x-cameraPos[0],y-cameraPos[1],z-cameraPos[2]),dtype=float); f/=np.linalg.norm(f)
            right=np.cross(f,np.array((0.,0.,1.)))
            if np.linalg.norm(right)<1e-9: right=np.array((1.,0.,0.))
            right/=np.linalg.norm(right); up=np.cross(right,f); center=np.array((x,y,z),dtype=float)
            verts=[tuple(center)]
            verts += [tuple(center+rad*(math.cos(2*math.pi*i/n)*right+math.sin(2*math.pi*i/n)*up)) for i in range(n+1)]
        add_vertices(verts,color,moderngl.TRIANGLE_FAN, 'circle')
        if is3D:
            vao,vbo,mode,_,center,alpha,kind=objects[-1]
            objects[-1]=(vao,vbo,mode,('circle',rad,x,y,z,color),(x,y,z),alpha,kind)

    def box(sx,sy,sz,color,x,y,z):
        hx,hy,hz=sx/2,sy/2,(sz/2 if is3D else 0)
        if not is3D:
            v=[(x-hx,y-hy,0),(x+hx,y-hy,0),(x+hx,y+hy,0),(x-hx,y-hy,0),(x+hx,y+hy,0),(x-hx,y+hy,0)]
        else:
            if sz==0 and sx!=0 and sy!=0:
                v=[(x-hx,y-hy,z),(x+hx,y-hy,z),(x+hx,y+hy,z),(x-hx,y-hy,z),(x+hx,y+hy,z),(x-hx,y+hy,z)]
                add_vertices(v,color,moderngl.TRIANGLES,'plane');return
            if sy==0 and sx!=0 and sz!=0:
                v=[(x-hx,y,z-hz),(x+hx,y,z-hz),(x+hx,y,z+hz),(x-hx,y,z-hz),(x+hx,y,z+hz),(x-hx,y,z+hz)]
                add_vertices(v,color,moderngl.TRIANGLES,'plane');return
            if sx==0 and sy!=0 and sz!=0:
                v=[(x,y-hy,z-hz),(x,y+hy,z-hz),(x,y+hy,z+hz),(x,y-hy,z-hz),(x,y+hy,z+hz),(x,y-hy,z+hz)]
                add_vertices(v,color,moderngl.TRIANGLES,'plane');return
            p=[(x+dx*hx,y+dy*hy,z+dz*hz) for dx,dy,dz in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
            faces=[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
            for a,b,c,d in faces:
                add_vertices([p[a],p[b],p[c],p[a],p[c],p[d]],color,moderngl.TRIANGLES,'box')
            return
        add_vertices(v,color,moderngl.TRIANGLES,'box')

    def line(width,color,x1,y1,z1,x2,y2,z2,persistent=False):
        # GL line width is intentionally only a hint on many core-profile drivers.
        add_vertices([(x1,y1,z1),(x2,y2,z2)],color,moderngl.LINES)
        vao,vbo,mode,_,center,alpha,kind=objects[-1]
        obj=(vao,vbo,mode,width,center,alpha,kind)
        objects[-1]=obj
        if persistent:
            objects.pop()
            persistentObjects.append(obj)

    def render(present=True):
        width=renderWidth if headless else window.width
        height=renderHeight if headless else window.height
        if headless: target.use()
        ctx.viewport=(0,0,width,height)
        ctx.clear(*bgf)
        prog['mvp'].write(np.array(_FlattenColumnMajor(mvp()),dtype='f4').tobytes())

        def draw(obj):
            extra=obj[3]
            if isinstance(extra,(int,float)):
                ctx.line_width=extra
            elif isinstance(extra,tuple) and extra[0]=='circle':
                _,rad,x,y,z,color=extra
                center=np.array((x,y,z),dtype=float)
                f=center-np.array(cameraPos,dtype=float)
                fn=np.linalg.norm(f)
                if fn>1e-12: f/=fn
                else: f=-camera_forward()
                right=np.cross(f,np.array((0.,0.,1.)))
                if np.linalg.norm(right)<1e-9: right=np.array((1.,0.,0.))
                right/=np.linalg.norm(right); up=np.cross(right,f)
                n=32
                verts=[tuple(center)]
                verts += [tuple(center+rad*(math.cos(2*math.pi*i/n)*right+math.sin(2*math.pi*i/n)*up)) for i in range(n+1)]
                c=tuple(v/255.0 for v in color)
                data=np.empty((len(verts),7),dtype='f4'); data[:,:3]=verts; data[:,3:]=c
                obj[1].write(data.tobytes())
            obj[0].render(obj[2])

        mvpBytes=np.array(_FlattenColumnMajor(mvp()),dtype='f4').tobytes()
        # Boxes are normally field/background cells, then circles are agents.
        # This also fixes the previous 2D path, which drew instanced circles
        # before unbatched boxes and could cover the agents.
        if boxVao is not None and boxInstances:
            boxProg['mvp'].write(mvpBytes)
            if is3D: ctx.enable(moderngl.CULL_FACE)
            boxVao.render(moderngl.TRIANGLES,instances=boxInstances)
            if is3D: ctx.disable(moderngl.CULL_FACE)

        if is3D and any(planeInstances):
            boxProg['mvp'].write(mvpBytes)
            ctx.disable(moderngl.CULL_FACE) # planes are visible from either side
            for planeVao,n in zip(planeVaos,planeInstances):
                if planeVao is not None and n: planeVao.render(moderngl.TRIANGLES,instances=n)

        if circleVao is not None and circleInstances:
            circleProg['mvp'].write(mvpBytes)
            circleProg['camera_pos'].value=tuple(float(v) for v in cameraPos)
            circleProg['billboard_3d'].value=1 if is3D else 0
            circleVao.render(moderngl.TRIANGLE_FAN,instances=circleInstances)

        for vao,vbo,width,count in lineBatches:
            ctx.line_width=width
            vao.render(moderngl.LINES,vertices=count)

        renderObjects=persistentObjects+objects
        opaque=[obj for obj in renderObjects if obj[5]>=255 and not (not is3D and obj[6]=='circle')]
        transparent=[obj for obj in renderObjects if obj[5]<255 and not (not is3D and obj[6]=='circle')]

        # Opaque 3D boxes use GPU back-face culling. Other primitives remain
        # double-sided. Each box face is its own object so transparent faces
        # can be sorted independently.
        for obj in opaque:
            if is3D and obj[6]=='box': ctx.enable(moderngl.CULL_FACE)
            else: ctx.disable(moderngl.CULL_FACE)
            draw(obj)
        ctx.disable(moderngl.CULL_FACE)

        if transparent:
            if is3D:
                transparent.sort(key=lambda obj: sum((obj[4][i]-cameraPos[i])**2 for i in range(3)),reverse=True)
                ctx.depth_mask=False
            # Transparent boxes use the same back-face culling as opaque boxes.
            for obj in transparent:
                if is3D and obj[6]=='box': ctx.enable(moderngl.CULL_FACE)
                else: ctx.disable(moderngl.CULL_FACE)
                draw(obj)
            ctx.disable(moderngl.CULL_FACE)
            if is3D: ctx.depth_mask=True

        if present and not headless: window.flip()

    def clear_objects():
        for obj in objects:
            obj[0].release()
            obj[1].release()
        objects.clear()

    if not headless: render()
    readyEvent.set()

    running=True
    while running:
        if not headless:
            window.dispatch_events()
            if window.has_exit and not visualClosed.is_set():
                visualClosed.set()
                window.set_visible(False)
        dirty=cameraDirty
        cameraDirty=False
        if is3D and not headless:
            dt=1/120
            f=camera_forward(); flat=np.array((f[0],f[1],0.),dtype=float)
            if np.linalg.norm(flat)>1e-12: flat/=np.linalg.norm(flat)
            right=np.array((flat[1],-flat[0],0.),dtype=float)
            delta=np.zeros(3,dtype=float)
            if keys[pyglet.window.key.W]: delta+=flat
            if keys[pyglet.window.key.S]: delta-=flat
            if keys[pyglet.window.key.D]: delta+=right
            if keys[pyglet.window.key.A]: delta-=right
            if keys[pyglet.window.key.SPACE]: delta[2]+=1
            if keys[pyglet.window.key.LSHIFT] or keys[pyglet.window.key.RSHIFT]: delta[2]-=1
            n=np.linalg.norm(delta)
            if n>0:
                cameraPos[:]=(np.array(cameraPos)+delta/n*moveSpeed*dt).tolist(); dirty=True
        # Consume the newest published frame before processing output commands.
        # This makes Save()/AddGifFrame() refer to the latest Update() even when
        # that frame has not yet reached the screen.
        currentFrame=seqShared.value
        if running and currentFrame!=seenFrame:
            with frameLock:
                currentFrame=seqShared.value
                n=countShared.value
                kinds=np.frombuffer(kindShared,dtype=np.int64,count=n).copy()
                data=np.frombuffer(dataShared,dtype=np.float64,count=n*7).reshape(n,7).copy()
                colors=np.frombuffer(colorShared,dtype=np.uint32,count=n).copy()
            seenFrame=currentFrame
            clear_objects()
            rebuild_circle_instances(kinds,data,colors)
            rebuild_box_instances(kinds,data,colors)
            rebuild_plane_instances(kinds,data,colors)
            rebuild_line_batches(kinds,data,colors)
            dirty=True

        while True:
            try: cmd=cmdQ.get_nowait()
            except queue.Empty: break
            op=cmd[0]
            if op=='circle': circle(*cmd[1:]); dirty=True
            elif op=='box': box(*cmd[1:]); dirty=True
            elif op=='line': line(*cmd[1:]); dirty=True
            elif op=='borders':
                width,color=cmd[1],cmd[2]
                corners=[(0,0,0),(xDim,0,0),(xDim,yDim,0),(0,yDim,0),(0,0,zDim),(xDim,0,zDim),(xDim,yDim,zDim),(0,yDim,zDim)]
                for obj in persistentObjects:
                    obj[0].release()
                    obj[1].release()
                persistentObjects.clear()
                for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
                    line(width,color,*corners[a],*corners[b],persistent=True)
                dirty=True
            elif op=='clear':
                clear_objects(); dirty=True
            elif op=='background': bgf=tuple(v/255.0 for v in cmd[1]); dirty=True
            elif op=='camera':
                cameraPos[:]=cmd[1:4]; cameraYaw=cmd[4]; cameraPitch=cmd[5]; dirty=True
            elif op=='save':
                path,token,kinds,data,colors=cmd[1:]
                clear_objects()
                rebuild_circle_instances(kinds,data,colors)
                rebuild_box_instances(kinds,data,colors)
                rebuild_plane_instances(kinds,data,colors)
                rebuild_line_batches(kinds,data,colors)
                # Draw into the back buffer, capture it before the double-buffer
                # swap, then present the exact immutable frame requested.
                render(False)
                data=(target if headless else ctx.screen).read(components=3,alignment=1)
                if headless:
                    from PIL import Image
                    frame=np.frombuffer(data,dtype=np.uint8).reshape(renderHeight,renderWidth,3)[::-1].copy()
                    Image.fromarray(frame).save(path)
                else:
                    image=pyglet.image.ImageData(window.width,window.height,'RGB',data,pitch=window.width*3)
                    image.save(path)
                    window.flip()
                ackQ.put(('save',token,None))
            elif op=='gif_start':
                gifPath,gifDelay,token=cmd[1],cmd[2],cmd[3]
                gifFrames=[]
                ackQ.put(('gif_start',token,None))
            elif op=='gif_frame':
                token,kinds,data,colors=cmd[1:]
                clear_objects()
                rebuild_circle_instances(kinds,data,colors)
                rebuild_box_instances(kinds,data,colors)
                rebuild_plane_instances(kinds,data,colors)
                rebuild_line_batches(kinds,data,colors)
                render(False)
                data=(target if headless else ctx.screen).read(components=3,alignment=1)
                height=renderHeight if headless else window.height
                width=renderWidth if headless else window.width
                frame=np.frombuffer(data,dtype=np.uint8).reshape(height,width,3)[::-1].copy()
                gifFrames.append(frame)
                if not headless: window.flip()
                ackQ.put(('gif_frame',token,None))
            elif op=='gif_stop':
                token=cmd[1]
                if gifPath is not None and gifFrames:
                    from PIL import Image
                    frames=[Image.fromarray(frame) for frame in gifFrames]
                    frames[0].save(gifPath,save_all=True,append_images=frames[1:],duration=gifDelay,loop=0)
                gifPath=None
                gifFrames=[]
                ackQ.put(('gif_stop',token,None))
            elif op=='close':
                if gifPath is not None and gifFrames:
                    from PIL import Image
                    frames=[Image.fromarray(frame) for frame in gifFrames]
                    frames[0].save(gifPath,save_all=True,append_images=frames[1:],duration=gifDelay,loop=0)
                running=False
                break

        if dirty and not headless: render(present=not visualClosed.is_set())
        time.sleep(1/120)
    if window is not None: window.close()
    if target is not None: target.release()
    ackQ.put(('closed',None,None))


class _OpenGLWindowSafe:
    """PAL 2D/3D primitive renderer. Coordinates are in model/world units."""
    def __init__(self,xDim,yDim,zDim=None,width=800,height=800,title='PAL',headless=False):
        self.xDim=float(xDim);self.yDim=float(yDim);self.zDim=None if zDim is None else float(zDim)
        self._draw=OpenGLDraw()
        self._cmdQ=mp.Queue();self._ackQ=mp.Queue();self._token=0
        self._frameShared=_NewFrameShared()
        self._readyEvent=mp.Event()
        self._visualClosed=mp.Event()
        self._headless=bool(headless)
        if self._headless:
            self._placementToken=None;screenX=screenY=0
        else:
            self._placementToken,screenX,screenY=_ReserveWindowPosition(int(width),int(height))
        self._process=mp.Process(target=_WindowProcess,args=(self._cmdQ,self._ackQ,self._frameShared,self.xDim,self.yDim,self.zDim,int(width),int(height),title,screenX,screenY,self._readyEvent,self._visualClosed,self._headless))
        self._process.daemon=True
        self._closed=False
        self._gifActive=False
        self._process.start()
        _RegisterWindowInit(self._readyEvent,self._process,"OpenGLWindow")
        atexit.register(self._Close)
        _RegisterCrashWindow(self)

    def _WaitReady(self,timeout=30):
        end=time.time()+timeout
        while not self._readyEvent.wait(min(.05,max(0,end-time.time()))):
            if not self._process.is_alive():
                self._process.join()
                raise RuntimeError(f"OpenGLWindow renderer exited during initialization exitcode:{self._process.exitcode}")
            if time.time()>=end: raise TimeoutError("timed out waiting for OpenGLWindow initialization")
        return self

    def IsOpen(self):
        alive=self._process.is_alive()
        if not alive: self._Cleanup()
        return alive and not self._visualClosed.is_set()

    def Update(self):
        _FlushDraw(self)
        return self

    def _Cleanup(self):
        if self._closed: return
        self._process.join()
        self._cmdQ.close();self._ackQ.close()
        if self._placementToken is not None: _ReleaseWindow(self._placementToken)
        _UnregisterCrashWindow(self)
        self._closed=True
        try: atexit.unregister(self._Close)
        except Exception: pass

    def Circle(self,rad,color,x,y,z=None):
        _Finite('radius and coordinates',rad,x,y)
        if rad<=0: raise Exception(f"radius must be positive rad:{rad}")
        color=_Color(color)
        if self.zDim is None:
            if z is not None: raise Exception("Circle on a 2D OpenGLWindow takes x and y only")
            z=0.0
        else:
            if z is None: raise Exception("Circle on a 3D OpenGLWindow requires x, y, and z")
            _Finite('z',z)
        return self._Circle(rad,color,x,y,z)

    def _Circle(self,rad,color,x,y,z=0.0):
        self._cmdQ.put(('circle',float(rad),color,float(x),float(y),float(z)));return self

    def Box(self,xLen,color,x,y,z=None,yLen=None,zLen=None):
        if yLen is None: yLen=xLen
        if self.zDim is None:
            if z is not None or zLen is not None: raise Exception("Box on a 2D OpenGLWindow takes x and y only; z/zLen are 3D arguments")
            z=0.0;zLen=0.0
        else:
            if z is None: raise Exception("Box on a 3D OpenGLWindow requires x, y, and z")
            if zLen is None: zLen=xLen
        _Finite('box lengths and coordinates',xLen,yLen,x,y,z,zLen)
        if self.zDim is None:
            if xLen<=0 or yLen<=0: raise Exception(f"box side lengths must be positive xLen:{xLen} yLen:{yLen}")
        else:
            if xLen<0 or yLen<0 or zLen<0: raise Exception(f"box side lengths cannot be negative xLen:{xLen} yLen:{yLen} zLen:{zLen}")
            if (xLen==0)+(yLen==0)+(zLen==0)>1: raise Exception(f"3D Box may have at most one zero side length xLen:{xLen} yLen:{yLen} zLen:{zLen}")
        color=_Color(color)
        return self._Box(xLen,color,x,y,z,yLen,zLen)

    def _Box(self,xLen,color,x,y,z=0.0,yLen=None,zLen=None):
        if yLen is None: yLen=xLen
        if zLen is None: zLen=xLen if self.zDim is not None else 0.0
        self._cmdQ.put(('box',float(xLen),float(yLen),float(zLen),color,float(x),float(y),float(z)));return self

    def BoxSQ(self,color,x,y,z=None):
        color=_Color(color)
        _Finite('square coordinates',x,y)
        if x!=int(x) or y!=int(y): raise Exception(f"BoxSQ coordinates must be integers x:{x} y:{y}")
        if self.zDim is None:
            if z is not None: raise Exception("BoxSQ on a 2D OpenGLWindow takes x and y only")
        else:
            if z is None: raise Exception("BoxSQ on a 3D OpenGLWindow requires x, y, and z")
            _Finite('z',z)
            if z!=int(z): raise Exception(f"BoxSQ coordinates must be integers x:{x} y:{y} z:{z}")
        return self._BoxSQ(color,x,y,z)

    def _BoxSQ(self,color,x,y,z=None):
        if self.zDim is None: return self._Box(1,color,x+0.5,y+0.5)
        return self._Box(1,color,x+0.5,y+0.5,z+0.5)

    def Line(self,width,color,x1,y1,x2,y2,z1=None,z2=None):
        _Finite('line width and coordinates',width,x1,y1,x2,y2)
        if width<=0: raise Exception(f"line width must be positive width:{width}")
        color=_Color(color)
        if self.zDim is None:
            if z1 is not None or z2 is not None: raise Exception("Line on a 2D OpenGLWindow takes x1, y1, x2, and y2 only")
            z1=z2=0.0
        else:
            if z1 is None or z2 is None: raise Exception("Line on a 3D OpenGLWindow requires z1 and z2")
            _Finite('line z coordinates',z1,z2)
        return self._Line(width,color,x1,y1,x2,y2,z1,z2)

    def _Line(self,width,color,x1,y1,x2,y2,z1=0.0,z2=0.0):
        self._cmdQ.put(('line',float(width),color,float(x1),float(y1),float(z1),float(x2),float(y2),float(z2)));return self

    def Borders(self,width,color):
        if self.zDim is None: raise Exception("Borders is only available on a 3D OpenGLWindow")
        _Finite('border width',width)
        if width<=0: raise Exception(f"border width must be positive width:{width}")
        return self._Borders(width,_Color(color))

    def _Borders(self,width,color):
        self._cmdQ.put(('borders',float(width),color));return self

    def Camera(self,x,y,z,yaw=None,pitch=None):
        if self.zDim is None: raise Exception("Camera is only available on a 3D OpenGLWindow")
        _Finite('camera position',x,y,z)
        if (yaw is None)!=(pitch is None): raise Exception("yaw and pitch must either both be provided or both be None")
        if yaw is not None:
            _Finite('camera angles',yaw,pitch)
            if pitch<=-90 or pitch>=90: raise Exception(f"camera pitch must be between -90 and 90 pitch:{pitch}")
        return self._Camera(x,y,z,yaw,pitch)

    def _Camera(self,x,y,z,yaw=None,pitch=None):
        self._cmdQ.put(('camera',float(x),float(y),float(z),None if yaw is None else float(yaw),None if pitch is None else float(pitch)));return self

    def Clear(self):
        if not self.IsOpen(): raise Exception("cannot clear a closed OpenGLWindow")
        return self._Clear()

    def _Clear(self): self._draw.Clear();self._cmdQ.put(('clear',));return self

    def Background(self,color):
        if not self.IsOpen(): raise Exception("cannot set background on a closed OpenGLWindow")
        return self._Background(_Color(color))

    def _Background(self,color): self._cmdQ.put(('background',color));return self

    def Save(self,path,block=False):
        if not self.IsOpen(): raise Exception("cannot save a closed OpenGLWindow")
        if not isinstance(path,(str,Path)): raise Exception(f"save path must be a string or Path path:{path}")
        if not isinstance(block,bool): raise Exception(f"block must be bool block:{block}")
        return self._Save(path,block)

    def _Save(self,path,block=False):
        kinds,data,colors=_OutputFrameSnapshot(self)
        path=str(Path(path))
        self._token+=1;token=self._token
        self._cmdQ.put(('save',path,token,kinds,data,colors))
        if block: return self._WaitAck('save',token,30)
        return self

    def Close(self):
        if self._closed: raise Exception("cannot close a closed OpenGLWindow")
        return self._Close()

    def _Close(self):
        if self._closed: return self
        _FlushDraw(self)
        if self._gifActive and self._process.is_alive():
            try: self.StopGif()
            except Exception: pass
        if self._process.is_alive():
            self._cmdQ.put(('close',))
            # Normal shutdown is lossless for accepted output commands. The
            # renderer drains its FIFO command queue before processing close.
            self._process.join()
        self._Cleanup()
        return self


class _OpenGLWindowFast:
    """PAL 2D/3D primitive renderer. Coordinates are in model/world units."""
    def __init__(self,xDim,yDim,zDim=None,width=800,height=800,title='PAL',headless=False):
        self.xDim=float(xDim);self.yDim=float(yDim);self.zDim=None if zDim is None else float(zDim)
        self._draw=OpenGLDraw()
        self._cmdQ=mp.Queue();self._ackQ=mp.Queue();self._token=0
        self._frameShared=_NewFrameShared()
        self._readyEvent=mp.Event()
        self._visualClosed=mp.Event()
        self._headless=bool(headless)
        if self._headless:
            self._placementToken=None;screenX=screenY=0
        else:
            self._placementToken,screenX,screenY=_ReserveWindowPosition(int(width),int(height))
        self._process=mp.Process(target=_WindowProcess,args=(self._cmdQ,self._ackQ,self._frameShared,self.xDim,self.yDim,self.zDim,int(width),int(height),title,screenX,screenY,self._readyEvent,self._visualClosed,self._headless))
        self._process.daemon=True
        self._closed=False
        self._gifActive=False
        self._process.start()
        _RegisterWindowInit(self._readyEvent,self._process,"OpenGLWindow")
        atexit.register(self._Close)

    def _WaitReady(self,timeout=30):
        end=time.time()+timeout
        while not self._readyEvent.wait(min(.05,max(0,end-time.time()))):
            if not self._process.is_alive():
                self._process.join()
                raise RuntimeError(f"OpenGLWindow renderer exited during initialization exitcode:{self._process.exitcode}")
            if time.time()>=end: raise TimeoutError("timed out waiting for OpenGLWindow initialization")
        return self

    def IsOpen(self):
        alive=self._process.is_alive()
        if not alive: self._Cleanup()
        return alive and not self._visualClosed.is_set()

    def Update(self):
        _FlushDraw(self)
        return self

    def _Cleanup(self):
        if self._closed: return
        self._process.join()
        self._cmdQ.close();self._ackQ.close()
        if self._placementToken is not None: _ReleaseWindow(self._placementToken)
        self._closed=True
        try: atexit.unregister(self._Close)
        except Exception: pass
    def Circle(self,rad,color,x,y,z=0.0): return self._Circle(rad,_Color(color),x,y,z)
    def Box(self,xLen,color,x,y,z=0.0,yLen=None,zLen=None): return self._Box(xLen,_Color(color),x,y,z,yLen,zLen)
    def BoxSQ(self,color,x,y,z=None): return self._BoxSQ(_Color(color),x,y,z)
    def Line(self,width,color,x1,y1,x2,y2,z1=0.0,z2=0.0): return self._Line(width,_Color(color),x1,y1,x2,y2,z1,z2)
    def Borders(self,width,color): return self._Borders(width,_Color(color))
    def Camera(self,x,y,z,yaw=None,pitch=None): return self._Camera(x,y,z,yaw,pitch)
    def Clear(self): return self._Clear()
    def Background(self,color): return self._Background(_Color(color))
    def Save(self,path,block=False): return self._Save(path,block)
    def StartGif(self,path,delay=100): return self._StartGif(path,delay)
    def AddGifFrame(self,block=False,timeout=30): return self._AddGifFrame(block,timeout)
    def StopGif(self,timeout=30): return self._StopGif(timeout)
    def Close(self): return self._Close()


    def _WaitAck(self,op,token,timeout):
        end=time.time()+timeout
        while time.time()<end:
            try:
                msg=self._ackQ.get(timeout=min(.1,max(0,end-time.time())))
                if msg[0]==op and msg[1]==token: return self
            except queue.Empty:
                if not self.IsOpen(): raise RuntimeError(f"OpenGLWindow closed while waiting for {op}")
        raise TimeoutError(f"timed out waiting for OpenGLWindow {op}")

    def _StartGif(self,path,delay=100):
        self._token+=1;token=self._token
        self._cmdQ.put(('gif_start',str(Path(path)),int(delay),token))
        self._WaitAck('gif_start',token,30)
        self._gifActive=True
        return self

    def _AddGifFrame(self,block=False,timeout=30):
        kinds,data,colors=_OutputFrameSnapshot(self)
        self._token+=1;token=self._token
        self._cmdQ.put(('gif_frame',token,kinds,data,colors))
        if block: return self._WaitAck('gif_frame',token,timeout)
        return self

    def _StopGif(self,timeout=30):
        self._token+=1;token=self._token
        self._cmdQ.put(('gif_stop',token))
        self._WaitAck('gif_stop',token,timeout)
        self._gifActive=False
        return self

    def _Circle(self,rad,color,x,y,z=0.0):
        self._cmdQ.put(('circle',float(rad),color,float(x),float(y),float(z)));return self


    def _Box(self,xLen,color,x,y,z=0.0,yLen=None,zLen=None):
        if yLen is None: yLen=xLen
        if zLen is None: zLen=xLen if self.zDim is not None else 0.0
        self._cmdQ.put(('box',float(xLen),float(yLen),float(zLen),color,float(x),float(y),float(z)));return self


    def _BoxSQ(self,color,x,y,z=None):
        if self.zDim is None: return self._Box(1,color,x+0.5,y+0.5)
        return self._Box(1,color,x+0.5,y+0.5,z+0.5)


    def _Line(self,width,color,x1,y1,x2,y2,z1=0.0,z2=0.0):
        self._cmdQ.put(('line',float(width),color,float(x1),float(y1),float(z1),float(x2),float(y2),float(z2)));return self


    def _Borders(self,width,color):
        self._cmdQ.put(('borders',float(width),color));return self


    def _Camera(self,x,y,z,yaw=None,pitch=None):
        self._cmdQ.put(('camera',float(x),float(y),float(z),None if yaw is None else float(yaw),None if pitch is None else float(pitch)));return self


    def _Clear(self): self._draw.Clear();self._cmdQ.put(('clear',));return self


    def _Background(self,color): self._cmdQ.put(('background',color));return self


    def _Save(self,path,block=False):
        kinds,data,colors=_OutputFrameSnapshot(self)
        path=str(Path(path))
        self._token+=1;token=self._token
        self._cmdQ.put(('save',path,token,kinds,data,colors))
        if block: return self._WaitAck('save',token,30)
        return self


    def _Close(self):
        if self._closed: return self
        _FlushDraw(self)
        if self._gifActive and self._process.is_alive():
            try: self._StopGif()
            except Exception: pass
        if self._process.is_alive():
            self._cmdQ.put(('close',))
            # Normal shutdown is lossless for accepted output commands. The
            # renderer drains its FIFO command queue before processing close.
            self._process.join()
        self._Cleanup()
        return self



def _StartOpenGLWindow(xDim,yDim,zDim,width,height,title,headless,windowClass):
    return windowClass(xDim,yDim,zDim,width,height,title,headless)


def StartOpenGLWindow(xDim,yDim,zDim=None,width=800,height=800,title='PAL',headless=False)-> tuple[OpenGLDraw,OpenGLWindow]:
    fast=_UseFastMode()
    if fast:
        win=_StartOpenGLWindow(xDim,yDim,zDim,width,height,title,headless,_OpenGLWindowFast)
        return win._draw,win
    _Finite('dimensions',xDim,yDim)
    if zDim is not None: _Finite('zDim',zDim)
    if xDim<=0 or yDim<=0 or zDim is not None and zDim<=0: raise Exception(f"window dimensions must be positive xDim:{xDim} yDim:{yDim} zDim:{zDim}")
    _Finite('pixel dimensions',width,height)
    if width!=int(width) or height!=int(height): raise Exception(f"pixel dimensions must be integers width:{width} height:{height}")
    if width<=0 or height<=0: raise Exception(f"pixel dimensions must be positive width:{width} height:{height}")
    if not isinstance(title,str): raise Exception(f"title must be a string title:{title}")
    if not isinstance(headless,bool): raise Exception(f"headless must be bool headless:{headless}")
    win=_StartOpenGLWindow(xDim,yDim,zDim,width,height,title,headless,_OpenGLWindowSafe)
    return win._draw,win
