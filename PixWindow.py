import atexit
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import operator
from .NativeCore import _ReserveWindowPosition,_ReleaseWindow,_RegisterWindowInit,_RegisterCrashWindow,_UnregisterCrashWindow,_UseFastMode
from typing import Protocol
from numba import types
from numba.extending import typeof_impl,register_model,models,make_attribute_wrapper,unbox,NativeValue,overload_method,overload



class Pix:
    """Write-oriented PAL pixel buffer backed by PixWindow shared memory."""
    def __init__(self,array):
        self._array=array
        self._flat=array.reshape(-1)
    def __setitem__(self,key,color):
        if isinstance(key,tuple): self._array[key]=color
        else: self._flat[key]=color
    def Xdim(self): return self._array.shape[0]
    def Ydim(self): return self._array.shape[1]
    def ToI(self,x,y): return x*self._array.shape[1]+y
    def ItoX(self,i): return i//self._array.shape[1]
    def ItoY(self,i): return i%self._array.shape[1]
    def __len__(self): return self._array.shape[0]*self._array.shape[1]


class _PixSafe(Pix):
    def __setitem__(self,key,color):
        if isinstance(key,tuple):
            if len(key)!=2: raise IndexError("Pix expects i or x,y")
            x,y=key
            if isinstance(x,slice) or isinstance(y,slice):
                self._array[x,y]=color
                return
            if x<0 or x>=self._array.shape[0] or y<0 or y>=self._array.shape[1]:
                raise IndexError(f"pixel coordinates out of bounds x:{x} y:{y}")
            self._array[x,y]=color
        else:
            if isinstance(key,slice):
                self._flat[key]=color
                return
            if key<0 or key>=self._flat.size:
                raise IndexError(f"pixel index out of bounds i:{key}")
            self._flat[key]=color


class _PixFast(Pix):
    pass


class _PixNumbaType(types.Type):
    def __init__(self,name,safe):
        self.safe=safe
        super().__init__(name=name)


_PIX_SAFE_NUMBA_TYPE=_PixNumbaType("PixSafe",True)
_PIX_FAST_NUMBA_TYPE=_PixNumbaType("PixFast",False)
_PIX_ARRAY_TYPE=types.Array(types.uint32,2,"C")
_PIX_FLAT_TYPE=types.Array(types.uint32,1,"C")


@typeof_impl.register(_PixSafe)
def _typeofPixSafe(val,c): return _PIX_SAFE_NUMBA_TYPE


@typeof_impl.register(_PixFast)
def _typeofPixFast(val,c): return _PIX_FAST_NUMBA_TYPE


@register_model(_PixNumbaType)
class _PixModel(models.StructModel):
    def __init__(self,dmm,feType): super().__init__(dmm,feType,[("array",_PIX_ARRAY_TYPE),("flat",_PIX_FLAT_TYPE)])


make_attribute_wrapper(_PixNumbaType,"array","_array")
make_attribute_wrapper(_PixNumbaType,"flat","_flat")


@unbox(_PixNumbaType)
def _unboxPix(typ,obj,c):
    proxy=c.context.make_helper(c.builder,typ)
    arrayObj=c.pyapi.object_getattr_string(obj,"_array")
    nativeArray=c.unbox(_PIX_ARRAY_TYPE,arrayObj)
    c.pyapi.decref(arrayObj)
    flatObj=c.pyapi.object_getattr_string(obj,"_flat")
    nativeFlat=c.unbox(_PIX_FLAT_TYPE,flatObj)
    c.pyapi.decref(flatObj)
    proxy.array=nativeArray.value
    proxy.flat=nativeFlat.value
    isError=c.builder.or_(nativeArray.is_error,nativeFlat.is_error)
    return NativeValue(proxy._getvalue(),is_error=isError)


@overload(operator.setitem)
def _olPixSetItem(pix,key,color):
    if isinstance(pix,_PixNumbaType):
        if isinstance(key,types.BaseTuple):
            hasSlice=any(isinstance(t,types.SliceType) for t in key.types)
            if hasSlice:
                def impl(pix,key,color):
                    pix._array[key[0],key[1]]=color
                return impl
            if pix.safe:
                def impl(pix,key,color):
                    x,y=key
                    if x<0 or x>=pix._array.shape[0] or y<0 or y>=pix._array.shape[1]:
                        raise IndexError("pixel coordinates out of bounds")
                    pix._array[x,y]=color
                return impl
            def impl(pix,key,color):
                pix._array[key[0],key[1]]=color
            return impl
        if isinstance(key,types.SliceType):
            def impl(pix,key,color):
                pix._flat[key]=color
            return impl
        if pix.safe:
            def impl(pix,key,color):
                if key<0 or key>=pix._flat.size:
                    raise IndexError("pixel index out of bounds")
                pix._flat[key]=color
            return impl
        def impl(pix,key,color):
            pix._flat[key]=color
        return impl


@overload_method(_PixNumbaType,"Xdim",inline="always")
def _olPixXdim(pix):
    def impl(pix): return pix._array.shape[0]
    return impl


@overload_method(_PixNumbaType,"Ydim",inline="always")
def _olPixYdim(pix):
    def impl(pix): return pix._array.shape[1]
    return impl


@overload_method(_PixNumbaType,"ToI",inline="always")
def _olPixToI(pix,x,y):
    def impl(pix,x,y): return x*pix._array.shape[1]+y
    return impl


@overload_method(_PixNumbaType,"ItoX",inline="always")
def _olPixItoX(pix,i):
    def impl(pix,i): return i//pix._array.shape[1]
    return impl


@overload_method(_PixNumbaType,"ItoY",inline="always")
def _olPixItoY(pix,i):
    def impl(pix,i): return i%pix._array.shape[1]
    return impl


@overload(len)
def _olPixLen(pix):
    if isinstance(pix,_PixNumbaType):
        def impl(pix):
            return pix._array.shape[0]*pix._array.shape[1]
        return impl



class PixWindow(Protocol):
    def IsOpen(self) -> bool: ...
    def Update(self): ...
    def Save(self,path:str,block=False): ...
    def StartGif(self,path:str,delay=100): ...
    def AddGifFrame(self,block=False): ...
    def StopGif(self): ...
    def Close(self): ...

def _WindowProcess(shmName,frameShmName,saveShmName,xDim,yDim,scale,title,screenX,screenY,closeRequest,visualClosed,closedEvent,savePath,saveEvent,saveDone,frameSeq,frameLock,framePending,readyEvent,outputQ,outputAckQ):
    import pygame
    shm=shared_memory.SharedMemory(name=shmName)
    frameShm=shared_memory.SharedMemory(name=frameShmName)
    saveShm=shared_memory.SharedMemory(name=saveShmName)
    pix=np.ndarray((xDim,yDim),dtype=np.uint32,buffer=shm.buf)
    framePix=np.ndarray((xDim,yDim),dtype=np.uint32,buffer=frameShm.buf)
    savePix=np.ndarray((xDim,yDim),dtype=np.uint32,buffer=saveShm.buf)
    displayPix=np.empty((xDim,yDim),dtype=np.uint32)
    rgb=np.empty((xDim,yDim,3),dtype=np.uint8)
    saveRgb=np.empty((xDim,yDim,3),dtype=np.uint8)
    import os
    os.environ['SDL_VIDEO_WINDOW_POS']=f'{screenX},{screenY}'
    pygame.display.init()
    screen=pygame.display.set_mode((xDim*scale,yDim*scale))
    pygame.display.set_caption(title)
    clock=pygame.time.Clock()
    surf=pygame.Surface((xDim,yDim),depth=24)
    screen.fill((0,0,0))
    pygame.display.flip()
    readyEvent.set()

    try:
        while True:
            if not visualClosed.is_set():
                for event in pygame.event.get():
                    if event.type==pygame.QUIT:
                        visualClosed.set()
                        pygame.display.quit()
                        break

            # Consume at most one completed frame. Copy the shared frame while
            # holding the lock, then release it before doing RGB conversion and
            # display work. Update() drops frames while one is already pending,
            # so a fast model can never starve the renderer by continually
            # invalidating the frame being converted.
            if framePending.is_set():
                with frameLock:
                    displayPix[:]=framePix
                    framePending.clear()
                rgb[:,:,0]=(displayPix>>16)&255
                rgb[:,:,1]=(displayPix>>8)&255
                rgb[:,:,2]=displayPix&255
                pygame.surfarray.blit_array(surf,rgb[:,::-1,:])
            while True:
                try:
                    output=outputQ.get_nowait()
                except Exception:
                    break
                if output[0]=='save':
                    _,path,frame,token=output
                    src=np.frombuffer(frame,dtype=np.uint32).reshape(xDim,yDim)
                    saveRgb[:,:,0]=(src>>16)&255
                    saveRgb[:,:,1]=(src>>8)&255
                    saveRgb[:,:,2]=src&255
                    saveSurf=pygame.surfarray.make_surface(saveRgb[:,::-1,:])
                    if scale!=1:
                        saveSurf=pygame.transform.scale(saveSurf,(xDim*scale,yDim*scale))
                    pygame.image.save(saveSurf,path)
                    outputAckQ.put(('save',token))
            if closeRequest.is_set():
                # Close is queued only after the parent has finished submitting
                # output. Drain every accepted output job before terminating.
                while True:
                    try:
                        output=outputQ.get_nowait()
                    except Exception:
                        break
                    if output[0]=='save':
                        _,path,frame,token=output
                        src=np.frombuffer(frame,dtype=np.uint32).reshape(xDim,yDim)
                        saveRgb[:,:,0]=(src>>16)&255
                        saveRgb[:,:,1]=(src>>8)&255
                        saveRgb[:,:,2]=src&255
                        saveSurf=pygame.surfarray.make_surface(saveRgb[:,::-1,:])
                        if scale!=1:
                            saveSurf=pygame.transform.scale(saveSurf,(xDim*scale,yDim*scale))
                        pygame.image.save(saveSurf,path)
                        outputAckQ.put(('save',token))
                break
            if not visualClosed.is_set():
                drawSurf=surf if scale==1 else pygame.transform.scale(surf,(xDim*scale,yDim*scale))
                screen.blit(drawSurf,(0,0))
                pygame.display.flip()
            clock.tick(60)
    finally:
        closedEvent.set()
        pygame.display.quit()
        shm.close()
        frameShm.close()
        saveShm.close()


def _HeadlessWindowProcess(outputQ,outputAckQ,xDim,yDim,scale,closeRequest,closedEvent,readyEvent):
    # Output-only worker: intentionally imports neither pygame nor any display API.
    from PIL import Image
    readyEvent.set()
    try:
        while True:
            try:
                output=outputQ.get(timeout=.05)
            except Exception:
                if closeRequest.is_set(): break
                continue
            if output[0]=='save':
                _,path,frame,token=output
                src=np.frombuffer(frame,dtype=np.uint32).reshape(xDim,yDim)
                rgb=np.empty((yDim,xDim,3),dtype=np.uint8)
                rgb[:,:,0]=(((src>>16)&255).T)[::-1,:]
                rgb[:,:,1]=(((src>>8)&255).T)[::-1,:]
                rgb[:,:,2]=((src&255).T)[::-1,:]
                image=Image.fromarray(rgb)
                if scale!=1: image=image.resize((xDim*scale,yDim*scale),resample=Image.Resampling.NEAREST)
                image.save(path)
                outputAckQ.put(('save',token))
            if closeRequest.is_set() and outputQ.empty(): break
    finally:
        closedEvent.set()


class _PixWindow:

    def __init__(self,shm,frameShm,saveShm,pix,framePix,savePix,process,placementToken,closeRequest,visualClosed,closedEvent,savePath,saveEvent,saveDone,frameSeq,frameLock,framePending,scale,outputQ,outputAckQ,headless=False):
        self.shm=shm
        self.frameShm=frameShm
        self.saveShm=saveShm
        self.pix=pix
        self.framePix=framePix
        self.savePix=savePix
        self.frameSeq=frameSeq
        self.frameLock=frameLock
        self.framePending=framePending
        self.scale=int(scale)
        self.process=process
        self.placementToken=placementToken
        self.closeRequest=closeRequest
        self.visualClosed=visualClosed
        self.closedEvent=closedEvent
        self.savePath=savePath
        self.saveEvent=saveEvent
        self.saveDone=saveDone
        self._gifPath=None
        self._gifDelay=100
        self._gifFrames=[]
        self._outputQ=outputQ
        self._outputAckQ=outputAckQ
        self._outputToken=0
        self._headless=bool(headless)
        self._cleaned=False
        atexit.register(self._Close)

    def IsOpen(self):
        isOpen=self.process.is_alive() and not self.visualClosed.is_set() and not self.closedEvent.is_set()
        if not self.process.is_alive():
            self.process.join()
            self._Cleanup()
        return isOpen

    def Update(self):
        # Publish the newest completed model frame. There is never a render
        # queue: a newer Update() replaces an older frame that has not yet been
        # consumed, so the display follows the model instead of lagging behind.
        # The lock is held only for the shared-memory copy; RGB conversion,
        # scaling, and display all happen after the renderer releases it.
        with self.frameLock:
            self.framePix[:]=self.pix
            self.frameSeq.value+=1
            self.framePending.set()
        return self

    def _WaitOutput(self,op,token):
        while True:
            msg=self._outputAckQ.get()
            if msg[0]==op and msg[1]==token:
                return self

    def _StartGif(self,path,delay=100):
        self._gifPath=str(path)
        self._gifDelay=int(delay)
        self._gifFrames=[]
        return self

    def _AddGifFrame(self,block=True):
        # Capture the exact newest frame published by Update(). The capture is
        # completed before returning; block controls compatibility with the
        # common output API and future asynchronous encoders.
        with self.frameLock:
            src=self.framePix.copy()
        frame=np.empty((src.shape[1],src.shape[0],3),dtype=np.uint8)
        frame[:,:,0]=(((src>>16)&255).T)[::-1,:]
        frame[:,:,1]=(((src>>8)&255).T)[::-1,:]
        frame[:,:,2]=((src&255).T)[::-1,:]
        if self.scale!=1:
            frame=np.repeat(np.repeat(frame,self.scale,axis=0),self.scale,axis=1)
        self._gifFrames.append(frame)
        return self

    def _StopGif(self):
        if self._gifPath is None: return self
        if self._gifFrames:
            from PIL import Image
            frames=[Image.fromarray(frame) for frame in self._gifFrames]
            frames[0].save(self._gifPath,save_all=True,append_images=frames[1:],duration=self._gifDelay,loop=0)
        self._gifPath=None
        self._gifFrames=[]
        return self

    def _Cleanup(self):
        if self._cleaned: return
        self.shm.close()
        self.frameShm.close()
        self.saveShm.close()
        try: self.shm.unlink()
        except FileNotFoundError: pass
        try: self.frameShm.unlink()
        except FileNotFoundError: pass
        try: self.saveShm.unlink()
        except FileNotFoundError: pass
        if self.placementToken is not None: _ReleaseWindow(self.placementToken)
        _UnregisterCrashWindow(self)
        self._cleaned=True
        try: atexit.unregister(self._Close)
        except Exception: pass

    def _Close(self):
        if not self._cleaned:
            self._StopGif()
            # Finish the multiprocessing Queue feeder before telling the child
            # to exit, so every accepted asynchronous save is visible to it.
            self._outputQ.close()
            self._outputQ.join_thread()
            self.closeRequest.set()
            if self.process.is_alive(): self.process.join()
            else: self.process.join()
            self._Cleanup()
        return self


class _PixWindowSafe(_PixWindow):

    def Save(self,path,block=False):
        if not self.IsOpen(): raise Exception("cannot save a closed PixWindow")
        if not isinstance(path,str): raise Exception(f"save path must be a string path:{path}")
        if not isinstance(block,(bool,np.bool_)): raise Exception(f"block must be bool block:{block}")
        return self._Save(path,block)

    def _Save(self,path,block=False):
        with self.frameLock:
            frame=self.framePix.tobytes()
        self._outputToken+=1
        token=self._outputToken
        self._outputQ.put(('save',str(path),frame,token))
        if block: return self._WaitOutput('save',token)
        return self

    def StartGif(self,path,delay=100):
        if not self.IsOpen(): raise Exception("cannot start GIF on a closed PixWindow")
        if self._gifPath is not None: raise Exception("a PixWindow GIF is already active")
        if not isinstance(path,str): raise Exception(f"GIF path must be a string path:{path}")
        if not np.isfinite(delay) or delay<=0: raise Exception(f"GIF delay must be positive delay:{delay}")
        return self._StartGif(path,delay)

    def AddGifFrame(self,block=False):
        if not self.IsOpen(): raise Exception("cannot add GIF frame from a closed PixWindow")
        if self._gifPath is None: raise Exception("StartGif must be called before AddGifFrame")
        if not isinstance(block,(bool,np.bool_)): raise Exception(f"block must be bool block:{block}")
        return self._AddGifFrame(block)

    def StopGif(self):
        if self._gifPath is None: raise Exception("no PixWindow GIF is active")
        return self._StopGif()

    def Close(self):
        if self._cleaned: raise Exception("cannot close a closed PixWindow")
        return self._Close()


class _PixWindowFast(_PixWindow):

    def Save(self,path,block=False): return self._Save(path,block)
    def StartGif(self,path,delay=100): return self._StartGif(path,delay)
    def AddGifFrame(self,block=False): return self._AddGifFrame(block)
    def StopGif(self): return self._StopGif()
    def Close(self): return self._Close()

    def _Save(self,path,block=False):
        with self.frameLock:
            frame=self.framePix.tobytes()
        self._outputToken+=1
        token=self._outputToken
        self._outputQ.put(('save',str(path),frame,token))
        if block: return self._WaitOutput('save',token)
        return self



def _StartPixWindow(xDim,yDim,scale,title,headless,windowClass,pixClass):
    nBytes=xDim*yDim*4
    shm=shared_memory.SharedMemory(create=True,size=nBytes)
    frameShm=shared_memory.SharedMemory(create=True,size=nBytes)
    saveShm=shared_memory.SharedMemory(create=True,size=nBytes)
    pix=np.ndarray((xDim,yDim),dtype=np.uint32,buffer=shm.buf)
    framePix=np.ndarray((xDim,yDim),dtype=np.uint32,buffer=frameShm.buf)
    savePix=np.ndarray((xDim,yDim),dtype=np.uint32,buffer=saveShm.buf)
    pix.fill(0)
    framePix.fill(0)
    savePix.fill(0)
    closeRequest=mp.Event()
    visualClosed=mp.Event()
    closedEvent=mp.Event()
    savePath=mp.Array('u',4096)
    saveEvent=mp.Event()
    saveDone=mp.Event()
    saveDone.set()
    frameSeq=mp.Value('q',0,lock=False)
    frameLock=mp.Lock()
    framePending=mp.Event()
    readyEvent=mp.Event()
    outputQ=mp.Queue()
    outputAckQ=mp.Queue()
    if headless:
        placementToken=None
        process=mp.Process(target=_HeadlessWindowProcess,args=(outputQ,outputAckQ,xDim,yDim,scale,closeRequest,closedEvent,readyEvent))
    else:
        placementToken,screenX,screenY=_ReserveWindowPosition(xDim*scale,yDim*scale)
        process=mp.Process(target=_WindowProcess,args=(shm.name,frameShm.name,saveShm.name,xDim,yDim,scale,title,screenX,screenY,closeRequest,visualClosed,closedEvent,savePath,saveEvent,saveDone,frameSeq,frameLock,framePending,readyEvent,outputQ,outputAckQ))
    process.daemon=True
    process.start()
    _RegisterWindowInit(readyEvent,process,"PixWindow")
    window=windowClass(shm,frameShm,saveShm,pix,framePix,savePix,process,placementToken,closeRequest,visualClosed,closedEvent,savePath,saveEvent,saveDone,frameSeq,frameLock,framePending,scale,outputQ,outputAckQ,headless)
    if windowClass is _PixWindowSafe: _RegisterCrashWindow(window)
    return pixClass(pix),window


def StartPixWindow(xDim,yDim,scale=1,title='PAL',headless=False) -> tuple[Pix,PixWindow]:
    fast=_UseFastMode()
    if fast:
        return _StartPixWindow(xDim,yDim,scale,title,headless,_PixWindowFast,_PixFast)
    if not isinstance(title,str): raise Exception(f"title must be a string title:{title}")
    if not isinstance(headless,bool): raise Exception(f"headless must be bool headless:{headless}")
    if not np.isfinite(xDim) or xDim<=0 or xDim!=int(xDim): raise Exception(f"xDim must be a positive integer xDim:{xDim}")
    if not np.isfinite(yDim) or yDim<=0 or yDim!=int(yDim): raise Exception(f"yDim must be a positive integer yDim:{yDim}")
    if not np.isfinite(scale) or scale<=0 or scale!=int(scale): raise Exception(f"scale must be a positive integer scale:{scale}")
    return _StartPixWindow(int(xDim),int(yDim),int(scale),title,headless,_PixWindowSafe,_PixSafe)
