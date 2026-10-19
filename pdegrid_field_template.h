/* Precision-specialized PDEGrid field updates compiled from one source. */
#ifndef PAL_PD_TYPE
#error "PAL_PD_TYPE must be defined"
#endif
#ifndef PAL_PD_FLOAT
#error "PAL_PD_FLOAT must be defined"
#endif
#ifndef PAL_PD_PREFIX
#error "PAL_PD_PREFIX must be defined"
#endif
#define PAL_PD_JOIN2(a,b) a##b
#define PAL_PD_JOIN(a,b) PAL_PD_JOIN2(a,b)
#define PAL_PD_FN(name) PAL_PD_JOIN(PAL_PD_PREFIX,name)

/* Conservative face-flux diffusion, with one edge contribution per face.
 * Both precision variants use double arithmetic for coefficients. */
#ifdef PAL_PD_DIFFUSION
PAL_API int32_t PAL_PD_FN(PAL_PD_DIFFUSION)(uintptr_t handle,double rate) {
    PAL_PD_TYPE *grid=(PAL_PD_TYPE *)handle;
    if (!grid || !isfinite(rate) || rate<0.0 || !isfinite(grid->dt) || grid->dt<0.0) return 0;
    const int32_t ny=grid->dimension>1?grid->dimensions[1]:1;
    const int32_t nz=grid->dimension>2?grid->dimensions[2]:1;
    const int32_t stride[3]={ny*nz,nz,1};
    const double spacing[3]={grid->dx,grid->dy,grid->dz};
    double factors[3]={0.0,0.0,0.0}, stability=0.0;
    for (int32_t axis=0;axis<grid->dimension;axis++) {
        if (!isfinite(spacing[axis]) || spacing[axis]<=0.0) return 0;
        factors[axis]=rate*grid->dt/(spacing[axis]*spacing[axis]);
        if (!isfinite(factors[axis])) return 0;
        if (grid->dimensions[axis]>1) stability+=factors[axis];
    }
    if (stability>0.5000005) return 0;
    for (int32_t axis=0;axis<grid->dimension;axis++) {
        const int32_t size=grid->dimensions[axis],step=stride[axis];
        if (size==1 || factors[axis]==0.0) continue;
        for (int32_t i=0;i<grid->length;i++) {
            const int32_t coordinate=(i/step)%size;
            if (coordinate==size-1 && !grid->wrap[axis]) continue;
            const int32_t j=coordinate==size-1?i-(size-1)*step:i+step;
            const PAL_PD_FLOAT flux=(PAL_PD_FLOAT)(factors[axis]*((double)grid->field[j]-grid->field[i]));
            grid->deltas[i]+=flux;
            grid->deltas[j]-=flux;
        }
    }
    return 1;
}
#endif

PAL_API int32_t PAL_PD_FN(dim)(uintptr_t handle,int32_t d) {
    PAL_PD_TYPE *grid=(PAL_PD_TYPE *)handle;
    return grid && d>=0 && d<grid->dimension ? grid->dimensions[d] : PAL_NULL;
}

PAL_API void PAL_PD_FN(steps)(uintptr_t handle,double dt,double dx,double dy,double dz) {
    PAL_PD_TYPE *grid=(PAL_PD_TYPE *)handle;
    grid->dt=dt; grid->dx=dx; grid->dy=dy; grid->dz=dz;
}
#ifdef PAL_PD_STEP_GETTERS
PAL_API double PAL_PD_FN(dt)(uintptr_t handle) { return ((PAL_PD_TYPE *)handle)->dt; }
PAL_API double PAL_PD_FN(dx)(uintptr_t handle) { return ((PAL_PD_TYPE *)handle)->dx; }
PAL_API double PAL_PD_FN(dy)(uintptr_t handle) { return ((PAL_PD_TYPE *)handle)->dy; }
PAL_API double PAL_PD_FN(dz)(uintptr_t handle) { return ((PAL_PD_TYPE *)handle)->dz; }
#endif

PAL_API double PAL_PD_FN(voxel)(uintptr_t handle) {
    PAL_PD_TYPE *grid=(PAL_PD_TYPE *)handle;
    if (grid->dimension==1) return grid->dx;
    if (grid->dimension==2) return grid->dx*grid->dy;
    return grid->dx*grid->dy*grid->dz;
}

PAL_API int32_t PAL_PD_FN(seti)(uintptr_t handle,int32_t i,PAL_PD_FLOAT value) {
    PAL_PD_TYPE *grid=(PAL_PD_TYPE *)handle;
    if (!grid || i<0 || i>=grid->length || !isfinite(value)) return 0;
    grid->field[i]=value;
    return 1;
}

PAL_API PAL_PD_FLOAT PAL_PD_FN(geti)(uintptr_t handle,int32_t i) {
    PAL_PD_TYPE *grid=(PAL_PD_TYPE *)handle;
    return grid && i>=0 && i<grid->length ? grid->field[i] : (PAL_PD_FLOAT)NAN;
}

PAL_API void PAL_PD_FN(update)(uintptr_t handle) {
    PAL_PD_TYPE *grid=(PAL_PD_TYPE *)handle;
    for (int32_t i=0;i<grid->length;i++) {
        grid->field[i]+=grid->deltas[i];
        grid->deltas[i]=(PAL_PD_FLOAT)0;
    }
}
PAL_API void PAL_PD_FN(clear)(uintptr_t handle,PAL_PD_FLOAT value) {
    PAL_PD_TYPE *grid=(PAL_PD_TYPE *)handle;
    for (int32_t i=0;i<grid->length;i++) {
        grid->field[i]=value;
        grid->deltas[i]=(PAL_PD_FLOAT)0;
    }
}
#undef PAL_PD_FN
#undef PAL_PD_JOIN
#undef PAL_PD_JOIN2
