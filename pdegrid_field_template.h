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

PAL_API int32_t PAL_PD_FN(dim)(uintptr_t handle,int32_t d) {
    PAL_PD_TYPE *grid=(PAL_PD_TYPE *)handle;
    return grid && d>=0 && d<grid->dimension ? grid->dimensions[d] : PAL_NULL;
}

PAL_API void PAL_PD_FN(steps)(uintptr_t handle,double dt,double dx,double dy,double dz) {
    PAL_PD_TYPE *grid=(PAL_PD_TYPE *)handle;
    grid->dt=dt; grid->dx=dx; grid->dy=dy; grid->dz=dz;
}
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
