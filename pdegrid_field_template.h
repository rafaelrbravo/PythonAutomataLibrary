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
