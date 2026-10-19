/* Shared AgentGrid occupancy operations; included once per native precision. */
#ifndef PAL_AG_TYPE
#error "PAL_AG_TYPE must be defined"
#endif
#ifndef PAL_AG_PREFIX
#error "PAL_AG_PREFIX must be defined"
#endif
#define PAL_AG_OCC_JOIN2(a,b) a##b
#define PAL_AG_OCC_JOIN(a,b) PAL_AG_OCC_JOIN2(a,b)
#define PAL_AG_OCC_FN(name) PAL_AG_OCC_JOIN(PAL_AG_PREFIX,name)

PAL_API int32_t PAL_AG_OCC_FN(count_i)(uintptr_t handle,int32_t location) {
    PAL_AG_TYPE *grid=(PAL_AG_TYPE *)handle;
    if (!grid || location<0 || location>=grid->length) return 0;
    return grid->stackable ? grid->counts[location] : (grid->grid[location]!=PAL_NULL);
}


PAL_API int32_t PAL_AG_OCC_FN(PAL_AG_OCC_LAST)(uintptr_t handle,int32_t location) {
    PAL_AG_TYPE *grid=(PAL_AG_TYPE *)handle;
    if (!grid || location<0 || location>=grid->length) return PAL_NULL;
    return grid->grid[location];
}


PAL_API int32_t PAL_AG_OCC_FN(PAL_AG_OCC_NFLOAT)(uintptr_t handle) {
    PAL_AG_TYPE *grid=(PAL_AG_TYPE *)handle;
    return grid ? grid->nFloatProperties : 0;
}

#undef PAL_AG_OCC_FN
#undef PAL_AG_OCC_JOIN
#undef PAL_AG_OCC_JOIN2
