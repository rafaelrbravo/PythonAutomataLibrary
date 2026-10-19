/* Shared point-coordinate bounds validation for both AgentGrid precisions. */
#ifndef PAL_AG_BOUNDS_TYPE
#error "PAL_AG_BOUNDS_TYPE must be defined"
#endif
#ifndef PAL_AG_BOUNDS_FLOAT
#error "PAL_AG_BOUNDS_FLOAT must be defined"
#endif
#ifndef PAL_AG_BOUNDS_NAME
#error "PAL_AG_BOUNDS_NAME must be defined"
#endif
static int PAL_AG_BOUNDS_NAME(const PAL_AG_BOUNDS_TYPE *g,PAL_AG_BOUNDS_FLOAT x,PAL_AG_BOUNDS_FLOAT y,PAL_AG_BOUNDS_FLOAT z) {
    PAL_AG_BOUNDS_FLOAT coords[3]={x,y,z};
    if (!g || !g->dimension) return 0;
    for (int32_t axis=0;axis<g->dimension;axis++)
        if (!isfinite(coords[axis]) || coords[axis]<0 || (double)coords[axis]>=(double)g->dimensions[axis]) return 0;
    return 1;
}
