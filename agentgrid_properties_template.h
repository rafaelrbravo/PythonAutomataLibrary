/* Instantiated by pal_native.c with PAL_AG_TYPE, PAL_AG_FLOAT and PAL_AG_PREFIX.
 * Keep precision-specific property operations in one source.
 */
#ifndef PAL_AG_TYPE
#error "PAL_AG_TYPE must be defined"
#endif
#ifndef PAL_AG_FLOAT
#error "PAL_AG_FLOAT must be defined"
#endif
#ifndef PAL_AG_PREFIX
#error "PAL_AG_PREFIX must be defined"
#endif
#define PAL_AG_JOIN2(a,b) a##b
#define PAL_AG_JOIN(a,b) PAL_AG_JOIN2(a,b)
#define PAL_AG_FN(name) PAL_AG_JOIN(PAL_AG_PREFIX,name)

#ifndef PAL_AG_DATA_ONLY
PAL_API int32_t PAL_AG_FN(set_property)(uintptr_t handle,int32_t agent,int32_t property,PAL_AG_FLOAT value) {
    PAL_AG_TYPE *g=(PAL_AG_TYPE *)handle;
    if (!g || agent<0 || agent>=g->nAgents || !g->intProperties[(size_t)agent*N_AGENT_INT_PROPERTIES+AGENT_ALIVE] || property<0 ||
        property>=g->nFloatProperties || !isfinite(value)) return 0;
    g->floatProperties[(size_t)agent*g->nFloatProperties+property]=value;
    return 1;
}
PAL_API PAL_AG_FLOAT PAL_AG_FN(get_property)(uintptr_t handle,int32_t agent,int32_t property) {
    PAL_AG_TYPE *g=(PAL_AG_TYPE *)handle;
    if (!g || agent<0 || agent>=g->nAgents || !g->intProperties[(size_t)agent*N_AGENT_INT_PROPERTIES+AGENT_ALIVE] || property<0 ||
        property>=g->nFloatProperties) return NAN;
    return g->floatProperties[(size_t)agent*g->nFloatProperties+property];
}
PAL_API int32_t PAL_AG_FN(add_property)(uintptr_t handle,int32_t agent,int32_t property,PAL_AG_FLOAT delta) {
    PAL_AG_TYPE *g=(PAL_AG_TYPE *)handle;
    if (!g || agent<0 || agent>=g->nAgents || !g->intProperties[(size_t)agent*N_AGENT_INT_PROPERTIES+AGENT_ALIVE] || property<0 ||
        property>=g->nFloatProperties || !isfinite(delta)) return 0;
    size_t i=(size_t)agent*g->nFloatProperties+property;
    PAL_AG_FLOAT next=g->floatProperties[i]+delta;
    if (!isfinite(next)) return 0;
    g->floatProperties[i]=next;
    return 1;
}

#endif /* PAL_AG_DATA_ONLY */
PAL_API PAL_AG_FLOAT *PAL_AG_FN(float_data)(uintptr_t handle) {
    PAL_AG_TYPE *grid=(PAL_AG_TYPE *)handle;
    return grid ? grid->floatProperties : NULL;
}

#undef PAL_AG_FN
#undef PAL_AG_JOIN
#undef PAL_AG_JOIN2
