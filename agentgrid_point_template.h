/* Shared AgentGrid point-coordinate assignment, instantiated for both precisions. */
#ifndef PAL_AG_POINT_TYPE
#error "PAL_AG_POINT_TYPE must be defined"
#endif
#ifndef PAL_AG_POINT_FLOAT
#error "PAL_AG_POINT_FLOAT must be defined"
#endif
#ifndef PAL_AG_POINT_NAME
#error "PAL_AG_POINT_NAME must be defined"
#endif
static void PAL_AG_POINT_NAME(PAL_AG_POINT_TYPE *g,int32_t agent,PAL_AG_POINT_FLOAT x,PAL_AG_POINT_FLOAT y,PAL_AG_POINT_FLOAT z) {
    PAL_AG_POINT_FLOAT coords[3]={x,y,z};
    for (int32_t axis=0;axis<g->dimension;axis++)
        g->floatProperties[(size_t)agent*g->nFloatProperties+axis]=coords[axis];
}
