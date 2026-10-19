/* Shared AgentGrid cell-center position assignment for float32 and float64.
 * Instantiate with PAL_AG_POS_TYPE, PAL_AG_POS_FLOAT and PAL_AG_POS_NAME.
 */
#ifndef PAL_AG_POS_TYPE
#error "PAL_AG_POS_TYPE must be defined"
#endif
#ifndef PAL_AG_POS_FLOAT
#error "PAL_AG_POS_FLOAT must be defined"
#endif
#ifndef PAL_AG_POS_NAME
#error "PAL_AG_POS_NAME must be defined"
#endif
static void PAL_AG_POS_NAME(PAL_AG_POS_TYPE *g,int32_t agent,int32_t location) {
    if (!g->dimension) return;
    int32_t coords[3]={0,0,0};
    if (g->dimension==1) coords[0]=location;
    else if (g->dimension==2) {
        coords[0]=location/g->dimensions[1];
        coords[1]=location%g->dimensions[1];
    } else {
        coords[0]=location/(g->dimensions[1]*g->dimensions[2]);
        coords[1]=(location/g->dimensions[2])%g->dimensions[1];
        coords[2]=location%g->dimensions[2];
    }
    for (int32_t axis=0;axis<g->dimension;axis++)
        g->floatProperties[(size_t)agent*g->nFloatProperties+axis]=(PAL_AG_POS_FLOAT)coords[axis]+(PAL_AG_POS_FLOAT)0.5;
}
