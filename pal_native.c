#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <math.h>

#if defined(_WIN32)
#define PAL_API __declspec(dllexport)
#else
#define PAL_API __attribute__((visibility("default")))
#endif

#define PAL_NULL -1

/* Public destructors are declared early because constructors use them on
 * allocation failure. */
PAL_API void pal_ag_free(uintptr_t handle);
PAL_API void pal_pg_free(uintptr_t handle);
PAL_API void pal_pd_free(uintptr_t handle);

/*
 * PAL native core
 * ---------------
 * This file intentionally favors straightforward C over cleverness.
 * Python/Numba owns the user-facing API.  C owns stable storage and workhorse
 * operations so user model functions do not have to compile PAL itself.
 */

/* ========================================================================== */
/* General utilities                                                          */
/* ========================================================================== */

/* RNG is implemented later in this file; IList uses the same PAL stream. */
PAL_API int64_t pal_rand_int(int64_t bound);

PAL_API uint32_t pal_rgb(int32_t r, int32_t g, int32_t b) {
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | (uint32_t)b;
}
PAL_API uint8_t pal_get_red(uint32_t color) { return (uint8_t)((color >> 16) & 255u); }
PAL_API uint8_t pal_get_green(uint32_t color) { return (uint8_t)((color >> 8) & 255u); }
PAL_API uint8_t pal_get_blue(uint32_t color) { return (uint8_t)(color & 255u); }



/* ========================================================================== */
/* IList                                                                  */
/* ========================================================================== */

typedef struct {
    int32_t length;
    int32_t capacity;
    int32_t *items;
} IList;

static int IList_Reserve(IList *list, int32_t required) {
    if (required <= list->capacity) return 1;

    int32_t newCapacity = list->capacity > 0 ? list->capacity : 16;
    while (newCapacity < required) newCapacity *= 2;

    int32_t *newItems = (int32_t *)realloc(list->items, (size_t)newCapacity * sizeof(int32_t));
    if (newItems == NULL) return 0;

    list->items = newItems;
    list->capacity = newCapacity;
    return 1;
}

PAL_API uintptr_t pal_q_new(int32_t unusedDimension) {
    (void)unusedDimension;
    IList *list = (IList *)calloc(1, sizeof(IList));
    return (uintptr_t)list;
}

PAL_API void pal_q_free(uintptr_t handle) {
    IList *list = (IList *)handle;
    if (list == NULL) return;
    free(list->items);
    free(list);
}

PAL_API int32_t pal_q_len(uintptr_t handle) {
    return ((IList *)handle)->length;
}

PAL_API int32_t *pal_q_data(uintptr_t handle) {
    return ((IList *)handle)->items;
}

PAL_API int32_t pal_q_get(uintptr_t handle, int32_t index) {
    IList *list = (IList *)handle;
    if (index < 0 || index >= list->length) return PAL_NULL;
    return list->items[index];
}

PAL_API int64_t pal_q_get_safe(uintptr_t handle, int32_t index) {
    IList *list = (IList *)handle;
    if (index < 0 || index >= list->length) return INT64_MIN;
    return (int64_t)list->items[index];
}

PAL_API int32_t pal_q_copy(uintptr_t handle, int32_t *out) {
    IList *list = (IList *)handle;
    if (list->length > 0) memcpy(out, list->items, (size_t)list->length * sizeof(int32_t));
    return list->length;
}

PAL_API void pal_q_set(uintptr_t handle, int32_t index, int32_t value) {
    IList *list = (IList *)handle;
    if (index >= 0 && index < list->length) list->items[index] = value;
}

PAL_API int32_t pal_q_add(uintptr_t handle, int32_t value) {
    IList *list = (IList *)handle;
    if (!IList_Reserve(list, list->length + 1)) return 0;
    list->items[list->length++] = value;
    return 1;
}

PAL_API void pal_q_clear(uintptr_t handle) {
    ((IList *)handle)->length = 0;
}

PAL_API int32_t pal_q_random(uintptr_t handle) {
    IList *list = (IList *)handle;
    if (list->length <= 0) return PAL_NULL;
    return list->items[pal_rand_int(list->length)];
}

PAL_API void pal_q_shuffle(uintptr_t handle) {
    IList *list = (IList *)handle;
    for (int32_t i = 0; i < list->length - 1; i++) {
        int32_t j = i + (int32_t)pal_rand_int((int64_t)list->length - i);
        int32_t temp = list->items[i]; list->items[i] = list->items[j]; list->items[j] = temp;
    }
}


/* ========================================================================== */
/* AgentGrid                                                                  */
/* ========================================================================== */

enum AgentIntProperty {
    AGENT_ALIVE = 0,
    AGENT_NEXT = 1,
    AGENT_PREV = 2,
    AGENT_LOCATION = 3,
    AGENT_ALIVE_POSITION = 4,
    N_AGENT_INT_PROPERTIES = 5
};

typedef struct {
    int32_t dimension;
    int32_t dimensions[3];
    int32_t wrap[3];
    int32_t length;
    int32_t stackable;

    int32_t nUserProperties;
    int32_t nFloatProperties;
    int32_t capacity;
    int32_t population;
    int32_t nAgents;
    int32_t nDead;
    uint64_t generation;

    float *floatProperties;
    int32_t *intProperties;
    int32_t *aliveAgents;
    int32_t *deadAgents;
    int32_t *grid;
    int32_t *counts;
} AgentGrid;

static int32_t AgentGrid_IntIndex(int32_t agent, int32_t property) {
    return agent * N_AGENT_INT_PROPERTIES + property;
}

static size_t AgentGrid_FloatIndex(const AgentGrid *grid, int32_t agent, int32_t property) {
    return (size_t)agent * (size_t)grid->nFloatProperties + (size_t)property;
}

static int32_t AgentGrid_ToI(const AgentGrid *grid, int32_t x, int32_t y, int32_t z) {
    if (grid->dimension == 1) return x;
    if (grid->dimension == 2) return x * grid->dimensions[1] + y;
    return (x * grid->dimensions[1] + y) * grid->dimensions[2] + z;
}

static int32_t AgentGrid_Wrap(const AgentGrid *grid, int32_t value, int32_t dimension) {
    int32_t size = grid->dimensions[dimension];
    if (value >= 0 && value < size) return value;
    if (!grid->wrap[dimension]) return PAL_NULL;
    value %= size;
    if (value < 0) value += size;
    return value;
}



static int AgentGrid_Grow(AgentGrid *grid) {
    int32_t oldCapacity = grid->capacity;
    if (oldCapacity > INT32_MAX / 2) return 0;
    int32_t newCapacity = oldCapacity > 0 ? oldCapacity * 2 : 1000;

    float *newFloatProperties = (float *)calloc(
        (size_t)newCapacity * (size_t)grid->nFloatProperties, sizeof(float));
    int32_t *newIntProperties = (int32_t *)malloc(
        (size_t)newCapacity * N_AGENT_INT_PROPERTIES * sizeof(int32_t));
    int32_t *newAliveAgents = (int32_t *)malloc((size_t)newCapacity * sizeof(int32_t));
    int32_t *newDeadAgents = (int32_t *)malloc((size_t)newCapacity * sizeof(int32_t));

    if (newFloatProperties == NULL || newIntProperties == NULL ||
        newAliveAgents == NULL || newDeadAgents == NULL) {
        free(newFloatProperties);
        free(newIntProperties);
        free(newAliveAgents);
        free(newDeadAgents);
        return 0;
    }

    memcpy(newFloatProperties, grid->floatProperties,
           (size_t)oldCapacity * (size_t)grid->nFloatProperties * sizeof(float));
    memcpy(newIntProperties, grid->intProperties,
           (size_t)oldCapacity * N_AGENT_INT_PROPERTIES * sizeof(int32_t));
    memcpy(newAliveAgents, grid->aliveAgents, (size_t)grid->population * sizeof(int32_t));
    memcpy(newDeadAgents, grid->deadAgents, (size_t)grid->nDead * sizeof(int32_t));

    for (int32_t agent = oldCapacity; agent < newCapacity; agent++) {
        for (int32_t property = 0; property < N_AGENT_INT_PROPERTIES; property++)
            newIntProperties[AgentGrid_IntIndex(agent, property)] = PAL_NULL;
        newIntProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE)] = 0;
    }

    free(grid->floatProperties);
    free(grid->intProperties);
    free(grid->aliveAgents);
    free(grid->deadAgents);
    grid->floatProperties = newFloatProperties;
    grid->intProperties = newIntProperties;
    grid->aliveAgents = newAliveAgents;
    grid->deadAgents = newDeadAgents;
    grid->capacity = newCapacity;
    return 1;
}

static int32_t AgentGrid_AllocateAgent(AgentGrid *grid) {
    int32_t agent;

    if (grid->nDead > 0) {
        agent = grid->deadAgents[--grid->nDead];
    } else {
        agent = grid->nAgents;
        if (agent >= grid->capacity && !AgentGrid_Grow(grid)) return PAL_NULL;
        grid->nAgents++;
    }

    grid->intProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE)] = 1;
    grid->intProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE_POSITION)] = grid->population;
    grid->aliveAgents[grid->population++] = agent;
    return agent;
}

static void AgentGrid_PutAgent(AgentGrid *grid, int32_t agent, int32_t location) {
    grid->intProperties[AgentGrid_IntIndex(agent, AGENT_LOCATION)] = location;
    if (grid->dimension == 0) return;

    if (grid->stackable) {
        int32_t previous = grid->grid[location];
        grid->intProperties[AgentGrid_IntIndex(agent, AGENT_PREV)] = previous;
        grid->intProperties[AgentGrid_IntIndex(agent, AGENT_NEXT)] = PAL_NULL;
        if (previous != PAL_NULL)
            grid->intProperties[AgentGrid_IntIndex(previous, AGENT_NEXT)] = agent;
        grid->grid[location] = agent;
        grid->counts[location]++;
    } else {
        grid->grid[location] = agent;
    }
}

static void AgentGrid_RemoveAgent(AgentGrid *grid, int32_t agent) {
    if (grid->dimension == 0) return;
    int32_t location = grid->intProperties[AgentGrid_IntIndex(agent, AGENT_LOCATION)];

    if (grid->stackable) {
        int32_t previous = grid->intProperties[AgentGrid_IntIndex(agent, AGENT_PREV)];
        int32_t next = grid->intProperties[AgentGrid_IntIndex(agent, AGENT_NEXT)];
        if (previous != PAL_NULL)
            grid->intProperties[AgentGrid_IntIndex(previous, AGENT_NEXT)] = next;
        if (next != PAL_NULL)
            grid->intProperties[AgentGrid_IntIndex(next, AGENT_PREV)] = previous;
        if (grid->grid[location] == agent) grid->grid[location] = previous;
        grid->counts[location]--;
        grid->intProperties[AgentGrid_IntIndex(agent, AGENT_PREV)] = PAL_NULL;
        grid->intProperties[AgentGrid_IntIndex(agent, AGENT_NEXT)] = PAL_NULL;
    } else {
        grid->grid[location] = PAL_NULL;
    }
}

static void AgentGrid_SetSquarePosition(AgentGrid *grid, int32_t agent, int32_t location) {
    if (grid->dimension == 0) return;

    int32_t x = 0, y = 0, z = 0;
    if (grid->dimension == 1) {
        x = location;
    } else if (grid->dimension == 2) {
        x = location / grid->dimensions[1];
        y = location % grid->dimensions[1];
    } else {
        x = location / (grid->dimensions[1] * grid->dimensions[2]);
        y = (location / grid->dimensions[2]) % grid->dimensions[1];
        z = location % grid->dimensions[2];
    }

    grid->floatProperties[AgentGrid_FloatIndex(grid, agent, 0)] = (float)x + 0.5f;
    if (grid->dimension > 1)
        grid->floatProperties[AgentGrid_FloatIndex(grid, agent, 1)] = (float)y + 0.5f;
    if (grid->dimension > 2)
        grid->floatProperties[AgentGrid_FloatIndex(grid, agent, 2)] = (float)z + 0.5f;
}

static int32_t AgentGrid_PointToI(const AgentGrid *grid, float x, float y, float z) {
    int32_t xi = (int32_t)floorf(x);
    int32_t yi = grid->dimension > 1 ? (int32_t)floorf(y) : 0;
    int32_t zi = grid->dimension > 2 ? (int32_t)floorf(z) : 0;
    return AgentGrid_ToI(grid, xi, yi, zi);
}

static int AgentGrid_PointInBounds(const AgentGrid *grid, float x, float y, float z) {
    if (!isfinite(x) || x < 0.0f || x >= (float)grid->dimensions[0]) return 0;
    if (grid->dimension > 1 && (!isfinite(y) || y < 0.0f || y >= (float)grid->dimensions[1])) return 0;
    if (grid->dimension > 2 && (!isfinite(z) || z < 0.0f || z >= (float)grid->dimensions[2])) return 0;
    return 1;
}

static void AgentGrid_SetPointPosition(AgentGrid *grid, int32_t agent, float x, float y, float z) {
    grid->floatProperties[AgentGrid_FloatIndex(grid, agent, 0)] = x;
    if (grid->dimension > 1) grid->floatProperties[AgentGrid_FloatIndex(grid, agent, 1)] = y;
    if (grid->dimension > 2) grid->floatProperties[AgentGrid_FloatIndex(grid, agent, 2)] = z;
}

PAL_API uintptr_t pal_ag_new(const int32_t *dimensions, int32_t dimension,
                             int32_t nUserProperties, int32_t stackable) {
    if (dimension < 0 || dimension > 3 || nUserProperties < 0 || nUserProperties > INT32_MAX - dimension) return 0;

    AgentGrid *grid = (AgentGrid *)calloc(1, sizeof(AgentGrid));
    if (grid == NULL) return 0;

    grid->dimension = dimension;
    grid->stackable = stackable != 0;
    grid->nUserProperties = nUserProperties;
    grid->nFloatProperties = dimension + nUserProperties;
    grid->capacity = 1000;
    grid->length = dimension == 0 ? 0 : 1;

    for (int32_t d = 0; d < dimension; d++) {
        int32_t value = dimensions[d];
        grid->wrap[d] = value < 0;
        grid->dimensions[d] = value < 0 ? -value : value;
        if (grid->dimensions[d] <= 0) { free(grid); return 0; }
        grid->length *= grid->dimensions[d];
    }

    grid->floatProperties = (float *)calloc(
        (size_t)grid->capacity * (size_t)grid->nFloatProperties, sizeof(float));
    grid->intProperties = (int32_t *)malloc(
        (size_t)grid->capacity * N_AGENT_INT_PROPERTIES * sizeof(int32_t));
    grid->aliveAgents = (int32_t *)malloc((size_t)grid->capacity * sizeof(int32_t));
    grid->deadAgents = (int32_t *)malloc((size_t)grid->capacity * sizeof(int32_t));

    if (grid->floatProperties == NULL || grid->intProperties == NULL ||
        grid->aliveAgents == NULL || grid->deadAgents == NULL) {
        pal_ag_free((uintptr_t)grid); return 0;
    }

    for (int32_t agent = 0; agent < grid->capacity; agent++) {
        for (int32_t property = 0; property < N_AGENT_INT_PROPERTIES; property++)
            grid->intProperties[AgentGrid_IntIndex(agent, property)] = PAL_NULL;
        grid->intProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE)] = 0;
    }

    if (dimension > 0) {
        grid->grid = (int32_t *)malloc((size_t)grid->length * sizeof(int32_t));
        if (grid->grid == NULL) { pal_ag_free((uintptr_t)grid); return 0; }
        for (int32_t i = 0; i < grid->length; i++) grid->grid[i] = PAL_NULL;

        if (grid->stackable) {
            grid->counts = (int32_t *)calloc((size_t)grid->length, sizeof(int32_t));
            if (grid->counts == NULL) { pal_ag_free((uintptr_t)grid); return 0; }
        }
    }

    return (uintptr_t)grid;
}

PAL_API void pal_ag_free(uintptr_t handle) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (grid == NULL) return;
    free(grid->floatProperties);
    free(grid->intProperties);
    free(grid->aliveAgents);
    free(grid->deadAgents);
    free(grid->grid);
    free(grid->counts);
    free(grid);
}

PAL_API int32_t pal_ag_dim(uintptr_t handle, int32_t dimension) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (dimension < 0 || dimension >= grid->dimension) return PAL_NULL;
    return grid->dimensions[dimension];
}
static double AgentGrid_InWrap(const AgentGrid *grid, double value, int32_t dimension) {
    double size=(double)grid->dimensions[dimension];
    if(value>=0.0 && value<size) return value;
    if(!grid->wrap[dimension]) return (double)PAL_NULL;
    value=fmod(value,size); if(value<0.0) value+=size; return value;
}
static double AgentGrid_DispWrap(const AgentGrid *grid, double a, double b, int32_t dimension) {
    double delta=b-a;
    if(!grid->wrap[dimension]) return delta;
    double size=(double)grid->dimensions[dimension];
    if(fabs(delta)*2.0>size) return delta>0.0?delta-size:delta+size;
    return delta;
}
PAL_API int32_t pal_ag_inwrap_sq_x(uintptr_t h,int32_t v){return AgentGrid_Wrap((AgentGrid*)h,v,0);}
PAL_API int32_t pal_ag_inwrap_sq_y(uintptr_t h,int32_t v){return AgentGrid_Wrap((AgentGrid*)h,v,1);}
PAL_API int32_t pal_ag_inwrap_sq_z(uintptr_t h,int32_t v){return AgentGrid_Wrap((AgentGrid*)h,v,2);}
PAL_API double pal_ag_inwrap_x(uintptr_t h,double v){return AgentGrid_InWrap((AgentGrid*)h,v,0);}
PAL_API double pal_ag_inwrap_y(uintptr_t h,double v){return AgentGrid_InWrap((AgentGrid*)h,v,1);}
PAL_API double pal_ag_inwrap_z(uintptr_t h,double v){return AgentGrid_InWrap((AgentGrid*)h,v,2);}
PAL_API double pal_ag_dispwrap_x(uintptr_t h,double a,double b){return AgentGrid_DispWrap((AgentGrid*)h,a,b,0);}
PAL_API double pal_ag_dispwrap_y(uintptr_t h,double a,double b){return AgentGrid_DispWrap((AgentGrid*)h,a,b,1);}
PAL_API double pal_ag_dispwrap_z(uintptr_t h,double a,double b){return AgentGrid_DispWrap((AgentGrid*)h,a,b,2);}


/* Fused safe entry points: validate and perform the operation in one native call. */
#define PAL_BAD_I INT32_MIN
static int AgentGrid_ValidAlive(const AgentGrid *g,int32_t a){return a>=0 && a<g->capacity && g->intProperties[AgentGrid_IntIndex(a,AGENT_ALIVE)]!=0;}
PAL_API int32_t pal_ag_i_safe(uintptr_t h,int32_t a){AgentGrid*g=(AgentGrid*)h;return AgentGrid_ValidAlive(g,a)?g->intProperties[AgentGrid_IntIndex(a,AGENT_LOCATION)]:PAL_BAD_I;}
PAL_API int32_t pal_ag_xsq_safe(uintptr_t h,int32_t a){AgentGrid*g=(AgentGrid*)h;if(!AgentGrid_ValidAlive(g,a)||g->dimension<1)return PAL_BAD_I;int32_t i=g->intProperties[AgentGrid_IntIndex(a,AGENT_LOCATION)];return g->dimension==1?i:g->dimension==2?i/g->dimensions[1]:i/(g->dimensions[1]*g->dimensions[2]);}
PAL_API int32_t pal_ag_ysq_safe(uintptr_t h,int32_t a){AgentGrid*g=(AgentGrid*)h;if(!AgentGrid_ValidAlive(g,a)||g->dimension<2)return PAL_BAD_I;int32_t i=g->intProperties[AgentGrid_IntIndex(a,AGENT_LOCATION)];return g->dimension==2?i%g->dimensions[1]:(i/g->dimensions[2])%g->dimensions[1];}
PAL_API int32_t pal_ag_zsq_safe(uintptr_t h,int32_t a){AgentGrid*g=(AgentGrid*)h;if(!AgentGrid_ValidAlive(g,a)||g->dimension<3)return PAL_BAD_I;return g->intProperties[AgentGrid_IntIndex(a,AGENT_LOCATION)]%g->dimensions[2];}
PAL_API float pal_ag_x_safe(uintptr_t h,int32_t a){AgentGrid*g=(AgentGrid*)h;return AgentGrid_ValidAlive(g,a)&&g->dimension>0?g->floatProperties[AgentGrid_FloatIndex(g,a,0)]:NAN;}
PAL_API float pal_ag_y_safe(uintptr_t h,int32_t a){AgentGrid*g=(AgentGrid*)h;return AgentGrid_ValidAlive(g,a)&&g->dimension>1?g->floatProperties[AgentGrid_FloatIndex(g,a,1)]:NAN;}
PAL_API float pal_ag_z_safe(uintptr_t h,int32_t a){AgentGrid*g=(AgentGrid*)h;return AgentGrid_ValidAlive(g,a)&&g->dimension>2?g->floatProperties[AgentGrid_FloatIndex(g,a,2)]:NAN;}
PAL_API float pal_ag_getp_safe(uintptr_t h,int32_t a,int32_t p){AgentGrid*g=(AgentGrid*)h;return AgentGrid_ValidAlive(g,a)&&p>=0&&p<g->nFloatProperties-g->dimension?g->floatProperties[AgentGrid_FloatIndex(g,a,g->dimension+p)]:NAN;}
PAL_API int32_t pal_ag_setp_safe(uintptr_t h,int32_t a,int32_t p,float v){AgentGrid*g=(AgentGrid*)h;if(!AgentGrid_ValidAlive(g,a)||p<0||p>=g->nFloatProperties-g->dimension||!isfinite(v))return 0;g->floatProperties[AgentGrid_FloatIndex(g,a,g->dimension+p)]=v;return 1;}
PAL_API double pal_ag_inwrap_safe(uintptr_t h,double v,int32_t d){AgentGrid*g=(AgentGrid*)h;if(d<0||d>=g->dimension||!isfinite(v))return NAN;return AgentGrid_InWrap(g,v,d);}
PAL_API double pal_ag_dispwrap_safe(uintptr_t h,double a,double b,int32_t d){AgentGrid*g=(AgentGrid*)h;if(d<0||d>=g->dimension||!isfinite(a)||!isfinite(b)||a<0||b<0||a>=g->dimensions[d]||b>=g->dimensions[d])return NAN;return AgentGrid_DispWrap(g,a,b,d);}
PAL_API int32_t pal_ag_inwrap_sq_safe(uintptr_t h,int32_t v,int32_t d){AgentGrid*g=(AgentGrid*)h;if(d<0||d>=g->dimension)return PAL_BAD_I;return AgentGrid_Wrap(g,v,d);}
PAL_API int32_t pal_ag_toi_safe(uintptr_t h,int32_t x,int32_t y,int32_t z){AgentGrid*g=(AgentGrid*)h;if(g->dimension<1||x<0||x>=g->dimensions[0])return PAL_BAD_I;if(g->dimension==1){if(y!=-1||z!=-1)return PAL_BAD_I;}else if(y<0||y>=g->dimensions[1])return PAL_BAD_I;if(g->dimension==2){if(z!=-1)return PAL_BAD_I;}else if(g->dimension==3&&(z<0||z>=g->dimensions[2]))return PAL_BAD_I;return AgentGrid_ToI(g,x,y,z);}
PAL_API int32_t pal_ag_itox_safe(uintptr_t h,int32_t i){AgentGrid*g=(AgentGrid*)h;if(g->dimension<2||i<0||i>=g->length)return PAL_BAD_I;return g->dimension==2?i/g->dimensions[1]:i/(g->dimensions[1]*g->dimensions[2]);}
PAL_API int32_t pal_ag_itoy_safe(uintptr_t h,int32_t i){AgentGrid*g=(AgentGrid*)h;if(g->dimension<2||i<0||i>=g->length)return PAL_BAD_I;return g->dimension==2?i%g->dimensions[1]:(i/g->dimensions[2])%g->dimensions[1];}
PAL_API int32_t pal_ag_itoz_safe(uintptr_t h,int32_t i){AgentGrid*g=(AgentGrid*)h;if(g->dimension<3||i<0||i>=g->length)return PAL_BAD_I;return i%g->dimensions[2];}
PAL_API int32_t pal_ag_pop(uintptr_t handle) { return ((AgentGrid *)handle)->population; }
PAL_API int32_t pal_ag_len(uintptr_t handle) { return ((AgentGrid *)handle)->length; }
PAL_API int32_t pal_ag_toi(uintptr_t handle, int32_t x, int32_t y, int32_t z) {
    return AgentGrid_ToI((AgentGrid *)handle, x, y, z);
}
PAL_API int32_t pal_ag_itox(uintptr_t handle, int32_t i) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (grid->dimension == 1) return i;
    if (grid->dimension == 2) return i / grid->dimensions[1];
    return i / (grid->dimensions[1] * grid->dimensions[2]);
}
PAL_API int32_t pal_ag_itoy(uintptr_t handle, int32_t i) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (grid->dimension == 2) return i % grid->dimensions[1];
    return (i / grid->dimensions[2]) % grid->dimensions[1];
}
PAL_API int32_t pal_ag_itoz(uintptr_t handle, int32_t i) {
    AgentGrid *grid = (AgentGrid *)handle;
    return i % grid->dimensions[2];
}
PAL_API int32_t pal_ag_all_copy(uintptr_t handle, int32_t *out, int32_t shuffle) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (grid->population > 0) memcpy(out, grid->aliveAgents, (size_t)grid->population * sizeof(int32_t));
    if (shuffle) {
        for (int32_t i = 0; i < grid->population - 1; i++) {
            int32_t j = i + (int32_t)pal_rand_int((int64_t)grid->population - i);
            int32_t temp = out[i]; out[i] = out[j]; out[j] = temp;
        }
    }
    return grid->population;
}
PAL_API int32_t pal_ag_alive(uintptr_t handle, int32_t agent) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (agent < 0 || agent >= grid->capacity) return 0;
    return grid->intProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE)] != 0;
}

PAL_API int32_t pal_ag_alive_safe(uintptr_t h,int32_t a){AgentGrid*g=(AgentGrid*)h;if(a<0||a>=g->nAgents)return -1;return AgentGrid_ValidAlive(g,a);}

PAL_API int32_t pal_ag_nagents(uintptr_t handle) { return ((AgentGrid *)handle)->nAgents; }
PAL_API int32_t *pal_ag_grid_data(uintptr_t handle) { return ((AgentGrid *)handle)->grid; }
PAL_API int32_t *pal_ag_int_data(uintptr_t handle) { return ((AgentGrid *)handle)->intProperties; }
PAL_API float *pal_ag_float_data(uintptr_t handle) { return ((AgentGrid *)handle)->floatProperties; }
PAL_API int32_t pal_ag_nfloatprops(uintptr_t handle) { return ((AgentGrid *)handle)->nFloatProperties; }
PAL_API int32_t pal_ag_stackable(uintptr_t handle) { return ((AgentGrid *)handle)->stackable; }
PAL_API int32_t pal_ag_wrap(uintptr_t handle, int32_t dimension) { AgentGrid*g=(AgentGrid*)handle; return dimension>=0&&dimension<g->dimension?g->wrap[dimension]:0; }
PAL_API uint64_t pal_ag_generation(uintptr_t handle) { return ((AgentGrid *)handle)->generation; }

PAL_API int32_t pal_ag_new_i(uintptr_t handle, int32_t location) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (grid->dimension > 0) {
        if (location < 0 || location >= grid->length) return PAL_NULL;
        if (!grid->stackable && grid->grid[location] != PAL_NULL) return PAL_NULL;
    }

    int32_t agent = AgentGrid_AllocateAgent(grid);
    if (agent == PAL_NULL) return PAL_NULL;
    AgentGrid_PutAgent(grid, agent, location);
    AgentGrid_SetSquarePosition(grid, agent, location);
    grid->generation++;
    return agent;
}

PAL_API int32_t pal_ag_new_sq(uintptr_t handle, int32_t x, int32_t y, int32_t z) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (grid->dimension == 0) return pal_ag_new_i(handle, PAL_NULL);

    x = AgentGrid_Wrap(grid, x, 0);
    if (x == PAL_NULL) return PAL_NULL;
    if (grid->dimension > 1) { y = AgentGrid_Wrap(grid, y, 1); if (y == PAL_NULL) return PAL_NULL; }
    if (grid->dimension > 2) { z = AgentGrid_Wrap(grid, z, 2); if (z == PAL_NULL) return PAL_NULL; }
    return pal_ag_new_i(handle, AgentGrid_ToI(grid, x, y, z));
}

PAL_API int32_t pal_ag_new_pt(uintptr_t handle, float x, float y, float z) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (grid->dimension == 0 || !AgentGrid_PointInBounds(grid, x, y, z)) return PAL_NULL;
    int32_t location = AgentGrid_PointToI(grid, x, y, z);
    if (!grid->stackable && grid->grid[location] != PAL_NULL) return PAL_NULL;
    int32_t agent = AgentGrid_AllocateAgent(grid);
    if (agent == PAL_NULL) return PAL_NULL;
    AgentGrid_PutAgent(grid, agent, location);
    AgentGrid_SetPointPosition(grid, agent, x, y, z);
    grid->generation++;
    return agent;
}

PAL_API void pal_ag_dispose(uintptr_t handle, int32_t agent) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (!pal_ag_alive(handle, agent)) return;

    AgentGrid_RemoveAgent(grid, agent);

    int32_t alivePosition = grid->intProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE_POSITION)];
    int32_t lastAgent = grid->aliveAgents[grid->population - 1];
    grid->population--;

    if (alivePosition < grid->population) {
        grid->aliveAgents[alivePosition] = lastAgent;
        grid->intProperties[AgentGrid_IntIndex(lastAgent, AGENT_ALIVE_POSITION)] = alivePosition;
    }

    grid->intProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE)] = 0;
    grid->intProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE_POSITION)] = PAL_NULL;
    grid->deadAgents[grid->nDead++] = agent;
    grid->generation++;
}

PAL_API int32_t pal_ag_i(uintptr_t handle, int32_t agent) {
    AgentGrid *grid = (AgentGrid *)handle;
    return grid->intProperties[AgentGrid_IntIndex(agent, AGENT_LOCATION)];
}

PAL_API int32_t pal_ag_xsq(uintptr_t handle, int32_t agent) {
    AgentGrid *grid = (AgentGrid *)handle;
    int32_t location = pal_ag_i(handle, agent);
    if (grid->dimension == 1) return location;
    if (grid->dimension == 2) return location / grid->dimensions[1];
    return location / (grid->dimensions[1] * grid->dimensions[2]);
}

PAL_API int32_t pal_ag_ysq(uintptr_t handle, int32_t agent) {
    AgentGrid *grid = (AgentGrid *)handle;
    int32_t location = pal_ag_i(handle, agent);
    if (grid->dimension == 2) return location % grid->dimensions[1];
    return (location / grid->dimensions[2]) % grid->dimensions[1];
}

PAL_API int32_t pal_ag_zsq(uintptr_t handle, int32_t agent) {
    AgentGrid *grid = (AgentGrid *)handle;
    return pal_ag_i(handle, agent) % grid->dimensions[2];
}

PAL_API float pal_ag_x(uintptr_t handle, int32_t agent) {
    AgentGrid *grid = (AgentGrid *)handle;
    return grid->floatProperties[AgentGrid_FloatIndex(grid, agent, 0)];
}
PAL_API float pal_ag_y(uintptr_t handle, int32_t agent) {
    AgentGrid *grid = (AgentGrid *)handle;
    return grid->floatProperties[AgentGrid_FloatIndex(grid, agent, 1)];
}
PAL_API float pal_ag_z(uintptr_t handle, int32_t agent) {
    AgentGrid *grid = (AgentGrid *)handle;
    return grid->floatProperties[AgentGrid_FloatIndex(grid, agent, 2)];
}

PAL_API float pal_ag_getp(uintptr_t handle, int32_t agent, int32_t property) {
    AgentGrid *grid = (AgentGrid *)handle;
    return grid->floatProperties[AgentGrid_FloatIndex(grid, agent, grid->dimension + property)];
}

PAL_API void pal_ag_setp(uintptr_t handle, int32_t agent, int32_t property, float value) {
    AgentGrid *grid = (AgentGrid *)handle;
    grid->floatProperties[AgentGrid_FloatIndex(grid, agent, grid->dimension + property)] = value;
}

PAL_API int32_t pal_ag_all_id(uintptr_t handle, int32_t position) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (position < 0 || position >= grid->population) return PAL_NULL;
    return grid->aliveAgents[position];
}

PAL_API int32_t pal_ag_count_i(uintptr_t handle, int32_t location) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (location < 0 || location >= grid->length) return 0;
    if (grid->stackable) return grid->counts[location];
    return grid->grid[location] != PAL_NULL;
}

PAL_API int32_t pal_ag_last_i(uintptr_t handle, int32_t location) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (location < 0 || location >= grid->length) return PAL_NULL;
    return grid->grid[location];
}

PAL_API int32_t pal_ag_move_i(uintptr_t handle, int32_t agent, int32_t location) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (!pal_ag_alive(handle, agent)) return 0;
    if (location < 0 || location >= grid->length) return 0;
    if (!grid->stackable && grid->grid[location] != PAL_NULL && grid->grid[location] != agent) return 0;

    AgentGrid_RemoveAgent(grid, agent);
    AgentGrid_PutAgent(grid, agent, location);
    AgentGrid_SetSquarePosition(grid, agent, location);
    grid->generation++;
    return 1;
}

PAL_API int32_t pal_ag_move_pt(uintptr_t handle, int32_t agent, float x, float y, float z) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (!pal_ag_alive(handle, agent) || grid->dimension == 0 || !AgentGrid_PointInBounds(grid, x, y, z)) return 0;
    int32_t location = AgentGrid_PointToI(grid, x, y, z);
    int32_t oldLocation = grid->intProperties[AgentGrid_IntIndex(agent, AGENT_LOCATION)];
    if (!grid->stackable && grid->grid[location] != PAL_NULL && grid->grid[location] != agent) return 0;
    if (location != oldLocation) {
        AgentGrid_RemoveAgent(grid, agent);
        AgentGrid_PutAgent(grid, agent, location);
    }
    AgentGrid_SetPointPosition(grid, agent, x, y, z);
    grid->generation++;
    return 1;
}

/* Fast entry points: these intentionally perform no argument validation. */
PAL_API int32_t pal_ag_new_i_(uintptr_t handle, int32_t location) {
    AgentGrid *grid = (AgentGrid *)handle;
    int32_t agent = AgentGrid_AllocateAgent(grid);
    if (agent == PAL_NULL) return PAL_NULL; /* allocation failure is not a safety check */
    AgentGrid_PutAgent(grid, agent, location);
    AgentGrid_SetSquarePosition(grid, agent, location);
    grid->generation++;
    return agent;
}

PAL_API int32_t pal_ag_new_sq_(uintptr_t handle, int32_t x, int32_t y, int32_t z) {
    AgentGrid *grid = (AgentGrid *)handle;
    if (grid->dimension == 0) return pal_ag_new_i_(handle, PAL_NULL);
    return pal_ag_new_i_(handle, AgentGrid_ToI(grid, x, y, z));
}

PAL_API int32_t pal_ag_new_pt_(uintptr_t handle, float x, float y, float z) {
    AgentGrid *grid = (AgentGrid *)handle;
    int32_t location = AgentGrid_PointToI(grid, x, y, z);
    int32_t agent = AgentGrid_AllocateAgent(grid);
    if (agent == PAL_NULL) return PAL_NULL;
    AgentGrid_PutAgent(grid, agent, location);
    AgentGrid_SetPointPosition(grid, agent, x, y, z);
    grid->generation++;
    return agent;
}

PAL_API void pal_ag_dispose_(uintptr_t handle, int32_t agent) {
    AgentGrid *grid = (AgentGrid *)handle;
    AgentGrid_RemoveAgent(grid, agent);

    int32_t alivePosition = grid->intProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE_POSITION)];
    int32_t lastAgent = grid->aliveAgents[grid->population - 1];
    grid->population--;

    if (alivePosition < grid->population) {
        grid->aliveAgents[alivePosition] = lastAgent;
        grid->intProperties[AgentGrid_IntIndex(lastAgent, AGENT_ALIVE_POSITION)] = alivePosition;
    }

    grid->intProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE)] = 0;
    grid->intProperties[AgentGrid_IntIndex(agent, AGENT_ALIVE_POSITION)] = PAL_NULL;
    grid->deadAgents[grid->nDead++] = agent;
    grid->generation++;
}

PAL_API int32_t pal_ag_move_i_(uintptr_t handle, int32_t agent, int32_t location) {
    AgentGrid *grid = (AgentGrid *)handle;
    AgentGrid_RemoveAgent(grid, agent);
    AgentGrid_PutAgent(grid, agent, location);
    AgentGrid_SetSquarePosition(grid, agent, location);
    grid->generation++;
    return 1;
}

PAL_API int32_t pal_ag_move_pt_(uintptr_t handle, int32_t agent, float x, float y, float z) {
    AgentGrid *grid = (AgentGrid *)handle;
    int32_t location = AgentGrid_PointToI(grid, x, y, z);
    int32_t oldLocation = grid->intProperties[AgentGrid_IntIndex(agent, AGENT_LOCATION)];
    if (location != oldLocation) {
        AgentGrid_RemoveAgent(grid, agent);
        AgentGrid_PutAgent(grid, agent, location);
    }
    AgentGrid_SetPointPosition(grid, agent, x, y, z);
    grid->generation++;
    return 1;
}

PAL_API float pal_ag_getp_(uintptr_t handle, int32_t agent, int32_t property) {
    AgentGrid *grid = (AgentGrid *)handle;
    return grid->floatProperties[AgentGrid_FloatIndex(grid, agent, grid->dimension + property)];
}

PAL_API void pal_ag_setp_(uintptr_t handle, int32_t agent, int32_t property, float value) {
    AgentGrid *grid = (AgentGrid *)handle;
    grid->floatProperties[AgentGrid_FloatIndex(grid, agent, grid->dimension + property)] = value;
}

PAL_API int32_t pal_ag_count_i_(uintptr_t handle, int32_t location) {
    AgentGrid *grid = (AgentGrid *)handle;
    return grid->stackable ? grid->counts[location] : grid->grid[location] != PAL_NULL;
}

PAL_API int32_t pal_ag_last_i_(uintptr_t handle, int32_t location) {
    return ((AgentGrid *)handle)->grid[location];
}


static int32_t AgentGrid_AddLocationToQuery(AgentGrid *grid, IList *list, int32_t location) {
    if (location < 0 || location >= grid->length) return list->length;
    int32_t agent=grid->grid[location];
    while(agent!=PAL_NULL){
        if(!IList_Reserve(list,list->length+1)) return PAL_NULL;
        list->items[list->length++]=agent;
        if(!grid->stackable) break;
        agent=grid->intProperties[AgentGrid_IntIndex(agent,AGENT_PREV)];
    }
    return list->length;
}
PAL_API int32_t pal_ag_add_i_q(uintptr_t gh,uintptr_t qh,int32_t i){return AgentGrid_AddLocationToQuery((AgentGrid*)gh,(IList*)qh,i);}
PAL_API int32_t pal_ag_get_i_q(uintptr_t gh,uintptr_t qh,int32_t i){IList*q=(IList*)qh;q->length=0;return AgentGrid_AddLocationToQuery((AgentGrid*)gh,q,i);}
PAL_API int32_t pal_ag_add_q(uintptr_t gh,uintptr_t qh,int32_t x,int32_t y,int32_t z){AgentGrid*g=(AgentGrid*)gh;
    if(x<0||x>=g->dimensions[0])return PAL_NULL;
    if(g->dimension==1){if(y!=-1||z!=-1)return PAL_NULL;}else if(y<0||y>=g->dimensions[1])return PAL_NULL;
    if(g->dimension==2){if(z!=-1)return PAL_NULL;}else if(g->dimension==3&&(z<0||z>=g->dimensions[2]))return PAL_NULL;
    return AgentGrid_AddLocationToQuery(g,(IList*)qh,AgentGrid_ToI(g,x,y,z));}
PAL_API int32_t pal_ag_get_q(uintptr_t gh,uintptr_t qh,int32_t x,int32_t y,int32_t z){IList*q=(IList*)qh;q->length=0;return pal_ag_add_q(gh,qh,x,y,z);}
static float AgentGrid_WrappedDelta(float d,int32_t size,int32_t wrap){if(!wrap)return d;float half=0.5f*(float)size;if(d>half)d-=(float)size;else if(d<-half)d+=(float)size;return d;}
PAL_API int32_t pal_ag_add_radius_q(uintptr_t gh,uintptr_t qh,double radius,double x,double y,double z,int32_t exclude){
    AgentGrid*g=(AgentGrid*)gh;IList*q=(IList*)qh;
    if(radius<0.0||!isfinite(radius)||!isfinite(x)||
       (g->dimension>1&&!isfinite(y))||(g->dimension>2&&!isfinite(z))||
       (exclude!=-1&&(exclude<0||exclude>=g->nAgents))) return PAL_NULL;
    if(x-radius<(double)INT32_MIN || x+radius>(double)INT32_MAX ||
       (g->dimension>1&&(y-radius<(double)INT32_MIN||y+radius>(double)INT32_MAX)) ||
       (g->dimension>2&&(z-radius<(double)INT32_MIN||z+radius>(double)INT32_MAX))) return PAL_NULL;
    double r2=radius*radius;if(!isfinite(r2))return PAL_NULL;
    int32_t x1=(int32_t)floor(x-radius),x2=(int32_t)floor(x+radius);
    int32_t y1=g->dimension>1?(int32_t)floor(y-radius):0,y2=g->dimension>1?(int32_t)floor(y+radius):0;
    int32_t z1=g->dimension>2?(int32_t)floor(z-radius):0,z2=g->dimension>2?(int32_t)floor(z+radius):0;
    int64_t nx=(int64_t)x2-x1+1,ny=g->dimension>1?(int64_t)y2-y1+1:1,nz=g->dimension>2?(int64_t)z2-z1+1:1;
    if(g->wrap[0]&&nx>g->dimensions[0])nx=g->dimensions[0];
    if(g->dimension>1&&g->wrap[1]&&ny>g->dimensions[1])ny=g->dimensions[1];
    if(g->dimension>2&&g->wrap[2]&&nz>g->dimensions[2])nz=g->dimensions[2];
    for(int64_t xo=0;xo<nx;xo++){int32_t xi=(int32_t)((int64_t)x1+xo);int32_t xx=AgentGrid_Wrap(g,xi,0);if(xx==PAL_NULL)continue;
      for(int64_t yo=0;yo<ny;yo++){int32_t yi=(int32_t)((int64_t)y1+yo);int32_t yy=g->dimension>1?AgentGrid_Wrap(g,yi,1):0;if(yy==PAL_NULL)continue;
       for(int64_t zo=0;zo<nz;zo++){int32_t zi=(int32_t)((int64_t)z1+zo);int32_t zz=g->dimension>2?AgentGrid_Wrap(g,zi,2):0;if(zz==PAL_NULL)continue;int32_t loc=AgentGrid_ToI(g,xx,yy,zz);int32_t a=g->grid[loc];
        if(a!=PAL_NULL){int32_t addCap=g->stackable?g->counts[loc]:1;if(addCap>INT32_MAX-q->length||!IList_Reserve(q,q->length+addCap))return PAL_NULL;}while(a!=PAL_NULL){float dx=AgentGrid_WrappedDelta(g->floatProperties[AgentGrid_FloatIndex(g,a,0)]-(float)x,g->dimensions[0],g->wrap[0]);float dy=0, dz=0;if(g->dimension>1)dy=AgentGrid_WrappedDelta(g->floatProperties[AgentGrid_FloatIndex(g,a,1)]-(float)y,g->dimensions[1],g->wrap[1]);if(g->dimension>2)dz=AgentGrid_WrappedDelta(g->floatProperties[AgentGrid_FloatIndex(g,a,2)]-(float)z,g->dimensions[2],g->wrap[2]);if(a!=exclude && (double)dx*dx+(double)dy*dy+(double)dz*dz<=r2){q->items[q->length++]=a;}if(!g->stackable)break;a=g->intProperties[AgentGrid_IntIndex(a,AGENT_PREV)];}
    }}}return q->length;
}
PAL_API int32_t pal_ag_get_radius_q(uintptr_t gh,uintptr_t qh,double radius,double x,double y,double z,int32_t exclude){((IList*)qh)->length=0;return pal_ag_add_radius_q(gh,qh,radius,x,y,z,exclude);}

/* ========================================================================== */
/* RNG + Multinomial                                                          */
/* ========================================================================== */

/* PAL owns one PCG32 stream per execution thread. Seed() changes the master
 * seed and bumps an epoch; each thread lazily rebuilds its stream on its next
 * draw. Stream IDs are assigned once per thread. */
#if defined(_MSC_VER)
#define PAL_TLS __declspec(thread)
#else
#define PAL_TLS _Thread_local
#endif

static uint64_t pal_rng_master_seed = UINT64_C(0x853c49e6748fea9b);
static uint64_t pal_rng_epoch = 1;
static uint64_t pal_rng_next_stream = 0;
PAL_TLS static uint64_t pal_rng_state = 0;
PAL_TLS static uint64_t pal_rng_inc = 0;
PAL_TLS static uint64_t pal_rng_seen_epoch = 0;
PAL_TLS static uint64_t pal_rng_stream_id = UINT64_MAX;

static uint64_t pal_splitmix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static uint32_t pal_pcg32(void) {
    uint64_t oldstate = pal_rng_state;
    pal_rng_state = oldstate * UINT64_C(6364136223846793005) + pal_rng_inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void pal_rng_init_thread(void) {
    if (pal_rng_stream_id == UINT64_MAX) {
#if defined(_MSC_VER)
        pal_rng_stream_id = (uint64_t)_InterlockedIncrement64((volatile long long *)&pal_rng_next_stream) - 1;
#else
        pal_rng_stream_id = __atomic_fetch_add(&pal_rng_next_stream, 1, __ATOMIC_RELAXED);
#endif
    }
    uint64_t s = pal_splitmix64(pal_rng_master_seed + pal_rng_stream_id * UINT64_C(0x9e3779b97f4a7c15));
    uint64_t seq = pal_splitmix64(pal_rng_master_seed ^ (pal_rng_stream_id + UINT64_C(0xda3e39cb94b95bdb)));
    pal_rng_state = 0;
    pal_rng_inc = (seq << 1u) | 1u;
    pal_pcg32();
    pal_rng_state += s;
    pal_pcg32();
    pal_rng_seen_epoch = pal_rng_epoch;
}

static uint32_t pal_rng_u32(void) {
    if (pal_rng_seen_epoch != pal_rng_epoch) pal_rng_init_thread();
    return pal_pcg32();
}

PAL_API void pal_seed(uint64_t seed) {
    pal_rng_master_seed = seed;
    pal_rng_next_stream = 0;
    pal_rng_epoch++;
    if (pal_rng_epoch == 0) pal_rng_epoch = 1;
    /* Calling thread is stream zero after an explicit seed. */
    pal_rng_stream_id = 0;
    pal_rng_next_stream = 1;
    pal_rng_seen_epoch = 0;
}

PAL_API double pal_random(void) {
    return (double)pal_rng_u32() * (1.0 / 4294967296.0);
}

PAL_API int64_t pal_rand_int(int64_t bound) {
    if (bound <= 0) return -1;
    if ((uint64_t)bound <= UINT64_C(0x100000000)) {
        uint32_t b = (uint32_t)bound;
        if ((uint64_t)bound == UINT64_C(0x100000000)) return (int64_t)pal_rng_u32();
        uint32_t threshold = (uint32_t)(-b) % b;
        for (;;) { uint32_t r = pal_rng_u32(); if (r >= threshold) return (int64_t)(r % b); }
    }
    uint64_t b = (uint64_t)bound;
    uint64_t threshold = (uint64_t)(-b) % b;
    for (;;) {
        uint64_t r = ((uint64_t)pal_rng_u32() << 32) | pal_rng_u32();
        if (r >= threshold) return (int64_t)(r % b);
    }
}

typedef struct {
    int32_t refs;
    int64_t n;
    double pRemaining;
} BinomialSolver;

typedef struct {
    BinomialSolver *solver;
    int32_t ownsSolver;
    int64_t nRemaining;
    double pRemaining;
} Multinomial;

/* Exact inversion for small mean; recursive beta-splitting for large cases.
 * The latter keeps the implementation compact while avoiding O(n) coin flips. */
static int64_t pal_binomial_sample(int64_t n, double p) {
    if (n <= 0 || p <= 0.0) return 0;
    if (p >= 1.0) return n;
    int flip = p > 0.5;
    double q = flip ? 1.0 - p : p;
    double mean = (double)n * q;
    if (mean < 30.0) {
        double s = exp((double)n * log1p(-q));
        double u = pal_random();
        int64_t x = 0;
        while (u > s) {
            u -= s;
            x++;
            s *= ((double)(n - x + 1) / (double)x) * (q / (1.0 - q));
        }
        return flip ? n - x : x;
    }
    /* BTPE-style normal proposal with exact log-PMF rejection. */
    double mu = (double)n * q;
    double sigma = sqrt(mu * (1.0 - q));
    for (;;) {
        double u1 = pal_random();
        double u2 = pal_random();
        if (u1 <= 0.0) continue;
        double z = sqrt(-2.0 * log(u1)) * cos(6.2831853071795864769 * u2);
        int64_t k = (int64_t)floor(mu + sigma * z + 0.5);
        if (k < 0 || k > n) continue;
        double logpmf = lgamma((double)n + 1.0) - lgamma((double)k + 1.0) - lgamma((double)(n-k) + 1.0)
                      + (double)k * log(q) + (double)(n-k) * log1p(-q);
        double logprop = -0.5*z*z - log(sigma) - 0.91893853320467274178;
        /* conservative envelope for discretized normal */
        double logM = 0.25;
        if (log(pal_random()) <= logpmf - logprop - logM) return flip ? n-k : k;
    }
}

PAL_API uintptr_t pal_multi_new(uintptr_t otherHandle) {
    Multinomial *m = (Multinomial *)calloc(1, sizeof(Multinomial));
    if (!m) return 0;
    if (otherHandle) {
        Multinomial *other = (Multinomial *)otherHandle;
        m->solver = other->solver;
        m->solver->refs++;
        m->ownsSolver = 0;
    } else {
        m->solver = (BinomialSolver *)calloc(1, sizeof(BinomialSolver));
        if (!m->solver) { free(m); return 0; }
        m->solver->refs = 1;
        m->ownsSolver = 1;
    }
    m->pRemaining = 1.0;
    return (uintptr_t)m;
}
PAL_API void pal_multi_free(uintptr_t handle) {
    Multinomial *m = (Multinomial *)handle;
    if (!m) return;
    if (m->solver && --m->solver->refs == 0) free(m->solver);
    free(m);
}
PAL_API int64_t pal_multi_binomial(uintptr_t handle, int64_t n, double p) {
    (void)handle;
    if (n < 0 || p < 0.0 || p > 1.0 || !isfinite(p)) return -1;
    return pal_binomial_sample(n, p);
}
PAL_API int32_t pal_multi_setup(uintptr_t handle, int64_t n) {
    if (!handle || n < 0) return 0;
    Multinomial *m = (Multinomial *)handle;
    m->nRemaining = n; m->pRemaining = 1.0; return 1;
}
PAL_API int64_t pal_multi_sample(uintptr_t handle, double p) {
    if (!handle) return -1;
    Multinomial *m = (Multinomial *)handle;
    if (!isfinite(p) || p < 0.0 || p > m->pRemaining + 1e-12) return -1;
    if (m->nRemaining == 0 || p == 0.0) { m->pRemaining -= p; return 0; }
    if (p >= m->pRemaining) {
        int64_t out = m->nRemaining; m->nRemaining = 0; m->pRemaining = 0.0; return out;
    }
    int64_t out = pal_binomial_sample(m->nRemaining, p / m->pRemaining);
    m->nRemaining -= out; m->pRemaining -= p; return out;
}



/* Bulk region access. Bounds are half-open and normalized by the Python/Numba API. */
PAL_API void pal_ag_counts_linear(uintptr_t h,int32_t i0,int32_t i1,int32_t*out){AgentGrid*g=(AgentGrid*)h;for(int32_t i=i0;i<i1;i++)out[i-i0]=g->counts[i];}
PAL_API void pal_ag_counts_region(uintptr_t h,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,int32_t*out){
    AgentGrid*g=(AgentGrid*)h; int32_t k=0;
    if(g->dimension==1){for(int32_t x=x0;x<x1;x++)out[k++]=g->counts[x];return;}
    if(g->dimension==2){for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)out[k++]=g->counts[AgentGrid_ToI(g,x,y,0)];return;}
    for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)for(int32_t z=z0;z<z1;z++)out[k++]=g->counts[AgentGrid_ToI(g,x,y,z)];
}
/* ========================================================================== */
/* PopGrid                                                                    */
/* ========================================================================== */

typedef enum { PG_FIELD_U8=1, PG_FIELD_U16=2, PG_FIELD_U32=4, PG_FIELD_I64=8 } PopFieldType;
typedef enum { PG_DELTA_I16=2, PG_DELTA_I32=4, PG_DELTA_I64=8 } PopDeltaType;

typedef struct {
    int32_t dimension;
    int32_t dimensions[3];
    int32_t wrap[3];
    int32_t length;
    int64_t capacity;
    int64_t population;
    uint8_t fieldType;
    uint8_t deltaType;
    void *field;
    void *deltas;
} PopGrid;

static int32_t PopGrid_ToI(const PopGrid *grid, int32_t x, int32_t y, int32_t z) {
    if (grid->dimension == 1) return x;
    if (grid->dimension == 2) return x * grid->dimensions[1] + y;
    return (x * grid->dimensions[1] + y) * grid->dimensions[2] + z;
}
static int64_t PopGrid_FieldGet(const PopGrid *g,int32_t i){
    switch(g->fieldType){case PG_FIELD_U8:return ((uint8_t*)g->field)[i];case PG_FIELD_U16:return ((uint16_t*)g->field)[i];case PG_FIELD_U32:return ((uint32_t*)g->field)[i];default:return ((int64_t*)g->field)[i];}
}
static void PopGrid_FieldSet(PopGrid *g,int32_t i,int64_t v){
    switch(g->fieldType){case PG_FIELD_U8:((uint8_t*)g->field)[i]=(uint8_t)v;break;case PG_FIELD_U16:((uint16_t*)g->field)[i]=(uint16_t)v;break;case PG_FIELD_U32:((uint32_t*)g->field)[i]=(uint32_t)v;break;default:((int64_t*)g->field)[i]=v;}
}
static int64_t PopGrid_DeltaGet(const PopGrid *g,int32_t i){
    switch(g->deltaType){case PG_DELTA_I16:return ((int16_t*)g->deltas)[i];case PG_DELTA_I32:return ((int32_t*)g->deltas)[i];default:return ((int64_t*)g->deltas)[i];}
}
static void PopGrid_DeltaSet(PopGrid *g,int32_t i,int64_t v){
    switch(g->deltaType){case PG_DELTA_I16:((int16_t*)g->deltas)[i]=(int16_t)v;break;case PG_DELTA_I32:((int32_t*)g->deltas)[i]=(int32_t)v;break;default:((int64_t*)g->deltas)[i]=v;}
}
static int PopGrid_DeltaFits(const PopGrid *g,int64_t current,int64_t add,int64_t *out){
    if ((add > 0 && current > INT64_MAX-add) || (add < 0 && current < INT64_MIN-add)) return 0;
    int64_t v=current+add;
    if(g->deltaType==PG_DELTA_I16 && (v<INT16_MIN || v>INT16_MAX)) return 0;
    if(g->deltaType==PG_DELTA_I32 && (v<INT32_MIN || v>INT32_MAX)) return 0;
    *out=v; return 1;
}

PAL_API uintptr_t pal_pg_new(const int32_t *dimensions, int32_t dimension, int64_t capacity) {
    if (dimension < 1 || dimension > 3 || capacity < 0) return 0;
    PopGrid *grid = (PopGrid *)calloc(1, sizeof(PopGrid)); if (!grid) return 0;
    grid->dimension=dimension; grid->capacity=capacity; grid->length=1;
    for(int32_t d=0;d<dimension;d++){int32_t value=dimensions[d];grid->wrap[d]=value<0;grid->dimensions[d]=value<0?-value:value;if(grid->dimensions[d]<=0){free(grid);return 0;}grid->length*=grid->dimensions[d];}
    if(capacity<=UINT8_MAX){grid->fieldType=PG_FIELD_U8;grid->deltaType=PG_DELTA_I16;}
    else if(capacity<=UINT16_MAX){grid->fieldType=PG_FIELD_U16;grid->deltaType=PG_DELTA_I32;}
    else if((uint64_t)capacity<=UINT32_MAX){grid->fieldType=PG_FIELD_U32;grid->deltaType=PG_DELTA_I64;}
    else {grid->fieldType=PG_FIELD_I64;grid->deltaType=PG_DELTA_I64;}
    grid->field=calloc((size_t)grid->length,(size_t)grid->fieldType);
    grid->deltas=calloc((size_t)grid->length,(size_t)grid->deltaType);
    if(!grid->field||!grid->deltas){pal_pg_free((uintptr_t)grid);return 0;} return (uintptr_t)grid;
}
PAL_API void pal_pg_free(uintptr_t handle){PopGrid*g=(PopGrid*)handle;if(!g)return;free(g->field);free(g->deltas);free(g);}
PAL_API int32_t pal_pg_dim(uintptr_t h,int32_t d){PopGrid*g=(PopGrid*)h;return d>=0&&d<g->dimension?g->dimensions[d]:PAL_NULL;}
static int32_t PopGrid_InWrap(const PopGrid*g,int32_t value,int32_t dimension){int32_t size=g->dimensions[dimension];if(value>=0&&value<size)return value;if(!g->wrap[dimension])return PAL_NULL;value%=size;if(value<0)value+=size;return value;}
PAL_API int32_t pal_pg_inwrap_x(uintptr_t h,int32_t v){return PopGrid_InWrap((PopGrid*)h,v,0);}
PAL_API int32_t pal_pg_inwrap_y(uintptr_t h,int32_t v){return PopGrid_InWrap((PopGrid*)h,v,1);}
PAL_API int32_t pal_pg_inwrap_z(uintptr_t h,int32_t v){return PopGrid_InWrap((PopGrid*)h,v,2);}
PAL_API void pal_pg_copy(uintptr_t h,int64_t*out){PopGrid*g=(PopGrid*)h;for(int32_t i=0;i<g->length;i++)out[i]=PopGrid_FieldGet(g,i);}
PAL_API int32_t pal_pg_len(uintptr_t h){return ((PopGrid*)h)->length;}
PAL_API int32_t pal_pg_itox(uintptr_t h,int32_t i){PopGrid*g=(PopGrid*)h;return g->dimension==1?i:g->dimension==2?i/g->dimensions[1]:i/(g->dimensions[1]*g->dimensions[2]);}
PAL_API int32_t pal_pg_itoy(uintptr_t h,int32_t i){PopGrid*g=(PopGrid*)h;return g->dimension==2?i%g->dimensions[1]:(i/g->dimensions[2])%g->dimensions[1];}
PAL_API int32_t pal_pg_itoz(uintptr_t h,int32_t i){PopGrid*g=(PopGrid*)h;return i%g->dimensions[2];}
PAL_API int32_t pal_pg_toi_safe(uintptr_t h,int32_t x,int32_t y,int32_t z){PopGrid*g=(PopGrid*)h;if(x<0||x>=g->dimensions[0])return PAL_BAD_I;if(g->dimension==1){if(y!=-1||z!=-1)return PAL_BAD_I;}else if(y<0||y>=g->dimensions[1])return PAL_BAD_I;if(g->dimension==2){if(z!=-1)return PAL_BAD_I;}else if(g->dimension==3&&(z<0||z>=g->dimensions[2]))return PAL_BAD_I;return PopGrid_ToI(g,x,y,z);}
PAL_API int32_t pal_pg_itox_safe(uintptr_t h,int32_t i){PopGrid*g=(PopGrid*)h;if(g->dimension<2||i<0||i>=g->length)return PAL_BAD_I;return g->dimension==2?i/g->dimensions[1]:i/(g->dimensions[1]*g->dimensions[2]);}
PAL_API int32_t pal_pg_itoy_safe(uintptr_t h,int32_t i){PopGrid*g=(PopGrid*)h;if(g->dimension<2||i<0||i>=g->length)return PAL_BAD_I;return g->dimension==2?i%g->dimensions[1]:(i/g->dimensions[2])%g->dimensions[1];}
PAL_API int32_t pal_pg_itoz_safe(uintptr_t h,int32_t i){PopGrid*g=(PopGrid*)h;if(g->dimension<3||i<0||i>=g->length)return PAL_BAD_I;return i%g->dimensions[2];}
PAL_API int32_t pal_pg_toi(uintptr_t h,int32_t x,int32_t y,int32_t z){return PopGrid_ToI((PopGrid*)h,x,y,z);}
PAL_API int64_t pal_pg_geti(uintptr_t h,int32_t i){PopGrid*g=(PopGrid*)h;return i>=0&&i<g->length?PopGrid_FieldGet(g,i):-1;}
PAL_API int64_t pal_pg_get(uintptr_t h,int32_t x,int32_t y,int32_t z){PopGrid*g=(PopGrid*)h;return PopGrid_FieldGet(g,PopGrid_ToI(g,x,y,z));}

/* Safe API: validates population bounds and delta accumulator overflow. */
PAL_API int32_t pal_pg_seti(uintptr_t h,int32_t i,int64_t value){PopGrid*g=(PopGrid*)h;if(i<0||i>=g->length||value<0||value>g->capacity)return 0;int64_t old=PopGrid_FieldGet(g,i),diff=value-old;if(diff>0&&g->population>INT64_MAX-diff)return 0;g->population+=diff;PopGrid_FieldSet(g,i,value);return 1;}
PAL_API int32_t pal_pg_addi(uintptr_t h,int32_t i,int64_t value){PopGrid*g=(PopGrid*)h;if(i<0||i>=g->length)return 0;int64_t next;if(!PopGrid_DeltaFits(g,PopGrid_DeltaGet(g,i),value,&next))return 0;PopGrid_DeltaSet(g,i,next);return 1;}
PAL_API int32_t pal_pg_set(uintptr_t h,int32_t x,int32_t y,int32_t z,int64_t value){PopGrid*g=(PopGrid*)h;return pal_pg_seti(h,PopGrid_ToI(g,x,y,z),value);}
PAL_API int32_t pal_pg_add(uintptr_t h,int32_t x,int32_t y,int32_t z,int64_t value){PopGrid*g=(PopGrid*)h;return pal_pg_addi(h,PopGrid_ToI(g,x,y,z),value);}
PAL_API int32_t pal_pg_update(uintptr_t h){PopGrid*g=(PopGrid*)h;int64_t newPop=g->population;for(int32_t i=0;i<g->length;i++){int64_t f=PopGrid_FieldGet(g,i),d=PopGrid_DeltaGet(g,i);if((d>0&&f>g->capacity-d)||(d<0&&(d==INT64_MIN||f<-d)))return 0;if(d>0&&newPop>INT64_MAX-d)return 0;if(d<0&&newPop < -d)return 0;newPop+=d;}for(int32_t i=0;i<g->length;i++){int64_t d=PopGrid_DeltaGet(g,i);PopGrid_FieldSet(g,i,PopGrid_FieldGet(g,i)+d);PopGrid_DeltaSet(g,i,0);}g->population=newPop;return 1;}

/* Fast API: same storage, no safety checks in Add/Set/Update hot paths. */
PAL_API int32_t pal_pg_seti_(uintptr_t h,int32_t i,int64_t value){PopGrid*g=(PopGrid*)h;int64_t old=PopGrid_FieldGet(g,i);g->population+=value-old;PopGrid_FieldSet(g,i,value);return 1;}
PAL_API int32_t pal_pg_addi_(uintptr_t h,int32_t i,int64_t value){PopGrid*g=(PopGrid*)h;PopGrid_DeltaSet(g,i,PopGrid_DeltaGet(g,i)+value);return 1;}
PAL_API int32_t pal_pg_set_(uintptr_t h,int32_t x,int32_t y,int32_t z,int64_t value){PopGrid*g=(PopGrid*)h;return pal_pg_seti_(h,PopGrid_ToI(g,x,y,z),value);}
PAL_API int32_t pal_pg_add_(uintptr_t h,int32_t x,int32_t y,int32_t z,int64_t value){PopGrid*g=(PopGrid*)h;return pal_pg_addi_(h,PopGrid_ToI(g,x,y,z),value);}
PAL_API int32_t pal_pg_update_(uintptr_t h){PopGrid*g=(PopGrid*)h;for(int32_t i=0;i<g->length;i++){int64_t d=PopGrid_DeltaGet(g,i);PopGrid_FieldSet(g,i,PopGrid_FieldGet(g,i)+d);g->population+=d;PopGrid_DeltaSet(g,i,0);}return 1;}

PAL_API int64_t pal_pg_pop(uintptr_t h){return ((PopGrid*)h)->population;}
PAL_API void pal_pg_clear(uintptr_t h){PopGrid*g=(PopGrid*)h;memset(g->field,0,(size_t)g->length*g->fieldType);memset(g->deltas,0,(size_t)g->length*g->deltaType);g->population=0;}
PAL_API int32_t pal_pg_clear_value(uintptr_t h,int64_t value){PopGrid*g=(PopGrid*)h;if(value<0||value>g->capacity)return 0;if(value>0&&g->length>0&&value>INT64_MAX/g->length)return 0;for(int32_t i=0;i<g->length;i++)PopGrid_FieldSet(g,i,value);memset(g->deltas,0,(size_t)g->length*g->deltaType);g->population=value*g->length;return 1;}
PAL_API void pal_pg_clear_value_(uintptr_t h,int64_t value){PopGrid*g=(PopGrid*)h;for(int32_t i=0;i<g->length;i++)PopGrid_FieldSet(g,i,value);memset(g->deltas,0,(size_t)g->length*g->deltaType);g->population=value*g->length;}

PAL_API int32_t pal_pg_field_bytes(uintptr_t h){return ((PopGrid*)h)->fieldType;}
PAL_API int32_t pal_pg_delta_bytes(uintptr_t h){return ((PopGrid*)h)->deltaType;}



PAL_API void pal_pg_linear_get(uintptr_t h,int32_t i0,int32_t i1,int64_t*out){PopGrid*g=(PopGrid*)h;for(int32_t i=i0;i<i1;i++)out[i-i0]=PopGrid_FieldGet(g,i);}
PAL_API void pal_pg_region_get(uintptr_t h,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,int64_t*out){
    PopGrid*g=(PopGrid*)h; int32_t k=0;
    if(g->dimension==1){for(int32_t x=x0;x<x1;x++)out[k++]=PopGrid_FieldGet(g,x);return;}
    if(g->dimension==2){for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)out[k++]=PopGrid_FieldGet(g,PopGrid_ToI(g,x,y,0));return;}
    for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)for(int32_t z=z0;z<z1;z++)out[k++]=PopGrid_FieldGet(g,PopGrid_ToI(g,x,y,z));
}
static int32_t PopGrid_RegionSetScalar(PopGrid*g,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,int64_t v,int safe){
    if(safe && (v<0||v>g->capacity))return 0; int64_t diff=0;
    for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)for(int32_t z=z0;z<z1;z++){int32_t i=PopGrid_ToI(g,x,y,z);int64_t old=PopGrid_FieldGet(g,i);if((v-old>0&&diff>INT64_MAX-(v-old))||(v-old<0&&diff<INT64_MIN-(v-old)))return 0;diff+=v-old;}
    if(safe && ((diff>0&&g->population>INT64_MAX-diff)||(diff<0&&g->population < -diff)))return 0;
    for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)for(int32_t z=z0;z<z1;z++)PopGrid_FieldSet(g,PopGrid_ToI(g,x,y,z),v);
    g->population+=diff; return 1;
}
static int32_t PopGrid_LinearSetScalar(PopGrid*g,int32_t i0,int32_t i1,int64_t v,int safe){if(safe&&(v<0||v>g->capacity))return 0;int64_t diff=0;for(int32_t i=i0;i<i1;i++){int64_t d=v-PopGrid_FieldGet(g,i);if((d>0&&diff>INT64_MAX-d)||(d<0&&diff<INT64_MIN-d))return 0;diff+=d;}if(safe&&((diff>0&&g->population>INT64_MAX-diff)||(diff<0&&g->population < -diff)))return 0;for(int32_t i=i0;i<i1;i++)PopGrid_FieldSet(g,i,v);g->population+=diff;return 1;}
PAL_API int32_t pal_pg_linear_set_scalar(uintptr_t h,int32_t i0,int32_t i1,int64_t v){return PopGrid_LinearSetScalar((PopGrid*)h,i0,i1,v,1);}
PAL_API int32_t pal_pg_linear_set_scalar_(uintptr_t h,int32_t i0,int32_t i1,int64_t v){return PopGrid_LinearSetScalar((PopGrid*)h,i0,i1,v,0);}
PAL_API int32_t pal_pg_region_set_scalar(uintptr_t h,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,int64_t v){return PopGrid_RegionSetScalar((PopGrid*)h,x0,x1,y0,y1,z0,z1,v,1);}
PAL_API int32_t pal_pg_region_set_scalar_(uintptr_t h,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,int64_t v){return PopGrid_RegionSetScalar((PopGrid*)h,x0,x1,y0,y1,z0,z1,v,0);}
static int32_t PopGrid_RegionSetArray(PopGrid*g,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,const int64_t*in,int safe){
    int32_t k=0; int64_t diff=0;
    for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)for(int32_t z=z0;z<z1;z++,k++){int64_t v=in[k];if(safe&&(v<0||v>g->capacity))return 0;int64_t old=PopGrid_FieldGet(g,PopGrid_ToI(g,x,y,z)),d=v-old;if((d>0&&diff>INT64_MAX-d)||(d<0&&diff<INT64_MIN-d))return 0;diff+=d;}
    if(safe&&((diff>0&&g->population>INT64_MAX-diff)||(diff<0&&g->population < -diff)))return 0;
    k=0;for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)for(int32_t z=z0;z<z1;z++,k++)PopGrid_FieldSet(g,PopGrid_ToI(g,x,y,z),in[k]);g->population+=diff;return 1;
}
static int32_t PopGrid_LinearSetArray(PopGrid*g,int32_t i0,int32_t i1,const int64_t*in,int safe){int64_t diff=0;for(int32_t i=i0;i<i1;i++){int64_t v=in[i-i0];if(safe&&(v<0||v>g->capacity))return 0;int64_t d=v-PopGrid_FieldGet(g,i);if((d>0&&diff>INT64_MAX-d)||(d<0&&diff<INT64_MIN-d))return 0;diff+=d;}if(safe&&((diff>0&&g->population>INT64_MAX-diff)||(diff<0&&g->population < -diff)))return 0;for(int32_t i=i0;i<i1;i++)PopGrid_FieldSet(g,i,in[i-i0]);g->population+=diff;return 1;}
PAL_API int32_t pal_pg_linear_set_array(uintptr_t h,int32_t i0,int32_t i1,const int64_t*in){return PopGrid_LinearSetArray((PopGrid*)h,i0,i1,in,1);}
PAL_API int32_t pal_pg_linear_set_array_(uintptr_t h,int32_t i0,int32_t i1,const int64_t*in){return PopGrid_LinearSetArray((PopGrid*)h,i0,i1,in,0);}
PAL_API int32_t pal_pg_region_set_array(uintptr_t h,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,const int64_t*in){return PopGrid_RegionSetArray((PopGrid*)h,x0,x1,y0,y1,z0,z1,in,1);}
PAL_API int32_t pal_pg_region_set_array_(uintptr_t h,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,const int64_t*in){return PopGrid_RegionSetArray((PopGrid*)h,x0,x1,y0,y1,z0,z1,in,0);}
/* ========================================================================== */
/* PDEgrid                                                                    */
/* ========================================================================== */

typedef struct {
    int32_t dimension;
    int32_t dimensions[3];
    int32_t wrap[3];
    int32_t length;
    float *field;
    float *deltas;
    float dt, dx, dy, dz;
    float scalarBC[6];
} PDEGrid;

static int32_t PDEGrid_ToI(const PDEGrid *grid, int32_t x, int32_t y, int32_t z) {
    if (grid->dimension == 1) return x;
    if (grid->dimension == 2) return x * grid->dimensions[1] + y;
    return (x * grid->dimensions[1] + y) * grid->dimensions[2] + z;
}

static int32_t PDEGrid_Wrap(const PDEGrid *grid, int32_t value, int32_t dimension) {
    int32_t size = grid->dimensions[dimension];
    if (value >= 0 && value < size) return value;
    if (!grid->wrap[dimension]) return PAL_NULL;
    value %= size;
    if (value < 0) value += size;
    return value;
}

PAL_API uintptr_t pal_pd_new(const int32_t *dimensions, int32_t dimension) {
    if (dimension < 1 || dimension > 3) return 0;
    PDEGrid *grid = (PDEGrid *)calloc(1, sizeof(PDEGrid));
    if (grid == NULL) return 0;

    grid->dimension = dimension;
    grid->length = 1;
    for (int32_t d = 0; d < dimension; d++) {
        int32_t value = dimensions[d];
        grid->wrap[d] = value < 0;
        grid->dimensions[d] = value < 0 ? -value : value;
        if (grid->dimensions[d] <= 0) { free(grid); return 0; }
        grid->length *= grid->dimensions[d];
    }

    grid->field = (float *)calloc((size_t)grid->length, sizeof(float));
    grid->deltas = (float *)calloc((size_t)grid->length, sizeof(float));
    if (grid->field == NULL || grid->deltas == NULL) { pal_pd_free((uintptr_t)grid); return 0; }
    grid->dt = grid->dx = grid->dy = grid->dz = 1.0f;
    return (uintptr_t)grid;
}

PAL_API void pal_pd_free(uintptr_t handle) {
    PDEGrid *grid = (PDEGrid *)handle;
    if (grid == NULL) return;
    free(grid->field); free(grid->deltas); free(grid);
}
PAL_API int32_t pal_pd_dim(uintptr_t handle, int32_t d) {
    PDEGrid *grid = (PDEGrid *)handle;
    return d >= 0 && d < grid->dimension ? grid->dimensions[d] : PAL_NULL;
}
PAL_API int32_t pal_pd_len(uintptr_t handle) { return ((PDEGrid *)handle)->length; }
PAL_API int32_t pal_pd_itox(uintptr_t h,int32_t i){PDEGrid*g=(PDEGrid*)h;return g->dimension==1?i:g->dimension==2?i/g->dimensions[1]:i/(g->dimensions[1]*g->dimensions[2]);}
PAL_API int32_t pal_pd_itoy(uintptr_t h,int32_t i){PDEGrid*g=(PDEGrid*)h;return g->dimension==2?i%g->dimensions[1]:(i/g->dimensions[2])%g->dimensions[1];}
PAL_API int32_t pal_pd_itoz(uintptr_t h,int32_t i){PDEGrid*g=(PDEGrid*)h;return i%g->dimensions[2];}
PAL_API int32_t pal_pd_toi_safe(uintptr_t h,int32_t x,int32_t y,int32_t z){PDEGrid*g=(PDEGrid*)h;if(x<0||x>=g->dimensions[0])return PAL_BAD_I;if(g->dimension==1){if(y!=-1||z!=-1)return PAL_BAD_I;}else if(y<0||y>=g->dimensions[1])return PAL_BAD_I;if(g->dimension==2){if(z!=-1)return PAL_BAD_I;}else if(g->dimension==3&&(z<0||z>=g->dimensions[2]))return PAL_BAD_I;return PDEGrid_ToI(g,x,y,z);}
PAL_API int32_t pal_pd_itox_safe(uintptr_t h,int32_t i){PDEGrid*g=(PDEGrid*)h;if(g->dimension<2||i<0||i>=g->length)return PAL_BAD_I;return g->dimension==2?i/g->dimensions[1]:i/(g->dimensions[1]*g->dimensions[2]);}
PAL_API int32_t pal_pd_itoy_safe(uintptr_t h,int32_t i){PDEGrid*g=(PDEGrid*)h;if(g->dimension<2||i<0||i>=g->length)return PAL_BAD_I;return g->dimension==2?i%g->dimensions[1]:(i/g->dimensions[2])%g->dimensions[1];}
PAL_API int32_t pal_pd_itoz_safe(uintptr_t h,int32_t i){PDEGrid*g=(PDEGrid*)h;if(g->dimension<3||i<0||i>=g->length)return PAL_BAD_I;return i%g->dimensions[2];}
PAL_API int32_t pal_pd_toi(uintptr_t handle, int32_t x, int32_t y, int32_t z) { return PDEGrid_ToI((PDEGrid *)handle, x, y, z); }
PAL_API float pal_pd_geti(uintptr_t handle, int32_t i) { PDEGrid*g=(PDEGrid*)handle; return i>=0&&i<g->length?g->field[i]:NAN; }
PAL_API float pal_pd_get(uintptr_t handle,int32_t x,int32_t y,int32_t z) { PDEGrid *g=(PDEGrid *)handle; return g->field[PDEGrid_ToI(g,x,y,z)]; }
PAL_API int32_t pal_pd_seti(uintptr_t handle, int32_t i, float value) { PDEGrid*g=(PDEGrid*)handle; if(i<0||i>=g->length||!isfinite(value))return 0;g->field[i]=value;return 1; }
PAL_API float pal_pd_geti_(uintptr_t h,int32_t i){return ((PDEGrid*)h)->field[i];}
PAL_API void pal_pd_seti_(uintptr_t h,int32_t i,float v){((PDEGrid*)h)->field[i]=v;}
PAL_API void pal_pd_addi(uintptr_t handle, int32_t i, float value) { ((PDEGrid *)handle)->deltas[i] += value; }
PAL_API void pal_pd_set(uintptr_t handle,int32_t x,int32_t y,int32_t z,float value) { PDEGrid *g=(PDEGrid *)handle; g->field[PDEGrid_ToI(g,x,y,z)]=value; }
PAL_API void pal_pd_add(uintptr_t handle,int32_t x,int32_t y,int32_t z,float value) { PDEGrid *g=(PDEGrid *)handle; g->deltas[PDEGrid_ToI(g,x,y,z)]+=value; }
PAL_API void pal_pd_update(uintptr_t handle) {
    PDEGrid *grid = (PDEGrid *)handle;
    for (int32_t i = 0; i < grid->length; i++) {
        grid->field[i] += grid->deltas[i];
        grid->deltas[i] = 0.0f;
    }
}
PAL_API void pal_pd_steps(uintptr_t handle, float dt, float dx, float dy, float dz) {
    PDEGrid *grid = (PDEGrid *)handle;
    grid->dt = dt; grid->dx = dx; grid->dy = dy; grid->dz = dz;
}
PAL_API float pal_pd_voxel(uintptr_t handle) {
    PDEGrid *grid = (PDEGrid *)handle;
    if (grid->dimension == 1) return grid->dx;
    if (grid->dimension == 2) return grid->dx * grid->dy;
    return grid->dx * grid->dy * grid->dz;
}
PAL_API void pal_pd_clear(uintptr_t handle, float value) {
    PDEGrid *grid = (PDEGrid *)handle;
    for (int32_t i = 0; i < grid->length; i++) { grid->field[i] = value; grid->deltas[i] = 0.0f; }
}

/* Explicit diffusion with zero-flux boundaries on non-wrapped axes. */


PAL_API void pal_pd_linear_get(uintptr_t h,int32_t i0,int32_t i1,float*out){PDEGrid*g=(PDEGrid*)h;memcpy(out,g->field+(size_t)i0,(size_t)(i1-i0)*sizeof(float));}
PAL_API void pal_pd_linear_set_scalar(uintptr_t h,int32_t i0,int32_t i1,float v){PDEGrid*g=(PDEGrid*)h;for(int32_t i=i0;i<i1;i++)g->field[i]=v;}
PAL_API void pal_pd_linear_set_array(uintptr_t h,int32_t i0,int32_t i1,const float*in){PDEGrid*g=(PDEGrid*)h;memcpy(g->field+(size_t)i0,in,(size_t)(i1-i0)*sizeof(float));}
PAL_API void pal_pd_region_get(uintptr_t h,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,float*out){
    PDEGrid*g=(PDEGrid*)h;int32_t k=0;
    if(g->dimension==1){for(int32_t x=x0;x<x1;x++)out[k++]=g->field[x];return;}
    if(g->dimension==2){for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)out[k++]=g->field[PDEGrid_ToI(g,x,y,0)];return;}
    for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)for(int32_t z=z0;z<z1;z++)out[k++]=g->field[PDEGrid_ToI(g,x,y,z)];
}
PAL_API void pal_pd_region_set_scalar(uintptr_t h,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,float v){PDEGrid*g=(PDEGrid*)h;for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)for(int32_t z=z0;z<z1;z++)g->field[PDEGrid_ToI(g,x,y,z)]=v;}
PAL_API void pal_pd_region_set_array(uintptr_t h,int32_t x0,int32_t x1,int32_t y0,int32_t y1,int32_t z0,int32_t z1,const float*in){PDEGrid*g=(PDEGrid*)h;int32_t k=0;for(int32_t x=x0;x<x1;x++)for(int32_t y=y0;y<y1;y++)for(int32_t z=z0;z<z1;z++,k++)g->field[PDEGrid_ToI(g,x,y,z)]=in[k];}
PAL_API int32_t pal_pd_diffusion(uintptr_t handle, float rate) {
    PDEGrid *grid = (PDEGrid *)handle;
    float sx = grid->dt * rate / (grid->dx * grid->dx);
    float sy = grid->dimension > 1 ? grid->dt * rate / (grid->dy * grid->dy) : 0.0f;
    float sz = grid->dimension > 2 ? grid->dt * rate / (grid->dz * grid->dz) : 0.0f;
    if (2.0f * (sx + sy + sz) > 1.000001f) return 0;

    int32_t yMax = grid->dimension > 1 ? grid->dimensions[1] : 1;
    int32_t zMax = grid->dimension > 2 ? grid->dimensions[2] : 1;

    for (int32_t x = 0; x < grid->dimensions[0]; x++) {
        for (int32_t y = 0; y < yMax; y++) {
            for (int32_t z = 0; z < zMax; z++) {
                int32_t i = PDEGrid_ToI(grid, x, y, z);
                float center = grid->field[i];
                float delta = 0.0f;

                int32_t low = PDEGrid_Wrap(grid, x - 1, 0);
                int32_t high = PDEGrid_Wrap(grid, x + 1, 0);
                if (low != PAL_NULL) delta += sx * (grid->field[PDEGrid_ToI(grid, low, y, z)] - center);
                if (high != PAL_NULL) delta += sx * (grid->field[PDEGrid_ToI(grid, high, y, z)] - center);

                if (grid->dimension > 1) {
                    low = PDEGrid_Wrap(grid, y - 1, 1);
                    high = PDEGrid_Wrap(grid, y + 1, 1);
                    if (low != PAL_NULL) delta += sy * (grid->field[PDEGrid_ToI(grid, x, low, z)] - center);
                    if (high != PAL_NULL) delta += sy * (grid->field[PDEGrid_ToI(grid, x, high, z)] - center);
                }

                if (grid->dimension > 2) {
                    low = PDEGrid_Wrap(grid, z - 1, 2);
                    high = PDEGrid_Wrap(grid, z + 1, 2);
                    if (low != PAL_NULL) delta += sz * (grid->field[PDEGrid_ToI(grid, x, y, low)] - center);
                    if (high != PAL_NULL) delta += sz * (grid->field[PDEGrid_ToI(grid, x, y, high)] - center);
                }

                grid->deltas[i] += delta;
            }
        }
    }
    return 1;
}

/* Extended PDEgrid API. Face BC pointers are NULL for zero-flux; otherwise
 * arrays laid out over the corresponding face. Wrapped axes ignore face BCs. */
static int32_t PDEGrid_InWrap(const PDEGrid *g,int32_t v,int32_t d){return PDEGrid_Wrap(g,v,d);}
static int32_t PDEGrid_FaceIndex(const PDEGrid*g,int32_t axis,int32_t x,int32_t y,int32_t z){
    if(axis==0) return g->dimension==1?0:(g->dimension==2?y:y*g->dimensions[2]+z);
    if(axis==1) return g->dimension==2?x:x*g->dimensions[2]+z;
    return x*g->dimensions[1]+y;
}
static float PDEGrid_BC(const PDEGrid *g,const float *bc,int32_t idx){
    if(bc>=g->scalarBC&&bc<g->scalarBC+6)return *bc;
    return bc[idx];
}
PAL_API uintptr_t pal_pd_bc_scalar(uintptr_t h,int32_t slot,float value){PDEGrid*g=(PDEGrid*)h;g->scalarBC[slot]=value;return (uintptr_t)&g->scalarBC[slot];}
static int PDEGrid_ExplicitStable(const PDEGrid*g,float rate){
    float s=2.0f*rate*g->dt/(g->dx*g->dx);
    if(g->dimension>1)s+=2.0f*rate*g->dt/(g->dy*g->dy);
    if(g->dimension>2)s+=2.0f*rate*g->dt/(g->dz*g->dz);
    return s<=1.000001f;
}
PAL_API int32_t pal_pd_inwrap_x(uintptr_t h,int32_t v){return PDEGrid_InWrap((PDEGrid*)h,v,0);}
PAL_API int32_t pal_pd_inwrap_y(uintptr_t h,int32_t v){return PDEGrid_InWrap((PDEGrid*)h,v,1);}
PAL_API int32_t pal_pd_inwrap_z(uintptr_t h,int32_t v){return PDEGrid_InWrap((PDEGrid*)h,v,2);}
PAL_API float pal_pd_dx(uintptr_t h){return ((PDEGrid*)h)->dx;}
PAL_API float pal_pd_dy(uintptr_t h){return ((PDEGrid*)h)->dy;}
PAL_API float pal_pd_dz(uintptr_t h){return ((PDEGrid*)h)->dz;}
PAL_API float pal_pd_dt(uintptr_t h){return ((PDEGrid*)h)->dt;}
PAL_API void pal_pd_copy(uintptr_t h,float*out){PDEGrid*g=(PDEGrid*)h;memcpy(out,g->field,(size_t)g->length*sizeof(float));}

static int32_t PDEGrid_DiffusionConstant(PDEGrid*g,float rate,const float*x0,const float*x1,const float*y0,const float*y1,const float*z0,const float*z1,int checked){
    const int D=g->dimension,X=g->dimensions[0],Y=D>1?g->dimensions[1]:1,Z=D>2?g->dimensions[2]:1;
    const int sxStride=Y*Z,syStride=Z;
    const float sx=rate*g->dt/(g->dx*g->dx),sy=D>1?rate*g->dt/(g->dy*g->dy):0.0f,sz=D>2?rate*g->dt/(g->dz*g->dz):0.0f;
    const float*bc0[3]={x0,y0,z0},*bc1[3]={x1,y1,z1};const float sc[3]={sx,sy,sz};const int ns[3]={X,Y,Z};
    if(checked){
        float diffusionCourant=0.0f;
        for(int a=0;a<D;a++){
            float w=sc[a],axis=0.0f;int n=ns[a];
            if(g->wrap[a])axis=2.0f*w;
            else if(n==1)axis=2.0f*w*((bc0[a]?1.0f:0.0f)+(bc1[a]?1.0f:0.0f));
            else{axis=n>=3?2.0f*w:0.0f;float lo=w+(bc0[a]?2.0f*w:0.0f),hi=w+(bc1[a]?2.0f*w:0.0f);if(lo>axis)axis=lo;if(hi>axis)axis=hi;}
            diffusionCourant+=axis;
        }
        if(diffusionCourant>1.000001f)return 0;
    }
    float *restrict f=g->field,*restrict d=g->deltas;
    if(D==1){
        for(int x=0;x<X;x++){
            float c=f[x],v=0.0f;
            if(x+1<X)v+=(f[x+1]-c)*sx;else if(g->wrap[0])v+=(f[0]-c)*sx;else if(x1)v+=2.0f*(PDEGrid_BC(g,x1,0)-c)*sx;
            if(x>0)v+=(f[x-1]-c)*sx;else if(g->wrap[0])v+=(f[X-1]-c)*sx;else if(x0)v+=2.0f*(PDEGrid_BC(g,x0,0)-c)*sx;
            d[x]+=v;
        }
        return 1;
    }
    if(D==2){
        for(int x=0;x<X;x++){
            int row=x*Y;
            for(int y=0;y<Y;y++){
                int i=row+y;float c=f[i],v=0.0f;
                if(x+1<X)v+=(f[i+Y]-c)*sx;else if(g->wrap[0])v+=(f[y]-c)*sx;else if(x1)v+=2.0f*(PDEGrid_BC(g,x1,y)-c)*sx;
                if(x>0)v+=(f[i-Y]-c)*sx;else if(g->wrap[0])v+=(f[(X-1)*Y+y]-c)*sx;else if(x0)v+=2.0f*(PDEGrid_BC(g,x0,y)-c)*sx;
                if(y+1<Y)v+=(f[i+1]-c)*sy;else if(g->wrap[1])v+=(f[row]-c)*sy;else if(y1)v+=2.0f*(PDEGrid_BC(g,y1,x)-c)*sy;
                if(y>0)v+=(f[i-1]-c)*sy;else if(g->wrap[1])v+=(f[row+Y-1]-c)*sy;else if(y0)v+=2.0f*(PDEGrid_BC(g,y0,x)-c)*sy;
                d[i]+=v;
            }
        }
        return 1;
    }
    for(int x=0;x<X;x++)for(int y=0;y<Y;y++){
        int base=(x*Y+y)*Z;
        for(int z=0;z<Z;z++){
            int i=base+z;float c=f[i],v=0.0f;
            int xFace=y*Z+z,yFace=x*Z+z,zFace=x*Y+y;
            if(x+1<X)v+=(f[i+sxStride]-c)*sx;else if(g->wrap[0])v+=(f[y*Z+z]-c)*sx;else if(x1)v+=2.0f*(PDEGrid_BC(g,x1,xFace)-c)*sx;
            if(x>0)v+=(f[i-sxStride]-c)*sx;else if(g->wrap[0])v+=(f[((X-1)*Y+y)*Z+z]-c)*sx;else if(x0)v+=2.0f*(PDEGrid_BC(g,x0,xFace)-c)*sx;
            if(y+1<Y)v+=(f[i+syStride]-c)*sy;else if(g->wrap[1])v+=(f[(x*Y)*Z+z]-c)*sy;else if(y1)v+=2.0f*(PDEGrid_BC(g,y1,yFace)-c)*sy;
            if(y>0)v+=(f[i-syStride]-c)*sy;else if(g->wrap[1])v+=(f[(x*Y+Y-1)*Z+z]-c)*sy;else if(y0)v+=2.0f*(PDEGrid_BC(g,y0,yFace)-c)*sy;
            if(z+1<Z)v+=(f[i+1]-c)*sz;else if(g->wrap[2])v+=(f[base]-c)*sz;else if(z1)v+=2.0f*(PDEGrid_BC(g,z1,zFace)-c)*sz;
            if(z>0)v+=(f[i-1]-c)*sz;else if(g->wrap[2])v+=(f[base+Z-1]-c)*sz;else if(z0)v+=2.0f*(PDEGrid_BC(g,z0,zFace)-c)*sz;
            d[i]+=v;
        }
    }
    return 1;
}
PAL_API int32_t pal_pd_diffusion_bc(uintptr_t h,float r,uintptr_t x0,uintptr_t x1,uintptr_t y0,uintptr_t y1,uintptr_t z0,uintptr_t z1){return PDEGrid_DiffusionConstant((PDEGrid*)h,r,(float*)x0,(float*)x1,(float*)y0,(float*)y1,(float*)z0,(float*)z1,1);}
PAL_API int32_t pal_pd_diffusion_bc_fast(uintptr_t h,float r,uintptr_t x0,uintptr_t x1,uintptr_t y0,uintptr_t y1,uintptr_t z0,uintptr_t z1){return PDEGrid_DiffusionConstant((PDEGrid*)h,r,(float*)x0,(float*)x1,(float*)y0,(float*)y1,(float*)z0,(float*)z1,0);}

static int32_t PDEGrid_DiffusionMask(PDEGrid*g,float rate,float*m,int checked){
    if(checked){for(int32_t i=0;i<g->length;i++)if(!isfinite(m[i]))return -1;if(!PDEGrid_ExplicitStable(g,rate))return 0;}
    int X=g->dimensions[0],Y=g->dimension>1?g->dimensions[1]:1,Z=g->dimension>2?g->dimensions[2]:1;float s[3]={rate*g->dt/(g->dx*g->dx),rate*g->dt/(g->dy*g->dy),rate*g->dt/(g->dz*g->dz)};
    for(int x=0;x<X;x++)for(int y=0;y<Y;y++)for(int z=0;z<Z;z++){int i=PDEGrid_ToI(g,x,y,z);if(m[i]<0)continue;float c=g->field[i],v=0;int q[3]={x,y,z};for(int a=0;a<g->dimension;a++)for(int sg=-1;sg<=1;sg+=2){int t=PDEGrid_Wrap(g,q[a]+sg,a);if(t==PAL_NULL)continue;int xx=x,yy=y,zz=z;if(a==0)xx=t;else if(a==1)yy=t;else zz=t;int j=PDEGrid_ToI(g,xx,yy,zz);if(m[j]>=0)v+=(g->field[j]-c)*s[a];}g->deltas[i]+=v;}return 1;
}
PAL_API int32_t pal_pd_diffusion_mask(uintptr_t h,float rate,uintptr_t mh){return PDEGrid_DiffusionMask((PDEGrid*)h,rate,(float*)mh,1);}
PAL_API int32_t pal_pd_diffusion_mask_fast(uintptr_t h,float rate,uintptr_t mh){return PDEGrid_DiffusionMask((PDEGrid*)h,rate,(float*)mh,0);}
static float PDEGrid_Harmonic(float a,float b){return a>0&&b>0?2*a*b/(a+b):0;}
static int32_t PDEGrid_DiffusionField(PDEGrid*g,uintptr_t rh,uintptr_t x0h,uintptr_t x1h,uintptr_t y0h,uintptr_t y1h,uintptr_t z0h,uintptr_t z1h,int checked){float*r=(float*)rh,*bc0[3]={(float*)x0h,(float*)y0h,(float*)z0h},*bc1[3]={(float*)x1h,(float*)y1h,(float*)z1h};int X=g->dimensions[0],Y=g->dimension>1?g->dimensions[1]:1,Z=g->dimension>2?g->dimensions[2]:1;float ds[3]={g->dt/(g->dx*g->dx),g->dt/(g->dy*g->dy),g->dt/(g->dz*g->dz)};
    /* Safe mode is transactional: validate the entire rate field and CFL bound before touching deltas. */
    if(checked){
        for(int32_t i=0;i<g->length;i++)if(r[i]<0||!isfinite(r[i]))return -1;
        for(int x=0;x<X;x++)for(int y=0;y<Y;y++)for(int z=0;z<Z;z++){int i=PDEGrid_ToI(g,x,y,z),q[3]={x,y,z};float sum=0;for(int a=0;a<g->dimension;a++)for(int sg=-1;sg<=1;sg+=2){int t=PDEGrid_Wrap(g,q[a]+sg,a);float rr=0,mult=1.0f;if(t!=PAL_NULL){int xx=x,yy=y,zz=z;if(a==0)xx=t;else if(a==1)yy=t;else zz=t;int j=PDEGrid_ToI(g,xx,yy,zz);rr=PDEGrid_Harmonic(r[i],r[j]);}else{float*bc=sg<0?bc0[a]:bc1[a];if(!bc)continue;rr=r[i];mult=2.0f;}sum+=rr*ds[a]*mult;}if(sum>1.000001f)return 0;}
    }
    for(int x=0;x<X;x++)for(int y=0;y<Y;y++)for(int z=0;z<Z;z++){int i=(x*Y+y)*Z+z;float c=g->field[i],v=0;for(int a=0;a<g->dimension;a++)for(int sg=-1;sg<=1;sg+=2){int j=-1;if(a==0){if(sg<0){if(x>0)j=i-Y*Z;else if(g->wrap[0])j=i+(X-1)*Y*Z;}else{if(x+1<X)j=i+Y*Z;else if(g->wrap[0])j=i-(X-1)*Y*Z;}}else if(a==1){if(sg<0){if(y>0)j=i-Z;else if(g->wrap[1])j=i+(Y-1)*Z;}else{if(y+1<Y)j=i+Z;else if(g->wrap[1])j=i-(Y-1)*Z;}}else{if(sg<0){if(z>0)j=i-1;else if(g->wrap[2])j=i+Z-1;}else{if(z+1<Z)j=i+1;else if(g->wrap[2])j=i-Z+1;}}float rr,nv,mult=1.0f;if(j>=0){rr=PDEGrid_Harmonic(r[i],r[j]);nv=g->field[j];}else{float*bc=sg<0?bc0[a]:bc1[a];if(!bc)continue;rr=r[i];nv=PDEGrid_BC(g,bc,PDEGrid_FaceIndex(g,a,x,y,z));mult=2.0f;}v+=(nv-c)*rr*ds[a]*mult;}g->deltas[i]+=v;}return 1;
}
PAL_API int32_t pal_pd_diffusion_field(uintptr_t h,uintptr_t rh,uintptr_t x0h,uintptr_t x1h,uintptr_t y0h,uintptr_t y1h,uintptr_t z0h,uintptr_t z1h){return PDEGrid_DiffusionField((PDEGrid*)h,rh,x0h,x1h,y0h,y1h,z0h,z1h,1);}
PAL_API int32_t pal_pd_diffusion_field_fast(uintptr_t h,uintptr_t rh,uintptr_t x0h,uintptr_t x1h,uintptr_t y0h,uintptr_t y1h,uintptr_t z0h,uintptr_t z1h){return PDEGrid_DiffusionField((PDEGrid*)h,rh,x0h,x1h,y0h,y1h,z0h,z1h,0);}
static int32_t PDEGrid_DiffusionInterfaces(PDEGrid*g,uintptr_t rxh,uintptr_t ryh,uintptr_t rzh,uintptr_t x0h,uintptr_t x1h,uintptr_t y0h,uintptr_t y1h,uintptr_t z0h,uintptr_t z1h,int checked){float*rr[3]={(float*)rxh,(float*)ryh,(float*)rzh},*bc0[3]={(float*)x0h,(float*)y0h,(float*)z0h},*bc1[3]={(float*)x1h,(float*)y1h,(float*)z1h};int X=g->dimensions[0],Y=g->dimension>1?g->dimensions[1]:1,Z=g->dimension>2?g->dimensions[2]:1;float ds[3]={g->dt/(g->dx*g->dx),g->dt/(g->dy*g->dy),g->dt/(g->dz*g->dz)};
    /* Validate every interface and every local stability sum before accumulating any delta. */
    if(checked){
        for(int a=0;a<g->dimension;a++){if(!rr[a])return -1;for(int32_t i=0;i<g->length;i++)if(rr[a][i]<0||!isfinite(rr[a][i]))return -1;}
        for(int x=0;x<X;x++)for(int y=0;y<Y;y++)for(int z=0;z<Z;z++){int i=PDEGrid_ToI(g,x,y,z),q[3]={x,y,z};float sum=0;for(int a=0;a<g->dimension;a++)for(int sg=-1;sg<=1;sg+=2){int t=PDEGrid_Wrap(g,q[a]+sg,a),ri=i;float mult=1.0f;if(t!=PAL_NULL){int xx=x,yy=y,zz=z;if(a==0)xx=t;else if(a==1)yy=t;else zz=t;int j=PDEGrid_ToI(g,xx,yy,zz);if(sg<0)ri=j;}else{float*bc=sg<0?bc0[a]:bc1[a];if(!bc)continue;mult=2.0f;}sum+=rr[a][ri]*ds[a]*mult;}if(sum>1.000001f)return 0;}
    }
    for(int x=0;x<X;x++)for(int y=0;y<Y;y++)for(int z=0;z<Z;z++){int i=(x*Y+y)*Z+z;float c=g->field[i],v=0;for(int a=0;a<g->dimension;a++)for(int sg=-1;sg<=1;sg+=2){int j=-1,ri=i;if(a==0){if(sg<0){if(x>0)j=i-Y*Z;else if(g->wrap[0])j=i+(X-1)*Y*Z;}else{if(x+1<X)j=i+Y*Z;else if(g->wrap[0])j=i-(X-1)*Y*Z;}}else if(a==1){if(sg<0){if(y>0)j=i-Z;else if(g->wrap[1])j=i+(Y-1)*Z;}else{if(y+1<Y)j=i+Z;else if(g->wrap[1])j=i-(Y-1)*Z;}}else{if(sg<0){if(z>0)j=i-1;else if(g->wrap[2])j=i+Z-1;}else{if(z+1<Z)j=i+1;else if(g->wrap[2])j=i-Z+1;}}float nv,mult=1.0f;if(j>=0){if(sg<0)ri=j;nv=g->field[j];}else{float*bc=sg<0?bc0[a]:bc1[a];if(!bc)continue;nv=PDEGrid_BC(g,bc,PDEGrid_FaceIndex(g,a,x,y,z));mult=2.0f;}float rate=rr[a][ri];v+=(nv-c)*rate*ds[a]*mult;}g->deltas[i]+=v;}return 1;
}
PAL_API int32_t pal_pd_diffusion_interfaces(uintptr_t h,uintptr_t rxh,uintptr_t ryh,uintptr_t rzh,uintptr_t x0h,uintptr_t x1h,uintptr_t y0h,uintptr_t y1h,uintptr_t z0h,uintptr_t z1h){return PDEGrid_DiffusionInterfaces((PDEGrid*)h,rxh,ryh,rzh,x0h,x1h,y0h,y1h,z0h,z1h,1);}
PAL_API int32_t pal_pd_diffusion_interfaces_fast(uintptr_t h,uintptr_t rxh,uintptr_t ryh,uintptr_t rzh,uintptr_t x0h,uintptr_t x1h,uintptr_t y0h,uintptr_t y1h,uintptr_t z0h,uintptr_t z1h){return PDEGrid_DiffusionInterfaces((PDEGrid*)h,rxh,ryh,rzh,x0h,x1h,y0h,y1h,z0h,z1h,0);}
static int32_t PDEGrid_DiffusionRadial(PDEGrid*g,float rate,const float*outerBC,int dim,int checked){
    if(checked&&(g->dimension!=1||g->wrap[0]||g->length<2))return -1;
    if(checked&&(rate<0||!isfinite(rate)))return -2;
    float base=rate*g->dt/(g->dx*g->dx),*f=g->field,*delta=g->deltas;int X=g->length;
    double worstGeom=dim==2?2.0:3.0;
    if(outerBC){double i=(double)(X-1),outer;if(dim==2)outer=(3.0*i+2.0)/(i+0.5);else outer=(i*i+2.0*(i+1.0)*(i+1.0))/(i*i+i+1.0/3.0);if(outer>worstGeom)worstGeom=outer;}
    if(checked&&(double)base*worstGeom>1.000001)return 0;
    if(dim==2){
        for(int i=0;i<X;i++){float den=(float)i+0.5f,left=i>0?(float)i/den:0.0f,right=((float)i+1.0f)/den,v=0.0f;if(i>0)v+=base*left*(f[i-1]-f[i]);if(i<X-1)v+=base*right*(f[i+1]-f[i]);else if(outerBC)v+=2.0f*base*right*((*outerBC)-f[i]);delta[i]+=v;}
    }else{
        for(int i=0;i<X;i++){float fi=(float)i,den=fi*fi+fi+1.0f/3.0f,left=i>0?fi*fi/den:0.0f,fp=fi+1.0f,right=fp*fp/den,v=0.0f;if(i>0)v+=base*left*(f[i-1]-f[i]);if(i<X-1)v+=base*right*(f[i+1]-f[i]);else if(outerBC)v+=2.0f*base*right*((*outerBC)-f[i]);delta[i]+=v;}
    }
    return 1;
}
PAL_API int32_t pal_pd_diffusion_radial_circle(uintptr_t h,float rate,uintptr_t outer){return PDEGrid_DiffusionRadial((PDEGrid*)h,rate,(float*)outer,2,1);}
PAL_API int32_t pal_pd_diffusion_radial_sphere(uintptr_t h,float rate,uintptr_t outer){return PDEGrid_DiffusionRadial((PDEGrid*)h,rate,(float*)outer,3,1);}
PAL_API int32_t pal_pd_diffusion_radial_circle_fast(uintptr_t h,float rate,uintptr_t outer){return PDEGrid_DiffusionRadial((PDEGrid*)h,rate,(float*)outer,2,0);}
PAL_API int32_t pal_pd_diffusion_radial_sphere_fast(uintptr_t h,float rate,uintptr_t outer){return PDEGrid_DiffusionRadial((PDEGrid*)h,rate,(float*)outer,3,0);}


/* Conservative first-order upwind advection. mode: 0=uniform, 1=center field, 2=+face interfaces. */
static int32_t PDEGrid_AdvectionUniform(PDEGrid*g,const float uv[3],float*bc0[3],float*bc1[3],int checked){
    const int D=g->dimension,X=g->dimensions[0],Y=D>1?g->dimensions[1]:1,Z=D>2?g->dimensions[2]:1;
    const int xs=Y*Z,ys=Z;const float ds[3]={g->dt/g->dx,g->dt/g->dy,g->dt/g->dz};
    if(checked){float cfl=0.0f;for(int a=0;a<D;a++){float v=uv[a];if(!isfinite(v))return -1;int n=g->dimensions[a];if(g->wrap[a]||n>1||(v>0.0f?bc1[a]:v<0.0f?bc0[a]:0))cfl+=fabsf(v)*ds[a];}if(cfl>1.000001f)return 0;}
    float *restrict f=g->field,*restrict d=g->deltas;
    const float ax=uv[0]*ds[0],ay=D>1?uv[1]*ds[1]:0.0f,az=D>2?uv[2]*ds[2]:0.0f;
    if(D==1){
        for(int x=0;x<X;x++){
            float c=f[x],dv=0.0f;
            if(ax>0.0f){
                if(x>0)dv+=ax*f[x-1];else if(g->wrap[0])dv+=ax*f[X-1];else if(bc0[0])dv+=ax*PDEGrid_BC(g,bc0[0],0);
                if(x+1<X||g->wrap[0]||bc1[0])dv-=ax*c;
            }else if(ax<0.0f){
                if(x>0||g->wrap[0]||bc0[0])dv+=ax*c;
                if(x+1<X)dv-=ax*f[x+1];else if(g->wrap[0])dv-=ax*f[0];else if(bc1[0])dv-=ax*PDEGrid_BC(g,bc1[0],0);
            }
            d[x]+=dv;
        }
        return 1;
    }
    for(int x=0;x<X;x++)for(int y=0;y<Y;y++){
        int base=(x*Y+y)*Z;
        for(int z=0;z<Z;z++){
            int i=base+z;float c=f[i],dv=0.0f,src;int face;
            /* Low x face: positive velocity is inflow, negative is outflow. */
            if(x>0)src=f[i-xs];else if(g->wrap[0])src=f[((X-1)*Y+y)*Z+z];else if(bc0[0]){face=D==1?0:(D==2?y:y*Z+z);src=PDEGrid_BC(g,bc0[0],face);}else src=NAN;
            if(ax>0.0f){if(!isnan(src))dv+=ax*src;}else if(ax<0.0f){if(!isnan(src))dv+=ax*c;}
            /* High x face: positive velocity is outflow, negative is inflow. */
            if(x+1<X)src=f[i+xs];else if(g->wrap[0])src=f[y*Z+z];else if(bc1[0]){face=D==1?0:(D==2?y:y*Z+z);src=PDEGrid_BC(g,bc1[0],face);}else src=NAN;
            if(ax>0.0f){if(!isnan(src))dv-=ax*c;}else if(ax<0.0f){if(!isnan(src))dv-=ax*src;}
            if(D>1){
                if(y>0)src=f[i-ys];else if(g->wrap[1])src=f[(x*Y+Y-1)*Z+z];else if(bc0[1]){face=D==2?x:x*Z+z;src=PDEGrid_BC(g,bc0[1],face);}else src=NAN;
                if(ay>0.0f){if(!isnan(src))dv+=ay*src;}else if(ay<0.0f){if(!isnan(src))dv+=ay*c;}
                if(y+1<Y)src=f[i+ys];else if(g->wrap[1])src=f[(x*Y)*Z+z];else if(bc1[1]){face=D==2?x:x*Z+z;src=PDEGrid_BC(g,bc1[1],face);}else src=NAN;
                if(ay>0.0f){if(!isnan(src))dv-=ay*c;}else if(ay<0.0f){if(!isnan(src))dv-=ay*src;}
            }
            if(D>2){
                if(z>0)src=f[i-1];else if(g->wrap[2])src=f[base+Z-1];else if(bc0[2]){face=x*Y+y;src=PDEGrid_BC(g,bc0[2],face);}else src=NAN;
                if(az>0.0f){if(!isnan(src))dv+=az*src;}else if(az<0.0f){if(!isnan(src))dv+=az*c;}
                if(z+1<Z)src=f[i+1];else if(g->wrap[2])src=f[base];else if(bc1[2]){face=x*Y+y;src=PDEGrid_BC(g,bc1[2],face);}else src=NAN;
                if(az>0.0f){if(!isnan(src))dv-=az*c;}else if(az<0.0f){if(!isnan(src))dv-=az*src;}
            }
            d[i]+=dv;
        }
    }
    return 1;
}
static int32_t PDEGrid_Advection(PDEGrid*g,int mode,float uv[3],float*vf[3],float*bc0[3],float*bc1[3],int checked){
    if(mode==0)return PDEGrid_AdvectionUniform(g,uv,bc0,bc1,checked);
    int X=g->dimensions[0],Y=g->dimension>1?g->dimensions[1]:1,Z=g->dimension>2?g->dimensions[2]:1;float ds[3]={g->dt/g->dx,g->dt/g->dy,g->dt/g->dz};
    /* Safe mode validates before touching deltas. Uniform velocity has an exact
     * O(1) CFL bound; spatial fields/interfaces require the lattice preflight. */
    if(checked){
        if(mode==0){
            float advectionCourant=0.0f;
            for(int a=0;a<g->dimension;a++){
                float vel=uv[a];if(!isfinite(vel))return -1;
                int n=g->dimensions[a];
                if(g->wrap[a]||n>1||(vel>0.0f?bc1[a]:vel<0.0f?bc0[a]:0))advectionCourant+=fabsf(vel)*ds[a];
            }
            if(advectionCourant>1.000001f)return 0;
        }else{
            for(int a=0;a<g->dimension;a++){if(!vf[a])return -1;for(int32_t i=0;i<g->length;i++)if(!isfinite(vf[a][i]))return -1;}
            for(int x=0;x<X;x++)for(int y=0;y<Y;y++)for(int z=0;z<Z;z++){int i=PDEGrid_ToI(g,x,y,z),q[3]={x,y,z};float localCourant=0;for(int a=0;a<g->dimension;a++)for(int sg=-1;sg<=1;sg+=2){int t=PDEGrid_Wrap(g,q[a]+sg,a),j=-1,fi=i;if(t!=PAL_NULL){int xx=x,yy=y,zz=z;if(a==0)xx=t;else if(a==1)yy=t;else zz=t;j=PDEGrid_ToI(g,xx,yy,zz);if(sg<0)fi=j;}else if(!(sg<0?bc0[a]:bc1[a]))continue;float vel=mode==2?vf[a][fi]:(j>=0?.5f*(vf[a][i]+vf[a][j]):vf[a][i]);float oriented=sg*vel;if(oriented>0)localCourant+=oriented*ds[a];}if(localCourant>1.000001f)return 0;}
        }
    }
    for(int x=0;x<X;x++)for(int y=0;y<Y;y++)for(int z=0;z<Z;z++){int i=(x*Y+y)*Z+z;float c=g->field[i],delta=0;for(int a=0;a<g->dimension;a++)for(int sg=-1;sg<=1;sg+=2){int j=-1,fi=i;if(a==0){if(sg<0){if(x>0)j=i-Y*Z;else if(g->wrap[0])j=i+(X-1)*Y*Z;}else{if(x+1<X)j=i+Y*Z;else if(g->wrap[0])j=i-(X-1)*Y*Z;}}else if(a==1){if(sg<0){if(y>0)j=i-Z;else if(g->wrap[1])j=i+(Y-1)*Z;}else{if(y+1<Y)j=i+Z;else if(g->wrap[1])j=i-(Y-1)*Z;}}else{if(sg<0){if(z>0)j=i-1;else if(g->wrap[2])j=i+Z-1;}else{if(z+1<Z)j=i+1;else if(g->wrap[2])j=i-Z+1;}}if(j>=0){if(sg<0)fi=j;}else if(!(sg<0?bc0[a]:bc1[a]))continue;float vel=mode==2?vf[a][fi]:(j>=0?.5f*(vf[a][i]+vf[a][j]):vf[a][i]);float oriented=sg*vel;if(oriented>0)delta-=oriented*c*ds[a];else if(oriented<0){float src=j>=0?g->field[j]:PDEGrid_BC(g,(sg<0?bc0[a]:bc1[a]),PDEGrid_FaceIndex(g,a,x,y,z));delta+=(-oriented)*src*ds[a];}}g->deltas[i]+=delta;}return 1;
}
PAL_API int32_t pal_pd_advection(uintptr_t h,float vx,float vy,float vz,uintptr_t x0,uintptr_t x1,uintptr_t y0,uintptr_t y1,uintptr_t z0,uintptr_t z1){float u[3]={vx,vy,vz};float*b0[3]={(float*)x0,(float*)y0,(float*)z0},*b1[3]={(float*)x1,(float*)y1,(float*)z1};float*v[3]={0};return PDEGrid_Advection((PDEGrid*)h,0,u,v,b0,b1,1);}
PAL_API int32_t pal_pd_advection_field(uintptr_t h,uintptr_t vx,uintptr_t vy,uintptr_t vz,uintptr_t x0,uintptr_t x1,uintptr_t y0,uintptr_t y1,uintptr_t z0,uintptr_t z1){float u[3]={0};float*v[3]={(float*)vx,(float*)vy,(float*)vz},*b0[3]={(float*)x0,(float*)y0,(float*)z0},*b1[3]={(float*)x1,(float*)y1,(float*)z1};return PDEGrid_Advection((PDEGrid*)h,1,u,v,b0,b1,1);}
PAL_API int32_t pal_pd_advection_interfaces(uintptr_t h,uintptr_t vx,uintptr_t vy,uintptr_t vz,uintptr_t x0,uintptr_t x1,uintptr_t y0,uintptr_t y1,uintptr_t z0,uintptr_t z1){float u[3]={0};float*v[3]={(float*)vx,(float*)vy,(float*)vz},*b0[3]={(float*)x0,(float*)y0,(float*)z0},*b1[3]={(float*)x1,(float*)y1,(float*)z1};return PDEGrid_Advection((PDEGrid*)h,2,u,v,b0,b1,1);}

PAL_API int32_t pal_pd_advection_fast(uintptr_t h,float vx,float vy,float vz,uintptr_t x0,uintptr_t x1,uintptr_t y0,uintptr_t y1,uintptr_t z0,uintptr_t z1){float u[3]={vx,vy,vz};float*b0[3]={(float*)x0,(float*)y0,(float*)z0},*b1[3]={(float*)x1,(float*)y1,(float*)z1};float*v[3]={0};return PDEGrid_Advection((PDEGrid*)h,0,u,v,b0,b1,0);}
PAL_API int32_t pal_pd_advection_field_fast(uintptr_t h,uintptr_t vx,uintptr_t vy,uintptr_t vz,uintptr_t x0,uintptr_t x1,uintptr_t y0,uintptr_t y1,uintptr_t z0,uintptr_t z1){float u[3]={0};float*v[3]={(float*)vx,(float*)vy,(float*)vz},*b0[3]={(float*)x0,(float*)y0,(float*)z0},*b1[3]={(float*)x1,(float*)y1,(float*)z1};return PDEGrid_Advection((PDEGrid*)h,1,u,v,b0,b1,0);}
PAL_API int32_t pal_pd_advection_interfaces_fast(uintptr_t h,uintptr_t vx,uintptr_t vy,uintptr_t vz,uintptr_t x0,uintptr_t x1,uintptr_t y0,uintptr_t y1,uintptr_t z0,uintptr_t z1){float u[3]={0};float*v[3]={(float*)vx,(float*)vy,(float*)vz},*b0[3]={(float*)x0,(float*)y0,(float*)z0},*b1[3]={(float*)x1,(float*)y1,(float*)z1};return PDEGrid_Advection((PDEGrid*)h,2,u,v,b0,b1,0);}

/* ADI: 1D Crank-Nicolson, 2D Peaceman-Rachford. 3D uses Douglas-Gunn directional solves. */
static void PDEGrid_SolveTri(int n,float r,int wrap,int minSet,int maxSet,float minVal,float maxVal,int addBC,float*rhs,float*out,float*cp,float*dp,float*z,float*u,float*bb){
    if(n==1){float diag=1;float d=rhs[0];if(!wrap&&minSet){diag+=r;d+=addBC?r*minVal:0;}if(!wrap&&maxSet){diag+=r;d+=addBC?r*maxVal:0;}out[0]=d/diag;return;}
    float off=-r*.5f;
    if(wrap){/* cyclic tridiagonal via Sherman-Morrison; work arrays are preallocated by the checked caller */float diag=1+r,gamma=-diag,alpha=off,beta=off;for(int i=0;i<n;i++){bb[i]=diag;u[i]=0.0f;}bb[0]-=gamma;bb[n-1]-=alpha*beta/gamma;u[0]=gamma;u[n-1]=alpha;cp[0]=off/bb[0];dp[0]=rhs[0]/bb[0];for(int i=1;i<n;i++){float den=bb[i]-off*cp[i-1];cp[i]=i<n-1?off/den:0;dp[i]=(rhs[i]-off*dp[i-1])/den;}out[n-1]=dp[n-1];for(int i=n-2;i>=0;i--)out[i]=dp[i]-cp[i]*out[i+1];cp[0]=off/bb[0];dp[0]=u[0]/bb[0];for(int i=1;i<n;i++){float den=bb[i]-off*cp[i-1];cp[i]=i<n-1?off/den:0;dp[i]=(u[i]-off*dp[i-1])/den;}z[n-1]=dp[n-1];for(int i=n-2;i>=0;i--)z[i]=dp[i]-cp[i]*z[i+1];float fact=(out[0]+beta*out[n-1]/gamma)/(1+z[0]+beta*z[n-1]/gamma);for(int i=0;i<n;i++)out[i]-=fact*z[i];return;}
    float b=1+r+(minSet?r*.5f:-r*.5f);float d=rhs[0]+(minSet&&addBC?r*minVal:0);cp[0]=off/b;dp[0]=d/b;for(int i=1;i<n;i++){b=1+r;if(i==n-1)b+=maxSet?r*.5f:-r*.5f;d=rhs[i]+(i==n-1&&maxSet&&addBC?r*maxVal:0);float den=b-off*cp[i-1];cp[i]=i<n-1?off/den:0;dp[i]=(d-off*dp[i-1])/den;}out[n-1]=dp[n-1];for(int i=n-2;i>=0;i--)out[i]=dp[i]-cp[i]*out[i+1];
}
PAL_API int32_t pal_pd_diffusion_adi(uintptr_t h,float rate,uintptr_t x0h,uintptr_t x1h,uintptr_t y0h,uintptr_t y1h,uintptr_t z0h,uintptr_t z1h){
    PDEGrid*g=(PDEGrid*)h;float*b0[3]={(float*)x0h,(float*)y0h,(float*)z0h},*b1[3]={(float*)x1h,(float*)y1h,(float*)z1h};int X=g->dimensions[0],Y=g->dimension>1?g->dimensions[1]:1,Z=g->dimension>2?g->dimensions[2]:1,N=g->length,M=X;if(Y>M)M=Y;if(Z>M)M=Z;float *s1=(float*)malloc((size_t)N*sizeof(float)),*s2=(float*)malloc((size_t)N*sizeof(float)),*rhs=(float*)malloc((size_t)M*sizeof(float)),*out=(float*)malloc((size_t)M*sizeof(float)),*cp=(float*)malloc((size_t)M*sizeof(float)),*dp=(float*)malloc((size_t)M*sizeof(float)),*zwork=(float*)malloc((size_t)M*sizeof(float)),*uwork=(float*)malloc((size_t)M*sizeof(float)),*bbwork=(float*)malloc((size_t)M*sizeof(float));if(!s1||!s2||!rhs||!out||!cp||!dp||!zwork||!uwork||!bbwork){free(s1);free(s2);free(rhs);free(out);free(cp);free(dp);free(zwork);free(uwork);free(bbwork);return -2;}float rx=rate*g->dt/(g->dx*g->dx),ry=rate*g->dt/(g->dy*g->dy),rz=rate*g->dt/(g->dz*g->dz);
    if(g->dimension==1){for(int x=0;x<X;x++){float c=g->field[x],v=0;int n=PDEGrid_Wrap(g,x+1,0);if(n!=PAL_NULL)v+=g->field[n]-c;else if(b1[0])v+=2.0f*(PDEGrid_BC(g,b1[0],0)-c);n=PDEGrid_Wrap(g,x-1,0);if(n!=PAL_NULL)v+=g->field[n]-c;else if(b0[0])v+=2.0f*(PDEGrid_BC(g,b0[0],0)-c);rhs[x]=c+.5f*rx*v;}PDEGrid_SolveTri(X,rx,g->wrap[0],b0[0]!=0,b1[0]!=0,b0[0]?PDEGrid_BC(g,b0[0],0):0,b1[0]?PDEGrid_BC(g,b1[0],0):0,1,rhs,out,cp,dp,zwork,uwork,bbwork);for(int x=0;x<X;x++)g->deltas[x]+=out[x]-g->field[x];}
    else if(g->dimension==2){for(int y=0;y<Y;y++){for(int x=0;x<X;x++){int i=x*Y+y;float c=g->field[i],v=0;int n=PDEGrid_Wrap(g,y+1,1);if(n!=PAL_NULL)v+=g->field[x*Y+n]-c;else if(b1[1])v+=2.0f*(PDEGrid_BC(g,b1[1],x)-c);n=PDEGrid_Wrap(g,y-1,1);if(n!=PAL_NULL)v+=g->field[x*Y+n]-c;else if(b0[1])v+=2.0f*(PDEGrid_BC(g,b0[1],x)-c);rhs[x]=c+.5f*ry*v;}PDEGrid_SolveTri(X,rx,g->wrap[0],b0[0]!=0,b1[0]!=0,b0[0]?PDEGrid_BC(g,b0[0],y):0,b1[0]?PDEGrid_BC(g,b1[0],y):0,1,rhs,out,cp,dp,zwork,uwork,bbwork);for(int x=0;x<X;x++)s1[x*Y+y]=out[x];}for(int x=0;x<X;x++){for(int y=0;y<Y;y++){int i=x*Y+y;float c=s1[i],v=0;int n=PDEGrid_Wrap(g,x+1,0);if(n!=PAL_NULL)v+=s1[n*Y+y]-c;else if(b1[0])v+=2.0f*(PDEGrid_BC(g,b1[0],y)-c);n=PDEGrid_Wrap(g,x-1,0);if(n!=PAL_NULL)v+=s1[n*Y+y]-c;else if(b0[0])v+=2.0f*(PDEGrid_BC(g,b0[0],y)-c);rhs[y]=c+.5f*rx*v;}PDEGrid_SolveTri(Y,ry,g->wrap[1],b0[1]!=0,b1[1]!=0,b0[1]?PDEGrid_BC(g,b0[1],x):0,b1[1]?PDEGrid_BC(g,b1[1],x):0,1,rhs,out,cp,dp,zwork,uwork,bbwork);for(int y=0;y<Y;y++){int i=x*Y+y;g->deltas[i]+=out[y]-g->field[i];}}}
    else{/* Douglas-Gunn: explicit full increment followed by three implicit half-step solves */for(int x=0;x<X;x++)for(int y=0;y<Y;y++)for(int z=0;z<Z;z++){int i=PDEGrid_ToI(g,x,y,z),q[3]={x,y,z};float c=g->field[i],v=0,sc[3]={rx,ry,rz};for(int a=0;a<3;a++)for(int sg=-1;sg<=1;sg+=2){int t=PDEGrid_Wrap(g,q[a]+sg,a);if(t!=PAL_NULL){int xx=x,yy=y,zz=z;if(a==0)xx=t;else if(a==1)yy=t;else zz=t;v+=(g->field[PDEGrid_ToI(g,xx,yy,zz)]-c)*sc[a];}else{float*bc=sg<0?b0[a]:b1[a];if(bc)v+=2.0f*(PDEGrid_BC(g,bc,PDEGrid_FaceIndex(g,a,x,y,z))-c)*sc[a];}}s1[i]=v;}for(int y=0;y<Y;y++)for(int z=0;z<Z;z++){for(int x=0;x<X;x++)rhs[x]=s1[PDEGrid_ToI(g,x,y,z)];PDEGrid_SolveTri(X,rx,g->wrap[0],b0[0]!=0,b1[0]!=0,0,0,0,rhs,out,cp,dp,zwork,uwork,bbwork);for(int x=0;x<X;x++)s2[PDEGrid_ToI(g,x,y,z)]=out[x];}for(int x=0;x<X;x++)for(int z=0;z<Z;z++){for(int y=0;y<Y;y++)rhs[y]=s2[PDEGrid_ToI(g,x,y,z)];PDEGrid_SolveTri(Y,ry,g->wrap[1],b0[1]!=0,b1[1]!=0,0,0,0,rhs,out,cp,dp,zwork,uwork,bbwork);for(int y=0;y<Y;y++)s1[PDEGrid_ToI(g,x,y,z)]=out[y];}for(int x=0;x<X;x++)for(int y=0;y<Y;y++){for(int z=0;z<Z;z++)rhs[z]=s1[PDEGrid_ToI(g,x,y,z)];PDEGrid_SolveTri(Z,rz,g->wrap[2],b0[2]!=0,b1[2]!=0,0,0,0,rhs,out,cp,dp,zwork,uwork,bbwork);for(int z=0;z<Z;z++)g->deltas[PDEGrid_ToI(g,x,y,z)]+=out[z];}}
    free(s1);free(s2);free(rhs);free(out);free(cp);free(dp);free(zwork);free(uwork);free(bbwork);return 1;
}
PAL_API int32_t pal_pd_iswrap(uintptr_t h,int32_t d){PDEGrid*g=(PDEGrid*)h;return d>=0&&d<g->dimension?g->wrap[d]:0;}

/* ========================================================================== */
/* Pickle snapshots                                                           */
/* ========================================================================== */
#define PAL_SNAP_MAGIC UINT32_C(0x50414c31)
#define PAL_SNAP_VERSION UINT32_C(1)
typedef struct { uint32_t magic,version,kind,reserved; } PalSnapHeader;
static void pal_snap_header(uint8_t **p,uint32_t kind){PalSnapHeader h={PAL_SNAP_MAGIC,PAL_SNAP_VERSION,kind,0};memcpy(*p,&h,sizeof(h));*p+=sizeof(h);}
static int pal_snap_check(const uint8_t **p,size_t n,uint32_t kind){if(n<sizeof(PalSnapHeader))return 0;PalSnapHeader h;memcpy(&h,*p,sizeof(h));if(h.magic!=PAL_SNAP_MAGIC||h.version!=PAL_SNAP_VERSION||h.kind!=kind)return 0;*p+=sizeof(h);return 1;}
#define SNAP_PUT(p,x) do{memcpy((p),&(x),sizeof(x));(p)+=sizeof(x);}while(0)
#define SNAP_GET(p,x) do{memcpy(&(x),(p),sizeof(x));(p)+=sizeof(x);}while(0)

PAL_API size_t pal_q_snapshot_size(uintptr_t h){IList*q=(IList*)h;return sizeof(PalSnapHeader)+sizeof(int32_t)+(size_t)q->length*sizeof(int32_t);}
PAL_API int32_t pal_q_snapshot(uintptr_t h,uint8_t*out,size_t n){IList*q=(IList*)h;if(n<pal_q_snapshot_size(h))return 0;uint8_t*p=out;pal_snap_header(&p,1);SNAP_PUT(p,q->length);if(q->length)memcpy(p,q->items,(size_t)q->length*4);return 1;}
PAL_API uintptr_t pal_q_restore(const uint8_t*data,size_t n){const uint8_t*p=data;if(!pal_snap_check(&p,n,1))return 0;int32_t len;SNAP_GET(p,len);size_t need=sizeof(PalSnapHeader)+4+(size_t)len*4;if(n<need||len<0)return 0;IList*q=(IList*)pal_q_new(0);if(!q)return 0;if(!IList_Reserve(q,len)){pal_q_free((uintptr_t)q);return 0;}q->length=len;if(len)memcpy(q->items,p,(size_t)len*4);return (uintptr_t)q;}

PAL_API size_t pal_multi_snapshot_size(uintptr_t h){(void)h;return sizeof(PalSnapHeader)+sizeof(int64_t)+sizeof(double);}
PAL_API int32_t pal_multi_snapshot(uintptr_t h,uint8_t*out,size_t n){if(n<pal_multi_snapshot_size(h))return 0;Multinomial*m=(Multinomial*)h;uint8_t*p=out;pal_snap_header(&p,2);SNAP_PUT(p,m->nRemaining);SNAP_PUT(p,m->pRemaining);return 1;}
PAL_API uintptr_t pal_multi_restore(const uint8_t*data,size_t n){const uint8_t*p=data;if(!pal_snap_check(&p,n,2)||n<sizeof(PalSnapHeader)+16)return 0;int64_t nr;double pr;SNAP_GET(p,nr);SNAP_GET(p,pr);uintptr_t h=pal_multi_new(0);if(!h)return 0;Multinomial*m=(Multinomial*)h;m->nRemaining=nr;m->pRemaining=pr;return h;}

PAL_API size_t pal_pg_snapshot_size(uintptr_t h){PopGrid*g=(PopGrid*)h;return sizeof(PalSnapHeader)+sizeof(PopGrid)-2*sizeof(void*)+(size_t)g->length*(g->fieldType+g->deltaType);}
PAL_API int32_t pal_pg_snapshot(uintptr_t h,uint8_t*out,size_t n){PopGrid*g=(PopGrid*)h;if(n<pal_pg_snapshot_size(h))return 0;uint8_t*p=out;pal_snap_header(&p,3);SNAP_PUT(p,g->dimension);memcpy(p,g->dimensions,12);p+=12;memcpy(p,g->wrap,12);p+=12;SNAP_PUT(p,g->length);SNAP_PUT(p,g->capacity);SNAP_PUT(p,g->population);SNAP_PUT(p,g->fieldType);SNAP_PUT(p,g->deltaType);memcpy(p,g->field,(size_t)g->length*g->fieldType);p+=(size_t)g->length*g->fieldType;memcpy(p,g->deltas,(size_t)g->length*g->deltaType);return 1;}
PAL_API uintptr_t pal_pg_restore(const uint8_t*data,size_t n){const uint8_t*p=data;if(!pal_snap_check(&p,n,3))return 0;int32_t dim,dims[3],wrap[3],len;int64_t cap,pop;uint8_t ft,dt;SNAP_GET(p,dim);memcpy(dims,p,12);p+=12;memcpy(wrap,p,12);p+=12;SNAP_GET(p,len);SNAP_GET(p,cap);SNAP_GET(p,pop);SNAP_GET(p,ft);SNAP_GET(p,dt);size_t need=sizeof(PalSnapHeader)+4+24+4+8+8+1+1+(size_t)len*(ft+dt);if(n<need||dim<1||dim>3)return 0;int32_t sd[3];for(int d=0;d<dim;d++)sd[d]=wrap[d]?-dims[d]:dims[d];uintptr_t h=pal_pg_new(sd,dim,cap);if(!h)return 0;PopGrid*g=(PopGrid*)h;if(g->length!=len||g->fieldType!=ft||g->deltaType!=dt){pal_pg_free(h);return 0;}g->population=pop;memcpy(g->field,p,(size_t)len*ft);p+=(size_t)len*ft;memcpy(g->deltas,p,(size_t)len*dt);return h;}

PAL_API size_t pal_pd_snapshot_size(uintptr_t h){PDEGrid*g=(PDEGrid*)h;return sizeof(PalSnapHeader)+4+24+4+16+(size_t)g->length*8;}
PAL_API int32_t pal_pd_snapshot(uintptr_t h,uint8_t*out,size_t n){PDEGrid*g=(PDEGrid*)h;if(n<pal_pd_snapshot_size(h))return 0;uint8_t*p=out;pal_snap_header(&p,4);SNAP_PUT(p,g->dimension);memcpy(p,g->dimensions,12);p+=12;memcpy(p,g->wrap,12);p+=12;SNAP_PUT(p,g->length);SNAP_PUT(p,g->dt);SNAP_PUT(p,g->dx);SNAP_PUT(p,g->dy);SNAP_PUT(p,g->dz);memcpy(p,g->field,(size_t)g->length*4);p+=(size_t)g->length*4;memcpy(p,g->deltas,(size_t)g->length*4);return 1;}
PAL_API uintptr_t pal_pd_restore(const uint8_t*data,size_t n){const uint8_t*p=data;if(!pal_snap_check(&p,n,4))return 0;int32_t dim,dims[3],wrap[3],len;float dt,dx,dy,dz;SNAP_GET(p,dim);memcpy(dims,p,12);p+=12;memcpy(wrap,p,12);p+=12;SNAP_GET(p,len);SNAP_GET(p,dt);SNAP_GET(p,dx);SNAP_GET(p,dy);SNAP_GET(p,dz);size_t need=sizeof(PalSnapHeader)+4+24+4+16+(size_t)len*8;if(n<need||dim<1||dim>3)return 0;int32_t sd[3];for(int d=0;d<dim;d++)sd[d]=wrap[d]?-dims[d]:dims[d];uintptr_t h=pal_pd_new(sd,dim);if(!h)return 0;PDEGrid*g=(PDEGrid*)h;if(g->length!=len){pal_pd_free(h);return 0;}g->dt=dt;g->dx=dx;g->dy=dy;g->dz=dz;memcpy(g->field,p,(size_t)len*4);p+=(size_t)len*4;memcpy(g->deltas,p,(size_t)len*4);return h;}

PAL_API size_t pal_ag_snapshot_size(uintptr_t h){AgentGrid*g=(AgentGrid*)h;size_t s=sizeof(PalSnapHeader)+4+24+4*8; s+=(size_t)g->capacity*g->nFloatProperties*4+(size_t)g->capacity*N_AGENT_INT_PROPERTIES*4+(size_t)g->population*4+(size_t)g->nDead*4;if(g->dimension>0){s+=(size_t)g->length*4;if(g->stackable)s+=(size_t)g->length*4;}return s;}
PAL_API int32_t pal_ag_snapshot(uintptr_t h,uint8_t*out,size_t n){AgentGrid*g=(AgentGrid*)h;if(n<pal_ag_snapshot_size(h))return 0;uint8_t*p=out;pal_snap_header(&p,5);SNAP_PUT(p,g->dimension);memcpy(p,g->dimensions,12);p+=12;memcpy(p,g->wrap,12);p+=12;SNAP_PUT(p,g->length);SNAP_PUT(p,g->stackable);SNAP_PUT(p,g->nUserProperties);SNAP_PUT(p,g->nFloatProperties);SNAP_PUT(p,g->capacity);SNAP_PUT(p,g->population);SNAP_PUT(p,g->nAgents);SNAP_PUT(p,g->nDead);size_t nf=(size_t)g->capacity*g->nFloatProperties*4,ni=(size_t)g->capacity*N_AGENT_INT_PROPERTIES*4;memcpy(p,g->floatProperties,nf);p+=nf;memcpy(p,g->intProperties,ni);p+=ni;memcpy(p,g->aliveAgents,(size_t)g->population*4);p+=(size_t)g->population*4;memcpy(p,g->deadAgents,(size_t)g->nDead*4);p+=(size_t)g->nDead*4;if(g->dimension>0){memcpy(p,g->grid,(size_t)g->length*4);p+=(size_t)g->length*4;if(g->stackable)memcpy(p,g->counts,(size_t)g->length*4);}return 1;}
PAL_API uintptr_t pal_ag_restore(const uint8_t*data,size_t n){const uint8_t*p=data;if(!pal_snap_check(&p,n,5))return 0;int32_t dim,dims[3],wrap[3],len,stack,nup,nfp,cap,pop,na,nd;SNAP_GET(p,dim);memcpy(dims,p,12);p+=12;memcpy(wrap,p,12);p+=12;SNAP_GET(p,len);SNAP_GET(p,stack);SNAP_GET(p,nup);SNAP_GET(p,nfp);SNAP_GET(p,cap);SNAP_GET(p,pop);SNAP_GET(p,na);SNAP_GET(p,nd);if(dim<0||dim>3||cap<0||pop<0||nd<0)return 0;size_t need=sizeof(PalSnapHeader)+4+24+32+(size_t)cap*nfp*4+(size_t)cap*N_AGENT_INT_PROPERTIES*4+(size_t)(pop+nd)*4+(dim>0?(size_t)len*4:0)+(dim>0&&stack?(size_t)len*4:0);if(n<need)return 0;int32_t sd[3];for(int d=0;d<dim;d++)sd[d]=wrap[d]?-dims[d]:dims[d];uintptr_t h=pal_ag_new(sd,dim,nup,stack);if(!h)return 0;AgentGrid*g=(AgentGrid*)h;if(g->capacity<cap){while(g->capacity<cap)if(!AgentGrid_Grow(g)){pal_ag_free(h);return 0;}}g->population=pop;g->nAgents=na;g->nDead=nd;size_t nf=(size_t)cap*nfp*4,ni=(size_t)cap*N_AGENT_INT_PROPERTIES*4;memcpy(g->floatProperties,p,nf);p+=nf;memcpy(g->intProperties,p,ni);p+=ni;memcpy(g->aliveAgents,p,(size_t)pop*4);p+=(size_t)pop*4;memcpy(g->deadAgents,p,(size_t)nd*4);p+=(size_t)nd*4;if(dim>0){memcpy(g->grid,p,(size_t)len*4);p+=(size_t)len*4;if(stack)memcpy(g->counts,p,(size_t)len*4);}return h;}
PAL_API int32_t pal_pg_all_count(uintptr_t h){PopGrid*g=(PopGrid*)h;int32_t n=0;for(int32_t i=0;i<g->length;i++)if(PopGrid_FieldGet(g,i)!=0)n++;return n;}
PAL_API int32_t pal_pg_all_copy(uintptr_t h,int32_t*out){PopGrid*g=(PopGrid*)h;int32_t n=0;for(int32_t i=0;i<g->length;i++)if(PopGrid_FieldGet(g,i)!=0)out[n++]=i;return n;}

/* ========================================================================== */
/* Grid: generic typed lattice storage                                        */
/* ========================================================================== */
typedef struct {
    int32_t dimension;
    int32_t dimensions[3];
    int32_t wrap[3];
    int32_t length;
    int32_t type;
    size_t itemSize;
    void *data;
} Grid;

PAL_API uintptr_t pal_grid_new(const int32_t *dimensions,int32_t dimension,int32_t type,size_t itemSize){
    if(dimension<1||dimension>3||itemSize==0)return 0;
    Grid*g=(Grid*)calloc(1,sizeof(Grid)); if(!g)return 0;
    g->dimension=dimension;g->length=1;g->type=type;g->itemSize=itemSize;
    for(int32_t d=0;d<dimension;d++){
        int32_t v=dimensions[d];g->wrap[d]=v<0;g->dimensions[d]=v<0?-v:v;
        if(g->dimensions[d]<=0||g->length>INT32_MAX/g->dimensions[d]){free(g);return 0;}
        g->length*=g->dimensions[d];
    }
    g->data=calloc((size_t)g->length,itemSize); if(!g->data){free(g);return 0;}
    return (uintptr_t)g;
}
PAL_API void pal_grid_free(uintptr_t h){Grid*g=(Grid*)h;if(!g)return;free(g->data);free(g);}
PAL_API uintptr_t pal_grid_data(uintptr_t h){return (uintptr_t)((Grid*)h)->data;}
PAL_API int32_t pal_grid_len(uintptr_t h){return ((Grid*)h)->length;}
PAL_API int32_t pal_grid_dim(uintptr_t h,int32_t d){Grid*g=(Grid*)h;return d>=0&&d<g->dimension?g->dimensions[d]:PAL_NULL;}
