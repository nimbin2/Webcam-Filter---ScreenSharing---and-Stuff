#include "vcam.h"
#include <string.h>
#include <stdio.h>

void registry_init(OverlayRegistry *r)
{
    memset(r, 0, sizeof *r);
    pthread_mutex_init(&r->lock, NULL);
}

void registry_destroy(OverlayRegistry *r)
{
    /* Grab all overlays under the lock, then free them outside it.
     * This avoids holding the lock during overlay_free, which can block
     * (e.g. waitpid on a screen-capture child). */
    Overlay *to_free[MAX_OVERLAYS];
    int n = 0;

    pthread_mutex_lock(&r->lock);
    for (int i = 0; i < r->count; i++)
        to_free[n++] = r->slots[i].overlay;
    r->count = 0;
    pthread_mutex_unlock(&r->lock);

    for (int i = 0; i < n; i++)
        overlay_free(to_free[i]);

    pthread_mutex_destroy(&r->lock);
}

bool registry_add(OverlayRegistry *r, const char *id, Overlay *o)
{
    pthread_mutex_lock(&r->lock);
    if (r->count >= MAX_OVERLAYS) {
        pthread_mutex_unlock(&r->lock);
        fprintf(stderr, "[registry] full (max %d)\n", MAX_OVERLAYS);
        return false;
    }
    for (int i = 0; i < r->count; i++) {
        if (strcmp(r->slots[i].id, id) == 0) {
            pthread_mutex_unlock(&r->lock);
            fprintf(stderr, "[registry] id '%s' already exists\n", id);
            return false;
        }
    }
    OverlaySlot *s = &r->slots[r->count++];
    strncpy(s->id, id, sizeof s->id - 1);
    s->overlay = o;
    pthread_mutex_unlock(&r->lock);
    return true;
}

bool registry_remove(OverlayRegistry *r, const char *id)
{
    /* Pluck the overlay out of the list under the lock, then free it
     * outside the lock so overlay_free can block safely (e.g. waitpid). */
    Overlay *victim = NULL;

    pthread_mutex_lock(&r->lock);
    for (int i = 0; i < r->count; i++) {
        if (strcmp(r->slots[i].id, id) == 0) {
            victim = r->slots[i].overlay;
            for (int j = i; j < r->count - 1; j++)
                r->slots[j] = r->slots[j+1];
            r->count--;
            break;
        }
    }
    pthread_mutex_unlock(&r->lock);

    if (!victim) return false;
    overlay_free(victim);   /* may block (waitpid), but lock is released */
    return true;
}

Overlay *registry_find(OverlayRegistry *r, const char *id)
{
    /* Caller must hold r->lock */
    for (int i = 0; i < r->count; i++)
        if (strcmp(r->slots[i].id, id) == 0)
            return r->slots[i].overlay;
    return NULL;
}

/* Move overlay with this id to the END of the registry list (drawn last
 * = on top). Returns false if id not found. */
bool registry_raise(OverlayRegistry *r, const char *id)
{
    pthread_mutex_lock(&r->lock);
    int idx = -1;
    for (int i = 0; i < r->count; i++)
        if (strcmp(r->slots[i].id, id) == 0) { idx = i; break; }
    if (idx < 0) { pthread_mutex_unlock(&r->lock); return false; }
    if (idx == r->count - 1) { pthread_mutex_unlock(&r->lock); return true; }

    OverlaySlot tmp = r->slots[idx];
    for (int i = idx; i < r->count - 1; i++) r->slots[i] = r->slots[i+1];
    r->slots[r->count - 1] = tmp;
    pthread_mutex_unlock(&r->lock);
    return true;
}

/* Move overlay with this id to the FRONT (drawn first = at the bottom). */
bool registry_lower(OverlayRegistry *r, const char *id)
{
    pthread_mutex_lock(&r->lock);
    int idx = -1;
    for (int i = 0; i < r->count; i++)
        if (strcmp(r->slots[i].id, id) == 0) { idx = i; break; }
    if (idx < 0) { pthread_mutex_unlock(&r->lock); return false; }
    if (idx == 0)  { pthread_mutex_unlock(&r->lock); return true; }

    OverlaySlot tmp = r->slots[idx];
    for (int i = idx; i > 0; i--) r->slots[i] = r->slots[i-1];
    r->slots[0] = tmp;
    pthread_mutex_unlock(&r->lock);
    return true;
}

bool registry_set_z(OverlayRegistry *r, const char *id, int target)
{
    pthread_mutex_lock(&r->lock);
    int idx = -1;
    for (int i = 0; i < r->count; i++)
        if (strcmp(r->slots[i].id, id) == 0) { idx = i; break; }
    if (idx < 0) { pthread_mutex_unlock(&r->lock); return false; }

    /* Negative target counts from the top: -1 = top, -2 = second-from-top */
    if (target < 0) target = r->count + target;
    if (target < 0)             target = 0;
    if (target >= r->count)     target = r->count - 1;
    if (target == idx) { pthread_mutex_unlock(&r->lock); return true; }

    OverlaySlot tmp = r->slots[idx];
    if (target > idx) {
        for (int i = idx; i < target; i++) r->slots[i] = r->slots[i+1];
    } else {
        for (int i = idx; i > target; i--) r->slots[i] = r->slots[i-1];
    }
    r->slots[target] = tmp;
    pthread_mutex_unlock(&r->lock);
    return true;
}
