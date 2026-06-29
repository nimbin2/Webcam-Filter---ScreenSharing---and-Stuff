#include "vcam.h"
#include <stdlib.h>
#include <string.h>

Frame *frame_alloc(int width, int height)
{
    Frame *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    f->width  = width;
    f->height = height;
    f->stride = width * 3; /* BGR24 */
    f->data   = malloc((size_t)f->stride * height);
    if (!f->data) { free(f); return NULL; }
    return f;
}

void frame_free(Frame *f)
{
    if (!f) return;
    free(f->data);
    free(f);
}
