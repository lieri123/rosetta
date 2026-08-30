#include "geom.h"
#include "threshold.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Gaussian elimination with partial pivoting on an n x (n+1) augmented matrix.
 * n is 8 here, so the cubic cost is irrelevant and clarity wins. */
static int solve_linear(double *a, int n, double *x)
{
    for (int col = 0; col < n; col++) {
        int pivot = col;
        double best = fabs(a[col * (n + 1) + col]);
        for (int row = col + 1; row < n; row++) {
            double v = fabs(a[row * (n + 1) + col]);
            if (v > best) {
                best = v;
                pivot = row;
            }
        }
        if (best < 1e-12) return -1;

        if (pivot != col) {
            for (int c = col; c <= n; c++) {
                double tmp = a[col * (n + 1) + c];
                a[col * (n + 1) + c] = a[pivot * (n + 1) + c];
                a[pivot * (n + 1) + c] = tmp;
            }
        }

        double diag = a[col * (n + 1) + col];
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = a[row * (n + 1) + col] / diag;
            if (factor == 0.0) continue;
            for (int c = col; c <= n; c++)
                a[row * (n + 1) + c] -= factor * a[col * (n + 1) + c];
        }
    }

    for (int i = 0; i < n; i++) x[i] = a[i * (n + 1) + n] / a[i * (n + 1) + i];
    return 0;
}

int homography_solve(const Point from[4], const Point to[4], double h[9])
{
    /* Each correspondence contributes two rows of the DLT system:
     *   x' = (h0 x + h1 y + h2) / (h6 x + h7 y + 1)
     *   y' = (h3 x + h4 y + h5) / (h6 x + h7 y + 1)
     * cleared of denominators. */
    double a[8 * 9];
    memset(a, 0, sizeof(a));

    for (int i = 0; i < 4; i++) {
        double x = from[i].x, y = from[i].y;
        double u = to[i].x, v = to[i].y;

        double *r0 = a + (2 * i) * 9;
        r0[0] = x; r0[1] = y; r0[2] = 1;
        r0[6] = -u * x; r0[7] = -u * y; r0[8] = u;

        double *r1 = a + (2 * i + 1) * 9;
        r1[3] = x; r1[4] = y; r1[5] = 1;
        r1[6] = -v * x; r1[7] = -v * y; r1[8] = v;
    }

    double sol[8];
    if (solve_linear(a, 8, sol) != 0) return -1;
    memcpy(h, sol, sizeof(sol));
    h[8] = 1.0;
    return 0;
}

Point homography_apply(const double h[9], Point p)
{
    double denom = h[6] * p.x + h[7] * p.y + h[8];
    if (fabs(denom) < 1e-12) denom = 1e-12;
    Point out;
    out.x = (h[0] * p.x + h[1] * p.y + h[2]) / denom;
    out.y = (h[3] * p.x + h[4] * p.y + h[5]) / denom;
    return out;
}

double quad_area(const Quad *q)
{
    /* Shoelace, absolute value so winding order does not matter. */
    double s = 0.0;
    for (int i = 0; i < 4; i++) {
        const Point *a = &q->c[i];
        const Point *b = &q->c[(i + 1) % 4];
        s += a->x * b->y - b->x * a->y;
    }
    return fabs(s) * 0.5;
}

static double dist(Point a, Point b)
{
    double dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

void quad_output_size(const Quad *q, int *w, int *h)
{
    double top = dist(q->c[0], q->c[1]);
    double bottom = dist(q->c[3], q->c[2]);
    double left = dist(q->c[0], q->c[3]);
    double right = dist(q->c[1], q->c[2]);

    /* The longer of each opposing pair: the near edge of a tilted page is the
     * one closest to its true length, so taking the max avoids squashing. */
    int ow = (int)((top > bottom ? top : bottom) + 0.5);
    int oh = (int)((left > right ? left : right) + 0.5);
    if (ow < 1) ow = 1;
    if (oh < 1) oh = 1;
    if (w) *w = ow;
    if (h) *h = oh;
}

/* Largest 4-connected component of pixels brighter than `threshold`, returned
 * as a mask. Iterative flood fill; the explicit stack keeps a 12MP photo from
 * blowing the call stack. */
static unsigned char *largest_bright_component(const Image *im, int threshold, long *size_out)
{
    size_t n = (size_t)im->w * (size_t)im->h;
    unsigned char *label = (unsigned char *)calloc(n, 1);
    unsigned char *best = (unsigned char *)calloc(n, 1);
    int *stack = (int *)malloc(n * sizeof(int));
    if (!label || !best || !stack) {
        free(label);
        free(best);
        free(stack);
        return NULL;
    }

    long best_size = 0;
    for (size_t seed = 0; seed < n; seed++) {
        if (label[seed] || im->px[seed] <= threshold) continue;

        long count = 0;
        int top = 0;
        stack[top++] = (int)seed;
        label[seed] = 1;

        /* Mark the component with 2, then promote it to `best` if it wins. */
        while (top > 0) {
            int idx = stack[--top];
            label[idx] = 2;
            count++;
            int x = idx % im->w;
            int y = idx / im->w;

            const int dx[4] = {1, -1, 0, 0};
            const int dy[4] = {0, 0, 1, -1};
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx < 0 || ny < 0 || nx >= im->w || ny >= im->h) continue;
                size_t ni = (size_t)ny * im->w + nx;
                if (label[ni] || im->px[ni] <= threshold) continue;
                label[ni] = 1;
                stack[top++] = (int)ni;
            }
        }

        if (count > best_size) {
            best_size = count;
            for (size_t i = 0; i < n; i++) best[i] = (label[i] == 2) ? 1 : 0;
        }
        /* Retire this component so later seeds do not revisit it. */
        for (size_t i = 0; i < n; i++)
            if (label[i] == 2) label[i] = 3;
    }

    free(label);
    free(stack);
    if (size_out) *size_out = best_size;
    return best;
}

int detect_page_quad(const Image *im, Quad *out)
{
    if (!img_ok(im) || im->w < 16 || im->h < 16) return -1;

    /* Detection runs on a downscaled copy: the corners only need to be
     * accurate to a pixel or two at full resolution, and the flood fill is the
     * expensive step. */
    double scale = 1.0;
    Image small = img_downscale_to(im, 640, &scale);
    if (!img_ok(&small)) return -1;

    /* Valley rather than Otsu: the page and the surface it sits on have very
     * different brightness spreads once the lighting is uneven, and Otsu
     * responds by splitting the page itself. See threshold.h. */
    int t = valley_threshold(&small);
    long area = 0;
    unsigned char *mask = largest_bright_component(&small, t, &area);
    if (!mask) {
        img_free(&small);
        return -1;
    }

    long total = (long)small.w * small.h;
    /* Too small to be a page, or so large that the photo is already a scan. */
    if (area < total / 10 || area > (long)(total * 0.985)) {
        free(mask);
        img_free(&small);
        return -1;
    }

    /* Corners as extremes of the rotated coordinates x+y and x-y. This is the
     * cheap standard trick: it is exact for a convex quad and degrades
     * gracefully on a ragged mask, where a full convex hull would not. */
    double best_sum_min = 1e18, best_sum_max = -1e18;
    double best_dif_min = 1e18, best_dif_max = -1e18;
    Point tl = {0, 0}, br = {0, 0}, tr = {0, 0}, bl = {0, 0};

    for (int y = 0; y < small.h; y++) {
        for (int x = 0; x < small.w; x++) {
            if (!mask[(size_t)y * small.w + x]) continue;
            double s = x + y, d = x - y;
            if (s < best_sum_min) { best_sum_min = s; tl.x = x; tl.y = y; }
            if (s > best_sum_max) { best_sum_max = s; br.x = x; br.y = y; }
            if (d > best_dif_max) { best_dif_max = d; tr.x = x; tr.y = y; }
            if (d < best_dif_min) { best_dif_min = d; bl.x = x; bl.y = y; }
        }
    }
    free(mask);
    img_free(&small);

    Quad q;
    q.c[0] = tl;
    q.c[1] = tr;
    q.c[2] = br;
    q.c[3] = bl;

    /* Back to full-resolution coordinates. */
    for (int i = 0; i < 4; i++) {
        q.c[i].x /= scale;
        q.c[i].y /= scale;
    }

    /* Reject degenerate quads: any edge shorter than 5% of the short side is
     * almost certainly a mask artefact rather than a page corner. */
    double shortest_side = im->w < im->h ? im->w : im->h;
    for (int i = 0; i < 4; i++) {
        if (dist(q.c[i], q.c[(i + 1) % 4]) < shortest_side * 0.05) return -1;
    }
    double frame = (double)im->w * (double)im->h;
    if (quad_area(&q) < frame * 0.10) return -1;
    /* A flatbed scan has no background to find: the "page" is the whole frame,
     * and warping it would be a resample for nothing. Decline rather than
     * pretend we corrected something. Note this tests the quad, not the
     * component -- ink makes holes in the component but does not move the
     * corners. */
    if (quad_area(&q) > frame * 0.97) return -1;

    *out = q;
    return 0;
}

Image warp_quad(const Image *im, const Quad *quad, int out_w, int out_h)
{
    Image out = {0, 0, NULL};
    if (!img_ok(im)) return out;

    if (out_w <= 0 || out_h <= 0) quad_output_size(quad, &out_w, &out_h);

    /* Solve destination -> source directly so the loop below is a gather: for
     * every output pixel we know exactly which source point to sample, and no
     * output pixel is left unwritten (which a forward scatter would do). */
    Point dst[4] = {
        {0.0, 0.0},
        {(double)out_w - 1, 0.0},
        {(double)out_w - 1, (double)out_h - 1},
        {0.0, (double)out_h - 1},
    };
    double h[9];
    if (homography_solve(dst, quad->c, h) != 0) return out;

    out = img_new(out_w, out_h);
    if (!out.px) return out;

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            Point p = {(double)x, (double)y};
            Point s = homography_apply(h, p);
            /* Anything outside the source is paper, not ink. */
            out.px[(size_t)y * out_w + x] = img_sample(im, s.x, s.y, 255);
        }
    }
    return out;
}
