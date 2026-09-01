#include "lines.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define INK_LEVEL 128

LineParams line_params_default(const Image *binary)
{
    LineParams p;
    p.max_angle_deg = 6.0;
    p.angle_step = 0.5;
    /* A strikethrough crosses at least one whole word, so the length floor is
     * set at word scale: about a fifteenth of the page width. This threshold
     * does most of the discriminating. Set it much lower and the detector
     * starts reporting the crossbar of a t, the middle of an e, or a couple of
     * collinear letter parts in adjacent characters -- all of which are
     * genuinely thin horizontal ink, and none of which are strikethroughs. */
    p.min_length = binary && binary->w > 0 ? binary->w / 15 : 50;
    if (p.min_length < 24) p.min_length = 24;
    p.max_thickness = binary && binary->h > 0 ? binary->h / 150 : 4;
    if (p.max_thickness < 3) p.max_thickness = 3;
    if (p.max_thickness > 14) p.max_thickness = 14;
    p.min_fill = 0.75;
    /* A stroke drawn through words is long relative to how thick it is. Two or
     * three letter parts that happen to line up are not: on a synthetic page
     * the real strikethrough came out at 128:1 and the false positives at 10:1
     * to 23:1. This is a guard rather than the decision -- the real backstop
     * is in the service, where a rule only deletes a word if it also bisects
     * that word's box at mid-height. Both exist because a false strikethrough
     * silently deletes text the writer wanted to keep, which is far worse than
     * leaving a crossed-out word in. */
    p.min_aspect = 18.0;
    p.max_lines = 64;
    return p;
}

static void segs_push(LineSegs *segs, LineSeg s)
{
    if (segs->count == segs->cap) {
        int cap = segs->cap ? segs->cap * 2 : 16;
        LineSeg *items = (LineSeg *)realloc(segs->items, (size_t)cap * sizeof(LineSeg));
        if (!items) return;
        segs->items = items;
        segs->cap = cap;
    }
    segs->items[segs->count++] = s;
}

void linesegs_free(LineSegs *segs)
{
    if (!segs) return;
    free(segs->items);
    segs->items = NULL;
    segs->count = segs->cap = 0;
}

static int is_ink(const Image *im, int x, int y)
{
    if (x < 0 || y < 0 || x >= im->w || y >= im->h) return 0;
    return im->px[(size_t)y * im->w + x] < INK_LEVEL;
}

/* Vertical extent of the ink blob through (x, y), capped so a black region
 * does not cost us a full column scan. */
static int stroke_thickness_at(const Image *im, int x, int y, int cap)
{
    if (!is_ink(im, x, y)) {
        /* Tolerate a one pixel miss: the line may sit between two rows. */
        if (is_ink(im, x, y - 1)) y -= 1;
        else if (is_ink(im, x, y + 1)) y += 1;
        else return 0;
    }
    int up = 0, down = 0;
    while (up < cap && is_ink(im, x, y - up - 1)) up++;
    while (down < cap && is_ink(im, x, y + down + 1)) down++;
    return up + down + 1;
}

static int cmp_int(const void *a, const void *b)
{
    int ia = *(const int *)a, ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

/* Walk one candidate line, cut it into ink runs, and keep the runs that look
 * like a drawn rule rather than a row of letters. */
static void extract_runs(const Image *im, double slope, double intercept,
                         double angle_deg, int votes, const LineParams *p,
                         LineSegs *out)
{
    int run_start = -1;
    int gap = 0;
    /* Allow short gaps: pen pressure varies and binarisation breaks strokes. */
    const int max_gap = p->min_length / 4 + 2;

    for (int x = 0; x <= im->w; x++) {
        int hit = 0;
        if (x < im->w) {
            int y = (int)(slope * x + intercept + 0.5);
            hit = is_ink(im, x, y) || is_ink(im, x, y - 1) || is_ink(im, x, y + 1);
        }

        if (hit) {
            if (run_start < 0) run_start = x;
            gap = 0;
            continue;
        }

        if (run_start < 0) continue;
        if (++gap <= max_gap && x < im->w) continue;

        int run_end = x - gap;
        int start = run_start;
        int length = run_end - start + 1;
        run_start = -1;
        gap = 0;
        if (length < p->min_length) continue;

        /* Measure fill and thickness over the run. A text row also produces a
         * long "line" of hits, but it is thick and patchy where a rule is thin
         * and solid; the thickness median is what separates them. */
        int step = length / 32;
        if (step < 1) step = 1;

        int samples = 0, filled = 0;
        int thick[64];
        int nthick = 0;

        for (int sx = start; sx <= run_end; sx += step) {
            int y = (int)(slope * sx + intercept + 0.5);
            samples++;
            int t = stroke_thickness_at(im, sx, y, p->max_thickness * 3 + 2);
            if (t > 0) {
                filled++;
                if (nthick < 64) thick[nthick++] = t;
            }
        }

        if (!samples || nthick < 3) continue;
        if ((double)filled / (double)samples < p->min_fill) continue;

        qsort(thick, (size_t)nthick, sizeof(int), cmp_int);
        int median = thick[nthick / 2];
        if (median > p->max_thickness) continue;
        if (median > 0 && (double)length / (double)median < p->min_aspect) continue;

        LineSeg seg;
        seg.x0 = start;
        seg.y0 = slope * seg.x0 + intercept;
        seg.x1 = run_end;
        seg.y1 = slope * seg.x1 + intercept;
        seg.thickness = median;
        seg.angle_deg = angle_deg;
        seg.votes = votes;
        segs_push(out, seg);
    }
}

/* One thick stroke supports several nearby (angle, intercept) cells, so the
 * same physical rule can come back two or three times at slightly different
 * angles. Keep the longest version of each and drop the rest: overlapping in x
 * and running within a couple of stroke widths of each other in y means the
 * same ink. */
static void dedupe_segments(LineSegs *segs, const LineParams *p)
{
    if (segs->count < 2) return;

    /* Longest first, so the survivor of each cluster is the best measurement
     * of it. Insertion sort: the list is at most max_lines long. */
    for (int i = 1; i < segs->count; i++) {
        LineSeg key = segs->items[i];
        double klen = key.x1 - key.x0;
        int j = i - 1;
        while (j >= 0 && (segs->items[j].x1 - segs->items[j].x0) < klen) {
            segs->items[j + 1] = segs->items[j];
            j--;
        }
        segs->items[j + 1] = key;
    }

    double tol = p->max_thickness * 2.5 + 1.0;
    int kept = 0;
    for (int i = 0; i < segs->count; i++) {
        LineSeg *cand = &segs->items[i];
        int duplicate = 0;

        for (int j = 0; j < kept; j++) {
            LineSeg *have = &segs->items[j];
            double lo = cand->x0 > have->x0 ? cand->x0 : have->x0;
            double hi = cand->x1 < have->x1 ? cand->x1 : have->x1;
            double overlap = hi - lo;
            if (overlap <= 0.0) continue;

            double cand_len = cand->x1 - cand->x0;
            if (overlap < 0.5 * cand_len) continue;

            /* Compare the two lines where they actually overlap, not at x=0:
             * a one degree difference in angle is metres apart by then. */
            double mid = (lo + hi) / 2.0;
            double cy = cand->y0 + (cand->y1 - cand->y0) *
                        (cand_len > 0 ? (mid - cand->x0) / cand_len : 0.0);
            double have_len = have->x1 - have->x0;
            double hy = have->y0 + (have->y1 - have->y0) *
                        (have_len > 0 ? (mid - have->x0) / have_len : 0.0);

            if (fabs(cy - hy) <= tol) {
                duplicate = 1;
                break;
            }
        }

        if (!duplicate) segs->items[kept++] = *cand;
    }
    segs->count = kept;
}

int detect_horizontal_rules(const Image *binary, const LineParams *params, LineSegs *out)
{
    if (!img_ok(binary) || !out) return -1;
    memset(out, 0, sizeof(*out));

    LineParams p = params ? *params : line_params_default(binary);
    if (p.angle_step <= 0.0) p.angle_step = 0.5;
    if (p.max_angle_deg <= 0.0) p.max_angle_deg = 6.0;
    if (p.min_length <= 0) p.min_length = line_params_default(binary).min_length;
    if (p.max_thickness <= 0) p.max_thickness = 4;
    if (p.min_fill <= 0.0) p.min_fill = 0.75;
    if (p.min_aspect <= 0.0) p.min_aspect = 18.0;
    if (p.max_lines <= 0) p.max_lines = 64;

    int ntheta = (int)(2 * p.max_angle_deg / p.angle_step) + 1;
    /* Parametrised by the intercept at x=0 rather than the usual perpendicular
     * distance: over a narrow angle band the two are equivalent, and the
     * intercept makes the walk in extract_runs trivial. */
    int margin = (int)(binary->w * tan(p.max_angle_deg * M_PI / 180.0)) + 2;
    int ncbins = binary->h + 2 * margin;
    if (ntheta <= 0 || ncbins <= 0) return -1;

    int *acc = (int *)calloc((size_t)ntheta * ncbins, sizeof(int));
    double *slopes = (double *)malloc((size_t)ntheta * sizeof(double));
    if (!acc || !slopes) {
        free(acc);
        free(slopes);
        return -1;
    }

    for (int t = 0; t < ntheta; t++) {
        double angle = -p.max_angle_deg + t * p.angle_step;
        slopes[t] = tan(angle * M_PI / 180.0);
    }

    /* Ignore a band around the edge. After perspective rectification the seam
     * where the warped page meets the fill is a long horizontal run of ink,
     * and nothing about it is a pen stroke. */
    int border_x = binary->w / 50 + 1;
    int border_y = binary->h / 50 + 1;

    for (int y = border_y; y < binary->h - border_y; y++) {
        const unsigned char *row = binary->px + (size_t)y * binary->w;
        for (int x = border_x; x < binary->w - border_x; x++) {
            if (row[x] >= INK_LEVEL) continue;
            for (int t = 0; t < ntheta; t++) {
                int c = (int)(y - slopes[t] * x + 0.5) + margin;
                if (c >= 0 && c < ncbins) acc[(size_t)t * ncbins + c]++;
            }
        }
    }

    /* Collect cells with enough support, strongest first, then suppress
     * neighbours so one thick rule does not report five times. */
    typedef struct { int t, c, votes; } Peak;
    Peak *peaks = NULL;
    int npeaks = 0, cappeaks = 0;
    int min_votes = (int)(p.min_length * p.min_fill);

    for (int t = 0; t < ntheta; t++) {
        for (int c = 0; c < ncbins; c++) {
            int v = acc[(size_t)t * ncbins + c];
            if (v < min_votes) continue;
            if (npeaks == cappeaks) {
                int cap = cappeaks ? cappeaks * 2 : 64;
                Peak *np = (Peak *)realloc(peaks, (size_t)cap * sizeof(Peak));
                if (!np) break;
                peaks = np;
                cappeaks = cap;
            }
            peaks[npeaks].t = t;
            peaks[npeaks].c = c;
            peaks[npeaks].votes = v;
            npeaks++;
        }
    }

    for (int i = 1; i < npeaks; i++) { /* insertion sort, descending votes */
        Peak key = peaks[i];
        int j = i - 1;
        while (j >= 0 && peaks[j].votes < key.votes) {
            peaks[j + 1] = peaks[j];
            j--;
        }
        peaks[j + 1] = key;
    }

    int suppress_c = p.max_thickness * 2 + 2;
    Peak *taken = (Peak *)malloc((size_t)(npeaks ? npeaks : 1) * sizeof(Peak));
    int ntaken = 0;

    for (int i = 0; i < npeaks && ntaken < p.max_lines; i++) {
        int ok = 1;
        for (int j = 0; j < ntaken; j++) {
            if (abs(peaks[i].c - taken[j].c) <= suppress_c) { ok = 0; break; }
        }
        if (!ok) continue;
        if (taken) taken[ntaken++] = peaks[i];

        double slope = slopes[peaks[i].t];
        double intercept = (double)(peaks[i].c - margin);
        double angle = -p.max_angle_deg + peaks[i].t * p.angle_step;
        extract_runs(binary, slope, intercept, angle, peaks[i].votes, &p, out);
        if (out->count >= p.max_lines) break;
    }

    free(taken);
    free(peaks);
    free(acc);
    free(slopes);

    dedupe_segments(out, &p);
    return 0;
}
