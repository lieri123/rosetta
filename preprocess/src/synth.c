#include "synth.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* A local LCG so tests are reproducible regardless of libc. */
static unsigned rng_state;
static void rng_seed(unsigned s) { rng_state = s ? s : 1u; }
static unsigned rng_next(void)
{
    rng_state = rng_state * 1103515245u + 12345u;
    return (rng_state >> 16) & 0x7fff;
}
static int rng_range(int lo, int hi)
{
    if (hi <= lo) return lo;
    return lo + (int)(rng_next() % (unsigned)(hi - lo + 1));
}

SynthOpts synth_defaults(void)
{
    SynthOpts o;
    o.width = 900;
    o.height = 1200;
    o.margin = 60;
    o.text_lines = 14;
    o.skew_deg = 0.0;
    o.perspective = 0.0;
    o.strike_line = -1;
    o.noise = 0.02;
    o.shading = 0.0;
    o.seed = 7u;
    return o;
}

/* Fill the rectangle [x0,x1]x[y0,y1] after rotating it about (cx, cy).
 * Iterating the rotated bounding box and testing membership in the unrotated
 * frame keeps the edges exact without a resampling pass. */
static void fill_rotated_rect(Image *im, double cx, double cy,
                              double x0, double y0, double x1, double y1,
                              double angle_deg, unsigned char value)
{
    double rad = angle_deg * M_PI / 180.0;
    double s = sin(rad), c = cos(rad);

    double corners[4][2] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
    double minx = 1e18, miny = 1e18, maxx = -1e18, maxy = -1e18;
    for (int i = 0; i < 4; i++) {
        double dx = corners[i][0] - cx, dy = corners[i][1] - cy;
        double rx = cx + dx * c - dy * s;
        double ry = cy + dx * s + dy * c;
        if (rx < minx) minx = rx;
        if (rx > maxx) maxx = rx;
        if (ry < miny) miny = ry;
        if (ry > maxy) maxy = ry;
    }

    int ix0 = (int)floor(minx), ix1 = (int)ceil(maxx);
    int iy0 = (int)floor(miny), iy1 = (int)ceil(maxy);
    if (ix0 < 0) ix0 = 0;
    if (iy0 < 0) iy0 = 0;
    if (ix1 >= im->w) ix1 = im->w - 1;
    if (iy1 >= im->h) iy1 = im->h - 1;

    for (int y = iy0; y <= iy1; y++) {
        for (int x = ix0; x <= ix1; x++) {
            double dx = x - cx, dy = y - cy;
            double ux = cx + dx * c + dy * s;   /* rotate back */
            double uy = cy - dx * s + dy * c;
            if (ux >= x0 && ux <= x1 && uy >= y0 && uy <= y1)
                im->px[(size_t)y * im->w + x] = value;
        }
    }
}

int synth_page(const SynthOpts *opts, SynthPage *out)
{
    if (!opts || !out) return -1;
    SynthOpts o = *opts;
    if (o.width < 64 || o.height < 64) return -1;
    if (o.margin < 0) o.margin = 0;
    if (o.margin * 2 >= o.width || o.margin * 2 >= o.height) o.margin = 0;

    memset(out, 0, sizeof(*out));
    rng_seed(o.seed);

    int pw = o.width - 2 * o.margin;
    int ph = o.height - 2 * o.margin;

    Image page = img_new(pw, ph);
    if (!img_ok(&page)) return -1;
    memset(page.px, 242, (size_t)pw * ph); /* paper */

    /* Pseudo-writing: each line is a row of word-shaped bars. They are not
     * letters, but the projection profile and the stroke-thickness statistics
     * that the preprocessor keys on are the same shape as the real thing. */
    int top = ph / 12;
    int line_height = (ph - 2 * top) / (o.text_lines > 0 ? o.text_lines : 1);
    int x_height = line_height / 3;
    if (x_height < 8) x_height = 8;
    int stroke = x_height / 7;
    if (stroke < 2) stroke = 2;
    int advance = (int)(x_height * 0.8);
    if (advance < stroke + 3) advance = stroke + 3;

    double cx = pw / 2.0, cy = ph / 2.0;

    for (int i = 0; i < o.text_lines; i++) {
        double baseline = top + i * (double)line_height;
        int x = pw / 12;
        int right = pw - pw / 12;
        int line_start = x, line_end = x;

        while (x < right - advance * 2) {
            int word = rng_range(3, 9) * advance;
            if (x + word > right) word = right - x;
            if (word < advance * 2) break;

            /* Letters are drawn as one or two near-vertical strokes with the
             * occasional connector, ascender or descender. They are not
             * readable, but the two properties the preprocessor keys on are
             * right: ink concentrated in bands at the line pitch, and thin
             * strokes separated by paper. A solid bar would satisfy neither,
             * and would hide a strikethrough inside itself. */
            for (int lx = x; lx + advance <= x + word; lx += advance) {
                int nstrokes = rng_range(1, 2);
                for (int k = 0; k < nstrokes; k++) {
                    int sx = lx + rng_range(0, advance - stroke - 1);
                    double t0 = baseline + rng_range(0, x_height / 4);
                    double t1 = baseline + x_height - rng_range(0, x_height / 5);
                    if (rng_range(0, 9) == 0) t0 = baseline - x_height * 0.8; /* ascender */
                    if (rng_range(0, 11) == 0) t1 = baseline + x_height * 1.7; /* descender */
                    fill_rotated_rect(&page, cx, cy, sx, t0, sx + stroke, t1,
                                      o.skew_deg, 30);
                }
                if (rng_range(0, 2) == 0) {
                    double cy_conn = baseline + rng_range(x_height / 4, 3 * x_height / 4);
                    fill_rotated_rect(&page, cx, cy, lx, cy_conn,
                                      lx + advance - 2, cy_conn + stroke - 1,
                                      o.skew_deg, 30);
                }
            }

            line_end = x + word;
            x += word + rng_range(advance, advance * 2);
        }

        if (o.strike_line == i) {
            /* A pen stroke straight through the middle of the line: thin,
             * unbroken, and spanning several words. */
            double sy = baseline + x_height / 2.0;
            double half = stroke / 2.0;
            fill_rotated_rect(&page, cx, cy, line_start, sy - half, line_end, sy + half,
                              o.skew_deg, 20);
            out->has_strike = 1;
            out->strike_x0 = line_start + o.margin;
            out->strike_y0 = sy + o.margin;
            out->strike_x1 = line_end + o.margin;
            out->strike_y1 = sy + o.margin;
        }
    }

    /* Illumination is multiplicative, not additive: a lamp to one side scales
     * reflectance rather than subtracting a constant. That distinction is the
     * whole reason a global threshold fails -- with a strong enough falloff the
     * paper on the dark side gets darker than the ink on the bright side, and
     * no single cut can separate them. */
    if (o.shading > 0.0) {
        for (int y = 0; y < ph; y++) {
            for (int x = 0; x < pw; x++) {
                double t = (double)x / (double)(pw - 1);
                double gain = 1.0 - o.shading * t;
                if (gain < 0.0) gain = 0.0;
                int v = (int)(page.px[(size_t)y * pw + x] * gain + 0.5);
                page.px[(size_t)y * pw + x] = (unsigned char)(v < 0 ? 0 : v);
            }
        }
    }

    Image canvas = img_new(o.width, o.height);
    if (!img_ok(&canvas)) {
        img_free(&page);
        return -1;
    }
    memset(canvas.px, 48, (size_t)o.width * o.height); /* desk, darker than paper */

    /* Where the page corners land in the output. With perspective 0 this is
     * just the margin rectangle. */
    double dx = o.perspective * pw;
    double dy = o.perspective * ph;
    Quad quad;
    quad.c[0].x = o.margin + dx * 0.6;  quad.c[0].y = o.margin;
    quad.c[1].x = o.margin + pw - 1;    quad.c[1].y = o.margin + dy * 0.35;
    quad.c[2].x = o.margin + pw - 1 - dx * 0.25; quad.c[2].y = o.margin + ph - 1;
    quad.c[3].x = o.margin;             quad.c[3].y = o.margin + ph - 1 - dy * 0.5;

    /* Map output pixels back into page space so every pixel inside the quad
     * gets written exactly once. */
    Point page_rect[4] = {
        {0.0, 0.0}, {(double)pw - 1, 0.0},
        {(double)pw - 1, (double)ph - 1}, {0.0, (double)ph - 1}
    };
    double h[9];
    if (homography_solve(quad.c, page_rect, h) != 0) {
        img_free(&page);
        img_free(&canvas);
        return -1;
    }

    for (int y = 0; y < o.height; y++) {
        for (int x = 0; x < o.width; x++) {
            Point p = {(double)x, (double)y};
            Point s = homography_apply(h, p);
            if (s.x < -0.5 || s.y < -0.5 || s.x > pw - 0.5 || s.y > ph - 0.5) continue;
            canvas.px[(size_t)y * o.width + x] = img_sample(&page, s.x, s.y, 242);
        }
    }

    if (o.noise > 0.0) {
        int amp = (int)(o.noise * 255.0);
        if (amp > 0) {
            for (size_t i = 0; i < (size_t)o.width * o.height; i++) {
                int v = canvas.px[i] + rng_range(-amp, amp);
                canvas.px[i] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
        }
    }

    img_free(&page);
    out->image = canvas;
    out->quad = quad;
    out->skew_deg = o.skew_deg;
    return 0;
}
