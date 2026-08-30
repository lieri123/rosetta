#include "deskew.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define INK_LEVEL 128 /* binarised input: ink is 0, paper is 255 */

double skew_profile_score(const Image *binary, double angle_deg)
{
    double rad = angle_deg * M_PI / 180.0;
    double s = sin(rad), c = cos(rad);

    /* Rotating the *coordinates* rather than the pixels: for each ink pixel we
     * compute which row it would land in after rotation and bump that bin.
     * One pass over the ink, no resampling. */
    int cx = binary->w / 2;
    int cy = binary->h / 2;
    int span = (int)(fabs(binary->w * s) + fabs(binary->h * c)) + 2;
    int offset = span / 2;

    double *profile = (double *)calloc((size_t)span + 1, sizeof(double));
    if (!profile) return 0.0;

    for (int y = 0; y < binary->h; y++) {
        const unsigned char *row = binary->px + (size_t)y * binary->w;
        double dy = y - cy;
        for (int x = 0; x < binary->w; x++) {
            if (row[x] >= INK_LEVEL) continue;
            double dx = x - cx;
            int bin = (int)(-dx * s + dy * c + offset + 0.5);
            if (bin >= 0 && bin < span) profile[bin] += 1.0;
        }
    }

    /* Sum of squared first differences. Variance of the profile is the textbook
     * choice, but it rewards any concentration of ink; the derivative rewards
     * sharp line/gap boundaries specifically, which is what a correct angle
     * actually produces. */
    double score = 0.0;
    for (int i = 1; i < span; i++) {
        double d = profile[i] - profile[i - 1];
        score += d * d;
    }

    free(profile);
    return score;
}

int estimate_skew(const Image *binary, double limit_deg, double *angle_deg)
{
    if (!img_ok(binary)) return -1;
    if (limit_deg <= 0.0) limit_deg = 10.0;

    /* Bail out on a blank page: with almost no ink the profile score is noise
     * and we would happily "correct" by a random few degrees. */
    size_t n = (size_t)binary->w * binary->h;
    size_t ink = 0;
    for (size_t i = 0; i < n; i++)
        if (binary->px[i] < INK_LEVEL) ink++;
    if (ink < 200 || (double)ink / (double)n < 0.0005) return -1;

    /* The search is O(angles * ink), so run it on a downscaled copy. Skew is a
     * global property; a 900px-wide view resolves it to well under a tenth of a
     * degree. */
    double scale = 1.0;
    Image small = img_downscale_to(binary, 900, &scale);
    if (!img_ok(&small)) return -1;
    /* Downscaling averages, so re-binarise to keep the ink test meaningful. */
    for (size_t i = 0; i < (size_t)small.w * small.h; i++)
        small.px[i] = small.px[i] < 200 ? 0 : 255;

    /* Coarse sweep, then two refinement passes around the winner. Sweeping the
     * whole range at 0.05 degrees would be 20x the work for the same answer. */
    double best_angle = 0.0;
    double best_score = -1.0;

    for (double a = -limit_deg; a <= limit_deg + 1e-9; a += 1.0) {
        double sc = skew_profile_score(&small, a);
        if (sc > best_score) {
            best_score = sc;
            best_angle = a;
        }
    }

    double step = 0.25;
    for (int pass = 0; pass < 2; pass++) {
        double lo = best_angle - step * 4;
        double hi = best_angle + step * 4;
        for (double a = lo; a <= hi + 1e-9; a += step) {
            if (a < -limit_deg || a > limit_deg) continue;
            double sc = skew_profile_score(&small, a);
            if (sc > best_score) {
                best_score = sc;
                best_angle = a;
            }
        }
        step /= 5.0; /* 0.25 -> 0.05 degrees */
    }

    img_free(&small);
    if (angle_deg) *angle_deg = best_angle;
    return 0;
}

Image rotate_image(const Image *im, double angle_deg, unsigned char fill)
{
    Image out = {0, 0, NULL};
    if (!img_ok(im)) return out;

    double rad = angle_deg * M_PI / 180.0;
    double s = sin(rad), c = cos(rad);

    /* Grow the canvas to the rotated bounding box so corners survive. */
    int ow = (int)(fabs(im->w * c) + fabs(im->h * s) + 0.5);
    int oh = (int)(fabs(im->w * s) + fabs(im->h * c) + 0.5);
    if (ow < 1) ow = 1;
    if (oh < 1) oh = 1;

    out = img_new(ow, oh);
    if (!out.px) return out;

    double scx = (im->w - 1) / 2.0, scy = (im->h - 1) / 2.0;
    double dcx = (ow - 1) / 2.0, dcy = (oh - 1) / 2.0;

    /* Inverse map: walk the destination, rotate back by -angle, sample. */
    for (int y = 0; y < oh; y++) {
        double dy = y - dcy;
        for (int x = 0; x < ow; x++) {
            double dx = x - dcx;
            double sx = dx * c + dy * s + scx;
            double sy = -dx * s + dy * c + scy;
            out.px[(size_t)y * ow + x] = img_sample(im, sx, sy, fill);
        }
    }
    return out;
}
