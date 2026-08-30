#include "threshold.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

Integral integral_build(const Image *im)
{
    Integral it = {0, 0, NULL, NULL};
    if (!img_ok(im)) return it;

    int w = im->w, h = im->h;
    size_t n = (size_t)(w + 1) * (size_t)(h + 1);
    it.sum = (double *)calloc(n, sizeof(double));
    it.sqsum = (double *)calloc(n, sizeof(double));
    if (!it.sum || !it.sqsum) {
        free(it.sum);
        free(it.sqsum);
        it.sum = it.sqsum = NULL;
        return it;
    }
    it.w = w;
    it.h = h;

    for (int y = 0; y < h; y++) {
        double row = 0.0, rowsq = 0.0;
        const unsigned char *src = im->px + (size_t)y * w;
        double *cur = it.sum + (size_t)(y + 1) * (w + 1);
        double *cursq = it.sqsum + (size_t)(y + 1) * (w + 1);
        const double *prev = it.sum + (size_t)y * (w + 1);
        const double *prevsq = it.sqsum + (size_t)y * (w + 1);
        for (int x = 0; x < w; x++) {
            double v = src[x];
            row += v;
            rowsq += v * v;
            cur[x + 1] = prev[x + 1] + row;
            cursq[x + 1] = prevsq[x + 1] + rowsq;
        }
    }
    return it;
}

void integral_free(Integral *it)
{
    if (!it) return;
    free(it->sum);
    free(it->sqsum);
    it->sum = it->sqsum = NULL;
    it->w = it->h = 0;
}

void integral_stats(const Integral *it, int x0, int y0, int x1, int y1,
                    double *mean, double *stddev)
{
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 >= it->w) x1 = it->w - 1;
    if (y1 >= it->h) y1 = it->h - 1;
    if (x1 < x0 || y1 < y0) {
        if (mean) *mean = 0.0;
        if (stddev) *stddev = 0.0;
        return;
    }

    int stride = it->w + 1;
    double area = (double)(x1 - x0 + 1) * (double)(y1 - y0 + 1);

    double s = it->sum[(size_t)(y1 + 1) * stride + (x1 + 1)]
             - it->sum[(size_t)y0 * stride + (x1 + 1)]
             - it->sum[(size_t)(y1 + 1) * stride + x0]
             + it->sum[(size_t)y0 * stride + x0];
    double sq = it->sqsum[(size_t)(y1 + 1) * stride + (x1 + 1)]
              - it->sqsum[(size_t)y0 * stride + (x1 + 1)]
              - it->sqsum[(size_t)(y1 + 1) * stride + x0]
              + it->sqsum[(size_t)y0 * stride + x0];

    double m = s / area;
    double var = sq / area - m * m;
    if (var < 0.0) var = 0.0; /* floating point noise on flat regions */
    if (mean) *mean = m;
    if (stddev) *stddev = sqrt(var);
}

int otsu_threshold(const Image *im)
{
    long hist[256];
    memset(hist, 0, sizeof(hist));
    size_t n = (size_t)im->w * (size_t)im->h;
    for (size_t i = 0; i < n; i++) hist[im->px[i]]++;

    double total = (double)n;
    double sum_all = 0.0;
    for (int i = 0; i < 256; i++) sum_all += (double)i * hist[i];

    double sum_bg = 0.0, weight_bg = 0.0;
    double best_var = -1.0;
    int best_t = 127;

    for (int t = 0; t < 256; t++) {
        weight_bg += hist[t];
        if (weight_bg == 0.0) continue;
        double weight_fg = total - weight_bg;
        if (weight_fg == 0.0) break;

        sum_bg += (double)t * hist[t];
        double mean_bg = sum_bg / weight_bg;
        double mean_fg = (sum_all - sum_bg) / weight_fg;
        double between = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if (between > best_var) {
            best_var = between;
            best_t = t;
        }
    }
    return best_t;
}

int valley_threshold(const Image *im)
{
    double hist[256] = {0};
    size_t n = (size_t)im->w * (size_t)im->h;
    for (size_t i = 0; i < n; i++) hist[im->px[i]] += 1.0;

    /* Smooth before looking for structure: sensor noise and JPEG quantisation
     * leave the raw histogram spiky enough that "peak" and "valley" would
     * otherwise mean whichever bin happened to win a coin flip. */
    double smooth[256];
    for (int pass = 0; pass < 3; pass++) {
        for (int i = 0; i < 256; i++) {
            double sum = 0.0;
            int count = 0;
            for (int d = -4; d <= 4; d++) {
                int j = i + d;
                if (j < 0 || j > 255) continue;
                sum += hist[j];
                count++;
            }
            smooth[i] = sum / count;
        }
        memcpy(hist, smooth, sizeof(hist));
    }

    int peak1 = 0;
    for (int i = 1; i < 256; i++)
        if (smooth[i] > smooth[peak1]) peak1 = i;

    /* The second mode must be far enough away to be a different material and
     * not the shoulder of the first. */
    const int min_separation = 40;
    int peak2 = -1;
    for (int i = 0; i < 256; i++) {
        if (abs(i - peak1) < min_separation) continue;
        if (peak2 < 0 || smooth[i] > smooth[peak2]) peak2 = i;
    }
    if (peak2 < 0 || smooth[peak2] <= 0.0) return otsu_threshold(im);

    /* A second "mode" that is really just background noise tells us nothing. */
    if (smooth[peak2] < smooth[peak1] * 0.01) return otsu_threshold(im);

    int lo = peak1 < peak2 ? peak1 : peak2;
    int hi = peak1 < peak2 ? peak2 : peak1;
    int valley = lo;
    for (int i = lo; i <= hi; i++)
        if (smooth[i] < smooth[valley]) valley = i;

    return valley;
}

Image sauvola_binarize(const Image *im, int window, double k)
{
    Image out = {0, 0, NULL};
    if (!img_ok(im)) return out;

    if (window <= 0) {
        /* A window of roughly 1/40 of the long edge lands near two x-heights
         * for a page photographed at a sane resolution, which is the range
         * Sauvola behaves best in. */
        int longest = im->w > im->h ? im->w : im->h;
        window = longest / 40;
    }
    if (window < 3) window = 3;
    if ((window & 1) == 0) window += 1;
    if (k <= 0.0) k = 0.34;

    Integral it = integral_build(im);
    if (!it.sum) return out;

    out = img_new(im->w, im->h);
    if (!out.px) {
        integral_free(&it);
        return out;
    }

    int r = window / 2;
    const double R = 128.0; /* dynamic range of the standard deviation */

    for (int y = 0; y < im->h; y++) {
        for (int x = 0; x < im->w; x++) {
            double mean, sd;
            integral_stats(&it, x - r, y - r, x + r, y + r, &mean, &sd);
            double t = mean * (1.0 + k * (sd / R - 1.0));
            unsigned char v = im->px[(size_t)y * im->w + x];
            out.px[(size_t)y * im->w + x] = (v <= t) ? 0 : 255;
        }
    }

    integral_free(&it);
    return out;
}

double img_ink_fraction(const Image *im, int threshold)
{
    size_t n = (size_t)im->w * (size_t)im->h;
    size_t ink = 0;
    for (size_t i = 0; i < n; i++)
        if (im->px[i] <= threshold) ink++;
    return n ? (double)ink / (double)n : 0.0;
}
