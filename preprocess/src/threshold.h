/* threshold.h -- global (Otsu) and local (Sauvola) binarisation.
 *
 * Handwriting photographed under a desk lamp has a brightness gradient across
 * the page, so a single global cut either eats the light corner or floods the
 * dark one. Sauvola adapts the cut per pixel using the local mean and standard
 * deviation, which is why it is the default here; Otsu is still useful as a
 * cheap page/background separator for perspective detection.
 */
#ifndef ROSETTA_THRESHOLD_H
#define ROSETTA_THRESHOLD_H

#include "img.h"

/* Summed-area tables for the local mean and variance. `sum` and `sqsum` hold
 * (w+1)*(h+1) entries so the rectangle query needs no bounds branching. */
typedef struct {
    int w;
    int h;
    double *sum;
    double *sqsum;
} Integral;

Integral integral_build(const Image *im);
void integral_free(Integral *it);
/* Mean and standard deviation over the inclusive rectangle [x0,x1]x[y0,y1]. */
void integral_stats(const Integral *it, int x0, int y0, int x1, int y1,
                    double *mean, double *stddev);

/* Otsu's method: returns the threshold maximising between-class variance.
 * Pixels <= the returned value are the dark class. */
int otsu_threshold(const Image *im);

/* Threshold at the histogram valley between the two dominant modes.
 *
 * Otsu assumes the two classes have comparable spread, and quietly does the
 * wrong thing when they do not: a photo lit from one side has a tight dark
 * mode for the desk and a paper class smeared across half the range, and Otsu
 * maximises between-class variance by cutting the *paper* in half rather than
 * separating paper from desk. Finding the valley between the two modes is
 * insensitive to that asymmetry. Falls back to Otsu when the histogram has no
 * clear second mode. */
int valley_threshold(const Image *im);

/* Binarise in place-ish: returns a new image where ink is 0 and paper is 255.
 * `window` is the side of the local square (odd, >= 3; <= 0 picks a size from
 * the image dimensions) and `k` is Sauvola's sensitivity, conventionally
 * 0.2-0.5 — higher keeps less ink. */
Image sauvola_binarize(const Image *im, int window, double k);

/* Fraction of pixels below `threshold`. Used to sanity check a binarisation
 * before trusting it downstream. */
double img_ink_fraction(const Image *im, int threshold);

#endif /* ROSETTA_THRESHOLD_H */
