/* synth.h -- synthetic page generator.
 *
 * Handwriting test data is expensive to label and impossible to check into a
 * repository, but most of what the preprocessor does has an exact ground truth
 * available for free if you generate the page yourself: you know the skew you
 * applied, the corners you projected to, and where you drew the strikethrough.
 * The unit tests and the `mkpage` demo tool both build on this.
 */
#ifndef ROSETTA_SYNTH_H
#define ROSETTA_SYNTH_H

#include "img.h"
#include "geom.h"

typedef struct {
    int width, height;   /* output image size */
    int margin;          /* background border around the page, in pixels */
    int text_lines;      /* rows of pseudo-writing */
    double skew_deg;     /* rotation of the writing within the page */
    double perspective;  /* corner displacement as a fraction of page size */
    int strike_line;     /* index of the line to strike through, -1 for none */
    double noise;        /* 0..1 sensor noise */
    double shading;      /* 0..1 brightness gradient across the page */
    unsigned seed;
} SynthOpts;

typedef struct {
    Image image;
    Quad quad;        /* true page corners in image space */
    double skew_deg;  /* the skew that was applied */
    /* Strikethrough endpoints in image space; valid when strike_line >= 0 and
     * perspective is 0 (with perspective they are only approximate). */
    double strike_x0, strike_y0, strike_x1, strike_y1;
    int has_strike;
} SynthPage;

SynthOpts synth_defaults(void);
int synth_page(const SynthOpts *opts, SynthPage *out);

#endif /* ROSETTA_SYNTH_H */
