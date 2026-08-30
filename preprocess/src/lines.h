/* lines.h -- Hough detection of near-horizontal rules (strikethrough hunting).
 *
 * Scope note: this file finds *candidate* horizontal strokes and nothing more.
 * Deciding whether a given stroke is a strikethrough needs the recogniser's
 * word boxes -- the test is whether the stroke bisects a box at mid-height --
 * and those live in the service layer. Keeping the geometry here and the
 * judgement there means this stays testable without an OCR round trip.
 *
 * The transform is restricted to a narrow band of angles around horizontal.
 * By the time this runs the page has been deskewed, so anything more than a
 * few degrees off is not a rule; that restriction turns an expensive general
 * Hough into a cheap one.
 */
#ifndef ROSETTA_LINES_H
#define ROSETTA_LINES_H

#include "img.h"

typedef struct {
    double x0, y0, x1, y1;
    double thickness;  /* median vertical stroke extent, in pixels */
    double angle_deg;  /* deviation from horizontal */
    int votes;         /* Hough accumulator support */
} LineSeg;

typedef struct {
    LineSeg *items;
    int count;
    int cap;
} LineSegs;

typedef struct {
    double max_angle_deg; /* half width of the angle sweep; default 6 */
    double angle_step;    /* default 0.5 */
    int min_length;       /* shortest run to report; <=0 derives from width */
    int max_thickness;    /* thickest stroke still a rule; <=0 derives */
    double min_fill;      /* fraction of the run that must be ink; default 0.75 */
    int max_lines;        /* cap on reported segments; default 64 */
} LineParams;

LineParams line_params_default(const Image *binary);

/* Detect near-horizontal strokes. Returns 0 on success; `out` must be freed
 * with linesegs_free. */
int detect_horizontal_rules(const Image *binary, const LineParams *p, LineSegs *out);

void linesegs_free(LineSegs *segs);

#endif /* ROSETTA_LINES_H */
