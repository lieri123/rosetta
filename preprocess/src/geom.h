/* geom.h -- page quad detection and perspective rectification.
 *
 * A phone photo of a notebook is a plane seen off-axis: straight rules bow
 * away from horizontal and letter height drifts across the page, which hurts
 * both the recogniser and the line grouping that the layout inference depends
 * on. Recovering the page quad and warping it back to a rectangle is a
 * homography, eight unknowns from four point correspondences.
 */
#ifndef ROSETTA_GEOM_H
#define ROSETTA_GEOM_H

#include "img.h"

typedef struct {
    double x, y;
} Point;

/* Corners in reading order: top-left, top-right, bottom-right, bottom-left. */
typedef struct {
    Point c[4];
} Quad;

/* Solve the homography mapping the four `from` points onto the four `to`
 * points. `h` receives the nine coefficients row major with h[8] fixed at 1.
 * Returns 0 on success, -1 if the system is singular (collinear points). */
int homography_solve(const Point from[4], const Point to[4], double h[9]);

/* Apply `h` to a point. */
Point homography_apply(const double h[9], Point p);

/* Locate the page: the largest bright connected region, reduced to its four
 * extreme corners. Returns 0 and fills `out` when a plausible quad is found,
 * -1 when the page fills the frame or no region qualifies (in which case the
 * caller should skip rectification rather than warp on a bad guess). */
int detect_page_quad(const Image *im, Quad *out);

/* Warp the region bounded by `quad` into a new image of the given size,
 * sampling bilinearly. Pass 0 for out_w/out_h to derive the output size from
 * the quad's own edge lengths. */
Image warp_quad(const Image *im, const Quad *quad, int out_w, int out_h);

/* Utilities shared with the tests. */
double quad_area(const Quad *q);
void quad_output_size(const Quad *q, int *w, int *h);

#endif /* ROSETTA_GEOM_H */
