/* deskew.h -- skew estimation by projection profile.
 *
 * Lines of handwriting are the strongest periodic structure on the page. Rotate
 * the ink by the correct angle and every row of the projection profile is
 * either dense (a line) or empty (the gap between lines); rotate it wrongly and
 * the ink smears across rows and the profile flattens. So the estimator is:
 * maximise the profile's roughness over a range of candidate angles.
 *
 * This beats a Hough line fit on handwriting, where baselines are implied by
 * letter bottoms rather than drawn, and there is no long straight edge to find.
 */
#ifndef ROSETTA_DESKEW_H
#define ROSETTA_DESKEW_H

#include "img.h"

/* Estimate the page skew in degrees. Positive means the text runs downhill to
 * the right and the page should be rotated by -angle to correct it.
 * `limit_deg` bounds the search (10 is a sane default for photographed notes).
 * Returns 0 on success and writes the angle; -1 if the image has too little
 * ink to judge, in which case the caller should leave the page alone. */
int estimate_skew(const Image *binary, double limit_deg, double *angle_deg);

/* Rotate about the image centre, expanding the canvas so nothing is clipped.
 * Empty area is filled with `fill` (255 for paper). */
Image rotate_image(const Image *im, double angle_deg, unsigned char fill);

/* Exposed for the tests: the profile score at one angle. Higher is sharper. */
double skew_profile_score(const Image *binary, double angle_deg);

#endif /* ROSETTA_DESKEW_H */
