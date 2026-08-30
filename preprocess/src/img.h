/* img.h -- 8-bit single channel image buffer plus PNG/JPEG IO.
 */
#ifndef ROSETTA_IMG_H
#define ROSETTA_IMG_H

#include <stddef.h>

typedef struct {
    int w;
    int h;
    unsigned char *px; 
} Image;

/* Allocation. img_new zeroes the buffer; both return an image with px == NULL
 * on failure (check with img_ok). */
Image img_new(int w, int h);
Image img_clone(const Image *src);
void img_free(Image *im);
int img_ok(const Image *im);

/* IO. img_load decodes any format stb_image handles and converts to gray.
 * img_save_png writes single channel PNG. Both return 0 on success. */
int img_load_gray(const char *path, Image *out);
int img_save_png(const char *path, const Image *im);

/* Sampling. img_at clamps to the border; img_sample does bilinear
 * interpolation at fractional coordinates and returns `fallback` for points
 * outside the source rectangle. */
unsigned char img_at(const Image *im, int x, int y);
unsigned char img_sample(const Image *im, double x, double y, unsigned char fallback);

/* Nearest-power-of-two-ish box downscale to fit within `max_dim`. Used to keep
 * the O(angles * pixels) searches cheap on 12 megapixel phone photos. */
Image img_downscale_to(const Image *src, int max_dim, double *scale_out);

#endif
