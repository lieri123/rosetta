#include "img.h"

#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include "../vendor/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../vendor/stb_image_write.h"

Image img_new(int w, int h)
{
    Image im = {0, 0, NULL};
    if (w <= 0 || h <= 0) return im;
    im.px = (unsigned char *)calloc((size_t)w * (size_t)h, 1);
    if (!im.px) return im;
    im.w = w;
    im.h = h;
    return im;
}

Image img_clone(const Image *src)
{
    Image im = img_new(src->w, src->h);
    if (im.px) memcpy(im.px, src->px, (size_t)src->w * (size_t)src->h);
    return im;
}

void img_free(Image *im)
{
    if (!im) return;
    free(im->px);
    im->px = NULL;
    im->w = im->h = 0;
}

int img_ok(const Image *im) { return im && im->px && im->w > 0 && im->h > 0; }

int img_load_gray(const char *path, Image *out)
{
    int w = 0, h = 0, n = 0;
    /* Ask stb for one channel; it applies the standard luma weights. */
    unsigned char *data = stbi_load(path, &w, &h, &n, 1);
    if (!data) return -1;
    out->w = w;
    out->h = h;
    out->px = data; /* stb's allocator is malloc, so free() below is fine */
    return 0;
}

int img_save_png(const char *path, const Image *im)
{
    if (!img_ok(im)) return -1;
    return stbi_write_png(path, im->w, im->h, 1, im->px, im->w) ? 0 : -1;
}

unsigned char img_at(const Image *im, int x, int y)
{
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= im->w) x = im->w - 1;
    if (y >= im->h) y = im->h - 1;
    return im->px[(size_t)y * im->w + x];
}

unsigned char img_sample(const Image *im, double x, double y, unsigned char fallback)
{
    if (x < -0.5 || y < -0.5 || x > im->w - 0.5 || y > im->h - 0.5) return fallback;

    int x0 = (int)((x < 0) ? x - 1 : x);
    int y0 = (int)((y < 0) ? y - 1 : y);
    double fx = x - x0;
    double fy = y - y0;

    double p00 = img_at(im, x0, y0);
    double p10 = img_at(im, x0 + 1, y0);
    double p01 = img_at(im, x0, y0 + 1);
    double p11 = img_at(im, x0 + 1, y0 + 1);

    double top = p00 + (p10 - p00) * fx;
    double bot = p01 + (p11 - p01) * fx;
    double v = top + (bot - top) * fy;

    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return (unsigned char)(v + 0.5);
}

Image img_downscale_to(const Image *src, int max_dim, double *scale_out)
{
    int longest = src->w > src->h ? src->w : src->h;
    if (max_dim <= 0 || longest <= max_dim) {
        if (scale_out) *scale_out = 1.0;
        return img_clone(src);
    }

    double scale = (double)max_dim / (double)longest;
    int dw = (int)(src->w * scale);
    int dh = (int)(src->h * scale);
    if (dw < 1) dw = 1;
    if (dh < 1) dh = 1;

    Image dst = img_new(dw, dh);
    if (!dst.px) return dst;

    /* Box filter over the source rectangle each destination pixel covers.
     * Averaging (rather than point sampling) matters: the skew estimator reads
     * ink density, and dropping pixels biases it. */
    for (int y = 0; y < dh; y++) {
        int sy0 = (int)((double)y * src->h / dh);
        int sy1 = (int)((double)(y + 1) * src->h / dh);
        if (sy1 <= sy0) sy1 = sy0 + 1;
        for (int x = 0; x < dw; x++) {
            int sx0 = (int)((double)x * src->w / dw);
            int sx1 = (int)((double)(x + 1) * src->w / dw);
            if (sx1 <= sx0) sx1 = sx0 + 1;
            unsigned long sum = 0;
            unsigned long count = 0;
            for (int sy = sy0; sy < sy1 && sy < src->h; sy++) {
                for (int sx = sx0; sx < sx1 && sx < src->w; sx++) {
                    sum += src->px[(size_t)sy * src->w + sx];
                    count++;
                }
            }
            dst.px[(size_t)y * dw + x] = (unsigned char)(count ? sum / count : 0);
        }
    }

    if (scale_out) *scale_out = (double)dw / (double)src->w;
    return dst;
}
