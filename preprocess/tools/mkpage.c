/* mkpage -- write a synthetic page image for demos and manual inspection.
 *
 * Not part of the service: it exists so `make demo` can show the pipeline
 * working end to end without anyone having to photograph a notebook first.
 */
#include "../src/synth.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    SynthOpts o = synth_defaults();
    const char *out = "page.png";

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has_next = i + 1 < argc;
        if (!strcmp(a, "-o") && has_next) out = argv[++i];
        else if (!strcmp(a, "--width") && has_next) o.width = atoi(argv[++i]);
        else if (!strcmp(a, "--height") && has_next) o.height = atoi(argv[++i]);
        else if (!strcmp(a, "--skew") && has_next) o.skew_deg = atof(argv[++i]);
        else if (!strcmp(a, "--perspective") && has_next) o.perspective = atof(argv[++i]);
        else if (!strcmp(a, "--strike") && has_next) o.strike_line = atoi(argv[++i]);
        else if (!strcmp(a, "--noise") && has_next) o.noise = atof(argv[++i]);
        else if (!strcmp(a, "--shading") && has_next) o.shading = atof(argv[++i]);
        else if (!strcmp(a, "--lines") && has_next) o.text_lines = atoi(argv[++i]);
        else if (!strcmp(a, "--seed") && has_next) o.seed = (unsigned)atoi(argv[++i]);
        else {
            fprintf(stderr,
                    "usage: %s [-o out.png] [--width N] [--height N] [--skew DEG]\n"
                    "          [--perspective F] [--strike LINE] [--noise F]\n"
                    "          [--shading F] [--lines N] [--seed N]\n", argv[0]);
            return 2;
        }
    }

    SynthPage page;
    if (synth_page(&o, &page) != 0) {
        fprintf(stderr, "error: could not synthesise page\n");
        return 1;
    }

    if (img_save_png(out, &page.image) != 0) {
        fprintf(stderr, "error: could not write %s\n", out);
        img_free(&page.image);
        return 1;
    }

    printf("wrote %s (%dx%d), skew %.2f deg, quad [", out,
           page.image.w, page.image.h, page.skew_deg);
    for (int i = 0; i < 4; i++)
        printf("%s(%.0f,%.0f)", i ? " " : "", page.quad.c[i].x, page.quad.c[i].y);
    printf("]\n");
    if (page.has_strike)
        printf("strikethrough at y=%.0f from x=%.0f to x=%.0f\n",
               page.strike_y0, page.strike_x0, page.strike_x1);

    img_free(&page.image);
    return 0;
}
