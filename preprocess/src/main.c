/* rosetta-preprocess -- page cleanup for handwriting recognition.
 *
 * Reads a photo or scan of a page, applies perspective correction, deskewing
 * and adaptive binarisation, and writes a cleaned PNG plus a JSON description
 * of what it did. The service layer shells out to this binary; the JSON is the
 * interface between the two, which keeps the C testable on its own and keeps
 * cgo out of the Go build.
 */
#include "img.h"
#include "geom.h"
#include "deskew.h"
#include "threshold.h"
#include "lines.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *in_path;
    const char *out_path;
    const char *json_path;
    const char *debug_dir;
    int do_perspective;
    int do_deskew;
    int do_threshold;
    int do_rules;
    int window;
    int min_rule_length;
    double k;
    double max_skew;
    int quiet;
} Options;

static void usage(FILE *f, const char *argv0)
{
    fprintf(f,
        "usage: %s [options] <image>\n"
        "\n"
        "Clean up a photographed page for handwriting recognition.\n"
        "\n"
        "options:\n"
        "  -o, --out PATH        write the processed page as PNG\n"
        "      --json PATH       write metadata JSON ('-' for stdout, the default)\n"
        "      --no-perspective  skip page quad detection and rectification\n"
        "      --no-deskew       skip rotation\n"
        "      --no-threshold    keep grayscale instead of binarising\n"
        "      --strikethrough   detect near-horizontal rules and report them\n"
        "      --window N        Sauvola window size in pixels (odd; default auto)\n"
        "      --min-rule-length N  shortest stroke reported as a rule (default: page width / 15)\n"
        "      --k F             Sauvola sensitivity, 0.2-0.5 (default 0.34)\n"
        "      --max-skew D      largest skew to search for, in degrees (default 10)\n"
        "      --debug-dir DIR   write the intermediate image after each step\n"
        "  -q, --quiet           suppress progress on stderr\n"
        "  -h, --help            this message\n",
        argv0);
}

static int parse_args(int argc, char **argv, Options *o)
{
    memset(o, 0, sizeof(*o));
    o->do_perspective = o->do_deskew = o->do_threshold = 1;
    o->json_path = "-";
    o->k = 0.34;
    o->max_skew = 10.0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has_next = (i + 1 < argc);

        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout, argv[0]);
            exit(0);
        } else if ((!strcmp(a, "-o") || !strcmp(a, "--out")) && has_next) {
            o->out_path = argv[++i];
        } else if (!strcmp(a, "--json") && has_next) {
            o->json_path = argv[++i];
        } else if (!strcmp(a, "--debug-dir") && has_next) {
            o->debug_dir = argv[++i];
        } else if (!strcmp(a, "--no-perspective")) {
            o->do_perspective = 0;
        } else if (!strcmp(a, "--no-deskew")) {
            o->do_deskew = 0;
        } else if (!strcmp(a, "--no-threshold")) {
            o->do_threshold = 0;
        } else if (!strcmp(a, "--strikethrough")) {
            o->do_rules = 1;
        } else if (!strcmp(a, "--window") && has_next) {
            o->window = atoi(argv[++i]);
        } else if (!strcmp(a, "--min-rule-length") && has_next) {
            o->min_rule_length = atoi(argv[++i]);
        } else if (!strcmp(a, "--k") && has_next) {
            o->k = atof(argv[++i]);
        } else if (!strcmp(a, "--max-skew") && has_next) {
            o->max_skew = atof(argv[++i]);
        } else if (!strcmp(a, "-q") || !strcmp(a, "--quiet")) {
            o->quiet = 1;
        } else if (a[0] == '-' && a[1] != '\0') {
            fprintf(stderr, "%s: unknown option %s\n", argv[0], a);
            return -1;
        } else if (!o->in_path) {
            o->in_path = a;
        } else {
            fprintf(stderr, "%s: unexpected argument %s\n", argv[0], a);
            return -1;
        }
    }

    if (!o->in_path) {
        usage(stderr, argv[0]);
        return -1;
    }
    return 0;
}

static void debug_dump(const Options *o, const char *name, const Image *im)
{
    if (!o->debug_dir) return;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.png", o->debug_dir, name);
    if (img_save_png(path, im) != 0)
        fprintf(stderr, "warning: could not write debug image %s\n", path);
}

/* Minimal JSON string escaping -- paths are the only free text we emit. */
static void json_string(FILE *f, const char *s)
{
    fputc('"', f);
    for (; s && *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"': fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        default:
            if (c < 0x20) fprintf(f, "\\u%04x", c);
            else fputc(c, f);
        }
    }
    fputc('"', f);
}

int main(int argc, char **argv)
{
    Options o;
    if (parse_args(argc, argv, &o) != 0) return 2;

    Image page;
    if (img_load_gray(o.in_path, &page) != 0) {
        fprintf(stderr, "error: cannot decode %s\n", o.in_path);
        return 1;
    }

    int src_w = page.w, src_h = page.h;
    debug_dump(&o, "00-gray", &page);

    int quad_found = 0;
    Quad quad;
    memset(&quad, 0, sizeof(quad));

    if (o.do_perspective) {
        if (detect_page_quad(&page, &quad) == 0) {
            Image warped = warp_quad(&page, &quad, 0, 0);
            if (img_ok(&warped)) {
                img_free(&page);
                page = warped;
                quad_found = 1;
                debug_dump(&o, "01-perspective", &page);
                if (!o.quiet) fprintf(stderr, "perspective: rectified to %dx%d\n", page.w, page.h);
            } else {
                fprintf(stderr, "warning: perspective warp failed, continuing\n");
            }
        } else if (!o.quiet) {
            fprintf(stderr, "perspective: no page quad found, leaving geometry alone\n");
        }
    }

    /* Deskew needs a binary view to measure ink; binarise once here and reuse
     * the result as the output when thresholding is enabled. */
    Image binary = sauvola_binarize(&page, o.window, o.k);
    if (!img_ok(&binary)) {
        fprintf(stderr, "error: binarisation failed (out of memory?)\n");
        img_free(&page);
        return 1;
    }
    debug_dump(&o, "02-binary", &binary);

    int deskewed = 0;
    double skew = 0.0;
    if (o.do_deskew && estimate_skew(&binary, o.max_skew, &skew) == 0) {
        /* Under a tenth of a degree is below what the estimator can resolve and
         * a rotation would cost a resample for nothing. */
        if (skew > 0.1 || skew < -0.1) {
            Image rp = rotate_image(&page, -skew, 255);
            Image rb = rotate_image(&binary, -skew, 255);
            if (img_ok(&rp) && img_ok(&rb)) {
                img_free(&page);
                img_free(&binary);
                page = rp;
                /* Rotation interpolates, so re-cut the binary to stay strictly
                 * two valued -- the rule detector depends on that. */
                for (size_t i = 0; i < (size_t)rb.w * rb.h; i++)
                    rb.px[i] = rb.px[i] < 128 ? 0 : 255;
                binary = rb;
                deskewed = 1;
                debug_dump(&o, "03-deskew", &binary);
                if (!o.quiet) fprintf(stderr, "deskew: rotated by %.2f degrees\n", -skew);
            } else {
                img_free(&rp);
                img_free(&rb);
                fprintf(stderr, "warning: rotation failed, continuing unrotated\n");
            }
        } else if (!o.quiet) {
            fprintf(stderr, "deskew: %.2f degrees, below correction threshold\n", skew);
        }
    }

    LineSegs rules;
    memset(&rules, 0, sizeof(rules));
    if (o.do_rules) {
        LineParams lp = line_params_default(&binary);
        if (o.min_rule_length > 0) lp.min_length = o.min_rule_length;
        if (detect_horizontal_rules(&binary, &lp, &rules) != 0)
            fprintf(stderr, "warning: rule detection failed\n");
        else if (!o.quiet)
            fprintf(stderr, "strikethrough: %d candidate rule(s)\n", rules.count);
    }

    const Image *result = o.do_threshold ? &binary : &page;
    double ink = img_ink_fraction(result, 128);

    if (o.out_path && img_save_png(o.out_path, result) != 0) {
        fprintf(stderr, "error: cannot write %s\n", o.out_path);
        img_free(&page);
        img_free(&binary);
        linesegs_free(&rules);
        return 1;
    }

    FILE *jf = NULL;
    if (o.json_path && strcmp(o.json_path, "-") == 0) {
        jf = stdout;
    } else if (o.json_path) {
        jf = fopen(o.json_path, "w");
        if (!jf) {
            fprintf(stderr, "error: cannot write %s\n", o.json_path);
            img_free(&page);
            img_free(&binary);
            linesegs_free(&rules);
            return 1;
        }
    }

    if (jf) {
        fputs("{\n  \"input\": ", jf);
        json_string(jf, o.in_path);
        fputs(",\n  \"output\": ", jf);
        if (o.out_path) json_string(jf, o.out_path); else fputs("null", jf);
        fprintf(jf, ",\n  \"source\": {\"width\": %d, \"height\": %d}", src_w, src_h);
        fprintf(jf, ",\n  \"result\": {\"width\": %d, \"height\": %d}", result->w, result->h);
        fprintf(jf, ",\n  \"ink_fraction\": %.5f", ink);

        fputs(",\n  \"perspective\": {", jf);
        fprintf(jf, "\"applied\": %s", quad_found ? "true" : "false");
        if (quad_found) {
            fputs(", \"quad\": [", jf);
            for (int i = 0; i < 4; i++)
                fprintf(jf, "%s[%.2f, %.2f]", i ? ", " : "", quad.c[i].x, quad.c[i].y);
            fputc(']', jf);
        }
        fputc('}', jf);

        fprintf(jf, ",\n  \"deskew\": {\"applied\": %s, \"angle_deg\": %.3f}",
                deskewed ? "true" : "false", deskewed ? -skew : 0.0);
        fprintf(jf, ",\n  \"binarize\": {\"applied\": %s, \"k\": %.3f, \"window\": %d}",
                o.do_threshold ? "true" : "false", o.k, o.window);

        fputs(",\n  \"rules\": [", jf);
        for (int i = 0; i < rules.count; i++) {
            const LineSeg *s = &rules.items[i];
            fprintf(jf,
                    "%s\n    {\"x0\": %.1f, \"y0\": %.1f, \"x1\": %.1f, \"y1\": %.1f,"
                    " \"thickness\": %.1f, \"angle_deg\": %.2f, \"votes\": %d}",
                    i ? "," : "", s->x0, s->y0, s->x1, s->y1,
                    s->thickness, s->angle_deg, s->votes);
        }
        fputs(rules.count ? "\n  ]\n}\n" : "]\n}\n", jf);
        if (jf != stdout) fclose(jf);
    }

    img_free(&page);
    img_free(&binary);
    linesegs_free(&rules);
    return 0;
}
