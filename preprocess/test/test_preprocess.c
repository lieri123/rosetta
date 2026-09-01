/* Unit tests for the preprocessing primitives.
 *
 * Everything here runs in memory against synthesised pages, so the suite needs
 * no fixture files and no recogniser. Where a transform has an exact inverse
 * (homography, rotation) the test checks the round trip; where it estimates
 * something (skew, page corners, rule position) the test generates the page
 * with a known answer and asserts the estimate lands inside a tolerance that
 * matters downstream rather than one that merely passes.
 */
#include "../src/img.h"
#include "../src/geom.h"
#include "../src/deskew.h"
#include "../src/threshold.h"
#include "../src/lines.h"
#include "../src/synth.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int checks = 0;
static const char *current_test = "";

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        checks++;                                                           \
        if (!(cond)) {                                                      \
            failures++;                                                     \
            fprintf(stderr, "FAIL %s:%d [%s] ", __FILE__, __LINE__,         \
                    current_test);                                          \
            fprintf(stderr, __VA_ARGS__);                                   \
            fputc('\n', stderr);                                            \
        }                                                                   \
    } while (0)

#define CHECK_NEAR(got, want, tol, label)                                   \
    CHECK(fabs((got) - (want)) <= (tol), "%s: got %.4f, want %.4f +/- %.4f", \
          label, (double)(got), (double)(want), (double)(tol))

static void test_homography_roundtrip(void)
{
    current_test = "homography_roundtrip";

    Point from[4] = {{0, 0}, {100, 0}, {100, 50}, {0, 50}};
    Point to[4] = {{12, 7}, {180, 30}, {160, 120}, {5, 95}};

    double h[9];
    CHECK(homography_solve(from, to, h) == 0, "solve failed on a valid quad");

    for (int i = 0; i < 4; i++) {
        Point p = homography_apply(h, from[i]);
        CHECK_NEAR(p.x, to[i].x, 1e-6, "corner x");
        CHECK_NEAR(p.y, to[i].y, 1e-6, "corner y");
    }

    /* The inverse mapping must undo it: this is the property warp_quad relies
     * on when it solves destination -> source and gathers. */
    double inv[9];
    CHECK(homography_solve(to, from, inv) == 0, "inverse solve failed");
    Point mid = {50, 25};
    Point there = homography_apply(h, mid);
    Point back = homography_apply(inv, there);
    CHECK_NEAR(back.x, mid.x, 1e-6, "round trip x");
    CHECK_NEAR(back.y, mid.y, 1e-6, "round trip y");

    /* Collinear points have no homography and must be reported, not fudged. */
    Point degenerate[4] = {{0, 0}, {10, 0}, {20, 0}, {30, 0}};
    CHECK(homography_solve(degenerate, to, h) != 0, "collinear input accepted");
}

static void test_integral_matches_bruteforce(void)
{
    current_test = "integral_matches_bruteforce";

    Image im = img_new(37, 23);
    for (int y = 0; y < im.h; y++)
        for (int x = 0; x < im.w; x++)
            im.px[(size_t)y * im.w + x] = (unsigned char)((x * 7 + y * 13) % 256);

    Integral it = integral_build(&im);
    CHECK(it.sum != NULL, "integral build failed");

    const int rects[][4] = {
        {0, 0, 36, 22}, {3, 4, 9, 11}, {10, 10, 10, 10}, {30, 18, 40, 30}
    };
    for (size_t r = 0; r < sizeof(rects) / sizeof(rects[0]); r++) {
        int x0 = rects[r][0], y0 = rects[r][1];
        int x1 = rects[r][2] < im.w ? rects[r][2] : im.w - 1;
        int y1 = rects[r][3] < im.h ? rects[r][3] : im.h - 1;

        double sum = 0.0, sqsum = 0.0;
        double n = 0.0;
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                double v = im.px[(size_t)y * im.w + x];
                sum += v;
                sqsum += v * v;
                n += 1.0;
            }
        }
        double want_mean = sum / n;
        double want_sd = sqrt(sqsum / n - want_mean * want_mean);

        double mean, sd;
        integral_stats(&it, x0, y0, x1, y1, &mean, &sd);
        CHECK_NEAR(mean, want_mean, 1e-9, "mean");
        CHECK_NEAR(sd, want_sd, 1e-6, "stddev");
    }

    integral_free(&it);
    img_free(&im);
}

static void test_otsu_splits_bimodal(void)
{
    current_test = "otsu_splits_bimodal";

    Image im = img_new(100, 100);
    for (size_t i = 0; i < (size_t)im.w * im.h; i++)
        im.px[i] = (i % 5 == 0) ? 40 : 210;

    /* The convention is that pixels <= t are the dark class, so t == 40 is
     * the correct answer here, not an off-by-one. */
    int t = otsu_threshold(&im);
    CHECK(t >= 40 && t < 210, "threshold %d did not land between the modes", t);
    img_free(&im);
}

static void test_sauvola_survives_shading(void)
{
    current_test = "sauvola_survives_shading";

    /* A page bright on the left, dark on the right, with ink of constant
     * contrast throughout. A global cut loses one end; Sauvola should not. */
    SynthOpts o = synth_defaults();
    o.width = 600;
    o.height = 800;
    o.margin = 0;
    o.shading = 0.9;
    o.noise = 0.0;
    SynthPage page;
    CHECK(synth_page(&o, &page) == 0, "synth failed");

    Image bin = sauvola_binarize(&page.image, 0, 0.34);
    CHECK(img_ok(&bin), "binarise failed");

    /* Count ink in the left and right thirds. If shading beat the threshold
     * one side would be almost empty or almost solid. */
    long ink_left = 0, ink_right = 0, n_left = 0, n_right = 0;
    for (int y = 0; y < bin.h; y++) {
        for (int x = 0; x < bin.w; x++) {
            int v = bin.px[(size_t)y * bin.w + x] < 128;
            if (x < bin.w / 3) { ink_left += v; n_left++; }
            else if (x > 2 * bin.w / 3) { ink_right += v; n_right++; }
        }
    }
    double fl = (double)ink_left / (double)n_left;
    double fr = (double)ink_right / (double)n_right;

    CHECK(fl > 0.02 && fl < 0.5, "left ink fraction %.3f out of range", fl);
    CHECK(fr > 0.02 && fr < 0.5, "right ink fraction %.3f out of range", fr);
    CHECK(fabs(fl - fr) < 0.10, "ink fractions diverge across the gradient: %.3f vs %.3f", fl, fr);

    /* The comparison that motivates the whole function. */
    int t = otsu_threshold(&page.image);
    long g_left = 0, g_right = 0;
    for (int y = 0; y < page.image.h; y++) {
        for (int x = 0; x < page.image.w; x++) {
            int v = page.image.px[(size_t)y * page.image.w + x] <= t;
            if (x < page.image.w / 3) g_left += v;
            else if (x > 2 * page.image.w / 3) g_right += v;
        }
    }
    double gl = (double)g_left / (double)n_left;
    double gr = (double)g_right / (double)n_right;
    CHECK(fabs(gl - gr) > fabs(fl - fr),
          "global threshold was not worse than Sauvola (%.3f vs %.3f)",
          fabs(gl - gr), fabs(fl - fr));

    img_free(&bin);
    img_free(&page.image);
}

static void test_skew_estimation(void)
{
    current_test = "skew_estimation";

    const double angles[] = {-6.0, -2.5, 0.0, 1.5, 4.0};
    for (size_t i = 0; i < sizeof(angles) / sizeof(angles[0]); i++) {
        SynthOpts o = synth_defaults();
        o.width = 700;
        o.height = 900;
        o.margin = 0;
        o.noise = 0.01;
        o.skew_deg = angles[i];
        SynthPage page;
        CHECK(synth_page(&o, &page) == 0, "synth failed at %.1f deg", angles[i]);

        Image bin = sauvola_binarize(&page.image, 0, 0.34);
        double est = 999.0;
        CHECK(estimate_skew(&bin, 10.0, &est) == 0, "estimate failed at %.1f deg", angles[i]);

        /* A quarter degree over a 900px page is under two pixels of drift end
         * to end, which is below what line grouping cares about. */
        CHECK(fabs(est - angles[i]) <= 0.25,
              "skew estimate %.2f for true %.2f", est, angles[i]);

        img_free(&bin);
        img_free(&page.image);
    }
}

static void test_blank_page_is_not_deskewed(void)
{
    current_test = "blank_page_is_not_deskewed";

    Image blank = img_new(400, 400);
    memset(blank.px, 255, (size_t)blank.w * blank.h);
    double angle = 42.0;
    CHECK(estimate_skew(&blank, 10.0, &angle) != 0,
          "a blank page produced an angle instead of an error");
    img_free(&blank);
}

static void test_rotation_roundtrip(void)
{
    current_test = "rotation_roundtrip";

    SynthOpts o = synth_defaults();
    o.width = 300;
    o.height = 300;
    o.margin = 0;
    o.noise = 0.0;
    SynthPage page;
    CHECK(synth_page(&o, &page) == 0, "synth failed");

    Image a = rotate_image(&page.image, 5.0, 255);
    Image b = rotate_image(&a, -5.0, 255);
    CHECK(img_ok(&b), "rotation failed");

    int ox = (b.w - page.image.w) / 2;
    int oy = (b.h - page.image.h) / 2;

    /* Two bilinear resamples visibly blur two-pixel pen strokes, so pixelwise
     * equality is the wrong bar. What must survive is the geometry: an error in
     * the inverse mapping (a sign flip, an off-by-one in the centre, a missing
     * half pixel) shows up as the ink centroid walking away from where it
     * started, which blurring alone cannot cause. */
    double cx_before = 0, cy_before = 0, m_before = 0;
    double cx_after = 0, cy_after = 0, m_after = 0;
    double err = 0.0;
    long n = 0;

    for (int y = 0; y < page.image.h; y++) {
        for (int x = 0; x < page.image.w; x++) {
            double wa = 255.0 - page.image.px[(size_t)y * page.image.w + x];
            double wb = 255.0 - img_at(&b, x + ox, y + oy);
            cx_before += x * wa; cy_before += y * wa; m_before += wa;
            cx_after += x * wb; cy_after += y * wb; m_after += wb;

            if (y > page.image.h / 4 && y < 3 * page.image.h / 4 &&
                x > page.image.w / 4 && x < 3 * page.image.w / 4) {
                err += fabs(wa - wb);
                n++;
            }
        }
    }

    CHECK(m_before > 0 && m_after > 0, "no ink to compare");
    CHECK_NEAR(cx_after / m_after, cx_before / m_before, 1.0, "ink centroid x");
    CHECK_NEAR(cy_after / m_after, cy_before / m_before, 1.0, "ink centroid y");
    /* Ink is conserved by an area-preserving rotation, give or take what the
     * corners lose off the edge of the frame. */
    CHECK_NEAR(m_after / m_before, 1.0, 0.10, "ink mass ratio");
    CHECK(err / n < 30.0, "round trip mean error %.2f too high", err / n);

    img_free(&a);
    img_free(&b);
    img_free(&page.image);
}

static void test_page_quad_detection(void)
{
    current_test = "page_quad_detection";

    SynthOpts o = synth_defaults();
    o.width = 800;
    o.height = 1000;
    o.margin = 70;
    o.perspective = 0.12;
    o.noise = 0.01;
    SynthPage page;
    CHECK(synth_page(&o, &page) == 0, "synth failed");

    Quad found;
    CHECK(detect_page_quad(&page.image, &found) == 0, "no quad detected");

    for (int i = 0; i < 4; i++) {
        double dx = found.c[i].x - page.quad.c[i].x;
        double dy = found.c[i].y - page.quad.c[i].y;
        double d = sqrt(dx * dx + dy * dy);
        /* Detection runs downscaled; a few pixels of corner error is expected
         * and harmless, since the warp resamples anyway. */
        CHECK(d < 12.0, "corner %d off by %.1f px", i, d);
    }

    Image rect = warp_quad(&page.image, &found, 0, 0);
    CHECK(img_ok(&rect), "warp failed");
    CHECK(rect.w > 500 && rect.h > 700, "warped page is %dx%d", rect.w, rect.h);

    /* After rectification the writing should be square to the frame again. */
    Image bin = sauvola_binarize(&rect, 0, 0.34);
    double residual = 99.0;
    CHECK(estimate_skew(&bin, 10.0, &residual) == 0, "skew estimate failed");
    CHECK(fabs(residual) < 1.0, "residual skew after rectification: %.2f deg", residual);

    img_free(&bin);
    img_free(&rect);
    img_free(&page.image);
}

static void test_page_quad_survives_uneven_lighting(void)
{
    current_test = "page_quad_survives_uneven_lighting";

    /* A lamp to one side: the far edge of the paper ends up dimmer than the
     * near edge by a factor of three. The page is still obviously a page to a
     * human, and the detector must find all four corners rather than stopping
     * where the paper gets dark -- cropping the shaded edge would silently
     * throw away text before the recogniser ever sees it. */
    SynthOpts o = synth_defaults();
    o.width = 900;
    o.height = 1200;
    o.margin = 60;
    o.perspective = 0.10;
    o.shading = 0.6;
    o.noise = 0.02;
    SynthPage page;
    CHECK(synth_page(&o, &page) == 0, "synth failed");

    Quad found;
    CHECK(detect_page_quad(&page.image, &found) == 0, "no quad detected");

    for (int i = 0; i < 4; i++) {
        double dx = found.c[i].x - page.quad.c[i].x;
        double dy = found.c[i].y - page.quad.c[i].y;
        double d = sqrt(dx * dx + dy * dy);
        CHECK(d < 15.0, "corner %d off by %.1f px under shading", i, d);
    }

    img_free(&page.image);
}

static void test_scanned_page_is_left_alone(void)
{
    current_test = "scanned_page_is_left_alone";

    /* A flatbed scan has no background to find. Detection must decline rather
     * than warp the page to whatever the largest bright blob happened to be. */
    SynthOpts o = synth_defaults();
    o.width = 600;
    o.height = 800;
    o.margin = 0;
    o.noise = 0.0;
    SynthPage page;
    CHECK(synth_page(&o, &page) == 0, "synth failed");

    Quad q;
    CHECK(detect_page_quad(&page.image, &q) != 0, "detected a quad on a full-frame page");
    img_free(&page.image);
}

static void test_strikethrough_detection(void)
{
    current_test = "strikethrough_detection";

    SynthOpts o = synth_defaults();
    o.width = 800;
    o.height = 600;
    o.margin = 0;
    o.text_lines = 8;
    o.noise = 0.0;
    o.strike_line = 3;
    SynthPage page;
    CHECK(synth_page(&o, &page) == 0, "synth failed");

    Image bin = sauvola_binarize(&page.image, 0, 0.34);
    LineParams lp = line_params_default(&bin);
    LineSegs segs;
    CHECK(detect_horizontal_rules(&bin, &lp, &segs) == 0, "detection failed");
    CHECK(segs.count > 0, "no rules found on a page with a strikethrough");

    /* The drawn stroke must be among the reported segments. */
    int matched = 0;
    for (int i = 0; i < segs.count; i++) {
        double ymid = (segs.items[i].y0 + segs.items[i].y1) / 2.0;
        if (fabs(ymid - page.strike_y0) <= 4.0 &&
            segs.items[i].x1 - segs.items[i].x0 > 0.5 * (page.strike_x1 - page.strike_x0)) {
            matched = 1;
            CHECK(segs.items[i].thickness <= lp.max_thickness,
                  "matched rule is %.0f px thick", segs.items[i].thickness);
        }
    }
    CHECK(matched, "the drawn strikethrough at y=%.0f was not reported", page.strike_y0);

    /* And the word rows, which are three times thicker, must not be. */
    int false_positives = 0;
    for (int i = 0; i < segs.count; i++) {
        double ymid = (segs.items[i].y0 + segs.items[i].y1) / 2.0;
        if (fabs(ymid - page.strike_y0) > 8.0) false_positives++;
    }
    CHECK(false_positives <= 2, "%d segments reported away from the strikethrough",
          false_positives);

    linesegs_free(&segs);
    img_free(&bin);
    img_free(&page.image);
}

static void test_no_rules_on_plain_text(void)
{
    current_test = "no_rules_on_plain_text";

    SynthOpts o = synth_defaults();
    o.width = 800;
    o.height = 600;
    o.margin = 0;
    o.text_lines = 8;
    o.noise = 0.0;
    o.strike_line = -1;
    SynthPage page;
    CHECK(synth_page(&o, &page) == 0, "synth failed");

    Image bin = sauvola_binarize(&page.image, 0, 0.34);
    LineParams lp = line_params_default(&bin);
    LineSegs segs;
    CHECK(detect_horizontal_rules(&bin, &lp, &segs) == 0, "detection failed");
    CHECK(segs.count <= 2, "%d rules reported on a page with none", segs.count);

    linesegs_free(&segs);
    img_free(&bin);
    img_free(&page.image);
}

static void test_stubby_marks_are_not_rules(void)
{
    current_test = "stubby_marks_are_not_rules";

    /* A short, thick horizontal blob -- the shape two or three letter parts
     * make when they happen to line up. It is horizontal ink of an acceptable
     * thickness, so only the length-to-thickness ratio separates it from a pen
     * stroke drawn through a word. */
    /* Page-shaped: the thickness limit is derived from the image height, and
     * a 200px-tall "page" would rule out strokes a real page allows. */
    Image page = img_new(800, 1000);
    memset(page.px, 255, (size_t)page.w * page.h);
    for (int y = 300; y < 306; y++)
        for (int x = 200; x < 260; x++)
            page.px[(size_t)y * page.w + x] = 0;

    LineParams lp = line_params_default(&page);
    LineSegs segs;
    CHECK(detect_horizontal_rules(&page, &lp, &segs) == 0, "detection failed");
    CHECK(segs.count == 0, "a 60x6 blob (10:1) was reported as a rule");
    linesegs_free(&segs);

    /* The same thickness drawn the length of a line is a strikethrough. */
    for (int y = 600; y < 606; y++)
        for (int x = 100; x < 700; x++)
            page.px[(size_t)y * page.w + x] = 0;

    CHECK(detect_horizontal_rules(&page, &lp, &segs) == 0, "detection failed");
    CHECK(segs.count == 1, "want the long stroke reported, got %d segment(s)", segs.count);
    if (segs.count == 1) {
        CHECK(segs.items[0].x1 - segs.items[0].x0 > 500,
              "reported segment is only %.0f px long", segs.items[0].x1 - segs.items[0].x0);
    }

    linesegs_free(&segs);
    img_free(&page);
}

static void test_downscale_preserves_brightness(void)
{
    current_test = "downscale_preserves_brightness";

    Image im = img_new(500, 400);
    for (size_t i = 0; i < (size_t)im.w * im.h; i++)
        im.px[i] = (unsigned char)(i % 251);

    double before = 0.0;
    for (size_t i = 0; i < (size_t)im.w * im.h; i++) before += im.px[i];
    before /= (double)im.w * im.h;

    double scale = 0.0;
    Image small = img_downscale_to(&im, 100, &scale);
    CHECK(img_ok(&small), "downscale failed");
    CHECK(small.w <= 100 && small.h <= 100, "downscale produced %dx%d", small.w, small.h);
    CHECK_NEAR(scale, (double)small.w / im.w, 1e-9, "reported scale");

    double after = 0.0;
    for (size_t i = 0; i < (size_t)small.w * small.h; i++) after += small.px[i];
    after /= (double)small.w * small.h;
    CHECK_NEAR(after, before, 3.0, "mean brightness");

    img_free(&small);
    img_free(&im);
}

int main(void)
{
    test_homography_roundtrip();
    test_integral_matches_bruteforce();
    test_otsu_splits_bimodal();
    test_sauvola_survives_shading();
    test_skew_estimation();
    test_blank_page_is_not_deskewed();
    test_rotation_roundtrip();
    test_page_quad_detection();
    test_page_quad_survives_uneven_lighting();
    test_scanned_page_is_left_alone();
    test_strikethrough_detection();
    test_no_rules_on_plain_text();
    test_stubby_marks_are_not_rules();
    test_downscale_preserves_brightness();

    if (failures) {
        fprintf(stderr, "\n%d of %d checks failed\n", failures, checks);
        return 1;
    }
    printf("all %d checks passed\n", checks);
    return 0;
}
