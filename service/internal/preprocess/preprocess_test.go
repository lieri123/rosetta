package preprocess

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakeBinary writes a shell script standing in for rosetta-preprocess, so the
// runner's argument handling and JSON parsing can be tested without building
// the C.
func fakeBinary(t *testing.T, script string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell script stand-in needs a POSIX shell")
	}
	path := filepath.Join(t.TempDir(), "fake-preprocess")
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"+script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestUnavailableBinaryIsReported(t *testing.T) {
	runner := NewRunner(filepath.Join(t.TempDir(), "does-not-exist"))
	if runner.Available() {
		t.Fatal("a missing binary reported as available")
	}
	_, err := runner.Run(context.Background(), "in.png", "out.png", Options{})
	if err != ErrUnavailable {
		t.Errorf("want ErrUnavailable, got %v", err)
	}
}

func TestRunParsesMetadata(t *testing.T) {
	binary := fakeBinary(t, `cat <<'JSON'
{"input":"in.png","output":"out.png","source":{"width":900,"height":1200},
 "result":{"width":823,"height":1086},"ink_fraction":0.053,
 "perspective":{"applied":true,"quad":[[106,60],[838,97],[819,1138],[60,1085]]},
 "deskew":{"applied":true,"angle_deg":-3.5},
 "binarize":{"applied":true,"k":0.34,"window":0},
 "rules":[{"x0":95,"y0":366,"x1":729,"y1":366,"thickness":4,"angle_deg":0,"votes":635}]}
JSON`)

	metadata, err := NewRunner(binary).Run(context.Background(), "in.png",
		filepath.Join(t.TempDir(), "out.png"), Options{DetectStrikethrough: true})
	if err != nil {
		t.Fatal(err)
	}

	if !metadata.Perspective.Applied || len(metadata.Perspective.Quad) != 4 {
		t.Errorf("perspective not parsed: %+v", metadata.Perspective)
	}
	if metadata.Deskew.AngleDeg != -3.5 {
		t.Errorf("skew angle not parsed: %+v", metadata.Deskew)
	}
	if len(metadata.Rules) != 1 || metadata.Rules[0].X1 != 729 {
		t.Errorf("rules not parsed: %+v", metadata.Rules)
	}
	if metadata.Result.Width != 823 {
		t.Errorf("result size not parsed: %+v", metadata.Result)
	}
}

func TestRunPassesTheRequestedFlags(t *testing.T) {
	dir := t.TempDir()
	argsFile := filepath.Join(dir, "args")
	binary := fakeBinary(t, `echo "$@" > `+argsFile+`
echo '{"rules":[]}'`)

	_, err := NewRunner(binary).Run(context.Background(), "in.png",
		filepath.Join(dir, "out.png"),
		Options{DetectStrikethrough: true, SkipDeskew: true})
	if err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	args := string(raw)

	for _, want := range []string{"--strikethrough", "--no-deskew", "--json -", "--quiet"} {
		if !strings.Contains(args, want) {
			t.Errorf("missing %q in: %s", want, args)
		}
	}
	// Options that were not asked for must not be sent: silently skipping
	// perspective correction would be very hard to notice from the output.
	if strings.Contains(args, "--no-perspective") {
		t.Errorf("sent --no-perspective when it was not requested: %s", args)
	}
}

func TestRunSurfacesStderrOnFailure(t *testing.T) {
	binary := fakeBinary(t, `echo "error: cannot decode in.png" >&2
exit 1`)

	_, err := NewRunner(binary).Run(context.Background(), "in.png",
		filepath.Join(t.TempDir(), "out.png"), Options{})
	if err == nil || !strings.Contains(err.Error(), "cannot decode") {
		t.Errorf("want the binary's own message, got %v", err)
	}
}

func TestRunRejectsUnparseableMetadata(t *testing.T) {
	binary := fakeBinary(t, `echo 'not json'`)
	_, err := NewRunner(binary).Run(context.Background(), "in.png",
		filepath.Join(t.TempDir(), "out.png"), Options{})
	if err == nil || !strings.Contains(err.Error(), "unparseable metadata") {
		t.Errorf("want a parse error, got %v", err)
	}
}

func TestRunCreatesTheOutputDirectory(t *testing.T) {
	binary := fakeBinary(t, `echo '{"rules":[]}'`)
	out := filepath.Join(t.TempDir(), "nested", "deeper", "out.png")

	if _, err := NewRunner(binary).Run(context.Background(), "in.png", out, Options{}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Dir(out)); err != nil {
		t.Errorf("output directory not created: %v", err)
	}
}

func TestRulesDefaultToEmptyNotNil(t *testing.T) {
	// The pipeline ranges over this; a nil slice would work in Go but
	// serialises as null, which the browser then has to special-case.
	binary := fakeBinary(t, `echo '{"deskew":{"applied":false}}'`)
	metadata, err := NewRunner(binary).Run(context.Background(), "in.png",
		filepath.Join(t.TempDir(), "out.png"), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if metadata.Rules == nil {
		t.Error("want an empty slice, got nil")
	}
}
