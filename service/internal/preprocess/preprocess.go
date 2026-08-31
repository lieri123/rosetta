// Package preprocess drives the C page-cleanup binary.
//
// Shelling out rather than binding through cgo. The C is a batch transform
// over a file with a JSON report -- a shape a subprocess models exactly -- and
// keeping it out of the Go build means the service still cross-compiles to a
// static binary, the C can be tested and profiled on its own, and a crash in
// the image code cannot take the server down with it. The cost is a process
// spawn per page, which is noise next to a recognition round trip.
package preprocess

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// ErrUnavailable means the binary is missing. Callers fall back to the
// original image rather than failing the page: unpreprocessed recognition is
// worse, but it is not nothing, and a missing build artefact should not make
// the service refuse work.
var ErrUnavailable = errors.New("preprocess binary not available")

type Runner struct {
	Binary  string
	Timeout time.Duration
}

func NewRunner(binary string) *Runner {
	return &Runner{Binary: binary, Timeout: 90 * time.Second}
}

func (r *Runner) Available() bool {
	if r.Binary == "" {
		return false
	}
	info, err := os.Stat(r.Binary)
	if err != nil {
		// Also accept a bare name resolved through PATH.
		_, lookErr := exec.LookPath(r.Binary)
		return lookErr == nil
	}
	return !info.IsDir()
}

type Size struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

type Perspective struct {
	Applied bool        `json:"applied"`
	Quad    [][]float64 `json:"quad,omitempty"`
}

type Deskew struct {
	Applied  bool    `json:"applied"`
	AngleDeg float64 `json:"angle_deg"`
}

// Rule is a near-horizontal stroke the C found. Whether it is a strikethrough
// depends on the word boxes, which only exist after recognition, so that
// judgement is made later in the layout package.
type Rule struct {
	X0        float64 `json:"x0"`
	Y0        float64 `json:"y0"`
	X1        float64 `json:"x1"`
	Y1        float64 `json:"y1"`
	Thickness float64 `json:"thickness"`
	AngleDeg  float64 `json:"angle_deg"`
	Votes     int     `json:"votes"`
}

type Metadata struct {
	Input       string      `json:"input"`
	Output      string      `json:"output"`
	Source      Size        `json:"source"`
	Result      Size        `json:"result"`
	InkFraction float64     `json:"ink_fraction"`
	Perspective Perspective `json:"perspective"`
	Deskew      Deskew      `json:"deskew"`
	Rules       []Rule      `json:"rules"`
}

type Options struct {
	DetectStrikethrough bool
	SkipPerspective     bool
	SkipDeskew          bool
	SkipThreshold       bool
	DebugDir            string
}

// Run cleans one page, writing the result to outPath.
func (r *Runner) Run(ctx context.Context, inPath, outPath string, options Options) (*Metadata, error) {
	if !r.Available() {
		return nil, ErrUnavailable
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return nil, err
	}

	args := []string{inPath, "-o", outPath, "--json", "-", "--quiet"}
	if options.DetectStrikethrough {
		args = append(args, "--strikethrough")
	}
	if options.SkipPerspective {
		args = append(args, "--no-perspective")
	}
	if options.SkipDeskew {
		args = append(args, "--no-deskew")
	}
	if options.SkipThreshold {
		args = append(args, "--no-threshold")
	}
	if options.DebugDir != "" {
		if err := os.MkdirAll(options.DebugDir, 0o755); err != nil {
			return nil, err
		}
		args = append(args, "--debug-dir", options.DebugDir)
	}

	timeout := r.Timeout
	if timeout <= 0 {
		timeout = 90 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var stdout, stderr bytes.Buffer
	command := exec.CommandContext(ctx, r.Binary, args...)
	command.Stdout = &stdout
	command.Stderr = &stderr

	if err := command.Run(); err != nil {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("preprocess timed out after %s", timeout)
		}
		// stderr carries the reason ("cannot decode", "cannot write"), which is
		// far more use in a log than the exit status.
		return nil, fmt.Errorf("preprocess failed: %w: %s", err, trim(stderr.String()))
	}

	var metadata Metadata
	if err := json.Unmarshal(stdout.Bytes(), &metadata); err != nil {
		return nil, fmt.Errorf("preprocess wrote unparseable metadata: %w", err)
	}
	if metadata.Rules == nil {
		metadata.Rules = []Rule{}
	}
	return &metadata, nil
}

func trim(s string) string {
	const limit = 400
	s = string(bytes.TrimSpace([]byte(s)))
	if len(s) > limit {
		return s[:limit] + "..."
	}
	return s
}
