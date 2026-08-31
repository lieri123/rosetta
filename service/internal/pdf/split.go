// Package pdf splits a PDF into page images.
//
// Rasterising a PDF properly means shipping a PDF renderer, and there is no
// good reason to carry one: every machine this runs on already has poppler or
// mupdf a package manager away, and both do the job better than anything worth
// writing here. So this shells out to whichever is present and says plainly
// what to install when neither is.
package pdf

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

type Tool struct {
	Name string
	Path string
}

// Find returns the first available rasteriser.
func Find() (Tool, error) {
	for _, name := range []string{"pdftoppm", "mutool"} {
		if path, err := exec.LookPath(name); err == nil {
			return Tool{Name: name, Path: path}, nil
		}
	}
	return Tool{}, fmt.Errorf(
		"no PDF rasteriser found: install poppler-utils (pdftoppm) or mupdf-tools (mutool), " +
			"or upload page images instead")
}

// Split renders every page of a PDF into outDir at the given DPI and returns
// the resulting image paths in page order.
func Split(ctx context.Context, pdfPath, outDir string, dpi int) ([]string, error) {
	tool, err := Find()
	if err != nil {
		return nil, err
	}
	if dpi <= 0 {
		// 300 is the usual floor for handwriting: below it, thin pen strokes
		// start dropping out during binarisation.
		dpi = 300
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return nil, err
	}

	prefix := filepath.Join(outDir, "page")

	var command *exec.Cmd
	switch tool.Name {
	case "pdftoppm":
		command = exec.CommandContext(ctx, tool.Path, "-png", "-r", fmt.Sprint(dpi), pdfPath, prefix)
	case "mutool":
		command = exec.CommandContext(ctx, tool.Path, "draw", "-o", prefix+"-%d.png",
			"-r", fmt.Sprint(dpi), pdfPath)
	}

	output, err := command.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("%s: %w: %s", tool.Name, err, strings.TrimSpace(string(output)))
	}

	matches, err := filepath.Glob(prefix + "*.png")
	if err != nil {
		return nil, err
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("%s produced no pages from %s", tool.Name, filepath.Base(pdfPath))
	}

	// Both tools zero-pad, but not identically, so sort on the numeric suffix
	// rather than lexically: page10 must not sort before page2.
	sort.Slice(matches, func(i, j int) bool {
		return pageNumber(matches[i]) < pageNumber(matches[j])
	})
	return matches, nil
}

func pageNumber(path string) int {
	base := strings.TrimSuffix(filepath.Base(path), ".png")
	index := strings.LastIndexByte(base, '-')
	if index < 0 {
		return 0
	}
	number := 0
	for _, char := range base[index+1:] {
		if char < '0' || char > '9' {
			return number
		}
		number = number*10 + int(char-'0')
	}
	return number
}

// IsPDF sniffs the magic bytes rather than trusting the filename.
func IsPDF(data []byte) bool {
	return len(data) >= 5 && string(data[:5]) == "%PDF-"
}
