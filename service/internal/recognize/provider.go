// Package recognize wraps the handwriting recognition APIs.
//
// Everything behind one interface, for two reasons. The obvious one is that
// providers can be swapped and compared without touching the pipeline. The
// less obvious one is that the mock provider makes the entire service testable
// and demonstrable offline -- recognition is the only step that costs money
// per page and sends ink to someone else's computer, and a system where you
// cannot run the pipeline without doing both is a system nobody will run.
package recognize

import (
	"context"
	"fmt"
	"sort"
)

// Box is a word's bounding box in the coordinate space of the image that was
// submitted -- which is the preprocessed image, not the original photo.
type Box struct {
	X0 float64 `json:"x0"`
	Y0 float64 `json:"y0"`
	X1 float64 `json:"x1"`
	Y1 float64 `json:"y1"`
}

func (b Box) Width() float64  { return b.X1 - b.X0 }
func (b Box) Height() float64 { return b.Y1 - b.Y0 }
func (b Box) MidY() float64   { return (b.Y0 + b.Y1) / 2 }

type Alternative struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
}

type Word struct {
	Text         string        `json:"text"`
	Confidence   float64       `json:"confidence"`
	Box          Box           `json:"box"`
	Alternatives []Alternative `json:"alternatives,omitempty"`
}

type Result struct {
	Provider string `json:"provider"`
	Words    []Word `json:"words"`
	// Raw text as the provider assembled it. Kept for comparison; the service
	// does its own layout from the boxes, because provider line breaking is
	// tuned for printed text and handles marginalia poorly.
	RawText string `json:"raw_text,omitempty"`
}

type Provider interface {
	Name() string
	Recognize(ctx context.Context, image []byte) (*Result, error)
}

// SortReadingOrder puts words in roughly the order a person would read them.
//
// Rough on purpose: this is a stable fallback ordering by vertical band and
// then horizontal position. Real line grouping happens in the layout package,
// which has the tolerances and can see the whole page at once.
func SortReadingOrder(words []Word) {
	sort.SliceStable(words, func(i, j int) bool {
		a, b := words[i].Box, words[j].Box
		// Same line if their vertical spans overlap by more than half the
		// shorter box: handwriting baselines wander, so an equality test on Y
		// would split every line in two.
		overlap := minFloat(a.Y1, b.Y1) - maxFloat(a.Y0, b.Y0)
		shorter := minFloat(a.Height(), b.Height())
		if shorter > 0 && overlap > shorter*0.5 {
			return a.X0 < b.X0
		}
		return a.Y0 < b.Y0
	})
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// New builds a provider by name.
func New(name, googleKey, azureEndpoint, azureKey string) (Provider, error) {
	switch name {
	case "mock", "":
		return NewMock(), nil
	case "google":
		return NewGoogleVision(googleKey), nil
	case "azure":
		return NewAzureRead(azureEndpoint, azureKey), nil
	default:
		return nil, fmt.Errorf("unknown recognition provider %q", name)
	}
}
