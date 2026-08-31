package recognize

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// MockProvider stands in for a paid API.
//
// It is not a stub that returns nothing: it produces a plausible page of
// words, with bounding boxes, confidences and the occasional confidently-wrong
// reading, deterministically derived from the bytes it is given. That is
// enough for the rest of the pipeline -- layout, rescoring, tiering, the
// editor -- to be exercised and demonstrated end to end without an API key,
// and enough for the tests to assert on exact output.
//
// Its errors are drawn from the same shape-collision families a real
// recogniser makes on handwriting, so what you see in the editor when running
// on the mock looks like what you see when running on Vision.
type MockProvider struct {
	fixture *Result
	lines   []string
}

var defaultMockLines = []string{
	"the noisy channel model treats recognition output as an observation",
	"every correction gives an aligned pair of predicted and corrected text",
	"accumulating those alignments builds a personal confusion matrix",
	"the matrix shows my own failure modes rather than an average writer's",
	"low confidence tokens get candidates from inverse confusions",
	"the decoder substitutes only when the margin is decisive",
}

func NewMock() *MockProvider {
	return &MockProvider{lines: defaultMockLines}
}

// NewMockFromFile replays a recorded provider response.
//
// Recording a real Vision response once and replaying it is how the expensive
// path gets regression tested: the parsing, the layout and the rescoring all
// run on genuine output, repeatedly, for free.
func NewMockFromFile(path string) (*MockProvider, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading mock fixture: %w", err)
	}
	var result Result
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("parsing mock fixture %s: %w", path, err)
	}
	return &MockProvider{fixture: &result}, nil
}

func (m *MockProvider) Name() string { return "mock" }

// Confusions the mock introduces, in the order they are tried. These are shape
// collisions -- the kind handwriting actually produces -- not random noise.
var mockConfusions = []struct{ from, to string }{
	{"m", "rn"},
	{"S", "5"},
	{"cl", "d"},
	{"a", "o"},
	{"i", "l"},
	{"n", "h"},
	{"t", "f"},
	{"e", "c"},
}

func (m *MockProvider) Recognize(ctx context.Context, image []byte) (*Result, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m.fixture != nil {
		clone := *m.fixture
		clone.Provider = m.Name()
		return &clone, nil
	}

	// Same bytes in, same words out: tests and demos need to be repeatable.
	digest := sha256.Sum256(image)
	seed := binary.LittleEndian.Uint64(digest[:8])
	random := &splitmix64{state: seed}

	const (
		lineHeight = 46.0
		charWidth  = 11.0
		originX    = 60.0
		originY    = 80.0
	)

	result := &Result{Provider: m.Name()}
	var raw strings.Builder

	for lineIndex, line := range m.lines {
		x := originX
		y := originY + float64(lineIndex)*lineHeight

		for _, word := range strings.Fields(line) {
			text := word
			confidence := 0.90 + random.float()*0.09

			// Roughly one word in five comes back misread, and a quarter of
			// those come back misread *and* confident. Those are the ones a
			// confidence threshold alone can never catch, and the reason the
			// amber tier exists.
			if random.float() < 0.20 {
				text = corrupt(word, random)
				if random.float() < 0.25 {
					confidence = 0.88 + random.float()*0.09
				} else {
					confidence = 0.30 + random.float()*0.45
				}
			}

			width := float64(len(text)) * charWidth
			result.Words = append(result.Words, Word{
				Text:       text,
				Confidence: confidence,
				Box:        Box{X0: x, Y0: y, X1: x + width, Y1: y + 28},
			})
			raw.WriteString(text)
			raw.WriteByte(' ')
			x += width + charWidth
		}
		raw.WriteByte('\n')
	}

	result.RawText = raw.String()
	return result, nil
}

func corrupt(word string, random *splitmix64) string {
	for attempt := 0; attempt < len(mockConfusions); attempt++ {
		pick := mockConfusions[int(random.next()%uint64(len(mockConfusions)))]
		if index := strings.Index(word, pick.from); index >= 0 {
			return word[:index] + pick.to + word[index+len(pick.from):]
		}
	}
	return word
}

// splitmix64 is a tiny deterministic generator. math/rand would do, but its
// stream is not guaranteed stable across Go releases, and these outputs are
// asserted on in tests.
type splitmix64 struct{ state uint64 }

func (s *splitmix64) next() uint64 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func (s *splitmix64) float() float64 {
	return float64(s.next()>>11) / float64(1<<53)
}
