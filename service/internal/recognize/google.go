package recognize

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// GoogleVision calls Cloud Vision's DOCUMENT_TEXT_DETECTION.
//
// That mode rather than TEXT_DETECTION because it returns the full
// block/paragraph/word/symbol hierarchy with a confidence at every level. The
// per-symbol confidences are the signal this whole project is built on, and
// the plain text mode does not return them.
type GoogleVision struct {
	APIKey string
	Client *http.Client
	// Endpoint is overridable so tests can point at a local server.
	Endpoint string
}

func NewGoogleVision(apiKey string) *GoogleVision {
	return &GoogleVision{
		APIKey:   apiKey,
		Client:   &http.Client{Timeout: 60 * time.Second},
		Endpoint: "https://vision.googleapis.com/v1/images:annotate",
	}
}

func (g *GoogleVision) Name() string { return "google-vision" }

type gcvVertex struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type gcvBoundingPoly struct {
	Vertices []gcvVertex `json:"vertices"`
}

type gcvSymbol struct {
	Text        string          `json:"text"`
	Confidence  float64         `json:"confidence"`
	BoundingBox gcvBoundingPoly `json:"boundingBox"`
	Property    *gcvProperty    `json:"property,omitempty"`
}

type gcvProperty struct {
	DetectedBreak *struct {
		Type string `json:"type"`
	} `json:"detectedBreak,omitempty"`
}

type gcvWord struct {
	Symbols     []gcvSymbol     `json:"symbols"`
	Confidence  float64         `json:"confidence"`
	BoundingBox gcvBoundingPoly `json:"boundingBox"`
}

type gcvResponse struct {
	Responses []struct {
		FullTextAnnotation struct {
			Text  string `json:"text"`
			Pages []struct {
				Blocks []struct {
					Paragraphs []struct {
						Words []gcvWord `json:"words"`
					} `json:"paragraphs"`
				} `json:"blocks"`
			} `json:"pages"`
		} `json:"fullTextAnnotation"`
		Error *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error,omitempty"`
	} `json:"responses"`
}

func (g *GoogleVision) Recognize(ctx context.Context, image []byte) (*Result, error) {
	if g.APIKey == "" {
		return nil, fmt.Errorf("google vision: no API key configured")
	}

	payload := map[string]any{
		"requests": []any{
			map[string]any{
				"image":    map[string]any{"content": base64.StdEncoding.EncodeToString(image)},
				"features": []any{map[string]any{"type": "DOCUMENT_TEXT_DETECTION"}},
				"imageContext": map[string]any{
					"languageHints": []string{"en"},
				},
			},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	url := g.Endpoint + "?key=" + g.APIKey
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := g.Client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("google vision: %w", err)
	}
	defer response.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(response.Body, 64<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("google vision: HTTP %d: %s", response.StatusCode, truncate(raw, 400))
	}

	return parseGoogleResponse(raw)
}

func parseGoogleResponse(raw []byte) (*Result, error) {
	var decoded gcvResponse
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, fmt.Errorf("google vision: parsing response: %w", err)
	}
	if len(decoded.Responses) == 0 {
		return nil, fmt.Errorf("google vision: empty response")
	}
	first := decoded.Responses[0]
	if first.Error != nil {
		return nil, fmt.Errorf("google vision: %s (code %d)", first.Error.Message, first.Error.Code)
	}

	result := &Result{Provider: "google-vision", RawText: first.FullTextAnnotation.Text}
	for _, page := range first.FullTextAnnotation.Pages {
		for _, block := range page.Blocks {
			for _, paragraph := range block.Paragraphs {
				for _, word := range paragraph.Words {
					result.Words = append(result.Words, convertGoogleWord(word))
				}
			}
		}
	}
	return result, nil
}

func convertGoogleWord(word gcvWord) Word {
	var text strings.Builder
	symbolConfidenceSum := 0.0
	for _, symbol := range word.Symbols {
		text.WriteString(symbol.Text)
		symbolConfidenceSum += symbol.Confidence
	}

	confidence := word.Confidence
	if confidence == 0 && len(word.Symbols) > 0 {
		// Vision sometimes omits word confidence while populating the symbols.
		// The mean over symbols is the honest reconstruction; taking the
		// minimum would flag every long word and taking 1.0 would hide real
		// uncertainty.
		confidence = symbolConfidenceSum / float64(len(word.Symbols))
	}

	return Word{
		Text:       text.String(),
		Confidence: confidence,
		Box:        boxFromVertices(word.BoundingBox.Vertices),
	}
}

func boxFromVertices(vertices []gcvVertex) Box {
	if len(vertices) == 0 {
		return Box{}
	}
	box := Box{
		X0: float64(vertices[0].X), Y0: float64(vertices[0].Y),
		X1: float64(vertices[0].X), Y1: float64(vertices[0].Y),
	}
	for _, vertex := range vertices[1:] {
		x, y := float64(vertex.X), float64(vertex.Y)
		if x < box.X0 {
			box.X0 = x
		}
		if x > box.X1 {
			box.X1 = x
		}
		if y < box.Y0 {
			box.Y0 = y
		}
		if y > box.Y1 {
			box.Y1 = y
		}
	}
	return box
}

func truncate(raw []byte, limit int) string {
	if len(raw) <= limit {
		return string(raw)
	}
	return string(raw[:limit]) + "..."
}
