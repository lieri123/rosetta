// Package rescore talks to the Python rescoring service.
//
// It is a separate process for one reason: the work inside it -- Levenshtein
// alignment, confusion matrix fitting, n-gram modelling and the evaluation
// harness around them -- is dramatically faster to write and to iterate on in
// Python, and the ecosystem for it is there. Doing that numerical work in Go
// would cost days and buy nothing. An HTTP hop on localhost costs under a
// millisecond next to a recognition round trip measured in seconds.
//
// The service is treated as optional at runtime. If it is down, pages still go
// through: recognition output is passed along and tiering falls back to
// recognition confidence alone. That is a worse product -- no substitutions,
// and no amber tier, because nothing is modelling context -- but it is a
// degradation rather than an outage.
package rescore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Alternative struct {
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
}

type Token struct {
	Text         string        `json:"text"`
	Confidence   float64       `json:"confidence"`
	Alternatives []Alternative `json:"alternatives,omitempty"`
}

type Decoded struct {
	Index       int     `json:"index"`
	Text        string  `json:"text"`
	Original    string  `json:"original"`
	Confidence  float64 `json:"confidence"`
	Tier        string  `json:"tier"`
	Reason      string  `json:"reason"`
	Suggestion  string  `json:"suggestion"`
	Surprisal   float64 `json:"surprisal"`
	Margin      float64 `json:"margin"`
	Substituted bool    `json:"substituted"`
}

type Pair struct {
	Predicted string `json:"predicted"`
	Corrected string `json:"corrected"`
}

type Client struct {
	BaseURL string
	HTTP    *http.Client
}

func New(baseURL string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		HTTP:    &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *Client) Healthy(ctx context.Context) bool {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/healthz", nil)
	if err != nil {
		return false
	}
	response, err := c.HTTP.Do(request)
	if err != nil {
		return false
	}
	defer response.Body.Close()
	io.Copy(io.Discard, response.Body)
	return response.StatusCode == http.StatusOK
}

func (c *Client) Rescore(ctx context.Context, tokens []Token) ([]Decoded, error) {
	var response struct {
		Tokens []Decoded `json:"tokens"`
		Text   string    `json:"text"`
	}
	if err := c.post(ctx, "/rescore", map[string]any{"tokens": tokens}, &response); err != nil {
		return nil, err
	}
	if len(response.Tokens) != len(tokens) {
		// The decoder is positional: token N of the reply describes token N of
		// the request. A length mismatch means that assumption is broken and
		// applying the reply would scramble the page.
		return nil, fmt.Errorf("rescorer returned %d tokens for %d sent",
			len(response.Tokens), len(tokens))
	}
	return response.Tokens, nil
}

func (c *Client) Learn(ctx context.Context, pairs []Pair, pageID int64) error {
	payload := map[string]any{"pairs": pairs}
	if pageID > 0 {
		payload["page_id"] = pageID
	}
	return c.post(ctx, "/learn", payload, nil)
}

func (c *Client) Ingest(ctx context.Context, text string) error {
	return c.post(ctx, "/ingest", map[string]any{"text": text}, nil)
}

func (c *Client) Stats(ctx context.Context) (map[string]any, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/stats", nil)
	if err != nil {
		return nil, err
	}
	response, err := c.HTTP.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	var stats map[string]any
	if err := json.NewDecoder(response.Body).Decode(&stats); err != nil {
		return nil, err
	}
	return stats, nil
}

func (c *Client) post(ctx context.Context, path string, payload any, out any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")

	response, err := c.HTTP.Do(request)
	if err != nil {
		return fmt.Errorf("rescorer %s: %w", path, err)
	}
	defer response.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(response.Body, 32<<20))
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("rescorer %s: HTTP %d: %s", path, response.StatusCode, string(raw))
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("rescorer %s: parsing reply: %w", path, err)
	}
	return nil
}

// FallbackDecode produces tier decisions without the rescorer.
//
// Recognition confidence is the only evidence available here, so this can
// mark the clearly-unreadable red and nothing else. The amber tier is
// deliberately absent rather than faked: amber means "a language model finds
// this improbable", and without the rescorer there is no language model to say
// so. Inventing an amber tier from confidence alone would put the same
// underline on a different claim.
func FallbackDecode(tokens []Token, lowConfidence float64) []Decoded {
	decoded := make([]Decoded, len(tokens))
	for i, token := range tokens {
		tier := "none"
		reason := ""
		if token.Confidence < lowConfidence {
			tier = "red"
			reason = "low recognition confidence"
		}
		decoded[i] = Decoded{
			Index:      i,
			Text:       token.Text,
			Original:   token.Text,
			Confidence: token.Confidence,
			Tier:       tier,
			Reason:     reason,
		}
	}
	return decoded
}
