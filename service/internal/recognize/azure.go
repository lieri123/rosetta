package recognize

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

// AzureRead calls Image Analysis 4.0's read feature.
//
// Kept alongside Vision so the two can be compared on the same pages with the
// same downstream pipeline. They differ in ways that matter here: Azure
// returns confidence per word but no symbol hierarchy, and its line grouping
// is generally better on tilted handwriting.
type AzureRead struct {
	Endpoint string
	Key      string
	Client   *http.Client
}

func NewAzureRead(endpoint, key string) *AzureRead {
	return &AzureRead{
		Endpoint: strings.TrimRight(endpoint, "/"),
		Key:      key,
		Client:   &http.Client{Timeout: 60 * time.Second},
	}
}

func (a *AzureRead) Name() string { return "azure-read" }

type azureResponse struct {
	ReadResult struct {
		Blocks []struct {
			Lines []struct {
				Text  string `json:"text"`
				Words []struct {
					Text            string  `json:"text"`
					Confidence      float64 `json:"confidence"`
					BoundingPolygon []struct {
						X float64 `json:"x"`
						Y float64 `json:"y"`
					} `json:"boundingPolygon"`
				} `json:"words"`
			} `json:"lines"`
		} `json:"blocks"`
	} `json:"readResult"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (a *AzureRead) Recognize(ctx context.Context, image []byte) (*Result, error) {
	if a.Endpoint == "" || a.Key == "" {
		return nil, fmt.Errorf("azure read: endpoint and key are both required")
	}

	url := a.Endpoint + "/computervision/imageanalysis:analyze?features=read&api-version=2023-10-01"
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(image))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/octet-stream")
	request.Header.Set("Ocp-Apim-Subscription-Key", a.Key)

	response, err := a.Client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("azure read: %w", err)
	}
	defer response.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(response.Body, 64<<20))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("azure read: HTTP %d: %s", response.StatusCode, truncate(raw, 400))
	}

	return parseAzureResponse(raw)
}

func parseAzureResponse(raw []byte) (*Result, error) {
	var decoded azureResponse
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return nil, fmt.Errorf("azure read: parsing response: %w", err)
	}
	if decoded.Error != nil {
		return nil, fmt.Errorf("azure read: %s: %s", decoded.Error.Code, decoded.Error.Message)
	}

	result := &Result{Provider: "azure-read"}
	var text strings.Builder
	for _, block := range decoded.ReadResult.Blocks {
		for _, line := range block.Lines {
			text.WriteString(line.Text)
			text.WriteByte('\n')
			for _, word := range line.Words {
				box := Box{}
				for index, point := range word.BoundingPolygon {
					if index == 0 {
						box = Box{X0: point.X, Y0: point.Y, X1: point.X, Y1: point.Y}
						continue
					}
					if point.X < box.X0 {
						box.X0 = point.X
					}
					if point.X > box.X1 {
						box.X1 = point.X
					}
					if point.Y < box.Y0 {
						box.Y0 = point.Y
					}
					if point.Y > box.Y1 {
						box.Y1 = point.Y
					}
				}
				result.Words = append(result.Words, Word{
					Text:       word.Text,
					Confidence: word.Confidence,
					Box:        box,
				})
			}
		}
	}
	result.RawText = text.String()
	return result, nil
}
