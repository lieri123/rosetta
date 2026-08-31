package rescore

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRescoreRoundTrip(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Tokens []Token `json:"tokens"`
		}
		json.NewDecoder(r.Body).Decode(&request)

		decoded := make([]Decoded, len(request.Tokens))
		for i, token := range request.Tokens {
			decoded[i] = Decoded{Index: i, Text: token.Text, Original: token.Text,
				Confidence: token.Confidence, Tier: "none"}
		}
		json.NewEncoder(w).Encode(map[string]any{"tokens": decoded})
	}))
	defer server.Close()

	result, err := New(server.URL).Rescore(context.Background(), []Token{
		{Text: "the", Confidence: 0.99},
		{Text: "matrix", Confidence: 0.61},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 2 || result[1].Text != "matrix" {
		t.Errorf("unexpected result: %+v", result)
	}
}

func TestRescoreRejectsALengthMismatch(t *testing.T) {
	// The mapping back onto the page is positional. A reply of the wrong
	// length means that assumption is broken, and applying it would scramble
	// the page rather than merely mis-tier a token.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"tokens": []Decoded{{Index: 0, Text: "only-one"}},
		})
	}))
	defer server.Close()

	_, err := New(server.URL).Rescore(context.Background(), []Token{
		{Text: "one"}, {Text: "two"},
	})
	if err == nil || !strings.Contains(err.Error(), "returned 1 tokens for 2") {
		t.Errorf("want a length mismatch error, got %v", err)
	}
}

func TestRescoreSurfacesServerErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"model not loaded"}`, http.StatusInternalServerError)
	}))
	defer server.Close()

	_, err := New(server.URL).Rescore(context.Background(), []Token{{Text: "x"}})
	if err == nil || !strings.Contains(err.Error(), "model not loaded") {
		t.Errorf("want the server's message, got %v", err)
	}
}

func TestLearnSendsPairs(t *testing.T) {
	var received struct {
		Pairs  []Pair `json:"pairs"`
		PageID int64  `json:"page_id"`
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&received)
		w.Write([]byte(`{}`))
	}))
	defer server.Close()

	err := New(server.URL).Learn(context.Background(),
		[]Pair{{Predicted: "the rnatrix", Corrected: "the matrix"}}, 7)
	if err != nil {
		t.Fatal(err)
	}
	if len(received.Pairs) != 1 || received.PageID != 7 {
		t.Errorf("wrong payload: %+v", received)
	}
}

func TestHealthyReportsFalseWhenDown(t *testing.T) {
	if New("http://127.0.0.1:1").Healthy(context.Background()) {
		t.Error("want false for an unreachable rescorer")
	}
}

func TestFallbackFlagsOnlyLowConfidence(t *testing.T) {
	decoded := FallbackDecode([]Token{
		{Text: "certain", Confidence: 0.98},
		{Text: "doubtful", Confidence: 0.20},
	}, 0.55)

	if decoded[0].Tier != "none" {
		t.Errorf("want a confident token left alone, got %q", decoded[0].Tier)
	}
	if decoded[1].Tier != "red" {
		t.Errorf("want a doubtful token red, got %q", decoded[1].Tier)
	}
}

func TestFallbackNeverInventsAmber(t *testing.T) {
	// Amber means a language model found the token improbable. With no
	// rescorer there is no language model, and putting the same underline on a
	// different claim would be a lie to the reader.
	decoded := FallbackDecode([]Token{
		{Text: "a", Confidence: 0.60},
		{Text: "b", Confidence: 0.70},
		{Text: "c", Confidence: 0.86},
	}, 0.55)

	for _, token := range decoded {
		if token.Tier == "amber" {
			t.Errorf("fallback produced an amber tier: %+v", token)
		}
	}
}

func TestFallbackPreservesTextAndOrder(t *testing.T) {
	tokens := []Token{{Text: "one", Confidence: 0.9}, {Text: "two", Confidence: 0.1}}
	decoded := FallbackDecode(tokens, 0.55)
	for i := range tokens {
		if decoded[i].Text != tokens[i].Text || decoded[i].Index != i {
			t.Errorf("token %d altered: %+v", i, decoded[i])
		}
	}
}
