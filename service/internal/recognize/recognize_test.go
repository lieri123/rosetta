package recognize

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestMockIsDeterministic(t *testing.T) {
	provider := NewMock()
	first, err := provider.Recognize(context.Background(), []byte("page-bytes"))
	if err != nil {
		t.Fatal(err)
	}
	second, err := provider.Recognize(context.Background(), []byte("page-bytes"))
	if err != nil {
		t.Fatal(err)
	}
	if len(first.Words) == 0 {
		t.Fatal("mock produced no words")
	}
	for i := range first.Words {
		a, b := first.Words[i], second.Words[i]
		if a.Text != b.Text || a.Confidence != b.Confidence || a.Box != b.Box {
			t.Fatalf("word %d differs between runs: %+v vs %+v", i, a, b)
		}
	}
}

func TestMockVariesWithInput(t *testing.T) {
	provider := NewMock()
	a, _ := provider.Recognize(context.Background(), []byte("page-one"))
	b, _ := provider.Recognize(context.Background(), []byte("page-two"))

	same := true
	for i := range a.Words {
		if i < len(b.Words) && a.Words[i].Text != b.Words[i].Text {
			same = false
			break
		}
	}
	if same {
		t.Error("two different images produced identical readings")
	}
}

func TestMockProducesConfidentlyWrongWords(t *testing.T) {
	// The mock has to reproduce the phenomenon the amber tier exists for. If
	// every error came back with low confidence, running on the mock would
	// make the tiering look better than it is.
	provider := NewMock()
	result, _ := provider.Recognize(context.Background(), []byte("a-page-of-notes"))

	known := map[string]bool{}
	for _, line := range defaultMockLines {
		for _, word := range strings.Fields(line) {
			known[word] = true
		}
	}

	confidentlyWrong := 0
	for _, word := range result.Words {
		if !known[word.Text] && word.Confidence > 0.85 {
			confidentlyWrong++
		}
	}
	if confidentlyWrong == 0 {
		t.Error("mock produced no confidently-wrong readings")
	}
}

func TestMockBoxesAreOrdered(t *testing.T) {
	provider := NewMock()
	result, _ := provider.Recognize(context.Background(), []byte("page"))
	for i, word := range result.Words {
		if word.Box.Width() <= 0 || word.Box.Height() <= 0 {
			t.Fatalf("word %d has an empty box: %+v", i, word.Box)
		}
	}
}

func TestMockFixtureReplay(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/fixture.json"
	fixture := Result{Words: []Word{{Text: "replayed", Confidence: 0.42,
		Box: Box{X0: 1, Y0: 2, X1: 3, Y1: 4}}}}
	raw, _ := json.Marshal(fixture)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	provider, err := NewMockFromFile(path)
	if err != nil {
		t.Fatal(err)
	}
	result, err := provider.Recognize(context.Background(), []byte("ignored"))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Words) != 1 || result.Words[0].Text != "replayed" {
		t.Errorf("fixture not replayed: %+v", result.Words)
	}
}

func TestGoogleParsesWordsAndBoxes(t *testing.T) {
	body := `{"responses":[{"fullTextAnnotation":{"text":"the matrix\n","pages":[{"blocks":[
		{"paragraphs":[{"words":[
			{"confidence":0.97,"boundingBox":{"vertices":[{"x":10,"y":20},{"x":50,"y":22},{"x":50,"y":48},{"x":10,"y":46}]},
			 "symbols":[{"text":"t","confidence":0.99},{"text":"h","confidence":0.98},{"text":"e","confidence":0.94}]},
			{"boundingBox":{"vertices":[{"x":60,"y":20},{"x":140,"y":20},{"x":140,"y":48},{"x":60,"y":48}]},
			 "symbols":[{"text":"m","confidence":0.60},{"text":"a","confidence":0.80}]}
		]}]}]}]}}]}`

	result, err := parseGoogleResponse([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Words) != 2 {
		t.Fatalf("want 2 words, got %d", len(result.Words))
	}
	if result.Words[0].Text != "the" || result.Words[0].Confidence != 0.97 {
		t.Errorf("first word wrong: %+v", result.Words[0])
	}

	// The bounding box is a quadrilateral, not a rectangle: a tilted page
	// gives four corners at different heights, and the box has to contain all
	// of them.
	box := result.Words[0].Box
	if box.X0 != 10 || box.Y0 != 20 || box.X1 != 50 || box.Y1 != 48 {
		t.Errorf("box not the extent of the vertices: %+v", box)
	}

	// Vision sometimes omits word confidence but fills in the symbols.
	// Averaging is the honest reconstruction.
	if got := result.Words[1].Confidence; got < 0.69 || got > 0.71 {
		t.Errorf("want the mean symbol confidence (0.70), got %v", got)
	}
}

func TestGoogleSurfacesAPIErrors(t *testing.T) {
	body := `{"responses":[{"error":{"code":7,"message":"permission denied"}}]}`
	if _, err := parseGoogleResponse([]byte(body)); err == nil ||
		!strings.Contains(err.Error(), "permission denied") {
		t.Errorf("want the API's own message, got %v", err)
	}
}

func TestGoogleSendsDocumentTextDetection(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&captured)
		w.Write([]byte(`{"responses":[{"fullTextAnnotation":{"text":"","pages":[]}}]}`))
	}))
	defer server.Close()

	provider := NewGoogleVision("test-key")
	provider.Endpoint = server.URL
	if _, err := provider.Recognize(context.Background(), []byte("image")); err != nil {
		t.Fatal(err)
	}

	// The per-symbol confidences this project runs on only come back in
	// DOCUMENT_TEXT_DETECTION mode.
	encoded, _ := json.Marshal(captured)
	if !strings.Contains(string(encoded), "DOCUMENT_TEXT_DETECTION") {
		t.Errorf("wrong feature requested: %s", encoded)
	}
}

func TestGoogleWithoutAKeyFailsFast(t *testing.T) {
	if _, err := NewGoogleVision("").Recognize(context.Background(), []byte("x")); err == nil {
		t.Error("want an error when no API key is configured")
	}
}

func TestAzureParsesWordsAndBoxes(t *testing.T) {
	body := `{"readResult":{"blocks":[{"lines":[{"text":"the matrix","words":[
		{"text":"the","confidence":0.95,"boundingPolygon":[{"x":10,"y":20},{"x":50,"y":20},{"x":50,"y":45},{"x":10,"y":45}]},
		{"text":"matrix","confidence":0.42,"boundingPolygon":[{"x":60,"y":18},{"x":140,"y":22},{"x":140,"y":47},{"x":60,"y":43}]}
	]}]}]}}`

	result, err := parseAzureResponse([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Words) != 2 || result.Words[1].Text != "matrix" {
		t.Fatalf("unexpected words: %+v", result.Words)
	}
	if result.Words[1].Confidence != 0.42 {
		t.Errorf("confidence lost: %+v", result.Words[1])
	}
	if box := result.Words[1].Box; box.Y0 != 18 || box.Y1 != 47 {
		t.Errorf("box does not span the polygon: %+v", box)
	}
}

func TestAzureSurfacesAPIErrors(t *testing.T) {
	body := `{"error":{"code":"401","message":"access denied"}}`
	if _, err := parseAzureResponse([]byte(body)); err == nil ||
		!strings.Contains(err.Error(), "access denied") {
		t.Errorf("want the API's own message, got %v", err)
	}
}

func TestNewRejectsUnknownProviders(t *testing.T) {
	if _, err := New("hieroglyphs", "", "", ""); err == nil {
		t.Error("want an error for an unknown provider")
	}
	if provider, err := New("", "", "", ""); err != nil || provider.Name() != "mock" {
		t.Errorf("empty provider name should default to mock, got %v %v", provider, err)
	}
}

func TestSortReadingOrder(t *testing.T) {
	words := []Word{
		{Text: "second", Box: Box{X0: 100, Y0: 10, X1: 160, Y1: 34}},
		{Text: "first", Box: Box{X0: 10, Y0: 12, X1: 60, Y1: 36}},
		{Text: "below", Box: Box{X0: 10, Y0: 80, X1: 60, Y1: 104}},
	}
	SortReadingOrder(words)
	if words[0].Text != "first" || words[1].Text != "second" || words[2].Text != "below" {
		t.Errorf("wrong reading order: %v %v %v", words[0].Text, words[1].Text, words[2].Text)
	}
}
