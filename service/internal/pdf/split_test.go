package pdf

import "testing"

func TestIsPDFSniffsMagicBytes(t *testing.T) {
	if !IsPDF([]byte("%PDF-1.7\n...")) {
		t.Error("a real PDF header was not recognised")
	}
	// The filename is not evidence: people rename things.
	if IsPDF([]byte("\x89PNG\r\n\x1a\n")) {
		t.Error("a PNG was mistaken for a PDF")
	}
	if IsPDF([]byte("%PD")) {
		t.Error("a short buffer was mistaken for a PDF")
	}
}

func TestPageNumberSortsNumericallyNotLexically(t *testing.T) {
	// page-10 must not sort before page-2, which is exactly what a plain
	// string sort would do and would silently shuffle a long document.
	cases := map[string]int{
		"/tmp/doc/page-1.png":  1,
		"/tmp/doc/page-02.png": 2,
		"/tmp/doc/page-10.png": 10,
		"/tmp/doc/page.png":    0,
	}
	for path, want := range cases {
		if got := pageNumber(path); got != want {
			t.Errorf("pageNumber(%q) = %d, want %d", path, got, want)
		}
	}
}
