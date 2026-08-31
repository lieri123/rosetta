package layout

import (
	"strings"
	"testing"

	"github.com/lieri123/rosetta/service/internal/recognize"
)

func word(text string, x0, y0, x1, y1 float64) recognize.Word {
	return recognize.Word{
		Text:       text,
		Confidence: 0.9,
		Box:        recognize.Box{X0: x0, Y0: y0, X1: x1, Y1: y1},
	}
}

func lineTexts(lines []Line) []string {
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		parts := make([]string, 0, len(line.Words))
		for _, w := range line.Words {
			parts = append(parts, w.Text)
		}
		out = append(out, strings.Join(parts, " "))
	}
	return out
}

func TestGroupOrdersWordsWithinALine(t *testing.T) {
	// Deliberately out of order: the provider's ordering is not guaranteed.
	lines := Group([]recognize.Word{
		word("channel", 120, 10, 200, 34),
		word("the", 10, 10, 50, 34),
		word("noisy", 60, 10, 115, 34),
	})

	if len(lines) != 1 {
		t.Fatalf("want 1 line, got %d: %v", len(lines), lineTexts(lines))
	}
	if got := lineTexts(lines)[0]; got != "the noisy channel" {
		t.Errorf("want %q, got %q", "the noisy channel", got)
	}
}

func TestGroupSurvivesAscendersAndDescenders(t *testing.T) {
	// A word with a descender is taller and starts higher than its neighbours.
	// Grouping on box edges would split this line in two; grouping on centres
	// does not.
	lines := Group([]recognize.Word{
		word("paging", 10, 4, 80, 46),
		word("the", 90, 12, 130, 34),
		word("queue", 140, 12, 200, 40),
	})

	if len(lines) != 1 {
		t.Fatalf("want 1 line, got %d: %v", len(lines), lineTexts(lines))
	}
}

func TestGroupSeparatesLines(t *testing.T) {
	lines := Group([]recognize.Word{
		word("first", 10, 10, 60, 34),
		word("second", 10, 60, 70, 84),
		word("third", 10, 110, 60, 134),
	})

	if len(lines) != 3 {
		t.Fatalf("want 3 lines, got %d: %v", len(lines), lineTexts(lines))
	}
}

func TestParagraphsBreakOnALargerGap(t *testing.T) {
	lines := Group([]recognize.Word{
		word("one", 10, 10, 60, 34),
		word("two", 10, 50, 60, 74),
		// A gap three times the usual one: a new paragraph.
		word("three", 10, 190, 70, 214),
		word("four", 10, 230, 60, 254),
	})

	if len(lines) != 4 {
		t.Fatalf("want 4 lines, got %d", len(lines))
	}
	if lines[0].Paragraph != lines[1].Paragraph {
		t.Errorf("lines 0 and 1 should share a paragraph")
	}
	if lines[2].Paragraph == lines[1].Paragraph {
		t.Errorf("line 2 should start a new paragraph")
	}
	if lines[2].Paragraph != lines[3].Paragraph {
		t.Errorf("lines 2 and 3 should share a paragraph")
	}
}

func TestGroupHandlesNoWords(t *testing.T) {
	if lines := Group(nil); lines != nil {
		t.Errorf("want nil for no words, got %v", lines)
	}
}

func TestStrikethroughRemovesABisectedWord(t *testing.T) {
	lines := Group([]recognize.Word{
		word("keep", 10, 10, 60, 40),
		word("delete", 70, 10, 130, 40),
		word("this", 140, 10, 180, 40),
	})

	// A stroke through the vertical middle of "delete" and nothing else.
	struck := ApplyStrikethrough(lines, []Rule{{X0: 68, Y0: 25, X1: 132, Y1: 25, Thickness: 2}})
	if struck != 1 {
		t.Fatalf("want 1 struck word, got %d", struck)
	}

	text, tokens := Assemble(lines, nil)
	if strings.Contains(text, "delete") {
		t.Errorf("struck word still in the text: %q", text)
	}
	if !strings.Contains(text, "keep") || !strings.Contains(text, "this") {
		t.Errorf("neighbouring words were lost: %q", text)
	}

	// It is still recorded, so the UI can show what was removed.
	var struckTokens int
	for _, token := range tokens {
		if token.Struck {
			struckTokens++
		}
	}
	if struckTokens != 1 {
		t.Errorf("want 1 struck token recorded, got %d", struckTokens)
	}
}

func TestUnderlineIsNotAStrikethrough(t *testing.T) {
	// The distinction the whole feature turns on. A stroke below the word is
	// emphasis; deleting it would throw away text the writer meant to keep.
	lines := Group([]recognize.Word{word("important", 10, 10, 100, 40)})
	struck := ApplyStrikethrough(lines, []Rule{{X0: 8, Y0: 42, X1: 102, Y1: 42, Thickness: 2}})
	if struck != 0 {
		t.Errorf("an underline was treated as a strikethrough")
	}
}

func TestPartialOverlapIsNotAStrikethrough(t *testing.T) {
	// A rule that clips the last letters of a word did not cross it out.
	lines := Group([]recognize.Word{word("important", 10, 10, 100, 40)})
	struck := ApplyStrikethrough(lines, []Rule{{X0: 85, Y0: 25, X1: 140, Y1: 25, Thickness: 2}})
	if struck != 0 {
		t.Errorf("a rule overlapping 15%% of the word was treated as a strikethrough")
	}
}

func TestSlopedRuleIsEvaluatedAtTheWordCentre(t *testing.T) {
	// The stroke starts above the word and ends below it, passing through the
	// middle. Sampling only its start point would miss it.
	lines := Group([]recognize.Word{word("crossed", 10, 10, 110, 40)})
	struck := ApplyStrikethrough(lines, []Rule{{X0: 5, Y0: 14, X1: 115, Y1: 36, Thickness: 2}})
	if struck != 1 {
		t.Errorf("want the sloped stroke to count, got %d struck", struck)
	}
}

func TestAssembleOffsetsIndexTheText(t *testing.T) {
	lines := Group([]recognize.Word{
		word("the", 10, 10, 50, 34),
		word("noisy", 60, 10, 115, 34),
		word("channel", 10, 60, 90, 84),
	})

	text, tokens := Assemble(lines, nil)
	if len(tokens) != 3 {
		t.Fatalf("want 3 tokens, got %d", len(tokens))
	}

	// The property the editor depends on: slicing the text by a token's
	// offsets yields exactly that token. If this drifts, every underline lands
	// on the wrong word.
	for _, token := range tokens {
		if token.Start < 0 || token.End > len(text) || token.Start > token.End {
			t.Fatalf("token %q has offsets %d..%d outside text of length %d",
				token.Text, token.Start, token.End, len(text))
		}
		if got := text[token.Start:token.End]; got != token.Text {
			t.Errorf("offsets %d..%d give %q, want %q", token.Start, token.End, got, token.Text)
		}
	}
}

func TestAssembleAppliesDecodedText(t *testing.T) {
	lines := Group([]recognize.Word{
		word("the", 10, 10, 50, 34),
		word("rnatrix", 60, 10, 130, 34),
	})

	text, tokens := Assemble(lines, []Decoded{
		{Text: "the", Original: "the", Confidence: 0.99, Tier: "none"},
		{Text: "matrix", Original: "rnatrix", Confidence: 0.6, Tier: "amber", Reason: "rescored"},
	})

	if text != "the matrix" {
		t.Errorf("want %q, got %q", "the matrix", text)
	}
	if tokens[1].Original != "rnatrix" || tokens[1].Tier != "amber" {
		t.Errorf("decoded fields not carried through: %+v", tokens[1])
	}
	// Offsets must describe the substituted text, not the original.
	if got := text[tokens[1].Start:tokens[1].End]; got != "matrix" {
		t.Errorf("offsets point at %q after substitution", got)
	}
}

func TestAssembleSeparatesParagraphsWithABlankLine(t *testing.T) {
	lines := Group([]recognize.Word{
		word("one", 10, 10, 60, 34),
		word("two", 10, 50, 60, 74),
		word("three", 10, 190, 70, 214),
	})

	text, _ := Assemble(lines, nil)
	if !strings.Contains(text, "\n\n") {
		t.Errorf("want a blank line between paragraphs, got %q", text)
	}
}

func TestWordsSkipsStruckText(t *testing.T) {
	lines := Group([]recognize.Word{
		word("keep", 10, 10, 60, 40),
		word("delete", 70, 10, 130, 40),
	})
	ApplyStrikethrough(lines, []Rule{{X0: 68, Y0: 25, X1: 132, Y1: 25, Thickness: 2}})

	words := Words(lines)
	if len(words) != 1 || words[0].Text != "keep" {
		t.Errorf("struck words should not reach the rescorer, got %v", words)
	}
}
