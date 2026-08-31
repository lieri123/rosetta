// Package layout turns a bag of word boxes into text.
//
// The recogniser returns words with positions, in an order that is only
// loosely reading order, and its own line breaking is tuned for printed
// documents. Handwriting needs different tolerances: baselines drift, letters
// vary in height within a line, and the gap that separates two columns is not
// much wider than the gap between two words.
//
// This package also owns the strikethrough decision. The C preprocessor finds
// horizontal strokes; only here, with word boxes in hand, can we ask the
// question that matters -- does this stroke pass through the middle of a word,
// rather than under it, and if so the word was crossed out and should not be
// in the text.
package layout

import (
	"sort"
	"strings"

	"github.com/lieri123/rosetta/service/internal/recognize"
)

// Rule is a horizontal stroke found by the preprocessor, in the coordinate
// space of the image that was recognised.
type Rule struct {
	X0, Y0, X1, Y1 float64
	Thickness      float64
}

type Line struct {
	Words     []recognize.Word
	Paragraph int
	Top       float64
	Bottom    float64
}

// Decoded is what the rescorer says about one word, positionally matched to
// the words handed to it.
type Decoded struct {
	Text       string
	Original   string
	Confidence float64
	Tier       string
	Reason     string
	Suggestion string
}

// Token is an assembled word: its final text, its box, and where it sits in
// the page's text so the editor can decorate exactly that range.
type Token struct {
	Index      int
	Text       string
	Original   string
	Confidence float64
	Tier       string
	Reason     string
	Suggestion string
	Start      int
	End        int
	Box        recognize.Box
	Line       int
	Paragraph  int
	Struck     bool
}

// Group sorts words into lines and lines into paragraphs.
func Group(words []recognize.Word) []Line {
	if len(words) == 0 {
		return nil
	}

	sorted := make([]recognize.Word, len(words))
	copy(sorted, words)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sorted[i].Box.MidY() < sorted[j].Box.MidY()
	})

	medianHeight := medianWordHeight(sorted)

	var lines []Line
	current := Line{Top: sorted[0].Box.Y0, Bottom: sorted[0].Box.Y1}

	for _, word := range sorted {
		if len(current.Words) == 0 {
			current.Words = append(current.Words, word)
			current.Top, current.Bottom = word.Box.Y0, word.Box.Y1
			continue
		}

		// Same line when the word's vertical centre sits inside the line's
		// band, widened by a fraction of typical letter height. Using the
		// centre rather than the edges is what makes this survive ascenders
		// and descenders, which otherwise start a new line every time someone
		// writes a 'g'.
		tolerance := medianHeight * 0.6
		centre := word.Box.MidY()
		lineCentre := (current.Top + current.Bottom) / 2

		if absFloat(centre-lineCentre) <= tolerance {
			current.Words = append(current.Words, word)
			if word.Box.Y0 < current.Top {
				current.Top = word.Box.Y0
			}
			if word.Box.Y1 > current.Bottom {
				current.Bottom = word.Box.Y1
			}
			continue
		}

		lines = append(lines, finishLine(current))
		current = Line{Words: []recognize.Word{word}, Top: word.Box.Y0, Bottom: word.Box.Y1}
	}
	lines = append(lines, finishLine(current))

	assignParagraphs(lines, medianHeight)
	return lines
}

func finishLine(line Line) Line {
	sort.SliceStable(line.Words, func(i, j int) bool {
		return line.Words[i].Box.X0 < line.Words[j].Box.X0
	})
	return line
}

// assignParagraphs breaks where the vertical gap between lines grows.
//
// Compared against the typical gap on this page rather than an absolute
// number, so it works the same on a photo of a pocket notebook and on an A4
// scan.
func assignParagraphs(lines []Line, medianHeight float64) {
	if len(lines) == 0 {
		return
	}

	gaps := make([]float64, 0, len(lines))
	for i := 1; i < len(lines); i++ {
		gaps = append(gaps, lines[i].Top-lines[i-1].Bottom)
	}

	// The lower quartile, not the median. Within-paragraph gaps are the small
	// ones and paragraph gaps are the large tail -- which is precisely what we
	// are trying to detect -- so a middle statistic is contaminated by the
	// thing it is meant to measure against. On a page with two paragraphs and
	// three lines, the median gap *is* the paragraph gap and nothing ever
	// breaks.
	typical := lowerQuartile(gaps)
	if typical <= 0 {
		typical = medianHeight * 0.4
	}

	paragraph := 0
	lines[0].Paragraph = 0
	for i := 1; i < len(lines); i++ {
		if lines[i].Top-lines[i-1].Bottom > typical*1.8+medianHeight*0.3 {
			paragraph++
		}
		lines[i].Paragraph = paragraph
	}
}

// ApplyStrikethrough marks words that a horizontal stroke passes through.
//
// The test is bisection, not proximity: a stroke that runs under a word is an
// underline and the word stays; a stroke through the middle of it is a
// deletion. The band is deliberately narrow -- a quarter of the word's height
// either side of centre -- because getting this wrong deletes text the writer
// meant to keep, which is far worse than leaving a crossed-out word in.
func ApplyStrikethrough(lines []Line, rules []Rule) int {
	if len(rules) == 0 {
		return 0
	}

	struck := 0
	for lineIndex := range lines {
		for wordIndex := range lines[lineIndex].Words {
			box := lines[lineIndex].Words[wordIndex].Box
			for _, rule := range rules {
				if !bisects(rule, box) {
					continue
				}
				lines[lineIndex].Words[wordIndex].Text = ""
				struck++
				break
			}
		}
	}
	return struck
}

func bisects(rule Rule, box recognize.Box) bool {
	// Horizontal overlap has to be substantial: a rule that clips the last
	// letter of a word did not cross it out.
	overlap := minFloat(rule.X1, box.X1) - maxFloat(rule.X0, box.X0)
	if box.Width() <= 0 || overlap < box.Width()*0.6 {
		return false
	}

	// Where the rule sits at the word's horizontal centre, since it may slope.
	centreX := (box.X0 + box.X1) / 2
	ruleY := rule.Y0
	if rule.X1 != rule.X0 {
		t := (centreX - rule.X0) / (rule.X1 - rule.X0)
		ruleY = rule.Y0 + t*(rule.Y1-rule.Y0)
	}

	band := box.Height() * 0.25
	return absFloat(ruleY-box.MidY()) <= band
}

// Assemble builds the page text and the tokens that index into it.
//
// The offsets are computed here, while the string is being built, and stored
// alongside the tokens. That is the whole reason the underlines line up: the
// browser never re-tokenises and never has to guess where a word starts.
func Assemble(lines []Line, decoded []Decoded) (string, []Token) {
	var text strings.Builder
	tokens := make([]Token, 0, len(decoded))

	decodedIndex := 0
	tokenIndex := 0

	for lineNumber, line := range lines {
		if lineNumber > 0 {
			if line.Paragraph != lines[lineNumber-1].Paragraph {
				text.WriteString("\n\n")
			} else {
				text.WriteString("\n")
			}
		}

		first := true
		for _, word := range line.Words {
			if word.Text == "" {
				// Struck through: recorded as a token so the UI can show what
				// was removed, but contributing nothing to the text.
				token := Token{
					Index: tokenIndex, Text: "", Original: word.Text,
					Confidence: word.Confidence, Tier: "none",
					Start: text.Len(), End: text.Len(),
					Box: word.Box, Line: lineNumber, Paragraph: line.Paragraph,
					Struck: true,
				}
				if decodedIndex < len(decoded) {
					token.Original = decoded[decodedIndex].Original
				}
				tokens = append(tokens, token)
				tokenIndex++
				continue
			}

			if !first {
				text.WriteString(" ")
			}
			first = false

			final := word.Text
			token := Token{
				Index:      tokenIndex,
				Original:   word.Text,
				Confidence: word.Confidence,
				Tier:       "none",
				Box:        word.Box,
				Line:       lineNumber,
				Paragraph:  line.Paragraph,
			}
			if decodedIndex < len(decoded) {
				item := decoded[decodedIndex]
				final = item.Text
				token.Original = item.Original
				token.Confidence = item.Confidence
				token.Tier = item.Tier
				token.Reason = item.Reason
				token.Suggestion = item.Suggestion
			}
			decodedIndex++

			token.Text = final
			token.Start = text.Len()
			text.WriteString(final)
			token.End = text.Len()

			tokens = append(tokens, token)
			tokenIndex++
		}
	}

	return text.String(), tokens
}

// Words flattens lines back into the sequence the rescorer should see, which
// is reading order with struck-through words already removed.
func Words(lines []Line) []recognize.Word {
	var words []recognize.Word
	for _, line := range lines {
		for _, word := range line.Words {
			if word.Text == "" {
				continue
			}
			words = append(words, word)
		}
	}
	return words
}

func medianWordHeight(words []recognize.Word) float64 {
	heights := make([]float64, 0, len(words))
	for _, word := range words {
		if word.Box.Height() > 0 {
			heights = append(heights, word.Box.Height())
		}
	}
	value := median(heights)
	if value <= 0 {
		return 20
	}
	return value
}

// lowerQuartile is the 25th percentile by nearest rank.
func lowerQuartile(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	return sorted[int(float64(len(sorted))*0.25)]
}

func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	return sorted[len(sorted)/2]
}

func absFloat(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
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
