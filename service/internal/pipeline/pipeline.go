// Package pipeline is what happens to one page.
//
// Clean it, recognise it, work out its layout, decide what was crossed out,
// rescore it, store the text and the spans. Each step degrades rather than
// fails where it sensibly can: a missing preprocessor means recognising the
// raw photo, an unreachable rescorer means confidence-only tiering. A page
// that cannot be recognised at all is a real failure and is recorded as one.
package pipeline

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"github.com/lieri123/rosetta/service/internal/events"
	"github.com/lieri123/rosetta/service/internal/layout"
	"github.com/lieri123/rosetta/service/internal/preprocess"
	"github.com/lieri123/rosetta/service/internal/recognize"
	"github.com/lieri123/rosetta/service/internal/rescore"
	"github.com/lieri123/rosetta/service/internal/store"
)

type Processor struct {
	Store         *store.Store
	Preprocess    *preprocess.Runner
	Provider      recognize.Provider
	Rescorer      *rescore.Client
	Broker        *events.Broker
	DataDir       string
	LowConfidence float64
	Logger        *log.Logger
}

func (p *Processor) logf(format string, args ...any) {
	if p.Logger != nil {
		p.Logger.Printf(format, args...)
	}
}

// ProcessPage runs the whole pipeline for one page.
func (p *Processor) ProcessPage(ctx context.Context, job store.Job) error {
	page, err := p.Store.GetPage(ctx, job.PageID)
	if err != nil {
		return fmt.Errorf("loading page %d: %w", job.PageID, err)
	}

	if err := p.Store.SetPageStatus(ctx, page.ID, store.StatusRunning, ""); err != nil {
		return err
	}
	p.publish(events.Event{
		Type: "started", DocumentID: page.DocumentID, PageID: page.ID,
		PageIndex: page.Index, Message: "processing",
	})

	imagePath, rules, err := p.clean(ctx, page)
	if err != nil {
		return p.fail(ctx, page, err)
	}

	imageBytes, err := os.ReadFile(imagePath)
	if err != nil {
		return p.fail(ctx, page, fmt.Errorf("reading %s: %w", imagePath, err))
	}

	result, err := p.Provider.Recognize(ctx, imageBytes)
	if err != nil {
		return p.fail(ctx, page, fmt.Errorf("recognition: %w", err))
	}
	p.publish(events.Event{
		Type: "recognised", DocumentID: page.DocumentID, PageID: page.ID,
		PageIndex: page.Index,
		Message:   fmt.Sprintf("%d words from %s", len(result.Words), result.Provider),
	})

	lines := layout.Group(result.Words)
	struck := layout.ApplyStrikethrough(lines, rules)
	if struck > 0 {
		p.logf("page %d: %d word(s) crossed out", page.ID, struck)
	}

	words := layout.Words(lines)
	decoded := p.decode(ctx, words)

	text, tokens := layout.Assemble(lines, decoded)
	if err := p.Store.SavePageResult(ctx, page.ID, text, toStoreTokens(tokens)); err != nil {
		return p.fail(ctx, page, fmt.Errorf("saving page: %w", err))
	}

	p.publish(events.Event{
		Type: "page", DocumentID: page.DocumentID, PageID: page.ID,
		PageIndex: page.Index,
		Message:   fmt.Sprintf("%d tokens", len(tokens)),
	})
	return nil
}

// clean runs the C preprocessor, falling back to the original image.
func (p *Processor) clean(ctx context.Context, page *store.Page) (string, []layout.Rule, error) {
	if p.Preprocess == nil || !p.Preprocess.Available() {
		p.logf("page %d: preprocessor unavailable, recognising the original image", page.ID)
		return page.SourcePath, nil, nil
	}

	outPath := filepath.Join(
		filepath.Dir(page.SourcePath),
		strconv.Itoa(page.Index)+"-clean.png",
	)
	metadata, err := p.Preprocess.Run(ctx, page.SourcePath, outPath, preprocess.Options{
		DetectStrikethrough: true,
	})
	if err != nil {
		// A page the preprocessor chokes on is still worth recognising: the
		// photo may be fine and the page detection merely confused.
		p.logf("page %d: preprocessing failed (%v), falling back to the original", page.ID, err)
		return page.SourcePath, nil, nil
	}

	if err := p.Store.SetPagePreprocessed(ctx, page.ID, outPath,
		metadata.Result.Width, metadata.Result.Height,
		metadata.Deskew.AngleDeg, metadata.Perspective.Applied); err != nil {
		return "", nil, err
	}

	rules := make([]layout.Rule, 0, len(metadata.Rules))
	for _, rule := range metadata.Rules {
		rules = append(rules, layout.Rule{
			X0: rule.X0, Y0: rule.Y0, X1: rule.X1, Y1: rule.Y1, Thickness: rule.Thickness,
		})
	}

	p.publish(events.Event{
		Type: "preprocessed", DocumentID: page.DocumentID, PageID: page.ID,
		PageIndex: page.Index,
		Message: fmt.Sprintf("deskewed %.2f deg, %d rule(s)",
			metadata.Deskew.AngleDeg, len(metadata.Rules)),
	})
	return outPath, rules, nil
}

func (p *Processor) decode(ctx context.Context, words []recognize.Word) []layout.Decoded {
	tokens := make([]rescore.Token, len(words))
	for i, word := range words {
		alternatives := make([]rescore.Alternative, 0, len(word.Alternatives))
		for _, alternative := range word.Alternatives {
			alternatives = append(alternatives, rescore.Alternative{
				Text: alternative.Text, Confidence: alternative.Confidence,
			})
		}
		tokens[i] = rescore.Token{
			Text: word.Text, Confidence: word.Confidence, Alternatives: alternatives,
		}
	}

	var decoded []rescore.Decoded
	if p.Rescorer != nil {
		result, err := p.Rescorer.Rescore(ctx, tokens)
		if err != nil {
			p.logf("rescorer unavailable (%v); falling back to confidence-only tiering", err)
		} else {
			decoded = result
		}
	}
	if decoded == nil {
		decoded = rescore.FallbackDecode(tokens, p.lowConfidence())
	}

	out := make([]layout.Decoded, len(decoded))
	for i, item := range decoded {
		out[i] = layout.Decoded{
			Text: item.Text, Original: item.Original, Confidence: item.Confidence,
			Tier: item.Tier, Reason: item.Reason, Suggestion: item.Suggestion,
		}
	}
	return out
}

func (p *Processor) lowConfidence() float64 {
	if p.LowConfidence > 0 {
		return p.LowConfidence
	}
	return 0.55
}

func (p *Processor) fail(ctx context.Context, page *store.Page, cause error) error {
	if err := p.Store.SetPageStatus(ctx, page.ID, store.StatusFailed, cause.Error()); err != nil {
		p.logf("page %d: could not record failure: %v", page.ID, err)
	}
	p.publish(events.Event{
		Type: "failed", DocumentID: page.DocumentID, PageID: page.ID,
		PageIndex: page.Index, Message: cause.Error(),
	})
	return cause
}

func (p *Processor) publish(event events.Event) {
	if p.Broker != nil {
		p.Broker.Publish(event)
	}
}

func toStoreTokens(tokens []layout.Token) []store.Token {
	out := make([]store.Token, len(tokens))
	for i, token := range tokens {
		out[i] = store.Token{
			Index: token.Index, Text: token.Text, Original: token.Original,
			Confidence: token.Confidence, Tier: token.Tier, Reason: token.Reason,
			Suggestion: token.Suggestion, Start: token.Start, End: token.End,
			X0: token.Box.X0, Y0: token.Box.Y0, X1: token.Box.X1, Y1: token.Box.Y1,
			LineIndex: token.Line, ParaIndex: token.Paragraph, Struck: token.Struck,
		}
	}
	return out
}
