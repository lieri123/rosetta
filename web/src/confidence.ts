// The confidence underlines.
//
// This is the point of the project's interface. Recognition gives a confidence
// per token; the rescorer adds a judgement about whether the token belongs
// where it is. Two different kinds of doubt, so two different marks:
//
//   red    the recogniser could barely read the ink. Probably garbage.
//   amber  the recogniser was confident, but a language model finds the token
//          improbable here, or the rescorer nearly picked something else.
//   none   confident and unsurprising. No mark at all.
//
// The amber tier is what a confidence threshold on its own cannot give you.
// A plausible-looking wrong word comes back with high confidence, because the
// recogniser found a clean reading -- just not the one on the page. Only the
// context model notices.
//
// Everything else here follows from one requirement: the marks must stay on
// their words while the text is being edited. So the spans live in a
// StateField and are mapped through every change, rather than being recomputed
// from offsets that stopped being true at the first keystroke.

import { EditorState, StateEffect, StateField, type Extension, type Range } from "@codemirror/state";
import { Decoration, EditorView, hoverTooltip, type DecorationSet, type Tooltip } from "@codemirror/view";

import type { Tier, Token } from "./api";

export interface Span {
  from: number;
  to: number;
  tier: Exclude<Tier, "none">;
  reason: string;
  confidence: number;
  original: string;
  suggestion?: string;
}

/** Replace every span, e.g. after loading a page. */
export const setSpans = StateEffect.define<Span[]>();
/** Drop one span, e.g. once its suggestion has been accepted or dismissed. */
export const clearSpan = StateEffect.define<{ from: number; to: number }>();

const redMark = Decoration.mark({ class: "cm-doubt cm-doubt-red" });
const amberMark = Decoration.mark({ class: "cm-doubt cm-doubt-amber" });

/** The span data, kept alongside the decorations so the tooltip can find it. */
const spanData = new WeakMap<DecorationSet, Span[]>();

function build(spans: Span[], state: EditorState): DecorationSet {
  const ranges: Range<Decoration>[] = [];

  for (const span of spans) {
    // Offsets come from the server and the document may have been edited
    // since; clamp rather than throw, because one stale span should not blank
    // the whole editor.
    const from = Math.max(0, Math.min(span.from, state.doc.length));
    const to = Math.max(from, Math.min(span.to, state.doc.length));
    if (from === to) continue;
    ranges.push((span.tier === "red" ? redMark : amberMark).range(from, to));
  }

  const set = Decoration.set(ranges, true);
  spanData.set(set, spans);
  return set;
}

export const doubtField = StateField.define<DecorationSet>({
  create() {
    return Decoration.none;
  },

  update(decorations, transaction) {
    // Map first: an edit before a mark shifts it, and an edit inside one
    // shrinks it. This is what keeps an underline attached to its word instead
    // of to a character offset that has moved on.
    decorations = decorations.map(transaction.changes);

    const previous = spanData.get(decorations) ?? [];
    let spans = previous.map((span) => ({
      ...span,
      from: transaction.changes.mapPos(span.from, 1),
      to: transaction.changes.mapPos(span.to, -1),
    }));

    for (const effect of transaction.effects) {
      if (effect.is(setSpans)) {
        spans = effect.value;
        decorations = build(spans, transaction.state);
      } else if (effect.is(clearSpan)) {
        spans = spans.filter(
          (span) => !(span.from === effect.value.from && span.to === effect.value.to),
        );
        decorations = build(spans, transaction.state);
      }
    }

    spanData.set(decorations, spans);
    return decorations;
  },

  provide: (field) => EditorView.decorations.from(field),
});

export function spansIn(state: EditorState): Span[] {
  return spanData.get(state.field(doubtField)) ?? [];
}

export function spansFromTokens(tokens: Token[]): Span[] {
  const spans: Span[] = [];
  for (const token of tokens) {
    if (token.tier === "none" || token.struck) continue;
    if (token.end <= token.start) continue;
    spans.push({
      from: token.start,
      to: token.end,
      tier: token.tier,
      reason: token.reason ?? "",
      confidence: token.confidence,
      original: token.original,
      suggestion: token.suggestion,
    });
  }
  return spans;
}

/** Accept a suggestion: replace the marked text and drop the mark. */
export function acceptSuggestion(view: EditorView, span: Span): void {
  if (!span.suggestion) return;
  view.dispatch({
    changes: { from: span.from, to: span.to, insert: span.suggestion },
    effects: clearSpan.of({ from: span.from, to: span.to }),
  });
  view.focus();
}

function tooltipFor(view: EditorView, position: number): Tooltip | null {
  const span = spansIn(view.state).find((item) => position >= item.from && position <= item.to);
  if (!span) return null;

  return {
    pos: span.from,
    end: span.to,
    above: true,
    create: () => {
      const dom = document.createElement("div");
      dom.className = "cm-doubt-tooltip";

      const heading = document.createElement("div");
      heading.className = `tooltip-heading ${span.tier}`;
      heading.textContent =
        span.tier === "red" ? "Barely readable" : "Doubtful in context";
      dom.appendChild(heading);

      if (span.reason) {
        const reason = document.createElement("div");
        reason.className = "tooltip-reason";
        reason.textContent = span.reason;
        dom.appendChild(reason);
      }

      const detail = document.createElement("div");
      detail.className = "tooltip-detail";
      detail.textContent = `read as "${span.original}" at ${(span.confidence * 100).toFixed(0)}% confidence`;
      dom.appendChild(detail);

      if (span.suggestion) {
        const button = document.createElement("button");
        button.className = "tooltip-accept";
        button.textContent = `Use "${span.suggestion}"`;
        button.addEventListener("mousedown", (event) => {
          // mousedown, not click: the tooltip is dismissed on blur, and by the
          // time a click lands the element is gone.
          event.preventDefault();
          acceptSuggestion(view, span);
        });
        dom.appendChild(button);
      }

      return { dom };
    },
  };
}

export function confidenceUnderlines(): Extension {
  return [doubtField, hoverTooltip(tooltipFor, { hoverTime: 80 })];
}

/** Counts for the status line: how much of the page needs a look. */
export function tierCounts(state: EditorState): { red: number; amber: number } {
  let red = 0;
  let amber = 0;
  for (const span of spansIn(state)) {
    if (span.tier === "red") red += 1;
    else amber += 1;
  }
  return { red, amber };
}
