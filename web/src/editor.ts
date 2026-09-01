// The editor view.
//
// A deliberately small CodeMirror setup: history, sensible keys, line
// wrapping, and the confidence underlines. No language mode, no syntax
// highlighting, no autocompletion -- this is prose, and every extension that
// tries to be clever about prose gets in the way of correcting it.

import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { EditorState, type Extension } from "@codemirror/state";
import { EditorView, keymap, placeholder } from "@codemirror/view";

import { confidenceUnderlines, setSpans, type Span } from "./confidence";

export interface EditorCallbacks {
  onChange: (text: string) => void;
  onSave: () => void;
}

const theme = EditorView.theme({
  "&": {
    fontSize: "16px",
    height: "100%",
    backgroundColor: "transparent",
  },
  ".cm-content": {
    fontFamily: "var(--font-prose)",
    lineHeight: "1.75",
    padding: "24px 28px",
    caretColor: "var(--ink)",
    maxWidth: "70ch",
  },
  ".cm-line": { padding: "0" },
  "&.cm-focused": { outline: "none" },
  ".cm-scroller": { overflow: "auto" },
  ".cm-cursor": { borderLeftColor: "var(--ink)" },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": {
    backgroundColor: "var(--selection)",
  },
});

export function createEditor(parent: HTMLElement, callbacks: EditorCallbacks): EditorView {
  const extensions: Extension[] = [
    history(),
    keymap.of([
      {
        // Explicit save, even though edits are also saved on a debounce. A
        // correction is training data, and people want to know it landed.
        key: "Mod-s",
        preventDefault: true,
        run: () => {
          callbacks.onSave();
          return true;
        },
      },
      ...defaultKeymap,
      ...historyKeymap,
    ]),
    EditorView.lineWrapping,
    placeholder("Add a page to get started."),
    confidenceUnderlines(),
    theme,
    EditorView.updateListener.of((update) => {
      if (update.docChanged) {
        callbacks.onChange(update.state.doc.toString());
      }
    }),
  ];

  return new EditorView({ state: EditorState.create({ extensions }), parent });
}

/**
 * Load a page into the editor.
 *
 * The document and its spans are replaced in one transaction, so there is no
 * moment where the old underlines are showing over the new text.
 */
export function loadPage(view: EditorView, text: string, spans: Span[]): void {
  view.dispatch({
    changes: { from: 0, to: view.state.doc.length, insert: text },
    effects: setSpans.of(spans),
    // A fresh page starts at the top with the cursor at the beginning, not
    // wherever the previous page happened to leave it.
    selection: { anchor: 0 },
    scrollIntoView: true,
  });
}

export function editorText(view: EditorView): string {
  return view.state.doc.toString();
}
