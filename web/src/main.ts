// Application wiring.
//
// Upload a page, watch it being worked on, read the text, fix what is wrong,
// and have the fixes sent back as training data. That last step is the point:
// a correction is not just an edit to a document, it is the evidence the
// rescorer learns from, and the system gets better at reading this particular
// handwriting every time it happens.

import type { EditorView } from "@codemirror/view";

import {
  followProgress,
  getDocument,
  getPage,
  listDocuments,
  pageImageURL,
  saveCorrection,
  upload,
  type Document,
  type Page,
  type ProgressEvent,
} from "./api";
import { spansFromTokens, tierCounts } from "./confidence";
import { createEditor, editorText, loadPage } from "./editor";

const elements = {
  file: document.getElementById("file") as HTMLInputElement,
  copy: document.getElementById("copy") as HTMLButtonElement,
  documents: document.getElementById("documents") as HTMLUListElement,
  status: document.getElementById("status") as HTMLDivElement,
  editor: document.getElementById("editor") as HTMLDivElement,
  pageTabs: document.getElementById("page-tabs") as HTMLDivElement,
  counts: document.getElementById("counts") as HTMLSpanElement,
  saveState: document.getElementById("save-state") as HTMLSpanElement,
  toggleImage: document.getElementById("toggle-image") as HTMLButtonElement,
  imagePane: document.getElementById("image-pane") as HTMLDivElement,
  pageImage: document.getElementById("page-image") as HTMLImageElement,
};

interface AppState {
  documentId: number | null;
  pageId: number | null;
  pages: Page[];
  loadedText: string;
  dirty: boolean;
  saving: boolean;
  stopFollowing: (() => void) | null;
}

const state: AppState = {
  documentId: null,
  pageId: null,
  pages: [],
  loadedText: "",
  dirty: false,
  saving: false,
  stopFollowing: null,
};

let view: EditorView;
let saveTimer: number | undefined;

// ----------------------------------------------------------------------

function showStatus(message: string, kind: "info" | "error" = "info"): void {
  elements.status.textContent = message;
  elements.status.className = `status ${kind}`;
  elements.status.hidden = false;
}

function hideStatus(): void {
  elements.status.hidden = true;
}

function updateCounts(): void {
  if (!state.pageId) {
    elements.counts.textContent = "";
    return;
  }
  const { red, amber } = tierCounts(view.state);
  if (red === 0 && amber === 0) {
    elements.counts.textContent = "nothing flagged";
    return;
  }
  const parts: string[] = [];
  if (red) parts.push(`${red} unreadable`);
  if (amber) parts.push(`${amber} doubtful`);
  elements.counts.textContent = parts.join(", ");
}

function setSaveState(text: string): void {
  elements.saveState.textContent = text;
}

// ----------------------------------------------------------------------

async function refreshDocuments(): Promise<void> {
  try {
    const { documents } = await listDocuments();
    renderDocuments(documents);
    if (!state.documentId && documents.length > 0) {
      await openDocument(documents[0].id);
    }
  } catch (error) {
    showStatus(`Could not list documents: ${(error as Error).message}`, "error");
  }
}

function renderDocuments(documents: Document[]): void {
  elements.documents.replaceChildren();

  for (const document_ of documents) {
    const item = document.createElement("li");
    item.className = "document";
    if (document_.id === state.documentId) item.classList.add("active");

    const button = document.createElement("button");
    button.className = "document-button";
    button.addEventListener("click", () => void openDocument(document_.id));

    const title = document.createElement("span");
    title.className = "document-title";
    title.textContent = document_.title;
    button.appendChild(title);

    const meta = document.createElement("span");
    meta.className = `document-meta ${document_.status}`;
    meta.textContent =
      document_.status === "done"
        ? `${document_.page_count} page${document_.page_count === 1 ? "" : "s"}`
        : document_.status;
    button.appendChild(meta);

    item.appendChild(button);
    elements.documents.appendChild(item);
  }
}

async function openDocument(id: number): Promise<void> {
  state.stopFollowing?.();
  state.stopFollowing = null;

  state.documentId = id;
  state.pageId = null;
  try {
    const { document: doc, pages } = await getDocument(id);
    state.pages = pages;
    renderDocuments(await listDocuments().then((r) => r.documents));
    renderPageTabs();

    if (doc.status === "running" || doc.status === "pending") {
      follow(id);
    } else {
      hideStatus();
    }

    const firstReady = pages.find((page) => page.status === "done") ?? pages[0];
    if (firstReady) await openPage(firstReady.id);
  } catch (error) {
    showStatus(`Could not open that document: ${(error as Error).message}`, "error");
  }
}

function renderPageTabs(): void {
  elements.pageTabs.replaceChildren();
  if (state.pages.length <= 1) return;

  for (const page of state.pages) {
    const button = document.createElement("button");
    button.className = "page-tab";
    if (page.id === state.pageId) button.classList.add("active");
    if (page.status !== "done") button.classList.add("pending");
    button.textContent = String(page.index + 1);
    button.title = `Page ${page.index + 1} (${page.status})`;
    button.addEventListener("click", () => void openPage(page.id));
    elements.pageTabs.appendChild(button);
  }
}

async function openPage(id: number): Promise<void> {
  // Anything unsaved goes first: switching pages must not quietly discard a
  // correction that has not been flushed yet.
  await flushSave();

  try {
    const { page, tokens } = await getPage(id);
    state.pageId = id;
    state.loadedText = page.text;
    state.dirty = false;

    loadPage(view, page.text, spansFromTokens(tokens));
    elements.pageImage.src = pageImageURL(id, page.status);
    elements.copy.disabled = page.text.length === 0;

    renderPageTabs();
    updateCounts();
    setSaveState(page.status === "done" ? "saved" : page.status);
  } catch (error) {
    showStatus(`Could not load that page: ${(error as Error).message}`, "error");
  }
}

// ----------------------------------------------------------------------

function follow(documentId: number): void {
  showStatus("Working on it...");
  state.stopFollowing = followProgress(documentId, (event: ProgressEvent) => {
    handleProgress(documentId, event);
  });
}

function handleProgress(documentId: number, event: ProgressEvent): void {
  switch (event.type) {
    case "preprocessed":
    case "recognised":
      showStatus(`Page ${(event.page_index ?? 0) + 1}: ${event.message ?? ""}`);
      break;
    case "page":
      showStatus(`Page ${(event.page_index ?? 0) + 1} is ready`);
      void refreshCurrentDocument(documentId);
      break;
    case "failed":
      showStatus(`Page ${(event.page_index ?? 0) + 1} failed: ${event.message ?? ""}`, "error");
      break;
    case "done":
      hideStatus();
      state.stopFollowing?.();
      state.stopFollowing = null;
      void refreshCurrentDocument(documentId);
      void refreshDocuments();
      break;
    default:
      break;
  }
}

async function refreshCurrentDocument(documentId: number): Promise<void> {
  if (state.documentId !== documentId) return;
  try {
    const { pages } = await getDocument(documentId);
    state.pages = pages;
    renderPageTabs();

    // Show the first finished page as soon as there is one, so a long document
    // becomes readable while the rest is still being worked on.
    if (!state.pageId) {
      const ready = pages.find((page) => page.status === "done");
      if (ready) await openPage(ready.id);
      return;
    }

    // The page on screen was opened while it was still being worked on -- the
    // usual case straight after an upload, where the only page exists but has
    // no text yet. Once it is finished, show the result. Skipped while there
    // are unsaved edits: a correction in progress outranks a refresh.
    const current = pages.find((page) => page.id === state.pageId);
    if (!state.dirty && current?.status === "done" && current.text !== state.loadedText) {
      await openPage(current.id);
    }
  } catch {
    // A refresh failing is not worth interrupting the user over; the next
    // event will try again.
  }
}

// ----------------------------------------------------------------------

function scheduleSave(): void {
  window.clearTimeout(saveTimer);
  // Long enough that a save is not fired mid-word, short enough that a
  // correction is not lost by closing the tab.
  saveTimer = window.setTimeout(() => void flushSave(), 1200);
}

async function flushSave(): Promise<void> {
  window.clearTimeout(saveTimer);
  if (!state.pageId || !state.dirty || state.saving) return;

  const text = editorText(view);
  if (text === state.loadedText) {
    state.dirty = false;
    setSaveState("saved");
    return;
  }

  state.saving = true;
  setSaveState("saving...");

  try {
    const result = await saveCorrection(state.pageId, text);
    state.loadedText = text;
    state.dirty = false;

    if (result.warning) {
      setSaveState("saved locally");
      showStatus(result.warning, "error");
    } else if (result.learned > 0) {
      // Worth saying plainly: this is the loop that makes the system adapt.
      setSaveState(
        `saved, ${result.learned} correction${result.learned === 1 ? "" : "s"} learned`,
      );
    } else {
      setSaveState("saved");
    }
  } catch (error) {
    setSaveState("not saved");
    showStatus(`Could not save: ${(error as Error).message}`, "error");
  } finally {
    state.saving = false;
  }
}

// ----------------------------------------------------------------------

async function handleUpload(file: File): Promise<void> {
  showStatus(`Uploading ${file.name}...`);
  try {
    const { id } = await upload(file);
    // Switch to the new document before refreshing the list, so the list does
    // not briefly reopen whatever was showing before.
    state.pageId = null;
    await openDocument(id);
    await refreshDocuments();
  } catch (error) {
    showStatus(`Upload failed: ${(error as Error).message}`, "error");
  }
}

function wireEvents(): void {
  elements.file.addEventListener("change", () => {
    const file = elements.file.files?.[0];
    if (file) void handleUpload(file);
    elements.file.value = "";
  });

  elements.copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(editorText(view));
      elements.copy.textContent = "Copied";
      window.setTimeout(() => (elements.copy.textContent = "Copy text"), 1500);
    } catch {
      showStatus("The browser would not give access to the clipboard.", "error");
    }
  });

  elements.toggleImage.addEventListener("click", () => {
    const showing = !elements.imagePane.hidden;
    elements.imagePane.hidden = showing;
    elements.toggleImage.textContent = showing ? "Show page" : "Hide page";
    document.body.classList.toggle("with-image", !showing);
  });

  // Drag a photo anywhere onto the window.
  document.addEventListener("dragover", (event) => event.preventDefault());
  document.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file) void handleUpload(file);
  });

  // A correction in flight is training data; do not lose it to a closed tab.
  window.addEventListener("beforeunload", () => {
    if (state.dirty) void flushSave();
  });
}

function main(): void {
  view = createEditor(elements.editor, {
    onChange: () => {
      state.dirty = true;
      setSaveState("editing...");
      updateCounts();
      scheduleSave();
    },
    onSave: () => void flushSave(),
  });

  wireEvents();
  void refreshDocuments();
}

main();
