// The API client, and the types that come back with it.
//
// Every field here has a counterpart in the Go service; keeping the shapes
// written out rather than inferred means a change on the server shows up as a
// type error here rather than as an undefined at runtime.

export type Tier = "none" | "amber" | "red";

export interface Token {
  index: number;
  text: string;
  /** What the recogniser read, before any rescoring. */
  original: string;
  confidence: number;
  tier: Tier;
  reason?: string;
  /** A better reading the rescorer found but did not apply on its own. */
  suggestion?: string;
  /** Character offsets into the page text. Computed by the service. */
  start: number;
  end: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  line: number;
  paragraph: number;
  struck?: boolean;
}

export interface Page {
  id: number;
  document_id: number;
  index: number;
  width: number;
  height: number;
  status: string;
  error?: string;
  text: string;
  skew_deg: number;
  rectified: boolean;
}

export interface Document {
  id: number;
  title: string;
  source_name: string;
  status: string;
  error?: string;
  page_count: number;
  created_at: number;
  updated_at: number;
}

export interface ProgressEvent {
  type: string;
  document_id: number;
  page_id?: number;
  page_index?: number;
  message?: string;
  completed: number;
  total: number;
  at: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body?.error) detail = body.error;
    } catch {
      // A non-JSON error body is still an error; the status text will do.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function listDocuments(): Promise<{ documents: Document[] }> {
  return request("/api/documents");
}

export function getDocument(id: number): Promise<{ document: Document; pages: Page[] }> {
  return request(`/api/documents/${id}`);
}

export function getPage(id: number): Promise<{ page: Page; tokens: Token[] }> {
  return request(`/api/pages/${id}`);
}

export function upload(file: File): Promise<{ id: number; pages: number }> {
  const body = new FormData();
  body.append("file", file);
  body.append("title", file.name);
  return request("/api/documents", { method: "POST", body });
}

export function saveCorrection(
  pageId: number,
  text: string,
): Promise<{ learned: number; pairs: number; warning?: string }> {
  return request(`/api/pages/${pageId}/corrections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

/**
 * The image behind a page.
 *
 * `version` is a cache buster, not a parameter the server reads: the same URL
 * serves the original photo while a page is queued and the cleaned-up version
 * once it has been through the preprocessor. Passing the page's status makes
 * the URL change exactly when the image does.
 */
export function pageImageURL(pageId: number, version = ""): string {
  const query = version ? `?v=${encodeURIComponent(version)}` : "";
  return `/api/pages/${pageId}/image${query}`;
}

/**
 * Follow a document's progress.
 *
 * Returns a function that closes the stream. The caller must call it: an
 * EventSource left open keeps reconnecting after the page it belonged to is
 * gone.
 */
export function followProgress(
  documentId: number,
  onEvent: (event: ProgressEvent) => void,
): () => void {
  const source = new EventSource(`/api/documents/${documentId}/events`);

  const handle = (event: MessageEvent) => {
    try {
      onEvent(JSON.parse(event.data) as ProgressEvent);
    } catch {
      // A malformed frame is not worth tearing the stream down for.
    }
  };

  for (const type of ["snapshot", "queued", "started", "preprocessed", "recognised", "page", "done", "failed"]) {
    source.addEventListener(type, handle as EventListener);
  }

  source.addEventListener("error", () => {
    // EventSource reconnects on its own. The one case worth acting on is the
    // server closing the stream for good, which shows up as CLOSED.
    if (source.readyState === EventSource.CLOSED) {
      onEvent({ type: "closed", document_id: documentId, completed: 0, total: 0, at: 0 });
    }
  });

  return () => source.close();
}
