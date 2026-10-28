// API client for the local ModelToolbox server.
export async function apiPost(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON error */ }
  if (!res.ok) {
    const msg = body && body.error ? body.error.message : `HTTP ${res.status}`;
    const err = new Error(msg);
    err.kind = body && body.error ? body.error.kind : "http_error";
    err.status = res.status;
    throw err;
  }
  return body;
}

export async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function apiDelete(path) {
  const res = await fetch(path, { method: "DELETE" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// Download an export by POSTing and saving the attachment.
export async function apiDownload(path, payload, filename) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try { const j = await res.json(); msg = j.error ? j.error.message : msg; } catch (e) {}
    throw new Error(msg);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export const api = { apiPost, apiGet, apiDelete, apiDownload };
