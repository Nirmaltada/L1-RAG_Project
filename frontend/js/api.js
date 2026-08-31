const backendOrigin = window.location.port === "8000"
  ? window.location.origin
  : "http://localhost:8000";

export const API_BASE = `${backendOrigin}/api`;

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch (_) { /* response was not JSON */ }
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

export const api = {
  health: () => request("/health"),
  listChats: () => request("/chats"),
  createChat: (title = "New chat") => request("/chats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  }),
  getChat: (chatId) => request(`/chats/${chatId}`),
  renameChat: (chatId, title) => request(`/chats/${chatId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  }),
  deleteChat: (chatId) => request(`/chats/${chatId}`, { method: "DELETE" }),
  listDocuments: (chatId) => request(`/chats/${chatId}/documents`),
  uploadDocument: (chatId, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/chats/${chatId}/documents`, { method: "POST", body: form });
  },
  deleteDocument: (chatId, documentId) => request(
    `/chats/${chatId}/documents/${documentId}`,
    { method: "DELETE" },
  ),
  async streamMessage(chatId, content, onEvent) {
    const response = await fetch(`${API_BASE}/chats/${chatId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }
    if (!response.body) throw new Error("Streaming is not supported by this browser");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finished = false;
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line.trim()) onEvent(JSON.parse(line));
        }
        if (done) {
          finished = true;
          break;
        }
      }
      if (buffer.trim()) onEvent(JSON.parse(buffer));
    } finally {
      if (!finished) await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  },
};
