import { renderMarkdown } from "./markdown.js";

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function renderChats(container, chats, activeId, handlers) {
  container.replaceChildren();
  for (const chat of chats) {
    const row = element("div", `chat-row${chat.id === activeId ? " active" : ""}`);
    const open = element("button", "chat-name", chat.title);
    open.title = chat.title;
    open.addEventListener("click", () => handlers.open(chat.id));
    const rename = element("button", "icon-button", "✎");
    rename.title = "Rename chat";
    rename.addEventListener("click", () => handlers.rename(chat));
    const remove = element("button", "icon-button", "×");
    remove.title = "Delete chat";
    remove.addEventListener("click", () => handlers.remove(chat));
    row.append(open, rename, remove);
    container.append(row);
  }
}

export function renderDocuments(container, documents, onDelete) {
  container.replaceChildren();
  if (!documents.length) {
    container.append(element("div", "document-name", "No documents in this chat"));
    return;
  }
  for (const document of documents) {
    const row = element("div", "document-row");
    const details = element("div", "document-name", document.filename);
    details.title = `${document.document_type} · ${document.chunk_count} chunks`;
    details.append(element("span", "document-meta", `${document.category} · ${document.chunk_count} chunks`));
    const remove = element("button", "icon-button", "×");
    remove.title = "Delete document";
    remove.addEventListener("click", () => onDelete(document));
    row.append(details, remove);
    container.append(row);
  }
}

export function appendMessage(container, role, content = "", metadata = {}) {
  const empty = container.querySelector(".empty-state");
  if (empty) empty.remove();
  const wrapper = element("article", `message ${role}`);
  wrapper.append(element("div", "message-label", role === "user" ? "You" : "Assistant"));
  const bubble = element("div", "bubble");
  setMessageContent(bubble, role, content);
  wrapper.append(bubble);
  if (metadata.sources?.length) renderSources(wrapper, metadata.sources);
  container.append(wrapper);
  container.scrollTop = container.scrollHeight;
  return { wrapper, bubble };
}

export function setMessageContent(bubble, role, content) {
  if (role === "assistant") {
    bubble.innerHTML = renderMarkdown(content);
  } else {
    bubble.textContent = content;
  }
}

export function renderHistory(container, messages) {
  container.replaceChildren();
  if (!messages.length) {
    const empty = element("div", "empty-state");
    empty.append(
      element("h2", "", "Start a conversation"),
      element("p", "", "Ask a general question, or upload a document to ground answers in your own material."),
    );
    container.append(empty);
    return;
  }
  for (const message of messages) appendMessage(container, message.role, message.content, message.metadata);
}

export function renderSources(wrapper, sources) {
  const old = wrapper.querySelector(".sources");
  if (old) old.remove();
  if (!sources.length) return;
  const section = element("div", "sources");
  section.append(element("div", "sources-title", "Sources"));
  for (const source of sources) {
    const page = source.page ? ` — page ${source.page}` : "";
    section.append(element("div", "source", `[${source.index}] ${source.filename}${page}`));
  }
  wrapper.append(section);
}
