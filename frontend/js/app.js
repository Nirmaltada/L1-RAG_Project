import { api } from "./api.js";
import {
  appendMessage,
  renderChats,
  renderDocuments,
  renderHistory,
  renderSources,
  setMessageContent,
} from "./chat.js";

const elements = {
  chatList: document.querySelector("#chat-list"),
  documentList: document.querySelector("#document-list"),
  messages: document.querySelector("#messages"),
  chatTitle: document.querySelector("#chat-title"),
  connection: document.querySelector("#connection-status"),
  newChat: document.querySelector("#new-chat"),
  uploadButton: document.querySelector("#upload-button"),
  fileInput: document.querySelector("#file-input"),
  uploadStatus: document.querySelector("#upload-status"),
  composer: document.querySelector("#composer"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  error: document.querySelector("#error-banner"),
};

const state = { chats: [], activeChatId: null, busy: false };

function showError(message = "") {
  elements.error.textContent = message;
  elements.error.hidden = !message;
}

function setBusy(busy) {
  state.busy = busy;
  elements.send.disabled = busy;
  elements.input.disabled = busy;
  elements.newChat.disabled = busy;
}

function chatHandlers() {
  return {
    open: openChat,
    async rename(chat) {
      const title = window.prompt("Rename chat", chat.title)?.trim();
      if (!title) return;
      try {
        await api.renameChat(chat.id, title);
        await refreshChats();
        if (chat.id === state.activeChatId) elements.chatTitle.textContent = title;
      } catch (error) { showError(error.message); }
    },
    async remove(chat) {
      if (!window.confirm(`Delete “${chat.title}” and all of its documents?`)) return;
      try {
        await api.deleteChat(chat.id);
        if (chat.id === state.activeChatId) state.activeChatId = null;
        await refreshChats();
        if (state.chats.length) await openChat(state.chats[0].id);
        else await createChat();
      } catch (error) { showError(error.message); }
    },
  };
}

async function refreshChats() {
  state.chats = await api.listChats();
  renderChats(elements.chatList, state.chats, state.activeChatId, chatHandlers());
}

async function openChat(chatId) {
  if (state.busy) return;
  showError();
  state.activeChatId = chatId;
  const [chat, documents] = await Promise.all([api.getChat(chatId), api.listDocuments(chatId)]);
  elements.chatTitle.textContent = chat.title;
  renderHistory(elements.messages, chat.messages);
  renderDocuments(elements.documentList, documents, deleteDocument);
  renderChats(elements.chatList, state.chats, chatId, chatHandlers());
  elements.input.focus();
}

async function createChat() {
  if (state.busy) return;
  try {
    const chat = await api.createChat();
    await refreshChats();
    await openChat(chat.id);
  } catch (error) { showError(error.message); }
}

async function deleteDocument(document) {
  if (!window.confirm(`Delete “${document.filename}”?`)) return;
  try {
    await api.deleteDocument(state.activeChatId, document.id);
    renderDocuments(elements.documentList, await api.listDocuments(state.activeChatId), deleteDocument);
  } catch (error) { showError(error.message); }
}

async function uploadSelected() {
  const file = elements.fileInput.files[0];
  elements.fileInput.value = "";
  if (!file || !state.activeChatId) return;
  elements.uploadStatus.textContent = `Processing ${file.name}…`;
  elements.uploadButton.disabled = true;
  showError();
  try {
    await api.uploadDocument(state.activeChatId, file);
    const documents = await api.listDocuments(state.activeChatId);
    renderDocuments(elements.documentList, documents, deleteDocument);
    elements.uploadStatus.textContent = `${file.name} is ready.`;
  } catch (error) {
    elements.uploadStatus.textContent = "";
    showError(error.message);
  } finally {
    elements.uploadButton.disabled = false;
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const content = elements.input.value.trim();
  if (!content || !state.activeChatId || state.busy) return;
  showError();
  setBusy(true);
  elements.input.value = "";
  elements.input.style.height = "auto";
  appendMessage(elements.messages, "user", content);
  const assistant = appendMessage(elements.messages, "assistant");
  assistant.bubble.classList.add("typing");
  let answer = "";
  try {
    await api.streamMessage(state.activeChatId, content, (streamEvent) => {
      if (streamEvent.type === "status" && !answer) {
        assistant.bubble.textContent = streamEvent.message;
      } else if (streamEvent.type === "delta") {
        answer += streamEvent.content;
        setMessageContent(assistant.bubble, "assistant", answer);
        elements.messages.scrollTop = elements.messages.scrollHeight;
      } else if (streamEvent.type === "done") {
        renderSources(assistant.wrapper, streamEvent.sources || []);
      } else if (streamEvent.type === "error") {
        throw new Error(streamEvent.message);
      }
    });
    if (!answer) assistant.wrapper.remove();
  } catch (error) {
    if (!answer) assistant.bubble.textContent = "The assistant could not complete this response.";
    showError(error.message);
  } finally {
    try {
      await refreshChats();
      const active = state.chats.find((chat) => chat.id === state.activeChatId);
      if (active) elements.chatTitle.textContent = active.title;
    } catch (_) { /* preserve the original stream result or error */ }
    assistant.bubble.classList.remove("typing");
    setBusy(false);
    elements.input.focus();
  }
}

elements.newChat.addEventListener("click", createChat);
elements.uploadButton.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", uploadSelected);
elements.composer.addEventListener("submit", sendMessage);
elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});
elements.input.addEventListener("input", () => {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 180)}px`;
});

async function initialize() {
  try {
    const health = await api.health();
    elements.connection.textContent = health.groq_configured && health.openrouter_configured
      ? "Backend connected"
      : "Backend connected · add API keys in backend/.env";
    await refreshChats();
    if (!state.chats.length) await createChat();
    else await openChat(state.chats[0].id);
  } catch (error) {
    elements.connection.textContent = "Backend unavailable";
    showError(`Cannot connect to the backend: ${error.message}`);
  }
}

initialize();
