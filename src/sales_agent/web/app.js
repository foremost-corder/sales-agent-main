const USER_ID = "local-demo-user";

const state = {
  conversations: [],
  activeConversationId: null,
  sending: false,
};

const elements = {
  sidebar: document.querySelector("#sidebar"),
  overlay: document.querySelector("#mobile-overlay"),
  menuButton: document.querySelector("#menu-button"),
  newChat: document.querySelector("#new-chat"),
  conversationList: document.querySelector("#conversation-list"),
  chatTitle: document.querySelector("#chat-title"),
  messages: document.querySelector("#messages"),
  emptyState: document.querySelector("#empty-state"),
  form: document.querySelector("#message-form"),
  input: document.querySelector("#message-input"),
  sendButton: document.querySelector("#send-button"),
  statusDot: document.querySelector("#status-dot"),
  serviceStatus: document.querySelector("#service-status"),
  modePill: document.querySelector("#mode-pill"),
};

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.detail || "请求失败，请稍后重试");
  }
  return body;
}

async function refreshSystemStatus() {
  try {
    const [database, model] = await Promise.all([
      request("/health/database"),
      request("/health/model"),
    ]);
    elements.statusDot.className = `status-dot${model.configured ? "" : " warning"}`;
    elements.serviceStatus.textContent = model.configured
      ? "数据库与模型已就绪"
      : "数据库已连接 · 模型待配置";
    elements.modePill.textContent = model.configured ? model.model : "需要 API Key";
    return database;
  } catch (error) {
    elements.statusDot.className = "status-dot error";
    elements.serviceStatus.textContent = "服务连接异常";
  }
}

function closeSidebar() {
  elements.sidebar.classList.remove("open");
  elements.overlay.classList.remove("open");
}

function renderConversationList() {
  elements.conversationList.replaceChildren();
  for (const conversation of state.conversations) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "conversation-item";
    button.textContent = conversation.title || "未命名对话";
    button.title = button.textContent;
    button.classList.toggle("active", conversation.id === state.activeConversationId);
    button.addEventListener("click", () => selectConversation(conversation));
    elements.conversationList.append(button);
  }
}

function appendMessage(role, content, extraClass = "") {
  elements.emptyState?.remove();
  const row = document.createElement("div");
  row.className = `message-row ${role} ${extraClass}`.trim();
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = content;
  row.append(bubble);
  elements.messages.append(row);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return row;
}

async function loadConversations(selectFirst = true) {
  state.conversations = await request(`/api/conversations?user_id=${encodeURIComponent(USER_ID)}`);
  renderConversationList();
  if (selectFirst && state.conversations.length > 0) {
    await selectConversation(state.conversations[0]);
  }
}

async function selectConversation(conversation) {
  state.activeConversationId = conversation.id;
  elements.chatTitle.textContent = conversation.title || "未命名对话";
  renderConversationList();
  closeSidebar();
  const messages = await request(`/api/conversations/${conversation.id}/messages?limit=100`);
  elements.messages.replaceChildren();
  if (messages.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.id = "empty-state";
    empty.innerHTML = '<div class="empty-icon">S</div><h2>开始这段对话</h2><p>消息会安全保存在本机 PostgreSQL 中。</p>';
    elements.messages.append(empty);
    elements.emptyState = empty;
  } else {
    for (const message of messages) appendMessage(message.role, message.content);
  }
  elements.input.focus();
}

async function createConversation() {
  const conversation = await request("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: USER_ID, title: "新对话" }),
  });
  state.conversations.unshift(conversation);
  await selectConversation(conversation);
}

async function ensureConversation() {
  if (state.activeConversationId) return state.activeConversationId;
  await createConversation();
  return state.activeConversationId;
}

async function sendMessage(content) {
  const conversationId = await ensureConversation();
  appendMessage("user", content);
  const pending = appendMessage("assistant", "正在思考…", "pending");
  try {
    const result = await request(`/api/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    pending.remove();
    appendMessage("assistant", result.assistant_message.content);
    await loadConversations(false);
    const active = state.conversations.find((item) => item.id === conversationId);
    if (active) elements.chatTitle.textContent = active.title || "未命名对话";
  } catch (error) {
    pending.remove();
    throw error;
  }
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const content = elements.input.value.trim();
  if (!content || state.sending) return;
  state.sending = true;
  elements.sendButton.disabled = true;
  elements.input.value = "";
  elements.input.style.height = "auto";
  try {
    await sendMessage(content);
  } catch (error) {
    appendMessage("assistant", error.message, "error");
  } finally {
    state.sending = false;
    elements.sendButton.disabled = false;
    elements.input.focus();
  }
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

elements.input.addEventListener("input", () => {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 180)}px`;
});

elements.newChat.addEventListener("click", createConversation);
elements.menuButton.addEventListener("click", () => {
  elements.sidebar.classList.add("open");
  elements.overlay.classList.add("open");
});
elements.overlay.addEventListener("click", closeSidebar);

refreshSystemStatus();
loadConversations().catch((error) => appendMessage("assistant", error.message, "error"));
