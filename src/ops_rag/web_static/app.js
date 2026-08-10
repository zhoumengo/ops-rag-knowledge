(function () {
  "use strict";

  const chatEl = document.getElementById("chat");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const statusEl = document.getElementById("status");
  const statusText = document.getElementById("status-text");
  const chips = Array.from(document.querySelectorAll(".chip"));
  const uploadBtn = document.getElementById("upload");
  const fileInput = document.getElementById("file-input");
  const uploadStatus = document.getElementById("upload-status");
  const uploadText = document.getElementById("upload-text");

  const history = [];
  let sending = false;
  let pollTimer = null;

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function inline(text) {
    return text
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
  }

  function renderMarkdown(src) {
    const lines = escapeHtml(src).split("\n");
    const out = [];
    let list = null;
    const closeList = () => {
      if (list) {
        out.push("</" + list + ">");
        list = null;
      }
    };

    for (const line of lines) {
      const heading = line.match(/^(#{1,3})\s+(.*)/);
      if (heading) {
        closeList();
        const level = heading[1].length;
        out.push("<h" + level + ">" + heading[2] + "</h" + level + ">");
        continue;
      }
      const ul = line.match(/^\s*[-*]\s+(.*)/);
      const ol = line.match(/^\s*\d+[.、]\s+(.*)/);
      if (ul || ol) {
        const tag = ul ? "ul" : "ol";
        if (list !== tag) {
          closeList();
          out.push("<" + tag + ">");
          list = tag;
        }
        out.push("<li>" + inline(ul ? ul[1] : ol[1]) + "</li>");
        continue;
      }
      closeList();
      if (line.trim()) out.push("<p>" + inline(line.trim()) + "</p>");
    }
    closeList();
    return out.join("");
  }

  function addBubble(role, content, isMarkdown) {
    const wrap = document.createElement("div");
    wrap.className = "bubble " + role;
    if (role === "bot") {
      const img = document.createElement("img");
      img.className = "avatar small";
      img.src = "/static/robot.svg";
      img.alt = "机器人头像";
      wrap.appendChild(img);
    }
    const msg = document.createElement("div");
    msg.className = "msg";
    if (isMarkdown) {
      msg.innerHTML = renderMarkdown(content);
    } else {
      msg.textContent = content;
    }
    wrap.appendChild(msg);
    chatEl.appendChild(wrap);
    chatEl.scrollTop = chatEl.scrollHeight;
    return wrap;
  }

  function addTyping() {
    const wrap = document.createElement("div");
    wrap.className = "bubble bot";
    const img = document.createElement("img");
    img.className = "avatar small";
    img.src = "/static/robot.svg";
    img.alt = "机器人头像";
    const msg = document.createElement("div");
    msg.className = "msg";
    msg.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    wrap.appendChild(img);
    wrap.appendChild(msg);
    chatEl.appendChild(wrap);
    chatEl.scrollTop = chatEl.scrollHeight;
    return wrap;
  }

  function setStatus(state, text) {
    statusEl.classList.toggle("online", state === "online");
    statusEl.classList.toggle("error", state === "error");
    statusText.textContent = text;
  }

  function showUploadStatus(text, kind) {
    uploadStatus.hidden = false;
    uploadStatus.className = "upload-status" + (kind ? " " + kind : "");
    uploadText.textContent = text;
  }

  async function pollTask(taskId) {
    try {
      const res = await fetch("/api/tasks/" + taskId);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "HTTP " + res.status);
      const kind = data.status === "error" ? "error" : data.status === "done" ? "ok" : "busy";
      showUploadStatus(data.message, kind);
      if (data.status === "done") {
        clearInterval(pollTimer);
        pollTimer = null;
        uploadBtn.disabled = false;
        addBubble("bot", "✅ **" + data.file_name + "** 已成功入库，可以开始提问了。", true);
      } else if (data.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        uploadBtn.disabled = false;
        addBubble("bot", "❌ " + data.message, false);
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      uploadBtn.disabled = false;
      showUploadStatus("查询任务状态失败：" + err.message, "error");
    }
  }

  async function uploadFile(file) {
    if (uploadBtn.disabled) return;
    uploadBtn.disabled = true;
    showUploadStatus("正在上传 " + file.name + "…", "");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "HTTP " + res.status);
      showUploadStatus("文件解析中…", "busy");
      pollTimer = setInterval(function () { pollTask(data.task_id); }, 2000);
    } catch (err) {
      uploadBtn.disabled = false;
      showUploadStatus("上传失败：" + err.message, "error");
    }
  }

  function autoResize() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  }

  async function send(text) {
    const question = (text || inputEl.value).trim();
    if (!question || sending) return;

    addBubble("user", question);
    inputEl.value = "";
    autoResize();
    history.push({ role: "user", content: question });

    const typing = addTyping();
    sending = true;
    sendBtn.disabled = true;
    setStatus("thinking", "思考中…");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, history: history.slice(-10) }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "HTTP " + res.status);
      }
      addBubble("bot", data.answer, true);
      history.push({ role: "assistant", content: data.answer });
      setStatus("online", "在线");
    } catch (err) {
      addBubble("bot", "出错了：" + err.message, false);
      setStatus("error", "异常");
    } finally {
      typing.remove();
      sending = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  sendBtn.addEventListener("click", function () {
    send();
  });

  inputEl.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  inputEl.addEventListener("input", autoResize);

  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      send(chip.textContent);
    });
  });

  uploadBtn.addEventListener("click", function () {
    fileInput.click();
  });

  fileInput.addEventListener("change", function () {
    const file = fileInput.files && fileInput.files[0];
    fileInput.value = "";
    if (file) uploadFile(file);
  });

  fetch("/api/health")
    .then(function (res) {
      return res.json();
    })
    .then(function (data) {
      if (data.status === "ok" && data.manifest) {
        setStatus("online", "在线 · " + data.documents + " 份文档");
      } else if (data.status === "ok") {
        setStatus("error", "未生成 manifest");
      } else {
        setStatus("error", "离线");
      }
    })
    .catch(function () {
      setStatus("error", "离线");
    });
})();
