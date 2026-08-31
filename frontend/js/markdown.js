const allowedTags = new Set([
  "A",
  "BLOCKQUOTE",
  "BR",
  "CODE",
  "EM",
  "H1",
  "H2",
  "H3",
  "H4",
  "H5",
  "H6",
  "HR",
  "LI",
  "OL",
  "P",
  "PRE",
  "STRONG",
  "TABLE",
  "TBODY",
  "TD",
  "TH",
  "THEAD",
  "TR",
  "UL",
]);

const allowedAttributes = new Set(["href", "target", "rel"]);

function sanitizeHtml(html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  for (const node of template.content.querySelectorAll("*")) {
    if (!allowedTags.has(node.tagName)) {
      node.replaceWith(document.createTextNode(node.textContent || ""));
      continue;
    }
    for (const attribute of [...node.attributes]) {
      if (!allowedAttributes.has(attribute.name)) {
        node.removeAttribute(attribute.name);
      }
    }
    if (node.tagName === "A") {
      const href = node.getAttribute("href") || "";
      if (!/^https?:\/\//i.test(href)) node.removeAttribute("href");
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noreferrer");
    }
  }
  return template.innerHTML;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderInline(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '<a href="$2" target="_blank" rel="noreferrer">$1</a>',
    );
}

function flushParagraph(lines, html) {
  if (!lines.length) return;
  html.push(`<p>${renderInline(lines.join(" "))}</p>`);
  lines.length = 0;
}

export function renderMarkdown(value) {
  if (window.marked?.parse) {
    window.marked.setOptions({ breaks: true, gfm: true });
    return sanitizeHtml(window.marked.parse(value));
  }

  const lines = value.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  const paragraph = [];
  let listOpen = false;
  let fenceOpen = false;
  let codeLines = [];

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      flushParagraph(paragraph, html);
      if (listOpen) {
        html.push("</ul>");
        listOpen = false;
      }
      if (fenceOpen) {
        html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        fenceOpen = false;
      } else {
        fenceOpen = true;
      }
      continue;
    }
    if (fenceOpen) {
      codeLines.push(line);
      continue;
    }

    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph(paragraph, html);
      if (listOpen) {
        html.push("</ul>");
        listOpen = false;
      }
      continue;
    }

    const heading = /^(#{1,3})\s+(.+)$/.exec(trimmed);
    if (heading) {
      flushParagraph(paragraph, html);
      if (listOpen) {
        html.push("</ul>");
        listOpen = false;
      }
      const level = heading[1].length + 2;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = /^[-*]\s+(.+)$/.exec(trimmed);
    const numbered = /^\d+\.\s+(.+)$/.exec(trimmed);
    if (bullet || numbered) {
      flushParagraph(paragraph, html);
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${renderInline((bullet || numbered)[1])}</li>`);
      continue;
    }

    paragraph.push(trimmed);
  }

  flushParagraph(paragraph, html);
  if (listOpen) html.push("</ul>");
  if (fenceOpen) html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
  return html.join("");
}
