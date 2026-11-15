(function () {
  function ready(fn) {
    if (document.readyState !== "loading") { fn(); } else { document.addEventListener("DOMContentLoaded", fn); }
  }

  function normalise(href) {
    var a = document.createElement("a");
    a.href = href;
    var p = a.pathname.replace(/index\.html$/, "");
    if (p.length > 1 && p.charAt(p.length - 1) !== "/") { p += "/"; }
    return p;
  }

  function setupToggle() {
    var btn = document.getElementById("rlb-theme-toggle");
    if (!btn || !window.rlbTheme) { return; }
    btn.addEventListener("click", function () {
      var next = window.rlbTheme.get() === "dark" ? "light" : "dark";
      window.rlbTheme.set(next);
      if (document.querySelector(".language-mermaid svg")) { window.location.reload(); }
    });
  }

  function setupPager() {
    var pager = document.querySelector(".rlb-pager");
    var nav = document.getElementById("site-nav");
    if (!pager || !nav) { return; }
    var seen = {};
    var links = [];
    nav.querySelectorAll("a.nav-list-link").forEach(function (a) {
      var key = normalise(a.getAttribute("href"));
      if (!seen[key] && a.hostname === window.location.hostname) {
        seen[key] = true;
        links.push({ href: a.getAttribute("href"), key: key, title: a.textContent.trim() });
      }
    });
    var here = normalise(window.location.href);
    var idx = -1;
    for (var i = 0; i < links.length; i++) { if (links[i].key === here) { idx = i; break; } }
    if (idx < 0) { return; }
    var prev = links[idx - 1];
    var next = links[idx + 1];
    var pa = pager.querySelector(".rlb-pager-prev");
    var na = pager.querySelector(".rlb-pager-next");
    if (prev) { pa.href = prev.href; pa.querySelector(".rlb-pager-title").textContent = prev.title; pa.hidden = false; }
    if (next) { na.href = next.href; na.querySelector(".rlb-pager-title").textContent = next.title; na.hidden = false; }
    if (prev || next) { pager.hidden = false; }
  }

  function setupCodeLabels() {
    document.querySelectorAll("div.highlighter-rouge").forEach(function (block) {
      if (block.closest(".rlb-codefile")) { return; }
      var m = block.className.match(/language-([\w+-]+)/);
      if (!m || m[1] === "plaintext" || m[1] === "mermaid") { return; }
      var tag = document.createElement("span");
      tag.className = "rlb-lang-tag";
      tag.textContent = m[1];
      block.appendChild(tag);
    });
  }

  ready(function () {
    setupToggle();
    setupPager();
    setupCodeLabels();
  });
})();
