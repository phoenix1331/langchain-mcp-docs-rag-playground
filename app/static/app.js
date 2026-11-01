// The htmx SSE extension reconnects when a stream drops. Reconnecting would re-ask the
// question (and spend model quota), so a dropped stream is closed and reported instead.
document.addEventListener("htmx:sseError", (event) => {
  const stream = event.target;
  event.detail.source.close();
  stream.removeAttribute("sse-connect");
  const body = stream.querySelector(".answer-body");
  if (body && !body.querySelector(".rendered, .error")) {
    body.innerHTML = '<p class="error" role="alert">The connection dropped before the answer finished. Ask again to retry.</p>';
  }
});
