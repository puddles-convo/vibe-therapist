// A link to a heading inside a closed <details> opens it first.
function reveal() {
  const target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (!target) return;
  for (let el = target; el; el = el.parentElement) if (el.tagName === "DETAILS") el.open = true;
  if (target.tagName === "A" && target.nextElementSibling && target.nextElementSibling.tagName === "DETAILS") target.nextElementSibling.open = true;
  target.scrollIntoView();
}
addEventListener("hashchange", reveal);
addEventListener("DOMContentLoaded", reveal);
