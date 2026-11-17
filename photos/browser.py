"""Collect photos from JavaScript-heavy pages by loading them in headless Chromium.

Plain HTML scraping misses photos that a page loads with JavaScript (single-page
apps, infinite scroll, CSS background images). Here the page is opened in a real
browser, scrolled to the bottom to trigger lazy loading, and every image the
browser downloads is captured. Paginated pages are followed by clicking their
"Next" / "Load more" control until no new photos appear. Image URLs still visible
in the DOM afterwards (e.g. not yet loaded, or set as CSS backgrounds) are returned
so the caller can fetch them too.
"""
import ipaddress
import socket
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List
from urllib.parse import urlparse

PAGE_TIMEOUT_MS = 45_000
SCROLL_PAUSE_MS = 900
MAX_SCROLLS = 60
# Stop scrolling once the page height hasn't grown for this many scrolls.
STABLE_SCROLLS = 3

# Image URLs referenced in the DOM: <img>, srcset candidates and CSS backgrounds.
DOM_IMAGE_URLS_JS = """
() => {
  const urls = new Set();
  for (const img of document.images) {
    if (img.currentSrc) urls.add(img.currentSrc);
    if (img.src) urls.add(img.src);
  }
  for (const el of document.querySelectorAll('*')) {
    const bg = getComputedStyle(el).backgroundImage;
    if (!bg || bg === 'none') continue;
    for (const match of bg.matchAll(/url\\(["']?([^"')]+)["']?\\)/g)) urls.add(match[1]);
  }
  return [...urls].filter(u => u.startsWith('http'));
}
"""

# Finds a usable "next page" / "load more" control, marks it for clicking and says how
# to follow it: {"click": true}, {"href": url} for <link rel=next>, or null if none.
FIND_NEXT_JS = """
() => {
  for (const el of document.querySelectorAll('[data-photo-next]')) el.removeAttribute('data-photo-next');
  const usable = el => {
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') return false;
    const cls = typeof el.className === 'string' ? el.className : '';
    if (/(^|[\\s_-])(disabled|inactive)($|[\\s_-])/i.test(cls)) return false;
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const NEXT_TEXT = /^(next( page)?|older( posts| entries)?|load more|show more|see more|view more|more photos|›|»|>|→|>>)\\s*[›»>→]*$/i;
  const NEXT_LABEL = /\\bnext\\b|\\bload more\\b|\\bshow more\\b/i;
  const NEXT_CLASS = /(^|[\\s_-])(next|load-?more|show-?more)($|[\\s_-])/i;
  const els = document.querySelectorAll('a[rel~="next"], a, button, [role="button"], input[type="button"]');
  for (const el of els) {
    if (!usable(el)) continue;
    const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
    const label = el.getAttribute('aria-label') || el.getAttribute('title') || el.value || '';
    const cls = typeof el.className === 'string' ? el.className : '';
    if (el.matches('a[rel~="next"]') || NEXT_TEXT.test(text) || NEXT_LABEL.test(label) || NEXT_CLASS.test(cls)) {
      el.setAttribute('data-photo-next', '');
      return {click: true};
    }
  }
  const link = document.querySelector('link[rel~="next"][href]');
  return link ? {href: link.href} : null;
}
"""


@dataclass
class CapturedImage:
    url: str
    content_type: str
    body: bytes


@dataclass
class BrowserResult:
    images: List[CapturedImage] = field(default_factory=list)
    dom_urls: List[str] = field(default_factory=list)
    pages: int = 0
    hit_page_limit: bool = False


class BrowserUnavailable(Exception):
    pass


def capture_images(url: str, user_agent: str, max_bytes: int, max_pages: int = 1) -> BrowserResult:
    try:
        from playwright.sync_api import Error as PlaywrightError, sync_playwright
    except ImportError:
        raise BrowserUnavailable("Playwright is not installed (pip install playwright).")

    responses = []
    result = BrowserResult()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                context = browser.new_context(user_agent=user_agent, viewport={"width": 1440, "height": 900})
                page = context.new_page()
                page.route("**/*", _block_private_hosts)
                page.on("response", lambda response: responses.append(response))

                page.goto(url, wait_until="networkidle", timeout=PAGE_TIMEOUT_MS)
                dom_urls, found_before = {}, -1
                while True:
                    _scroll_to_bottom(page)
                    result.pages += 1
                    for dom_url in page.evaluate(DOM_IMAGE_URLS_JS):
                        dom_urls.setdefault(dom_url, None)
                    # Stop once a "next" click brought nothing new (last page, or not real pagination).
                    found = len(dom_urls) + _image_response_count(responses)
                    if found == found_before:
                        break
                    found_before = found
                    if result.pages >= max_pages:
                        result.hit_page_limit = max_pages > 1 and page.evaluate(FIND_NEXT_JS) is not None
                        break
                    if not _go_to_next_page(page):
                        break

                result.dom_urls = list(dom_urls)
                result.images = _read_images(responses, max_bytes)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc):
            raise BrowserUnavailable("Chromium is not installed (python -m playwright install chromium).")
        if not result.images:
            raise
    return result


def _scroll_to_bottom(page):
    last_position, stable = None, 0
    for _ in range(MAX_SCROLLS):
        page.evaluate("window.scrollBy(0, window.innerHeight)")
        page.wait_for_timeout(SCROLL_PAUSE_MS)
        # At the bottom with nothing new loaded, neither value changes.
        position = page.evaluate("[window.scrollY, document.documentElement.scrollHeight]")
        stable = stable + 1 if position == last_position else 0
        if stable >= STABLE_SCROLLS:
            break
        last_position = position
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        pass  # some pages keep polling forever; take what has loaded


def _go_to_next_page(page) -> bool:
    target = page.evaluate(FIND_NEXT_JS)
    if not target:
        return False
    try:
        if target.get("href"):
            page.goto(target["href"], wait_until="networkidle", timeout=PAGE_TIMEOUT_MS)
        else:
            page.click("[data-photo-next]", timeout=5_000)
            page.wait_for_timeout(SCROLL_PAUSE_MS)
            page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        # The click may still have loaded more photos; the caller checks for progress.
        pass
    return True


def _image_response_count(responses) -> int:
    return len({
        r.url for r in responses
        if r.headers.get("content-type", "").startswith("image/") and r.status == 200
    })


def _read_images(responses, max_bytes: int) -> List[CapturedImage]:
    images: Dict[str, CapturedImage] = {}
    for response in responses:
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if not content_type.startswith("image/") or "svg" in content_type or response.status != 200:
            continue
        if response.url in images:
            continue
        try:
            body = response.body()
        except Exception:
            continue  # body no longer available (e.g. redirected or evicted)
        if len(body) <= max_bytes:
            images[response.url] = CapturedImage(response.url, content_type, body)
    return list(images.values())


def _block_private_hosts(route):
    host = urlparse(route.request.url).hostname or ""
    if route.request.url.startswith(("http://", "https://")) and _is_private(host):
        route.abort()
    else:
        route.continue_()


@lru_cache(maxsize=512)
def _is_private(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False  # let the browser report the failure
    return any(not ipaddress.ip_address(info[4][0].split("%")[0]).is_global for info in infos)
