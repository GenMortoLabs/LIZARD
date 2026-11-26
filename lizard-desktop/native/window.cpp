// ai: The desktop's window helpers (presenter.h; out of presenter.cpp 2026-11-25, built beside either presenter): the
// ai: window manager's window and states on X11, an X11 window's background and its bit gravity, Windows' full screen.
// ai: Each says it did nothing (false, 0, the window itself) where its system is not the one running.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include "presenter.h"

#include <map>
#include <mutex>

#if LIZ_X11 && !defined(_WIN32)
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#endif

namespace lizard {

namespace {

#if LIZ_X11 && !defined(_WIN32)
// ai: the window the manager manages above w: the ancestor with WM_STATE (a reparenting manager puts its own frame
// ai: between it and the root), else the root's child (no manager)
Window topLevel(Display* d, Window w) {
  const Atom wmState = XInternAtom(d, "WM_STATE", True);
  Window cur = w;
  for (;;) {
    if (wmState != None) {
      Atom type = None;
      int format = 0;
      unsigned long n = 0, after = 0;
      unsigned char* data = nullptr;
      if (XGetWindowProperty(d, cur, wmState, 0, 0, False, AnyPropertyType, &type, &format, &n, &after, &data) == Success) {
        if (data) XFree(data);
        if (type != None) return cur;
      }
    }
    Window root = 0, parent = 0, *kids = nullptr;
    unsigned int nk = 0;
    if (!XQueryTree(d, cur, &root, &parent, &kids, &nk)) return cur;
    if (kids) XFree(kids);
    if (!parent || parent == root) return cur;
    cur = parent;
  }
}
#endif

}  // namespace

// ai: the window manager's window around w (topLevel above), through a Display of its own (2026-10-23: the parent the
// ai: desktop's file chooser is attached to, tx_jni.cpp windowOf); w itself off X11
unsigned long x11TopLevel(unsigned long window) {
#if LIZ_X11 && !defined(_WIN32)
  Display* d = XOpenDisplay(nullptr);
  if (!d) return window;
  const unsigned long top = topLevel(d, window);
  XCloseDisplay(d);
  return top;
#else
  return window;
#endif
}

// ai: The window manager's own states for its window (x11TopLevel's), asked as other apps ask: EWMH _NET_WM_STATE client
// ai: messages, the way GTK, Qt and the browsers do it; the manager keeps the window's place, size and maximized state
// ai: and gives them back. `full` asks full screen (2026-10-23), `maximized` both maximized directions (2026-11-13: the
// ai: app keeps a full-screen window maximized underneath, FullScreen.kt), each where true, on one connection so the
// ai: manager takes them in order: added, maximized first (a manager refuses to maximize a full-screen window: Muffin's
// ai: recalc_features), removed, full screen first. true where the requests went; false off X11 (the app then asks
// ai: Compose).
// ai: Until 2026-10-23 the presenter set _NET_WM_BYPASS_COMPOSITOR and _NET_WM_STATE_ABOVE with the app's full screen,
// ai: for a vsync whose losses were the machine's power-saving profile.
bool x11WmState(unsigned long top, bool add, bool full, bool maximized) {
#if LIZ_X11 && !defined(_WIN32)
  Display* d = XOpenDisplay(nullptr);
  if (!d) return false;
  bool sent = true;
  auto ask = [&](const char* a, const char* b) {
    XEvent e{};
    e.xclient.type = ClientMessage;
    e.xclient.window = top;
    e.xclient.message_type = XInternAtom(d, "_NET_WM_STATE", False);
    e.xclient.format = 32;
    e.xclient.data.l[0] = add ? 1 : 0;   // ai: _NET_WM_STATE_ADD, _NET_WM_STATE_REMOVE
    e.xclient.data.l[1] = static_cast<long>(XInternAtom(d, a, False));
    e.xclient.data.l[2] = b ? static_cast<long>(XInternAtom(d, b, False)) : 0;
    e.xclient.data.l[3] = 1;   // ai: the source: an application
    sent = XSendEvent(d, DefaultRootWindow(d), False, SubstructureRedirectMask | SubstructureNotifyMask, &e) != 0 && sent;
  };
  if (add && maximized) ask("_NET_WM_STATE_MAXIMIZED_VERT", "_NET_WM_STATE_MAXIMIZED_HORZ");
  if (full) ask("_NET_WM_STATE_FULLSCREEN", nullptr);
  if (!add && maximized) ask("_NET_WM_STATE_MAXIMIZED_VERT", "_NET_WM_STATE_MAXIMIZED_HORZ");
  XFlush(d);
  XCloseDisplay(d);
  return sent;
#else
  (void)top; (void)add; (void)full; (void)maximized;
  return false;
#endif
}
// ai: the states the manager has the window in, from its _NET_WM_STATE: WM_FULL, WM_MAXIMIZED (both directions)
int x11WmStateOf(unsigned long top) {
#if LIZ_X11 && !defined(_WIN32)
  Display* d = XOpenDisplay(nullptr);
  if (!d) return 0;
  const Atom state = XInternAtom(d, "_NET_WM_STATE", False), full = XInternAtom(d, "_NET_WM_STATE_FULLSCREEN", False),
             vert = XInternAtom(d, "_NET_WM_STATE_MAXIMIZED_VERT", False), horz = XInternAtom(d, "_NET_WM_STATE_MAXIMIZED_HORZ", False);
  Atom type = None;
  int format = 0;
  unsigned long n = 0, after = 0;
  unsigned char* data = nullptr;
  int bits = 0;
  if (XGetWindowProperty(d, top, state, 0, 64, False, XA_ATOM, &type, &format, &n, &after, &data) == Success && data) {
    const Atom* atoms = reinterpret_cast<const Atom*>(data);
    bool v = false, h = false;
    for (unsigned long i = 0; i < n; i++) {
      if (atoms[i] == full) bits |= WM_FULL;
      v = v || atoms[i] == vert;
      h = h || atoms[i] == horz;
    }
    if (v && h) bits |= WM_MAXIMIZED;
    XFree(data);
  }
  XCloseDisplay(d);
  return bits;
#else
  (void)top;
  return 0;
#endif
}

bool x11Background(unsigned long window, unsigned rgb) {
#if LIZ_X11 && !defined(_WIN32)
  Display* d = XOpenDisplay(nullptr);
  if (!d) return false;
  XWindowAttributes wa{};
  bool ok = XGetWindowAttributes(d, static_cast<Window>(window), &wa) != 0;
  if (ok) {
    XColor c{};
    c.red = static_cast<unsigned short>(((rgb >> 16) & 0xff) * 257);
    c.green = static_cast<unsigned short>(((rgb >> 8) & 0xff) * 257);
    c.blue = static_cast<unsigned short>((rgb & 0xff) * 257);
    c.flags = DoRed | DoGreen | DoBlue;
    ok = XAllocColor(d, wa.colormap, &c) != 0;
    if (ok) XSetWindowBackground(d, static_cast<Window>(window), c.pixel);
  }
  XFlush(d);
  XCloseDisplay(d);
  return ok;
#else
  (void)window; (void)rgb;
  return false;
#endif
}

bool x11KeepOnResize(unsigned long window) {
#if LIZ_X11 && !defined(_WIN32)
  Display* d = XOpenDisplay(nullptr);
  if (!d) return false;
  XSetWindowAttributes a{};
  a.bit_gravity = NorthWestGravity;
  XChangeWindowAttributes(d, static_cast<Window>(window), CWBitGravity, &a);
  XFlush(d);
  XCloseDisplay(d);
  return true;
#else
  (void)window;
  return false;
#endif
}

#if defined(_WIN32)
namespace {
// ai: the windows full screen now, each with what it was before: its style, its rectangle as a normal window, maximized
struct Windowed { LONG_PTR style = 0; RECT rect{}; bool maximized = false; };
std::mutex gFullMu;
std::map<HWND, Windowed> gFull;
}  // namespace
#endif

// ai: Windows' full screen (2026-10-24; Compose's placement before, which left the window unmaximized and the app's
// ai: own state wrong), as Chromium makes it: a maximized window restored first (left maximized, its placement's
// ai: rectangle stays the monitor's and it comes back over the taskbar), its normal rectangle kept, WS_CAPTION and
// ai: WS_THICKFRAME off and the window over its monitor's whole rectangle, which also hides the taskbar; on leaving,
// ai: the style and the rectangle back and the window maximized again where it was. Called from the app's thread,
// ai: these reach the window's (AWT's) synchronously, as AWT's own peers do.
bool win32Fullscreen(void* hwnd, bool on) {
#if defined(_WIN32)
  HWND w = static_cast<HWND>(hwnd);
  if (!w || !IsWindow(w)) return false;
  std::lock_guard<std::mutex> l(gFullMu);
  auto at = gFull.find(w);
  if (on) {
    if (at != gFull.end()) return true;
    MONITORINFO mi{};
    mi.cbSize = sizeof mi;
    if (!GetMonitorInfoW(MonitorFromWindow(w, MONITOR_DEFAULTTONEAREST), &mi)) return false;
    Windowed was;
    was.maximized = IsZoomed(w) != 0;
    if (was.maximized) ShowWindow(w, SW_RESTORE);
    was.style = GetWindowLongPtrW(w, GWL_STYLE);
    GetWindowRect(w, &was.rect);
    gFull[w] = was;
    SetWindowLongPtrW(w, GWL_STYLE, was.style & ~static_cast<LONG_PTR>(WS_CAPTION | WS_THICKFRAME));
    const RECT& r = mi.rcMonitor;
    SetWindowPos(w, nullptr, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
  } else {
    if (at == gFull.end()) return true;
    const Windowed was = at->second;
    gFull.erase(at);
    SetWindowLongPtrW(w, GWL_STYLE, was.style);
    const RECT& r = was.rect;
    SetWindowPos(w, nullptr, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
    if (was.maximized) ShowWindow(w, SW_MAXIMIZE);
  }
  return true;
#else
  (void)hwnd; (void)on;
  return false;
#endif
}
bool win32IsFullscreen(void* hwnd) {
#if defined(_WIN32)
  std::lock_guard<std::mutex> l(gFullMu);
  return gFull.count(static_cast<HWND>(hwnd)) > 0;
#else
  (void)hwnd;
  return false;
#endif
}

}  // namespace lizard
