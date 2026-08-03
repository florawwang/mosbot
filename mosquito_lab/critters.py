"""Whimsical mosquito cursor + a few mosquitoes that amble around the app.

Fits the theme: it's a mosquito lab, so the pointer is a mosquito and a small,
calm swarm drifts across the screen — they crawl, rest often, and gently gather
toward the (mosquito) cursor. Honors prefers-reduced-motion, and can be toggled
off from the app (see render_mosquito_swarm's `enabled` flag).

The overlay is drawn on the *parent* Streamlit document (the visible page), not
the component iframe, so it covers the whole app. It's injected once per session
(guarded by a window flag), never blocks clicks (pointer-events:none), and can be
fully removed when disabled.
"""

from __future__ import annotations

import streamlit.components.v1 as components

# Mosquito, top-down, facing +x (east) so it rotates cleanly toward any heading.
_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 44 28'>"
    "<g stroke='#2b2b2b' stroke-width='1' stroke-linecap='round' fill='none'>"
    "<path d='M22 12 L10 2'/><path d='M20 12 L2 6'/><path d='M24 12 L34 3'/>"
    "<path d='M22 16 L10 26'/><path d='M20 16 L2 22'/><path d='M24 16 L34 25'/>"
    "</g>"
    "<g fill='#9fb7c9' fill-opacity='0.45' stroke='#6d8394' stroke-width='0.5'>"
    "<ellipse cx='16' cy='9' rx='10' ry='4' transform='rotate(18 16 9)'/>"
    "<ellipse cx='16' cy='19' rx='10' ry='4' transform='rotate(-18 16 19)'/>"
    "</g>"
    "<g fill='#2b2b2b'>"
    "<path d='M6 14 Q16 10 24 14 Q16 18 6 14 Z'/>"
    "<circle cx='26' cy='14' r='3.2'/><circle cx='30' cy='14' r='2'/>"
    "</g>"
    "<path d='M32 14 L42 14' stroke='#2b2b2b' stroke-width='1.2' stroke-linecap='round'/>"
    "</svg>"
)

# Injected (once) into the parent document; runs in the parent realm so it
# survives Streamlit reruns that recycle the component iframe. Calm movement,
# and mosquitoes gently steer toward the cursor to cluster around it.
_BLOB = """
(function(){
  var BUG_COUNT = __COUNT__;
  var svg = "__SVG__";
  var svgCursor = svg.replace("viewBox='0 0 44 28'", "viewBox='0 0 44 28' width='36' height='23'");
  var uri = 'data:image/svg+xml,' + encodeURIComponent(svgCursor);

  // Mosquito cursor everywhere.
  var stEl = document.createElement('style');
  stEl.setAttribute('data-mosbot', 'cursor');
  stEl.textContent = '*{cursor:url(' + JSON.stringify(uri) + ') 34 12, auto !important;}';
  document.head.appendChild(stEl);

  // Full-viewport, click-through overlay for the roaming mosquitoes.
  var layer = document.getElementById('mosbot-bug-layer');
  if (!layer) {
    layer = document.createElement('div');
    layer.id = 'mosbot-bug-layer';
    layer.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147483000;overflow:hidden;';
    document.body.appendChild(layer);
  }

  var reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var W = window.innerWidth, H = window.innerHeight;
  window.addEventListener('resize', function(){ W = window.innerWidth; H = window.innerHeight; });
  var mx = -9999, my = -9999;
  window.addEventListener('mousemove', function(e){ mx = e.clientX; my = e.clientY; }, {passive:true});
  function rnd(a,b){ return a + Math.random()*(b-a); }

  var bugs = [];
  for (var i=0;i<BUG_COUNT;i++){
    var el = document.createElement('div');
    el.style.cssText = 'position:absolute;left:0;top:0;width:30px;height:20px;will-change:transform;'
      + 'transform-origin:15px 10px;filter:drop-shadow(0 1px 1px rgba(0,0,0,0.25));';
    el.innerHTML = svg;
    var sv = el.firstChild; if (sv){ sv.setAttribute('width','30'); sv.setAttribute('height','20'); }
    layer.appendChild(el);
    var b = { el: el, x: rnd(30, Math.max(60, W-30)), y: rnd(30, Math.max(60, H-30)),
              a: rnd(0, 6.283), sp: 0, state: 'crawl', t: rnd(400, 2000) };
    b.el.style.transform = 'translate('+(b.x-15)+'px,'+(b.y-10)+'px) rotate('+(b.a*57.2958)+'deg)';
    bugs.push(b);
  }

  // Reduced motion: leave them resting where they landed.
  if (reduce) return;

  var last = 0;
  function tick(ts){
    if (!document.getElementById('mosbot-bug-layer')) return;   // toggled off -> stop the loop
    if (!last) last = ts;
    var dt = Math.min(48, ts - last); last = ts;
    for (var i=0;i<bugs.length;i++){
      var b = bugs[i];
      b.t -= dt;
      if (b.t <= 0){
        var r = Math.random();
        if (b.state === 'fly') { b.state = (r < 0.7) ? 'crawl' : 'rest'; }
        else if (r < 0.06) { b.state = 'fly'; }   // rare little hop
        else if (r < 0.5) { b.state = 'rest'; }   // rests often (calm)
        else { b.state = 'crawl'; }
        b.t = (b.state === 'rest') ? rnd(800, 2600) : rnd(1000, 3400);
        b.a += rnd(-0.6, 0.6);
      }
      // Gently cluster toward the (mosquito) cursor.
      if (mx > -9999){
        var dxc = mx - b.x, dyc = my - b.y, dist = Math.sqrt(dxc*dxc + dyc*dyc);
        if (dist < 260){
          var toC = Math.atan2(dyc, dxc);
          var diff = Math.atan2(Math.sin(toC - b.a), Math.cos(toC - b.a));
          b.a += diff * 0.06;                       // slow steer toward cursor
          b.state = (dist > 46) ? 'crawl' : 'rest'; // gather and settle near it
          if (b.t < 300) b.t = 300;
        }
      }

      var target = (b.state === 'rest') ? 0 : ((b.state === 'fly') ? 0.05 : 0.02); // px/ms (calm)
      b.sp += (target - b.sp) * 0.05;
      if (b.state !== 'rest') { b.a += Math.sin((ts + i*97)/900) * 0.012; }
      b.x += Math.cos(b.a) * b.sp * dt;
      b.y += Math.sin(b.a) * b.sp * dt;

      var m = 16; // keep on screen
      if (b.x < m){ b.x = m; b.a = Math.PI - b.a; }
      if (b.x > W - m){ b.x = W - m; b.a = Math.PI - b.a; }
      if (b.y < m){ b.y = m; b.a = -b.a; }
      if (b.y > H - m){ b.y = H - m; b.a = -b.a; }

      var wob = 0, bob = 0;
      if (b.state === 'crawl') { wob = Math.sin(ts/110 + i) * 3; }   // gentle leg-scuttle
      if (b.state === 'fly')   { bob = Math.sin(ts/70 + i) * 1.5; }  // soft flight bob
      var deg = b.a * 57.2958 + wob;
      b.el.style.transform = 'translate('+(b.x-15)+'px,'+(b.y-10+bob)+'px) rotate('+deg+'deg)';
      b.el.style.opacity = (b.state === 'rest') ? '0.55' : '0.8';
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
"""


def render_mosquito_swarm(enabled: bool = True, count: int = 4) -> None:
    """Inject (or remove) the mosquito cursor + roaming mosquitoes.

    Idempotent: when enabled, injects once per session; when disabled, tears the
    overlay + cursor back down so the app returns to a normal pointer.
    """
    if enabled:
        blob = _BLOB.replace("__COUNT__", str(int(count))).replace("__SVG__", _SVG)
        html = (
            "<script>(function(){try{"
            "var P=window.parent,D=P.document;"
            "if(P.__mosbotBugsInit)return;P.__mosbotBugsInit=true;"
            "var s=D.createElement('script');s.setAttribute('data-mosbot','bugs');"
            "s.textContent=" + _js_string(blob) + ";"
            "D.head.appendChild(s);"
            "}catch(e){}})();</script>"
        )
    else:
        html = (
            "<script>(function(){try{"
            "var P=window.parent,D=P.document;"
            "var l=D.getElementById('mosbot-bug-layer');if(l)l.remove();"
            "var c=D.querySelector('style[data-mosbot=cursor]');if(c)c.remove();"
            "var b=D.querySelector('script[data-mosbot=bugs]');if(b)b.remove();"
            "P.__mosbotBugsInit=false;"
            "}catch(e){}})();</script>"
        )
    components.html(html, height=0, width=0)


def _js_string(code: str) -> str:
    """Encode a Python str as a safe single-quoted JS string literal."""
    esc = (
        code.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "")
        .replace("</", "<\\/")  # avoid closing the host <script> early
    )
    return "'" + esc + "'"
