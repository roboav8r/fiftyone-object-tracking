/**
 * BEV Track Visualization — UMD panel.
 *
 * Runs inside the FiftyOne App; expects these globals to be present:
 *   window.React, window.recoil, window.__fos__, window.__foo__,
 *   window.__fop__, window.__mui__.
 *
 * Talks to the Python operators registered in __init__.py via
 * foo.useOperatorExecutor("@roboav8r/fiftyone-object-tracking/<name>").
 *
 * Rendering is hand-rolled SVG (no chart libs). Layout:
 *
 *   [ Header: scene picker | base/world toggle | class legend | open in modal ]
 *   [ BEVChart (full panel width)                                            ]
 *   [ Scrubber (full panel width)                                            ]
 *   [ TrackTimeline rows (per-instance presence strips, full panel width)    ]
 */

(function () {
  "use strict";

  var React  = window.React;
  var recoil = window.recoil;
  var fos    = window.__fos__ || window.fos;
  var foo    = window.__foo__ || window.foo;
  var fop    = window.__fop__ || window.fop;
  var fosp   = window.__fosp__ || window.fosp || {};   // @fiftyone/spaces
  var fopb   = window.__fopb__ || window.fopb || {};   // @fiftyone/playback
  var mui    = window.__mui__ || window.mui || {};

  if (!React || !fop) {
    console.error("[object-tracking-toolkit] missing React or fop globals");
    return;
  }

  var h = React.createElement;
  var useState  = React.useState;
  var useEffect = React.useEffect;
  var useMemo   = React.useMemo;
  var useRef    = React.useRef;
  var useCallback = React.useCallback;

  var PLUGIN = "@roboav8r/fiftyone-object-tracking";
  var OP = function (name) { return PLUGIN + "/" + name; };


  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------

  // Resolve a media path/URL into something an <img> can actually load.
  // The get_camera_frame_urls operator returns signed https URLs for cloud
  // media but raw filesystem paths for local media (which the browser can't
  // load directly). fos.getSampleSrc — the same helper the grid/looker use —
  // passes http(s)/signed URLs through unchanged and routes local paths
  // through the App's /media server. Falls back to the raw value on older
  // FOE builds that don't export it.
  function resolveMediaSrc(url) {
    if (!url) return url;
    try {
      if (fos && typeof fos.getSampleSrc === "function") {
        return fos.getSampleSrc(url);
      }
    } catch (e) {
      console.warn("[bev-panel] getSampleSrc failed for", url, e);
    }
    return url;
  }

  // Per-class hue palette + per-instance shade jitter. classHue assigns
  // hues to classes in first-seen order so the palette is stable per panel
  // mount. instanceColor jitters lightness + saturation deterministically
  // from the instance id so two `person` tracks share a hue but are
  // individually distinguishable.
  var CLASS_HUE_PALETTE = [
    200,  // azure
      0,  // red
    120,  // green
     40,  // amber
    280,  // violet
    170,  // teal
    320,  // magenta
     60,  // yellow
    240,  // blue
    100,  // lime
     20,  // orange
    300,  // pink
  ];
  var classHueCache = Object.create(null);
  function classHue(label) {
    var key = String(label || "unknown");
    if (key in classHueCache) return classHueCache[key];
    var n = Object.keys(classHueCache).length;
    if (n >= CLASS_HUE_PALETTE.length) {
      console.warn("[bev-panel] class palette exhausted; hue collision possible");
    }
    return (classHueCache[key] = CLASS_HUE_PALETTE[n % CLASS_HUE_PALETTE.length]);
  }
  function _hashStr(s) {
    var h = 0;
    s = String(s || "");
    for (var i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    return h;
  }
  function classColor(label) {
    return "hsl(" + classHue(label) + ", 70%, 55%)";
  }
  function instanceColor(label, instanceId) {
    var h = _hashStr(instanceId);
    var sJ = ((h & 0xFF) / 0xFF) * 2 - 1;          // [-1, +1]
    var lJ = (((h >> 8) & 0xFF) / 0xFF) * 2 - 1;   // [-1, +1]
    var sat   = Math.max(45, Math.min(85, 70 + sJ * 15));
    var light = Math.max(40, Math.min(70, 55 + lJ * 12));
    return "hsl(" + classHue(label) + ", " + sat.toFixed(0) + "%, " + light.toFixed(0) + "%)";
  }

  // Split a sorted array of int frame_idxs into [startIdx, endIdx] inclusive
  // pairs over the array, where each pair is a contiguous run.
  function contiguousRuns(frames) {
    var runs = [];
    if (!frames || !frames.length) return runs;
    var s = 0;
    for (var i = 1; i < frames.length; i++) {
      if (frames[i] !== frames[i - 1] + 1) {
        runs.push([s, i - 1]);
        s = i;
      }
    }
    runs.push([s, frames.length - 1]);
    return runs;
  }

  // Find the array index in inst.frames whose value is the latest frame
  // <= currentFrameIdx, falling back to the earliest frame > currentFrameIdx.
  // Returns -1 if frames is empty.
  function nearestFrameIdx(frames, currentFrameIdx) {
    if (!frames || !frames.length) return -1;
    var bestPrev = -1;
    var bestNext = -1;
    for (var i = 0; i < frames.length; i++) {
      if (frames[i] <= currentFrameIdx) bestPrev = i;
      else if (bestNext < 0) { bestNext = i; break; }
    }
    if (bestPrev >= 0) return bestPrev;
    return bestNext;
  }

  function isFiniteNum(x) {
    return typeof x === "number" && isFinite(x);
  }

  // Short, human-readable identity tag for an instance row.
  // Prefers the source-side tracking_id (e.g. dx3 int "12"); falls back to a
  // 6-char prefix of the FO instance hex. UUID-style tracking_ids (raillabel)
  // get the same 6-char truncation so the timeline label stays compact.
  function shortTrackTag(inst) {
    var tid = inst && inst.tracking_id;
    if (tid !== undefined && tid !== null && String(tid).length) {
      var s = String(tid);
      return s.length > 8 ? s.slice(0, 6) : s;
    }
    return (inst && inst.instance_id ? String(inst.instance_id) : "").slice(0, 6);
  }

  /**
   * Compute axis-aligned data bounds across every trajectory in a payload
   * for the given view mode. Falls back to a small box around the origin
   * if the payload is empty.
   */
  function computeBounds(payload, viewMode) {
    var xMin = Infinity, xMax = -Infinity;
    var yMin = Infinity, yMax = -Infinity;
    if (!payload || !payload.instances) return { xMin: -10, xMax: 10, yMin: -10, yMax: 10 };

    for (var i = 0; i < payload.instances.length; i++) {
      var bw = payload.instances[i][viewMode];
      if (!bw) continue;
      for (var j = 0; j < bw.x.length; j++) {
        var x = bw.x[j], y = bw.y[j];
        if (!isFiniteNum(x) || !isFiniteNum(y)) continue;
        if (x < xMin) xMin = x; if (x > xMax) xMax = x;
        if (y < yMin) yMin = y; if (y > yMax) yMax = y;
      }
    }
    if (viewMode === "world" && payload.ego_world) {
      var ex = payload.ego_world.x, ey = payload.ego_world.y;
      for (var k = 0; k < ex.length; k++) {
        if (isFiniteNum(ex[k]) && isFiniteNum(ey[k])) {
          if (ex[k] < xMin) xMin = ex[k]; if (ex[k] > xMax) xMax = ex[k];
          if (ey[k] < yMin) yMin = ey[k]; if (ey[k] > yMax) yMax = ey[k];
        }
      }
    }

    if (!isFinite(xMin)) return { xMin: -10, xMax: 10, yMin: -10, yMax: 10 };

    var pad = Math.max(2, 0.08 * Math.max(xMax - xMin, yMax - yMin));
    return { xMin: xMin - pad, xMax: xMax + pad,
             yMin: yMin - pad, yMax: yMax + pad };
  }

  /**
   * Build a screen<->data projector. Vehicle base convention:
   *   data x  = forward → screen y (inverted, so up = forward)
   *   data y  = left    → screen x (inverted, so left = +y)
   * The aspect ratio is preserved so 1m horizontal ≈ 1m vertical on screen.
   *
   * `panX, panY, zoom` are an interactive viewport transform applied
   * AFTER the data→pixel mapping. Default identity (0, 0, 1) recovers
   * the original fit-to-bounds projection. Zoom is centered on the
   * chart's geometric center so a fresh zoom feels symmetric; pan is
   * a screen-space translation in pixels.
   */
  function makeProjector(bounds, width, height, panX, panY, zoom) {
    panX = panX || 0;
    panY = panY || 0;
    zoom = zoom || 1;

    var dx = bounds.xMax - bounds.xMin;
    var dy = bounds.yMax - bounds.yMin;
    if (dx <= 0) dx = 1; if (dy <= 0) dy = 1;

    // Equal scale per axis to preserve aspect.
    var sx = width  / dy;
    var sy = height / dx;
    var s  = Math.min(sx, sy);

    // Center the projected box in the viewport.
    var ox = (width  - s * dy) / 2;
    var oy = (height - s * dx) / 2;

    var cx = width / 2, cy = height / 2;
    function applyView(px, py) {
      return [cx + (px - cx) * zoom + panX,
              cy + (py - cy) * zoom + panY];
    }
    function unapplyView(px, py) {
      return [(px - cx - panX) / zoom + cx,
              (py - cy - panY) / zoom + cy];
    }

    return {
      // (data x, data y) → (px, py)
      project: function (x, y) {
        var px0 = ox + s * (bounds.yMax - y);
        var py0 = oy + s * (bounds.xMax - x);
        return applyView(px0, py0);
      },
      // (px, py) → (data x, data y)
      unproject: function (px, py) {
        var p0 = unapplyView(px, py);
        var y = bounds.yMax - (p0[0] - ox) / s;
        var x = bounds.xMax - (p0[1] - oy) / s;
        return [x, y];
      },
      scale: s * zoom,
    };
  }


  // ---------------------------------------------------------------------------
  // BEVChart
  // ---------------------------------------------------------------------------
  function BEVChart(props) {
    var payload = props.payload;
    var viewMode = props.viewMode;
    var currentFrameIdx = props.currentFrameIdx;
    var selectedInstanceIds = props.selectedInstanceIds;
    var hoveredInstanceId = props.hoveredInstanceId;
    var onHoverInstance = props.onHoverInstance;
    var onSelectInstance = props.onSelectInstance;

    var width  = props.width  || 600;
    var height = props.height || 480;

    // ---- Interactive pan/zoom state ----
    // panX/panY are in screen pixels; zoom is a multiplier around the chart
    // center. Defaults (0, 0, 1) reproduce the auto-fit projection.
    var [panX, setPanX] = useState(0);
    var [panY, setPanY] = useState(0);
    var [zoom, setZoom] = useState(1);

    // Stable mirror of the view state for the wheel listener (which reads
    // through a closure registered once via addEventListener).
    var viewRef = useRef({ panX: 0, panY: 0, zoom: 1 });
    useEffect(function () {
      viewRef.current = { panX: panX, panY: panY, zoom: zoom };
    });

    function resetView() { setPanX(0); setPanY(0); setZoom(1); }

    // Reset on scene or view-mode change (data bounds shift).
    var sceneName = payload && payload.scene_name;
    useEffect(function () { resetView(); }, [viewMode, sceneName]);

    var bounds = useMemo(function () {
      return computeBounds(payload, viewMode);
    }, [payload, viewMode]);

    var projector = useMemo(function () {
      return makeProjector(bounds, width, height, panX, panY, zoom);
    }, [bounds, width, height, panX, panY, zoom]);

    var svgRef = useRef(null);
    var panStartRef = useRef(null);
    var suppressClickRef = useRef(false);

    // Index from frame_idx → array index for ego pose lookup.
    var frameIndexLookup = useMemo(function () {
      var m = {};
      if (payload && payload.frame_indices) {
        for (var i = 0; i < payload.frame_indices.length; i++) {
          m[payload.frame_indices[i]] = i;
        }
      }
      return m;
    }, [payload]);

    function handleMouseDown(e) {
      // Middle-mouse drag begins a pan. preventDefault stops Chromium's
      // auto-scroll cursor / paste-from-primary side effects.
      if (e.button !== 1) return;
      e.preventDefault();
      panStartRef.current = {
        cx: e.clientX, cy: e.clientY,
        panX: panX, panY: panY,
      };
    }

    function handleMouseMove(e) {
      // Active pan in progress: translate without doing hit-testing.
      if (panStartRef.current) {
        var dx = e.clientX - panStartRef.current.cx;
        var dy = e.clientY - panStartRef.current.cy;
        if (dx * dx + dy * dy > 4) suppressClickRef.current = true;
        setPanX(panStartRef.current.panX + dx);
        setPanY(panStartRef.current.panY + dy);
        return;
      }

      if (!svgRef.current || !payload || !payload.instances) return;
      var rect = svgRef.current.getBoundingClientRect();
      var px = e.clientX - rect.left;
      var py = e.clientY - rect.top;

      // Hit-test live markers first; ghost markers only count if no live
      // marker is within range. Threshold = 18 px in both passes.
      var bestLiveId = null, bestLiveDist = 18;
      var bestGhostId = null, bestGhostDist = 18;

      for (var i = 0; i < payload.instances.length; i++) {
        var inst = payload.instances[i];
        var bw = inst[viewMode];
        if (!bw) continue;

        var liveIdx = inst.frames.indexOf(currentFrameIdx);
        if (liveIdx >= 0) {
          var lx = bw.x[liveIdx], ly = bw.y[liveIdx];
          if (isFiniteNum(lx) && isFiniteNum(ly)) {
            var lp = projector.project(lx, ly);
            var ld = Math.hypot(lp[0] - px, lp[1] - py);
            if (ld < bestLiveDist) {
              bestLiveDist = ld;
              bestLiveId = inst.instance_id;
            }
          }
        } else {
          var nIdx = nearestFrameIdx(inst.frames, currentFrameIdx);
          if (nIdx < 0) continue;
          var gx = bw.x[nIdx], gy = bw.y[nIdx];
          if (!isFiniteNum(gx) || !isFiniteNum(gy)) continue;
          var gp = projector.project(gx, gy);
          var gd = Math.hypot(gp[0] - px, gp[1] - py);
          if (gd < bestGhostDist) {
            bestGhostDist = gd;
            bestGhostId = inst.instance_id;
          }
        }
      }

      var bestId = bestLiveId || bestGhostId;
      if (bestId !== hoveredInstanceId) {
        onHoverInstance && onHoverInstance(bestId);
      }
    }

    function handleMouseUp(e) {
      if (panStartRef.current && e.button === 1) {
        e.preventDefault();
        panStartRef.current = null;
      }
    }

    function handleMouseLeave() {
      panStartRef.current = null;
      if (hoveredInstanceId) onHoverInstance && onHoverInstance(null);
    }

    function handleClick(e) {
      // If a pan just ended (mouse moved >2 px while middle-down), suppress
      // the click-as-selection event that would otherwise fire on mouseup.
      if (suppressClickRef.current) {
        suppressClickRef.current = false;
        return;
      }
      if (hoveredInstanceId !== undefined) {
        // Ctrl/Cmd-click toggles the track in the multi-selection;
        // plain click replaces the selection with just the clicked
        // track. The parent (BEVPanel) interprets the flag.
        var additive = !!(e && (e.metaKey || e.ctrlKey || e.shiftKey));
        onSelectInstance && onSelectInstance(hoveredInstanceId, additive);
      }
    }

    function handleDoubleClick() {
      resetView();
    }

    // Wheel zoom (cursor-centered). React's onWheel is root-delegated
    // passive in newer versions, so e.preventDefault() there warns/no-ops.
    // Bind directly via addEventListener with { passive: false }.
    //
    // The `hasPayload` dep is critical: on first render payload is null
    // so we render a fallback <div> (no SVG); svgRef.current is null and
    // this effect early-returns. When payload arrives the SVG mounts and
    // the effect must re-run to actually attach the listener.
    var hasPayload = !!payload;
    useEffect(function () {
      var node = svgRef.current;
      if (!node) return;
      function onWheel(e) {
        e.preventDefault();
        var rect = node.getBoundingClientRect();
        var mx = e.clientX - rect.left;
        var my = e.clientY - rect.top;
        var v = viewRef.current;
        var factor = e.deltaY > 0 ? 0.9 : 1.1;
        var newZoom = Math.max(0.2, Math.min(20, v.zoom * factor));
        var cxv = width / 2, cyv = height / 2;
        // Cursor-centered: keep the data point under the cursor fixed.
        var newPanX = mx - cxv - ((mx - cxv - v.panX) / v.zoom) * newZoom;
        var newPanY = my - cyv - ((my - cyv - v.panY) / v.zoom) * newZoom;
        setZoom(newZoom);
        setPanX(newPanX);
        setPanY(newPanY);
      }
      node.addEventListener("wheel", onWheel, { passive: false });
      return function () { node.removeEventListener("wheel", onWheel); };
    }, [width, height, hasPayload]);

    if (!payload || payload.error) {
      return h("div", {
        style: {
          width: width, height: height,
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "#888", fontStyle: "italic",
        },
      }, payload && payload.error ? payload.error : "Loading scene…");
    }

    var instances = payload.instances || [];

    // Trajectory polylines, fragment-aware:
    //   - one solid polyline per contiguous run of frame_idxs
    //   - one dashed bridge polyline per gap between successive runs
    var trajectories = [];
    instances.forEach(function (inst) {
      var bw = inst[viewMode];
      if (!bw || !bw.x.length || !inst.frames) return;
      var color = instanceColor(inst.label, inst.instance_id);
      var isHover = inst.instance_id === hoveredInstanceId;
      var isSel   = selectedInstanceIds && selectedInstanceIds.has(inst.instance_id);
      var solidWidth = isHover || isSel ? 2.5 : 1.0;
      var solidAlpha = isSel ? 1.0 : (isHover ? 0.9 : 0.55);

      var runs = contiguousRuns(inst.frames);
      var lastEndPx = null; // (px, py) of the previous run's last point

      runs.forEach(function (run, ri) {
        var pts = [];
        var firstPx = null;
        for (var j = run[0]; j <= run[1]; j++) {
          if (!isFiniteNum(bw.x[j]) || !isFiniteNum(bw.y[j])) continue;
          var pp = projector.project(bw.x[j], bw.y[j]);
          pts.push(pp[0].toFixed(2) + "," + pp[1].toFixed(2));
          if (firstPx === null) firstPx = pp;
        }
        if (pts.length >= 2) {
          trajectories.push(h("polyline", {
            key: "traj-" + inst.instance_id + "-r" + ri,
            points: pts.join(" "),
            fill: "none",
            stroke: color,
            strokeWidth: solidWidth,
            strokeOpacity: solidAlpha,
          }));
        }
        // Dashed bridge from previous run end → this run start.
        if (lastEndPx && firstPx) {
          trajectories.push(h("line", {
            key: "bridge-" + inst.instance_id + "-r" + ri,
            x1: lastEndPx[0], y1: lastEndPx[1],
            x2: firstPx[0], y2: firstPx[1],
            stroke: color,
            strokeWidth: Math.max(0.8, solidWidth - 0.6),
            strokeOpacity: solidAlpha * 0.55,
            strokeDasharray: "4,4",
          }));
        }
        // Track the last finite point of this run.
        for (var k = run[1]; k >= run[0]; k--) {
          if (isFiniteNum(bw.x[k]) && isFiniteNum(bw.y[k])) {
            lastEndPx = projector.project(bw.x[k], bw.y[k]);
            break;
          }
        }
      });
    });

    // Current-frame markers (live) + ghost markers (last-known position
    // for instances absent at currentFrameIdx).
    var atTime = [];
    instances.forEach(function (inst) {
      var bw = inst[viewMode];
      if (!bw) return;
      var color = instanceColor(inst.label, inst.instance_id);
      var isHover = inst.instance_id === hoveredInstanceId;
      var isSel   = selectedInstanceIds && selectedInstanceIds.has(inst.instance_id);

      var idx = inst.frames.indexOf(currentFrameIdx);
      var isLive = idx >= 0;
      if (!isLive) {
        idx = nearestFrameIdx(inst.frames, currentFrameIdx);
        if (idx < 0) return;
      }
      var x = bw.x[idx], y = bw.y[idx];
      if (!isFiniteNum(x) || !isFiniteNum(y)) return;
      var pp = projector.project(x, y);

      if (isLive) {
        // Footprint rectangle.
        var corners = bw.corners[idx];
        if (corners) {
          var cornerPts = [];
          for (var c = 0; c < corners.length; c++) {
            var cc = corners[c];
            if (!isFiniteNum(cc[0]) || !isFiniteNum(cc[1])) { cornerPts = null; break; }
            var ppc = projector.project(cc[0], cc[1]);
            cornerPts.push(ppc[0].toFixed(2) + "," + ppc[1].toFixed(2));
          }
          if (cornerPts && cornerPts.length === 4) {
            atTime.push(h("polygon", {
              key: "foot-" + inst.instance_id,
              points: cornerPts.join(" "),
              fill: color,
              fillOpacity: isSel ? 0.35 : (isHover ? 0.25 : 0.15),
              stroke: color,
              strokeWidth: isSel ? 2.0 : (isHover ? 1.5 : 1.0),
            }));
          }
        }

        // Center dot.
        atTime.push(h("circle", {
          key: "dot-" + inst.instance_id,
          cx: pp[0], cy: pp[1],
          r: isSel ? 5.5 : (isHover ? 4.5 : 3.5),
          fill: color, stroke: "#fff", strokeWidth: 1.0,
        }));

        if (isHover || isSel) {
          atTime.push(h("text", {
            key: "lbl-" + inst.instance_id,
            x: pp[0] + 8, y: pp[1] - 8,
            fill: color, fontSize: 11, fontFamily: "ui-monospace, monospace",
            stroke: "#0008", strokeWidth: 0.4, paintOrder: "stroke",
          }, inst.label + " " + shortTrackTag(inst)));
        }
      } else {
        // Ghost dot at last-known position. No footprint, no label.
        atTime.push(h("circle", {
          key: "ghost-" + inst.instance_id,
          cx: pp[0], cy: pp[1],
          r: isSel ? 4.5 : (isHover ? 4.0 : 3.0),
          fill: color, fillOpacity: isHover || isSel ? 0.45 : 0.25,
          stroke: color, strokeOpacity: isHover || isSel ? 0.7 : 0.45,
          strokeWidth: 1.0, strokeDasharray: "2,2",
        }));
      }
    });

    // Ego marker.
    var egoNodes = [];
    var ex = 0, ey = 0, eyaw = 0;
    if (viewMode === "world" && payload.ego_world) {
      var ei = frameIndexLookup[currentFrameIdx];
      if (ei !== undefined) {
        ex = payload.ego_world.x[ei];
        ey = payload.ego_world.y[ei];
        eyaw = payload.ego_world.yaw[ei];
      }
    } else {
      ex = 0; ey = 0; eyaw = 0;
    }
    if (isFiniteNum(ex) && isFiniteNum(ey) && isFiniteNum(eyaw)) {
      // Approximate rail-vehicle footprint for visual reference (4m x 2m).
      var EGO_L = 4.0, EGO_W = 2.0;
      var c = Math.cos(eyaw), s = Math.sin(eyaw);
      var local = [[+EGO_L/2, +EGO_W/2], [+EGO_L/2, -EGO_W/2],
                   [-EGO_L/2, -EGO_W/2], [-EGO_L/2, +EGO_W/2]];
      var pts = local.map(function (lp) {
        var wx = ex + c * lp[0] - s * lp[1];
        var wy = ey + s * lp[0] + c * lp[1];
        var pp = projector.project(wx, wy);
        return pp[0].toFixed(2) + "," + pp[1].toFixed(2);
      });
      egoNodes.push(h("polygon", {
        key: "ego",
        points: pts.join(" "),
        fill: "#ff8c00", fillOpacity: 0.55,
        stroke: "#ff8c00", strokeWidth: 2.0,
      }));
      // Heading arrow (forward = +x_base).
      var nose = [ex + c * (EGO_L/2 + 1.5), ey + s * (EGO_L/2 + 1.5)];
      var tail = [ex, ey];
      var tp = projector.project(tail[0], tail[1]);
      var np = projector.project(nose[0], nose[1]);
      egoNodes.push(h("line", {
        key: "ego-heading",
        x1: tp[0], y1: tp[1], x2: np[0], y2: np[1],
        stroke: "#fff", strokeWidth: 2.2,
      }));
    }

    // Axes / origin grid (light).
    var origin = projector.project(0, 0);
    var axes = [
      h("line", { key: "ax-x", x1: 0, y1: origin[1], x2: width, y2: origin[1],
                  stroke: "#444", strokeDasharray: "3,4" }),
      h("line", { key: "ax-y", x1: origin[0], y1: 0, x2: origin[0], y2: height,
                  stroke: "#444", strokeDasharray: "3,4" }),
    ];

    // Compass label per view mode + interaction hint.
    var compassMain = viewMode === "world"
      ? "World ENU — up = north, right = east"
      : "Vehicle base — up = forward (+x), left = +y";
    var compass = h("g", { key: "compass" }, [
      h("text", { key: "compass-label",
        x: 8, y: 14, fill: "#aaa", fontSize: 11,
        fontFamily: "ui-sans-serif, system-ui",
      }, compassMain),
      h("text", { key: "compass-hint",
        x: 8, y: 28, fill: "#777", fontSize: 10,
        fontFamily: "ui-sans-serif, system-ui",
      }, "middle-drag pan · wheel zoom · dbl-click reset"),
    ]);

    var isPanning = !!panStartRef.current;
    var cursor = isPanning ? "grabbing"
                           : (hoveredInstanceId ? "pointer" : "crosshair");

    var svgEl = h("svg", {
      ref: svgRef,
      width: width, height: height,
      style: { background: "#1c1c1c", border: "1px solid #2c2c2c",
               borderRadius: 4, cursor: cursor, display: "block" },
      onMouseDown: handleMouseDown,
      onMouseMove: handleMouseMove,
      onMouseUp: handleMouseUp,
      onMouseLeave: handleMouseLeave,
      onClick: handleClick,
      onDoubleClick: handleDoubleClick,
      // Block the browser's middle-mouse auto-scroll (Chromium) on the SVG.
      onAuxClick: function (e) { if (e.button === 1) e.preventDefault(); },
    }, [].concat(axes, trajectories, atTime, egoNodes, [compass]));

    // Reset / zoom% overlay button in the chart's top-right corner.
    var zoomPct = Math.round(zoom * 100);
    var atDefault = panX === 0 && panY === 0 && zoom === 1;
    var resetBtn = h("button", {
      onClick: function (e) { e.stopPropagation(); resetView(); },
      title: atDefault ? "View at default" : "Reset pan/zoom",
      style: {
        position: "absolute", top: 8, right: 8,
        background: atDefault ? "#222" : V51_ORANGE,
        color: "#eee", border: "1px solid #444", borderRadius: 4,
        padding: "3px 8px", cursor: "pointer",
        fontFamily: "ui-monospace, monospace", fontSize: 11,
        opacity: 0.85,
      },
    }, zoomPct + "% · Reset");

    return h("div", {
      style: { position: "relative", width: width, height: height },
    }, [svgEl, resetBtn]);
  }


  // ---------------------------------------------------------------------------
  // Scrubber
  // ---------------------------------------------------------------------------
  function Scrubber(props) {
    var frameIndices = props.frameIndices || [];
    var currentFrameIdx = props.currentFrameIdx;
    var mFrameTimestamps = props.mFrameTimestamps || [];
    var onScrub = props.onScrub;
    var onCommit = props.onCommit;
    var width = props.width || 800;
    var height = 70;

    var svgRef = useRef(null);
    var draggingRef = useRef(false);

    var curIdx = frameIndices.indexOf(currentFrameIdx);
    if (curIdx < 0 && frameIndices.length) curIdx = 0;

    function pixelToTickIndex(px) {
      if (!frameIndices.length) return 0;
      var ratio = Math.max(0, Math.min(1, px / width));
      return Math.round(ratio * (frameIndices.length - 1));
    }

    function handleMouseDown(e) {
      draggingRef.current = true;
      var rect = svgRef.current.getBoundingClientRect();
      var idx = pixelToTickIndex(e.clientX - rect.left);
      onScrub && onScrub(frameIndices[idx], idx);
    }
    function handleMouseMove(e) {
      if (!draggingRef.current) return;
      var rect = svgRef.current.getBoundingClientRect();
      var idx = pixelToTickIndex(e.clientX - rect.left);
      onScrub && onScrub(frameIndices[idx], idx);
    }
    function handleMouseUp(e) {
      if (!draggingRef.current) return;
      draggingRef.current = false;
      var rect = svgRef.current.getBoundingClientRect();
      var idx = pixelToTickIndex(e.clientX - rect.left);
      onCommit && onCommit(frameIndices[idx], idx);
    }

    var ticks = [];
    var tickStride = Math.max(1, Math.ceil(frameIndices.length / 60));
    for (var i = 0; i < frameIndices.length; i += tickStride) {
      var px = (i / Math.max(1, frameIndices.length - 1)) * width;
      ticks.push(h("line", {
        key: "tk-" + i, x1: px, x2: px, y1: 28, y2: 38,
        stroke: i === curIdx ? "#fff" : "#666", strokeWidth: 1,
      }));
    }

    var indicatorX = curIdx >= 0
      ? (curIdx / Math.max(1, frameIndices.length - 1)) * width : 0;
    var ts = curIdx >= 0 ? (mFrameTimestamps[curIdx] || "") : "";
    var label = "Frame " + (frameIndices[curIdx] != null ? frameIndices[curIdx] : "—") +
                "  ·  " + ts;

    return h("svg", {
      ref: svgRef,
      width: width, height: height,
      style: { background: "#1c1c1c", border: "1px solid #2c2c2c", borderRadius: 4,
               cursor: "pointer", userSelect: "none" },
      onMouseDown: handleMouseDown,
      onMouseMove: handleMouseMove,
      onMouseUp: handleMouseUp,
      onMouseLeave: handleMouseUp,
    }, [
      h("rect", { key: "track", x: 0, y: 32, width: width, height: 2, fill: "#444" }),
      h("g", { key: "ticks" }, ticks),
      h("line", { key: "indicator",
        x1: indicatorX, y1: 8, x2: indicatorX, y2: 56,
        stroke: "#5af", strokeWidth: 2 }),
      h("text", { key: "lbl",
        x: 8, y: 14, fill: "#ddd", fontSize: 11,
        fontFamily: "ui-monospace, monospace" }, label),
    ]);
  }


  // ---------------------------------------------------------------------------
  // TrackTimeline — per-instance presence rows aligned with the scrubber.
  //
  // Each row: [class swatch + label (left ~180px) | colored presence bar
  // (right, full width)]. Each contiguous run of frame_idxs renders as a
  // filled rectangle in the instance's (per-class jittered) color. A single
  // overlay <line> across all rows marks the current scrubber position.
  // Hover/click on a row drives the same hovered/selected instance state
  // the BEV chart uses.
  // ---------------------------------------------------------------------------
  function TrackTimeline(props) {
    var payload = props.payload;
    var currentFrameIdx = props.currentFrameIdx;
    var hoveredInstanceId = props.hoveredInstanceId;
    var selectedInstanceIds = props.selectedInstanceIds;
    var onHoverInstance = props.onHoverInstance;
    var onSelectInstance = props.onSelectInstance;
    var width = props.width || 800;
    var rowHeight = props.rowHeight || 14;
    var maxHeight = props.maxHeight || 280;

    if (!payload || !payload.instances || !payload.instances.length) {
      return h("div", {
        style: {
          width: width, padding: 12, color: "#888", fontStyle: "italic",
          fontFamily: "ui-sans-serif, system-ui", fontSize: 12,
        },
      }, "No tracks in this scene.");
    }

    var labelColW = 180;
    var stripW = Math.max(120, width - labelColW - 12);

    // Sort by class then label so same-class rows cluster.
    var instances = payload.instances.slice().sort(function (a, b) {
      var ka = (a.label || "") + "|" + (a.instance_id || "");
      var kb = (b.label || "") + "|" + (b.instance_id || "");
      return ka < kb ? -1 : ka > kb ? 1 : 0;
    });

    var fIdxs = payload.frame_indices || [];
    var nF = fIdxs.length;
    var firstF = nF ? fIdxs[0] : 0;
    var lastF  = nF ? fIdxs[nF - 1] : 0;
    function frameToPx(f) {
      if (lastF === firstF) return 0;
      return ((f - firstF) / (lastF - firstF)) * stripW;
    }

    var indicatorPx = currentFrameIdx != null && nF
      ? frameToPx(currentFrameIdx) : null;

    var rows = instances.map(function (inst) {
      var color = instanceColor(inst.label, inst.instance_id);
      var isHover = inst.instance_id === hoveredInstanceId;
      var isSel   = selectedInstanceIds && selectedInstanceIds.has(inst.instance_id);

      // Build run rectangles in pixel space.
      var runs = contiguousRuns(inst.frames || []);
      var runRects = runs.map(function (run, ri) {
        var f0 = inst.frames[run[0]];
        var f1 = inst.frames[run[1]];
        var x0 = frameToPx(f0);
        var x1 = frameToPx(f1);
        var w  = Math.max(2, x1 - x0 + (lastF !== firstF ? stripW / Math.max(1, lastF - firstF) : 2));
        return h("rect", {
          key: "rr-" + inst.instance_id + "-" + ri,
          x: x0, y: 1, width: w, height: rowHeight - 2,
          fill: color,
          fillOpacity: isSel ? 1.0 : (isHover ? 0.9 : 0.7),
          stroke: isSel ? "#fff" : "none",
          strokeWidth: isSel ? 1 : 0,
        });
      });

      var labelText = inst.label + " " + shortTrackTag(inst) +
        " (" + (inst.frames || []).length + "f)";

      return h("div", {
        key: "row-" + inst.instance_id,
        onMouseEnter: function () { onHoverInstance && onHoverInstance(inst.instance_id); },
        onMouseLeave: function () { onHoverInstance && onHoverInstance(null); },
        onClick: function (e) {
          // Ctrl/Cmd/Shift-click adds/toggles this track in the
          // multi-selection; plain click replaces. The parent
          // interprets the additive flag.
          var additive = !!(e && (e.metaKey || e.ctrlKey || e.shiftKey));
          onSelectInstance && onSelectInstance(inst.instance_id, additive);
        },
        style: {
          display: "flex", alignItems: "center", height: rowHeight + 2,
          background: isSel ? "#262626" : (isHover ? "#1f1f1f" : "transparent"),
          cursor: "pointer", userSelect: "none",
        },
      }, [
        h("div", {
          key: "lbl", style: {
            width: labelColW, paddingLeft: 8, paddingRight: 6,
            color: "#ddd", fontSize: 11,
            fontFamily: "ui-monospace, monospace",
            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
            borderLeft: "3px solid " + classColor(inst.label),
          },
        }, labelText),
        h("svg", {
          key: "strip", width: stripW, height: rowHeight,
          style: { background: "#141414" },
        }, runRects),
      ]);
    });

    return h("div", {
      style: { position: "relative", width: width, maxHeight: maxHeight,
               overflowY: "auto", borderTop: "1px solid #2c2c2c" },
    }, [
      h("div", { key: "rows" }, rows),
      indicatorPx != null
        ? h("div", {
            key: "indicator",
            style: {
              position: "absolute", top: 0, bottom: 0,
              left: labelColW + indicatorPx, width: 2,
              background: "#5af", opacity: 0.7, pointerEvents: "none",
            },
          })
        : null,
    ]);
  }

  // Class legend — small swatches per class observed in the payload.
  function ClassLegend(props) {
    var instances = (props.payload && props.payload.instances) || [];
    var counts = {};
    instances.forEach(function (inst) {
      var lbl = inst.label || "unknown";
      counts[lbl] = (counts[lbl] || 0) + 1;
    });
    var labels = Object.keys(counts).sort();
    if (!labels.length) return null;
    return h("div", {
      style: { display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center",
               fontFamily: "ui-sans-serif, system-ui", fontSize: 11, color: "#bbb" },
    }, labels.map(function (lbl) {
      return h("span", {
        key: "lg-" + lbl,
        style: { display: "inline-flex", alignItems: "center", gap: 4 },
      }, [
        h("span", {
          key: "sw",
          style: {
            display: "inline-block", width: 10, height: 10, borderRadius: 2,
            background: classColor(lbl),
          },
        }),
        h("span", { key: "tx" }, lbl + " (" + counts[lbl] + ")"),
      ]);
    }));
  }


  // ---------------------------------------------------------------------------
  // Trajectories tab (stub) — Stage 1
  //
  // Will host the ephemeral-tracklet workflow: Build trajectories (extract
  // from the tracking dataset into an ExecutionStore), Filter trajectories
  // (select tracks meeting one or more conditions), a saved-filters picker,
  // and an in-panel SVG grid of the resulting tracklets. For now it is an
  // inert placeholder so the tab shell can be wired and verified.
  // ---------------------------------------------------------------------------
  // Voxel51 brand primary (design-system tokens aren't importable from a
  // standalone UMD plugin, so the canonical hex is inlined here).
  var V51_ORANGE = "#FF6D04";

  var BTN_STYLE = {
    background: V51_ORANGE, color: "#fff", border: "1px solid #444",
    borderRadius: 4, padding: "5px 10px", cursor: "pointer",
    fontFamily: "ui-sans-serif, system-ui", fontSize: 12,
  };

  var BTN_DISABLED = Object.assign({}, BTN_STYLE, {
    background: "#333", color: "#888", cursor: "not-allowed",
  });

  // Small ego-relative BEV plot of one tracklet's path, drawn from the
  // per-frame XY arrays the build operator serialized. Same axis
  // convention as the Scene tab's BEVChart (forward = up, +left = left).
  function TrajectoryThumb(props) {
    var t = props.tracklet;
    var size = props.size || 132;
    var isWorld = props.frame === "world";
    // World frame: plot the absolute world-frame path (ego origin is not
    // meaningful). Ego/base frame: ego sits at its base-frame origin so plot
    // its scene-local path; objects plot their base-frame path relative to
    // the ego at origin.
    var xy = isWorld
      ? (t.xy_world || [])
      : ((t.kind === "ego" && t.xy_scene_local && t.xy_scene_local.length)
          ? t.xy_scene_local : (t.xy_base || []));

    var xs = [], ys = [];
    for (var i = 0; i < xy.length; i++) {
      if (isFiniteNum(xy[i][0]) && isFiniteNum(xy[i][1])) {
        xs.push(xy[i][0]); ys.push(xy[i][1]);
      }
    }
    var bg = { width: size, height: size, background: "#0a0a0a",
               borderRadius: 3, display: "block" };
    if (xs.length === 0) return h("svg", { width: size, height: size, style: bg });

    var xMin = Math.min.apply(null, xs), xMax = Math.max.apply(null, xs);
    var yMin = Math.min.apply(null, ys), yMax = Math.max.apply(null, ys);
    // Objects (ego/base frame only): force-include the ego origin so the path
    // reads relative to it. In world frame the origin carries no meaning.
    if (t.kind !== "ego" && !isWorld) {
      xMin = Math.min(xMin, 0); xMax = Math.max(xMax, 0);
      yMin = Math.min(yMin, 0); yMax = Math.max(yMax, 0);
    }
    var span = Math.max(xMax - xMin, yMax - yMin, 1);
    var pad = 0.12 * span;
    xMin -= pad; xMax += pad; yMin -= pad; yMax += pad;
    var dX = xMax - xMin, dY = yMax - yMin;
    var s = Math.min(size / dY, size / dX);
    var ox = (size - s * dY) / 2, oy = (size - s * dX) / 2;
    function proj(x, y) {
      // forward (x) → screen y inverted; left (y) → screen x inverted.
      return [ox + s * (yMax - y), oy + s * (xMax - x)];
    }
    var pts = [];
    for (var k = 0; k < xs.length; k++) {
      var p = proj(xs[k], ys[k]);
      pts.push(p[0].toFixed(1) + "," + p[1].toFixed(1));
    }
    var color = instanceColor(t.tracking_name, t.instance_id);
    var p0 = proj(xs[0], ys[0]);
    var pN = proj(xs[xs.length - 1], ys[ys.length - 1]);
    var origin = proj(0, 0);
    var kids = [
      h("polyline", { key: "path", points: pts.join(" "), fill: "none",
                      stroke: color, strokeWidth: 1.6 }),
      h("circle", { key: "start", cx: p0[0], cy: p0[1], r: 3, fill: "none",
                    stroke: color, strokeWidth: 1.4 }),
      h("circle", { key: "end", cx: pN[0], cy: pN[1], r: 2.4, fill: color }),
    ];
    if (t.kind !== "ego" && !isWorld) {
      kids.push(h("circle", { key: "ego", cx: origin[0], cy: origin[1], r: 2.6,
                              fill: "#2bff7f", stroke: "#0a0a0a",
                              strokeWidth: 0.7 }));
    }
    return h("svg", { width: size, height: size, style: bg }, kids);
  }

  // ---------------------------------------------------------------------------
  // Trajectories tab — build ephemeral tracklets and browse them as a grid.
  // ---------------------------------------------------------------------------
  function TrajectoriesTab(props) {
    var getTrajOp = foo.useOperatorExecutor(OP("get_trajectories"));
    var viewPatchesOp = foo.useOperatorExecutor(OP("view_track_patches"));
    var filterOp = foo.useOperatorExecutor(OP("filter_trajectories"));
    var clearFilterOp = foo.useOperatorExecutor(OP("clear_trajectory_filter"));
    var listFiltersOp = foo.useOperatorExecutor(OP("list_trajectory_filters"));
    var deleteFilterOp = foo.useOperatorExecutor(OP("delete_trajectory_filter"));
    var tagTrajOp = foo.useOperatorExecutor(OP("tag_trajectories"));
    var exportTrajOp = foo.useOperatorExecutor(OP("export_trajectories"));
    var promptInput = foo.usePromptOperatorInput();

    var [rows, setRows] = useState([]);
    var [meta, setMeta] = useState(null);
    var [filterInfo, setFilterInfo] = useState(null);
    var [loaded, setLoaded] = useState(false);
    var [refreshTick, setRefreshTick] = useState(0);
    var [savedFilters, setSavedFilters] = useState([]);
    var [selectedSaved, setSelectedSaved] = useState("");
    var [trajFrame, setTrajFrame] = useState("base");  // "base" (ego) | "world"
    // Which scene(s) the grid shows; "__all__" = every built scene.
    var [trajScene, setTrajScene] = useState("__all__");
    // Multi-select for tag/export: a Set of "scene:track_idx" keys.
    // ctrl/cmd-click toggles one; shift-click range-selects from the
    // last-clicked anchor; plain click still opens the track's patches.
    var [selected, setSelected] = useState(function () { return new Set(); });
    var [tagText, setTagText] = useState("");
    var anchorRef = useRef(null);

    // Refresh helper: bumping refreshTick re-runs get_trajectories AND
    // list_trajectory_filters. Passed as the completion callback to every
    // mutating operator below, so the grid + saved-filters bar update as
    // soon as the operator finishes — no polling, no manual refresh.
    function bumpRefresh() { setRefreshTick(function (x) { return x + 1; }); }

    // Fire get_trajectories on scene change / explicit refresh.
    useEffect(function () {
      var sc = trajScene === "__all__" ? null : trajScene;
      try { getTrajOp.execute({ scene_name: sc }); }
      catch (e) { console.error("[obj-track] get_trajectories throw", e); }
    }, [trajScene, refreshTick]);

    // Consume get_trajectories result.
    var lastConsumed = useRef(null);
    useEffect(function () {
      var r = getTrajOp.result;
      if (!r || r === lastConsumed.current) return;
      lastConsumed.current = r;
      var out = r.result || r;
      setRows((out && out.tracklets) || []);
      setMeta((out && out.meta) || null);
      setFilterInfo((out && out.filter) || null);
      setLoaded(true);
    }, [getTrajOp.result]);

    // Fetch saved filters on mount + on refresh (after save/delete).
    useEffect(function () {
      try { listFiltersOp.execute({}); }
      catch (e) { console.error("[obj-track] list_trajectory_filters throw", e); }
    }, [refreshTick]);

    var lastFiltersConsumed = useRef(null);
    useEffect(function () {
      var r = listFiltersOp.result;
      if (!r || r === lastFiltersConsumed.current) return;
      lastFiltersConsumed.current = r;
      var out = r.result || r;
      setSavedFilters((out && out.filters) || []);
    }, [listFiltersOp.result]);

    function openCellPatches(t) {
      if (t.kind === "ego" || !t.scene_name) return;  // ego has no detections
      try {
        viewPatchesOp.execute({
          scene_name: t.scene_name, instance_ids: [t.instance_id],
        });
      } catch (e) { console.error("[obj-track] view_track_patches throw", e); }
    }

    // Filter selection (Stage 3): when active, show only matched tracklets.
    var filterActive = !!(filterInfo && filterInfo.active);
    var shown = filterActive ? rows.filter(function (t) { return t._matched; }) : rows;

    // ----- multi-selection (tag / export) -----
    function selKey(t) { return (t.scene_name || "") + ":" + t.track_idx; }

    function onCellClick(t, idx, e) {
      var shift = e && e.shiftKey;
      var ctrl = e && (e.ctrlKey || e.metaKey);
      if (shift && anchorRef.current != null) {
        var a = Math.min(anchorRef.current, idx);
        var b = Math.max(anchorRef.current, idx);
        setSelected(function (prev) {
          var next = new Set(prev);
          for (var i = a; i <= b; i++) {
            var tt = shown[i];
            if (tt && tt.kind !== "ego") next.add(selKey(tt));
          }
          return next;
        });
        return;
      }
      if (ctrl) {
        anchorRef.current = idx;
        setSelected(function (prev) {
          var next = new Set(prev);
          var k = selKey(t);
          if (next.has(k)) next.delete(k);
          else if (t.kind !== "ego") next.add(k);
          return next;
        });
        return;
      }
      anchorRef.current = idx;
      openCellPatches(t);  // plain click preserves the patches behavior
    }

    function selectAllShown() {
      setSelected(function () {
        var next = new Set();
        shown.forEach(function (t) {
          if (t.kind !== "ego") next.add(selKey(t));
        });
        return next;
      });
    }
    function clearSelection() { setSelected(new Set()); }

    // Resolve the selected keys back to {scene_name, instance_id, track_idx}
    // for the tag/export operators (from rows so selection survives filtering).
    var selItems = rows.filter(function (r) {
      return r.kind !== "ego" && selected.has(selKey(r));
    }).map(function (r) {
      return { scene_name: r.scene_name, instance_id: r.instance_id,
               track_idx: r.track_idx };
    });
    var nSel = selItems.length;

    function applyTags(mode) {
      var tags = tagText.split(",").map(function (s) { return s.trim(); })
        .filter(function (s) { return s.length; });
      if (!tags.length || nSel === 0) return;
      try {
        tagTrajOp.execute({ selection: selItems, tags: tags, mode: mode },
                          { callback: bumpRefresh });
      } catch (e) { console.error("[obj-track] tag_trajectories throw", e); }
    }

    function exportSelected() {
      if (nSel === 0) return;
      try {
        exportTrajOp.execute({ selection: selItems, include_xy: true });
      } catch (e) { console.error("[obj-track] export_trajectories throw", e); }
    }

    var status = !loaded ? "Loading…"
      : (rows.length === 0
          ? "No trajectories built yet."
          : (filterActive
              ? (shown.length + " of " + rows.length + " match the filter")
              : (rows.length + " trajectories"
                 + (meta && meta.scenes ? " · " + meta.scenes.length + " scene(s)" : ""))));

    function frameBtn(mode, label) {
      var active = trajFrame === mode;
      return h("button", {
        key: "frame-" + mode,
        onClick: function () { setTrajFrame(mode); },
        title: mode === "world"
          ? "Plot absolute world-frame paths"
          : "Plot ego-relative (base-frame) paths",
        style: {
          background: active ? V51_ORANGE : "transparent",
          color: active ? "#fff" : "#aaa",
          border: "1px solid " + (active ? V51_ORANGE : "#444"),
          borderRadius: 4, padding: "4px 8px", cursor: "pointer",
          fontFamily: "ui-sans-serif, system-ui", fontSize: 11,
        },
      }, label);
    }

    var toolbar = h("div", {
      key: "traj-toolbar",
      style: { display: "flex", gap: 8, padding: "8px 12px",
               alignItems: "center", borderBottom: "1px solid #2c2c2c",
               background: "#171717", flexWrap: "wrap" },
    }, [
      h("button", {
        key: "build", style: BTN_STYLE,
        title: "Extract trajectories from this dataset",
        onClick: function () {
          try {
            promptInput(OP("build_trajectories"),
                        { scene: trajScene || "__all__" },
                        { callback: bumpRefresh });
          } catch (e) { console.error("[obj-track] prompt build throw", e); }
        },
      }, "Build trajectories"),
      h("button", {
        key: "filter",
        style: (rows.length === 0) ? BTN_DISABLED : BTN_STYLE,
        disabled: rows.length === 0,
        title: (rows.length === 0)
          ? "Build trajectories first"
          : "Select trajectories matching one or more conditions",
        onClick: function () {
          if (rows.length === 0) return;
          try {
            promptInput(OP("filter_trajectories"), {}, { callback: bumpRefresh });
          } catch (e) { console.error("[obj-track] prompt filter throw", e); }
        },
      }, "Filter trajectories"),
      filterActive
        ? h("button", {
            key: "clear", style: BTN_STYLE, title: "Clear the active filter",
            onClick: function () {
              try {
                clearFilterOp.execute({}, { callback: bumpRefresh });
              } catch (e) { console.error("[obj-track] clear filter throw", e); }
            },
          }, "Clear filter")
        : null,
      h("button", {
        key: "refresh", style: BTN_STYLE, title: "Reload from the store",
        onClick: function () { setRefreshTick(function (x) { return x + 1; }); },
      }, "↻"),
      h("label", { key: "scene-pick", style: { display: "flex", gap: 4,
                   alignItems: "center", marginLeft: 4, fontSize: 11,
                   color: "#888", fontFamily: "ui-sans-serif, system-ui" } }, [
        "Scene:",
        h("select", {
          key: "scene-sel", value: trajScene,
          style: { background: "#222", color: "#eee", border: "1px solid #444",
                   borderRadius: 4, fontSize: 11, maxWidth: 220 },
          title: "Which scene(s) the grid shows",
          onChange: function (e) { setTrajScene(e.target.value); clearSelection(); },
        }, [h("option", { key: "__all__", value: "__all__" }, "All scenes")].concat(
          (((meta && meta.scenes) || [])).map(function (s) {
            return h("option", { key: s, value: s }, s);
          })
        )),
      ]),
      h("span", { key: "frame-toggle", style: { display: "flex", gap: 4,
                   alignItems: "center", marginLeft: 4 } }, [
        h("span", { key: "lbl", style: { fontSize: 11, color: "#888",
                     fontFamily: "ui-sans-serif, system-ui" } }, "Frame:"),
        frameBtn("base", "Ego"), frameBtn("world", "World"),
      ]),
      h("span", { key: "status", style: { marginLeft: 8, fontSize: 12,
                   color: "#aaa", fontFamily: "ui-sans-serif, system-ui" } },
        status + (getTrajOp.isExecuting ? " · working…" : "")),
      h("label", { key: "saved", style: { marginLeft: "auto", fontSize: 12,
                    color: "#ddd", fontFamily: "ui-sans-serif, system-ui",
                    display: "flex", alignItems: "center", gap: 6 } }, [
        "Saved filter: ",
        h("select", {
          key: "saved-sel", value: selectedSaved,
          style: { background: "#222", color: "#eee", border: "1px solid #444" },
          onChange: function (e) {
            // Selecting a saved filter no longer auto-applies it; the user
            // clicks "Apply filter" to run it (explicit, predictable).
            setSelectedSaved(e.target.value);
          },
        }, [h("option", { key: "_none", value: "" }, "(none)")].concat(
          savedFilters.map(function (f) {
            return h("option", { key: f.name, value: f.name }, f.name);
          })
        )),
        h("button", {
          key: "apply-saved",
          style: selectedSaved ? BTN_STYLE : BTN_DISABLED,
          disabled: !selectedSaved,
          title: selectedSaved
            ? "Apply the selected saved filter to the grid"
            : "Pick a saved filter first",
          onClick: function () {
            if (!selectedSaved) return;
            var spec = null;
            for (var i = 0; i < savedFilters.length; i++) {
              if (savedFilters[i].name === selectedSaved) {
                spec = savedFilters[i]; break;
              }
            }
            if (!spec) return;
            try {
              filterOp.execute({ combinator: spec.combinator,
                                 conditions: spec.conditions },
                               { callback: bumpRefresh });
            } catch (err) { console.error("[obj-track] apply saved throw", err); }
          },
        }, "Apply filter"),
        selectedSaved
          ? h("button", {
              key: "del-saved",
              style: Object.assign({}, BTN_STYLE, { padding: "2px 8px",
                       background: "#5a2a2a" }),
              title: "Delete this saved filter",
              onClick: function () {
                try {
                  deleteFilterOp.execute({ name: selectedSaved },
                                         { callback: bumpRefresh });
                } catch (err) { console.error("[obj-track] delete saved throw", err); }
                setSelectedSaved("");
              },
            }, "✕")
          : null,
      ]),
    ]);

    var selBtn = function (key, label, enabled, title, onClick) {
      return h("button", {
        key: key, style: enabled ? BTN_STYLE : BTN_DISABLED,
        disabled: !enabled, title: title, onClick: onClick,
      }, label);
    };

    var selToolbar = h("div", {
      key: "sel-toolbar",
      style: { display: "flex", gap: 8, padding: "8px 12px",
               alignItems: "center", borderBottom: "1px solid #2c2c2c",
               background: "#141414", flexWrap: "wrap" },
    }, [
      h("span", { key: "lbl", style: { fontSize: 12, color: "#ddd",
                   fontFamily: "ui-sans-serif, system-ui" } },
        nSel + " selected"),
      selBtn("sel-all", "Select all", shown.length > 0,
             "Select all shown object trajectories", selectAllShown),
      selBtn("sel-clear", "Clear", nSel > 0, "Clear the selection",
             clearSelection),
      h("input", {
        key: "tag-input", value: tagText, placeholder: "tag (comma-sep)",
        onChange: function (e) { setTagText(e.target.value); },
        onKeyDown: function (e) { if (e.key === "Enter") applyTags("add"); },
        style: { background: "#222", color: "#eee", border: "1px solid #444",
                 borderRadius: 4, padding: "4px 8px", fontSize: 12,
                 fontFamily: "ui-sans-serif, system-ui", width: 150,
                 marginLeft: 4 },
      }),
      selBtn("tag-add", "Add tag", nSel > 0 && tagText.trim().length > 0,
             "Add the tag(s) to the selected trajectories",
             function () { applyTags("add"); }),
      selBtn("tag-rm", "Remove tag", nSel > 0 && tagText.trim().length > 0,
             "Remove the tag(s) from the selected trajectories",
             function () { applyTags("remove"); }),
      h("span", { key: "hint", style: { fontSize: 11, color: "#777",
                   fontFamily: "ui-sans-serif, system-ui" } },
        "ctrl/⌘-click: toggle · shift-click: range"),
      h("button", {
        key: "export",
        style: Object.assign({}, (nSel > 0) ? BTN_STYLE : BTN_DISABLED,
                             { marginLeft: "auto" }),
        disabled: nSel === 0,
        title: (nSel > 0)
          ? "Download the selected trajectories as JSON"
          : "Select trajectories first",
        onClick: exportSelected,
      }, "Export selected (.json)"),
    ]);

    // Cap the number of DOM cells we render (a flat all-scenes grid can be
    // thousands). Select-all / filter / tag / export still act on the full
    // `shown`/`rows` set — only the rendered cells are capped.
    var MAX_CELLS = 600;
    var rendered = shown.slice(0, MAX_CELLS);
    var cells = rendered.map(function (t, idx) {
      var clickable = t.kind !== "ego";
      var isSel = selected.has(selKey(t));
      var tagChips = (t.tags || []).map(function (tg, ti) {
        return h("span", { key: "tg-" + ti, style: {
          background: "#3a2a12", color: "#ffb066", borderRadius: 3,
          padding: "1px 5px", fontSize: 9, fontFamily: "ui-sans-serif, system-ui",
        } }, tg);
      });
      return h("div", {
        key: (t.scene_name || "") + ":" + t.track_idx + ":" + idx,
        onClick: function (e) { onCellClick(t, idx, e); },
        title: clickable
          ? "Click: patches · ctrl/⌘-click: select · shift-click: range"
          : "Ego trajectory",
        style: {
          position: "relative",
          background: "#161616",
          border: "1px solid " + (isSel ? V51_ORANGE : "#2c2c2c"),
          borderRadius: 4,
          padding: 8, cursor: clickable ? "pointer" : "default",
          display: "flex", flexDirection: "column", gap: 4, width: 148,
        },
      }, [
        isSel
          ? h("div", { key: "badge", style: {
              position: "absolute", top: 4, right: 4, width: 16, height: 16,
              borderRadius: 8, background: V51_ORANGE, color: "#fff",
              fontSize: 11, lineHeight: "16px", textAlign: "center",
              fontFamily: "ui-sans-serif, system-ui" } }, "✓")
          : null,
        h(TrajectoryThumb, { key: "thumb", tracklet: t, size: 130,
                             frame: trajFrame }),
        h("div", { key: "lbl", style: { fontSize: 11, color: "#eee",
                    fontFamily: "ui-sans-serif, system-ui",
                    whiteSpace: "nowrap", overflow: "hidden",
                    textOverflow: "ellipsis" } },
          (t.tracking_name || "?") + " #" + t.track_idx),
        h("div", { key: "meta", style: { fontSize: 10, color: "#999",
                    fontFamily: "ui-monospace, monospace" } },
          t.n_frames + "f · gap " + (t.max_gap_s || 0).toFixed(1) + "s"
          + (t.n_distinct_classes > 1 ? " · ⚠cls" : "")),
        (trajScene === "__all__")
          ? h("div", { key: "scene", title: t.scene_name, style: { fontSize: 9,
                color: "#6a6a6a", fontFamily: "ui-monospace, monospace",
                whiteSpace: "nowrap", overflow: "hidden",
                textOverflow: "ellipsis" } }, t.scene_name)
          : null,
        tagChips.length
          ? h("div", { key: "tags", style: { display: "flex", flexWrap: "wrap",
                        gap: 3 } }, tagChips)
          : null,
      ]);
    });

    var grid = (shown.length === 0)
      ? h("div", { key: "empty", style: { padding: 24, color: "#888",
                    fontFamily: "ui-sans-serif, system-ui", fontSize: 13,
                    textAlign: "center" } },
          loaded
            ? (rows.length === 0
                ? "No trajectories yet. Click \"Build trajectories\" to populate this grid."
                : "No trajectories match the current filter.")
            : "Loading…")
      : h("div", { key: "grid", style: { display: "flex", flexWrap: "wrap",
                    gap: 10, padding: 12 } }, cells);

    var banner = (shown.length > MAX_CELLS)
      ? h("div", { key: "cap", style: { padding: "6px 12px", fontSize: 12,
            color: "#ffb066", background: "#241a0c",
            borderBottom: "1px solid #2c2c2c",
            fontFamily: "ui-sans-serif, system-ui" } },
          "Showing " + MAX_CELLS + " of " + shown.length
          + " trajectories — pick a scene or apply a filter to narrow. "
          + "Select all / tag / export still act on all " + shown.length + ".")
      : null;

    return h("div", { style: { display: "flex", flexDirection: "column" } },
             [toolbar, selToolbar, banner, grid]);
  }


  // ---------------------------------------------------------------------------
  // Clusters tab — DTW + hierarchical clustering dendrogram.
  // ---------------------------------------------------------------------------

  // Distinct hue per (1-based) cluster id; gray for the un-clustered "trunk".
  function clusterColor(id) {
    if (id == null || id <= 0) return "#777";
    return "hsl(" + CLASS_HUE_PALETTE[(id - 1) % CLASS_HUE_PALETTE.length]
      + ", 65%, 55%)";
  }

  // Cut a scipy linkage matrix Z at `threshold`, client-side, into flat
  // cluster labels (1-based, indexed by observation/leaf index 0..n-1).
  // Equivalent to fcluster(Z, t=threshold, criterion="distance"); lets the
  // threshold drag re-label instantly with no server round-trip. Union-find
  // over the merges whose height <= threshold (Z rows are height-ascending).
  function cutLinkage(Z, threshold, n) {
    var parent = new Array(2 * n - 1);
    for (var i = 0; i < parent.length; i++) parent[i] = i;
    function find(x) {
      while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
      return x;
    }
    for (var k = 0; k < Z.length; k++) {
      if (Z[k][2] > threshold) continue;
      var node = n + k;
      parent[find(Z[k][0] | 0)] = node;
      parent[find(Z[k][1] | 0)] = node;
    }
    var rootToId = Object.create(null), labels = new Array(n), next = 1;
    for (var j = 0; j < n; j++) {
      var r = find(j);
      if (!(r in rootToId)) rootToId[r] = next++;
      labels[j] = rootToId[r];
    }
    return labels;
  }

  // Mirror of _math.origin_normalize: translate a path to the origin and
  // rotate so the start→end chord points along +x. Lets the preview show the
  // same shape-normalized view the clustering uses (so straight / left / right
  // separate from a common origin). Takes/returns parallel xs/ys arrays.
  function originNormalize(xs, ys) {
    var n = xs.length;
    if (n < 1) return [xs, ys];
    var px = [], py = [], i;
    for (i = 0; i < n; i++) { px.push(xs[i] - xs[0]); py.push(ys[i] - ys[0]); }
    if (n < 2) return [px, py];
    var dx = px[n - 1], dy = py[n - 1];
    var chord = Math.hypot(dx, dy);
    if (chord < 1e-6) return [px, py];
    var c = dx / chord, s = dy / chord;   // rot = [[c, s], [-s, c]]; p @ rot.T
    var rx = [], ry = [];
    for (i = 0; i < n; i++) {
      rx.push(px[i] * c + py[i] * s);
      ry.push(-px[i] * s + py[i] * c);
    }
    return [rx, ry];
  }

  function ClustersTab(props) {
    var promptInput = foo.usePromptOperatorInput();
    var getClustersOp = foo.useOperatorExecutor(OP("get_clusters"));
    var getTrajOp = foo.useOperatorExecutor(OP("get_trajectories"));
    var selectTrajOp = foo.useOperatorExecutor(OP("select_trajectories"));
    var tagTrajOp = foo.useOperatorExecutor(OP("tag_trajectories"));
    var exportTrajOp = foo.useOperatorExecutor(OP("export_trajectories"));
    var applyRunOp = foo.useOperatorExecutor(OP("cluster_trajectories"));
    var listRunsOp = foo.useOperatorExecutor(OP("list_cluster_runs"));
    var deleteRunOp = foo.useOperatorExecutor(OP("delete_cluster_run"));

    var [clustersByScene, setClustersByScene] = useState({});
    var [clusterMeta, setClusterMeta] = useState(null);
    var [viewScene, setViewScene] = useState(props.selectedScene || "");
    var [refreshTick, setRefreshTick] = useState(0);
    var [threshold, setThreshold] = useState(null);
    // Multiple clusters can be selected at once (ctrl/⌘-click to add).
    var [selectedClusterIds, setSelectedClusterIds] = useState(function () { return new Set(); });
    var [trackById, setTrackById] = useState({});
    var [tagText, setTagText] = useState("");
    var [savedRuns, setSavedRuns] = useState([]);
    var [selectedRun, setSelectedRun] = useState("");
    // Preview in normalized shape-space (default) or raw frame coordinates.
    var [previewNorm, setPreviewNorm] = useState(true);
    var [loaded, setLoaded] = useState(false);

    var svgRef = useRef(null);
    var geomRef = useRef({});
    var draggingRef = useRef(false);

    function bump() { setRefreshTick(function (x) { return x + 1; }); }
    // Members can come from several scenes (pooled "All scenes"), so key the
    // tracklet map by (scene, track_idx), not track_idx alone.
    function keyOf(scene, idx) { return scene + "::" + idx; }
    // Uniform member list; compat shim for blobs from before the members field.
    function blobMembers(b) {
      if (!b) return [];
      if (b.members) return b.members;
      return (b.track_idxs || []).map(function (ti) {
        return { scene_name: b.scene_name, track_idx: ti };
      });
    }
    var poolView = viewScene === "__all__";

    // Load stored clusters on mount / refresh (cheap read; no DTW recompute).
    useEffect(function () {
      try { getClustersOp.execute({}); }
      catch (e) { console.error("[obj-track] get_clusters throw", e); }
    }, [refreshTick]);

    var lastClusters = useRef(null);
    useEffect(function () {
      var r = getClustersOp.result;
      if (!r || r === lastClusters.current) return;
      lastClusters.current = r;
      var out = r.result || r;
      var cs = (out && out.clusters) || {};
      setClustersByScene(cs);
      setClusterMeta((out && out.clusters_meta) || null);
      setLoaded(true);
      var keys = Object.keys(cs);
      setViewScene(function (prev) {
        if (prev && cs[prev]) return prev;
        if (props.selectedScene && cs[props.selectedScene]) return props.selectedScene;
        return keys[0] || "";
      });
    }, [getClustersOp.result]);

    var blob = clustersByScene[viewScene] || null;

    // Reset the cut line + selection when the blob changes.
    useEffect(function () {
      if (blob && !blob.error) {
        setThreshold(blob.threshold);
        setSelectedClusterIds(new Set());
      }
    }, [blob]);

    // Fetch tracklets for the preview / tag-export. A pooled view spans every
    // scene, so fetch all (scene_name=null); a single-scene view fetches one.
    useEffect(function () {
      if (!viewScene) return;
      try { getTrajOp.execute({ scene_name: poolView ? null : viewScene }); }
      catch (e) { console.error("[obj-track] get_trajectories throw", e); }
    }, [viewScene, refreshTick]);

    var lastTraj = useRef(null);
    useEffect(function () {
      var r = getTrajOp.result;
      if (!r || r === lastTraj.current) return;
      lastTraj.current = r;
      var out = r.result || r;
      var rows = (out && out.tracklets) || [];
      var map = {};
      rows.forEach(function (t) { map[keyOf(t.scene_name, t.track_idx)] = t; });
      setTrackById(map);
    }, [getTrajOp.result]);

    // Saved clustering runs (per user) for the recall dropdown.
    useEffect(function () {
      try { listRunsOp.execute({}); }
      catch (e) { console.error("[obj-track] list_cluster_runs throw", e); }
    }, [refreshTick]);
    var lastRuns = useRef(null);
    useEffect(function () {
      var r = listRunsOp.result;
      if (!r || r === lastRuns.current) return;
      lastRuns.current = r;
      var out = r.result || r;
      setSavedRuns((out && out.runs) || []);
    }, [listRunsOp.result]);

    // Threshold drag: window-level listeners read live geometry from geomRef.
    useEffect(function () {
      function onMove(e) {
        if (!draggingRef.current || !svgRef.current) return;
        var g = geomRef.current;
        if (!g.plotH) return;
        var rect = svgRef.current.getBoundingClientRect();
        var yView = (e.clientY - rect.top) * (g.H / rect.height);
        var d = g.dMaxH * (1 - (yView - g.top) / g.plotH);
        setThreshold(Math.max(0, Math.min(g.dMaxH, d)));
      }
      function onUp() { draggingRef.current = false; }
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
      return function () {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
    }, []);

    // Client-side cut → per-observation labels + cluster → [member] map.
    var derived = useMemo(function () {
      if (!blob || blob.error || threshold == null) return null;
      var n = blob.n_clustered;
      var members = blobMembers(blob);
      var labels = cutLinkage(blob.Z, threshold, n);
      var clusters = {};
      for (var r = 0; r < n; r++) {
        var c = labels[r];
        (clusters[c] = clusters[c] || []).push(members[r]);
      }
      return { labels: labels, clusters: clusters, members: members };
    }, [blob, threshold]);

    // Union of the selected clusters' members (each {scene_name, track_idx}).
    function unionMembers(idSet) {
      if (!derived) return [];
      var out = [];
      idSet.forEach(function (cid) {
        (derived.clusters[cid] || []).forEach(function (m) { out.push(m); });
      });
      return out;
    }
    function pushSelection(idSet) {
      var sel = unionMembers(idSet).map(function (m) {
        return { scene_name: m.scene_name, track_idx: m.track_idx };
      });
      try { selectTrajOp.execute({ selection: sel, source: "cluster" }); }
      catch (e) { console.error("[obj-track] select_trajectories throw", e); }
    }
    function onClusterClick(cid, e) {
      var additive = e && (e.ctrlKey || e.metaKey || e.shiftKey);
      var prev = selectedClusterIds;
      var next;
      if (additive) {
        next = new Set(prev);
        if (next.has(cid)) next.delete(cid); else next.add(cid);
      } else {
        next = (prev.size === 1 && prev.has(cid)) ? new Set() : new Set([cid]);
      }
      setSelectedClusterIds(next);
      pushSelection(next);
    }
    function clearClusterSelection() {
      setSelectedClusterIds(new Set());
      try { selectTrajOp.execute({ selection: [] }); }
      catch (e) { console.error("[obj-track] clear selection throw", e); }
    }

    // ----- act on the selected clusters in place (tag / export) -----
    var anySel = selectedClusterIds.size > 0;
    var selMembers = unionMembers(selectedClusterIds);
    // Export works on any track (scene + track_idx). Tagging needs the FO
    // instance id and is meaningless for ego (no detection labels), so it
    // resolves through the loaded tracklets and drops ego/missing.
    var exportItems = selMembers.map(function (m) {
      return { scene_name: m.scene_name, track_idx: m.track_idx };
    });
    var tagItems = selMembers
      .map(function (m) { return trackById[keyOf(m.scene_name, m.track_idx)]; })
      .filter(function (t) { return t && t.kind !== "ego" && t.instance_id; })
      .map(function (t) {
        return { scene_name: t.scene_name, instance_id: t.instance_id,
                 track_idx: t.track_idx };
      });

    function applyClusterTags(mode) {
      var tags = tagText.split(",").map(function (s) { return s.trim(); })
        .filter(function (s) { return s.length; });
      if (!tags.length || !tagItems.length) return;
      try {
        tagTrajOp.execute({ selection: tagItems, tags: tags, mode: mode },
                          { callback: bump });  // re-fetch so tag chips refresh
      } catch (e) { console.error("[obj-track] tag_trajectories throw", e); }
    }
    function exportSelected() {
      if (!exportItems.length) return;
      try {
        exportTrajOp.execute({ selection: exportItems, include_xy: true });
      } catch (e) { console.error("[obj-track] export_trajectories throw", e); }
    }

    // ----- saved-run recall -----
    function applyRun(name) {
      var run = null;
      for (var i = 0; i < savedRuns.length; i++) {
        if (savedRuns[i].name === name) { run = savedRuns[i]; break; }
      }
      if (!run) return;
      var p = Object.assign({}, run.params || {},
        { scene: run.scene, max_tracks: run.max_tracks });
      var targetView = run.scene === "__all__" ? "__all__" : run.scene;
      try {
        applyRunOp.execute(p, { callback: function () {
          setViewScene(targetView); bump();
        } });
      } catch (e) { console.error("[obj-track] apply run throw", e); }
    }
    function deleteRun(name) {
      try { deleteRunOp.execute({ name: name }, { callback: bump }); }
      catch (e) { console.error("[obj-track] delete_cluster_run throw", e); }
    }

    // ----- toolbar -----
    var scenes = clusterMeta && clusterMeta.scenes ? clusterMeta.scenes
      : Object.keys(clustersByScene);
    function sceneLabel(s) { return s === "__all__" ? "All scenes (pooled)" : s; }
    var toolbar = h("div", {
      key: "cl-toolbar",
      style: { display: "flex", gap: 8, padding: "8px 12px",
               alignItems: "center", borderBottom: "1px solid #2c2c2c",
               background: "#171717", flexWrap: "wrap" },
    }, [
      h("button", {
        key: "cluster", style: BTN_STYLE,
        title: "Compute DTW + hierarchical clustering (pick scene/classes in the form)",
        onClick: function () {
          try {
            var preset = {};
            var sc = (viewScene && viewScene !== "__all__")
              ? viewScene : props.selectedScene;
            if (sc) preset.scene = sc;
            promptInput(OP("cluster_trajectories"), preset, { callback: bump });
          } catch (e) { console.error("[obj-track] prompt cluster throw", e); }
        },
      }, "Cluster trajectories…"),
      h("button", {
        key: "refresh", style: BTN_STYLE, title: "Reload clusters from the store",
        onClick: bump,
      }, "↻"),
      scenes.length
        ? h("label", { key: "scene-pick", style: { display: "flex", gap: 4,
              alignItems: "center", marginLeft: 4, fontSize: 11, color: "#888",
              fontFamily: "ui-sans-serif, system-ui" } }, [
            "View:",
            h("select", {
              key: "scene-sel", value: viewScene,
              style: { background: "#222", color: "#eee",
                       border: "1px solid #444", borderRadius: 4, fontSize: 11,
                       maxWidth: 240 },
              onChange: function (e) { setViewScene(e.target.value); },
            }, scenes.map(function (s) {
              return h("option", { key: s, value: s }, sceneLabel(s));
            })),
          ])
        : null,
      h("span", { key: "status", style: { marginLeft: 8, fontSize: 12,
                   color: "#aaa", fontFamily: "ui-sans-serif, system-ui" } },
        getClustersOp.isExecuting || applyRunOp.isExecuting ? "working…"
          : (derived
              ? (Object.keys(derived.clusters).length + " clusters at this cut")
              : "")),
      h("span", { key: "deleg-hint", style: { marginLeft: "auto", fontSize: 11,
                   color: "#777", fontFamily: "ui-sans-serif, system-ui" } },
        "Large runs may be scheduled — use ↻ to refresh when they finish."),
    ]);

    // ----- saved-runs bar -----
    var runsBar = h("div", {
      key: "cl-runsbar",
      style: { display: "flex", gap: 8, padding: "6px 12px",
               alignItems: "center", borderBottom: "1px solid #2c2c2c",
               background: "#141414", flexWrap: "wrap" },
    }, [
      h("span", { key: "lbl", style: { fontSize: 11, color: "#888",
        fontFamily: "ui-sans-serif, system-ui" } }, "Saved run:"),
      h("select", {
        key: "run-sel", value: selectedRun,
        style: { background: "#222", color: "#eee", border: "1px solid #444",
                 borderRadius: 4, fontSize: 11, maxWidth: 220 },
        onChange: function (e) { setSelectedRun(e.target.value); },
      }, [h("option", { key: "_none", value: "" }, "(none)")].concat(
        savedRuns.map(function (rn) {
          return h("option", { key: rn.name, value: rn.name }, rn.name);
        }))),
      h("button", {
        key: "apply-run", style: selectedRun ? BTN_STYLE : BTN_DISABLED,
        disabled: !selectedRun,
        title: selectedRun ? "Re-run this saved configuration" : "Pick a saved run",
        onClick: function () { if (selectedRun) applyRun(selectedRun); },
      }, "Apply"),
      selectedRun
        ? h("button", { key: "del-run",
            style: Object.assign({}, BTN_STYLE, { padding: "3px 8px", background: "#5a2a2a" }),
            title: "Delete this saved run",
            onClick: function () { deleteRun(selectedRun); setSelectedRun(""); },
          }, "✕")
        : null,
      h("span", { key: "hint", style: { fontSize: 11, color: "#777",
        fontFamily: "ui-sans-serif, system-ui" } },
        "Save a run from the \"Cluster trajectories…\" form (Save run as)."),
    ]);

    // ----- banner (cap / warnings) -----
    var banner = null;
    if (blob && !blob.error && (blob.capped || (blob.warnings && blob.warnings.length))) {
      var msgs = [];
      if (blob.capped) {
        msgs.push("Clustered " + blob.n_clustered + " of " + blob.n_total
          + " trajectories (longest kept; raise Max trajectories to include more).");
      }
      if (blob.warnings && blob.warnings.length) msgs.push(blob.warnings.join("; "));
      banner = h("div", { key: "cl-banner", style: { padding: "6px 12px",
          fontSize: 12, color: "#ffb066", background: "#241a0c",
          borderBottom: "1px solid #2c2c2c",
          fontFamily: "ui-sans-serif, system-ui" } }, msgs.join(" "));
    }

    // ----- empty / error states -----
    if (loaded && (!scenes.length || !blob)) {
      return h("div", { style: { display: "flex", flexDirection: "column" } }, [
        toolbar, runsBar,
        h("div", { key: "empty", style: { padding: 24, color: "#888",
            fontFamily: "ui-sans-serif, system-ui", fontSize: 13,
            textAlign: "center" } },
          "No clusters yet. Build trajectories, then click "
          + "\"Cluster trajectories…\" to group them by shape "
          + "(pick a class subset, or All scenes to pool — e.g. ego across runs)."),
      ]);
    }
    if (blob && blob.error) {
      return h("div", { style: { display: "flex", flexDirection: "column" } }, [
        toolbar, runsBar,
        h("div", { key: "err", style: { padding: 24, color: "#e07a7a",
            fontFamily: "ui-sans-serif, system-ui", fontSize: 13,
            textAlign: "center" } }, blob.error),
      ]);
    }

    // ----- dendrogram SVG -----
    var dendro = null;
    if (blob && !blob.error && derived && threshold != null) {
      var n = blob.n_clustered, ico = blob.icoord, dco = blob.dcoord,
          leaves = blob.leaves, labels = derived.labels;
      var W = 900, H = 320, mL = 46, mR = 16, mT = 14, mB = 56;
      var plotW = W - mL - mR, plotH = H - mT - mB;
      var xMax = 10 * n;
      var dMax = 0;
      for (var di = 0; di < dco.length; di++) {
        for (var dj = 0; dj < dco[di].length; dj++) {
          if (dco[di][dj] > dMax) dMax = dco[di][dj];
        }
      }
      var dMaxH = dMax > 0 ? dMax * 1.05 : 1;
      geomRef.current = { H: H, top: mT, plotH: plotH, dMaxH: dMaxH };
      var px = function (icoVal) { return mL + (icoVal / xMax) * plotW; };
      var py = function (d) { return mT + (1 - d / dMaxH) * plotH; };

      var links = ico.map(function (seg, k) {
        var d4 = dco[k];
        var below = d4[1] <= threshold;
        var color = "#5a5a5a";
        if (below) {
          var p0 = Math.round((seg[0] - 5) / 10);
          p0 = Math.max(0, Math.min(n - 1, p0));
          var cid = labels[leaves[p0]];
          color = (anySel && !selectedClusterIds.has(cid))
            ? "#3a3a3a" : clusterColor(cid);
        }
        var pts = [[seg[0], d4[0]], [seg[1], d4[1]], [seg[2], d4[2]], [seg[3], d4[3]]]
          .map(function (p) { return px(p[0]).toFixed(1) + "," + py(p[1]).toFixed(1); })
          .join(" ");
        return h("polyline", { key: "lk" + k, points: pts, fill: "none",
          stroke: color, strokeWidth: below ? 1.7 : 1.0,
          strokeOpacity: below ? 0.95 : 0.45 });
      });

      var ty = py(threshold);
      var axis = h("text", { key: "ax", x: 13, y: mT + plotH / 2, fill: "#888",
        fontSize: 10, fontFamily: "ui-sans-serif, system-ui",
        transform: "rotate(-90 13," + (mT + plotH / 2) + ")",
        textAnchor: "middle" }, "DTW distance");
      var startDrag = function (e) { draggingRef.current = true; e.preventDefault(); };
      var lineGrab = h("line", { key: "thg", x1: mL, y1: ty, x2: W - mR, y2: ty,
        stroke: "transparent", strokeWidth: 12, style: { cursor: "ns-resize" },
        onMouseDown: startDrag });
      var threshLine = h("line", { key: "th", x1: mL, y1: ty, x2: W - mR, y2: ty,
        stroke: V51_ORANGE, strokeWidth: 1.5, strokeDasharray: "5,4",
        style: { pointerEvents: "none" } });
      var handle = h("rect", { key: "thh", x: W - mR - 7, y: ty - 5, width: 14,
        height: 10, rx: 2, fill: V51_ORANGE, style: { cursor: "ns-resize" },
        onMouseDown: startDrag });
      var threshLabel = h("text", { key: "thl", x: mL + 4, y: ty - 4,
        fill: V51_ORANGE, fontSize: 10, fontFamily: "ui-monospace, monospace",
        style: { pointerEvents: "none" } }, "cut " + threshold.toFixed(1));

      dendro = h("svg", { key: "dendro", ref: svgRef,
        viewBox: "0 0 " + W + " " + H, width: "100%", height: H,
        style: { background: "#141414", border: "1px solid #2c2c2c",
                 borderRadius: 4, display: "block", maxWidth: "100%" } },
        [].concat(links, [lineGrab, threshLine, handle, threshLabel, axis]));
    }

    // ----- BEV preview: clustered paths, colored by cluster -----
    // "Normalized" applies the same origin_normalize the clustering uses, so
    // straight / left / right separate from a common origin; "Raw" shows the
    // actual frame geography. (Ego in base frame is a single point — use World
    // or Scene-local for ego.)
    var preview = null;
    if (blob && !blob.error && derived) {
      var frameKey = blob.frame === "base" ? "xy_base"
        : (blob.frame === "scene_local" ? "xy_scene_local" : "xy_world");
      var paths = [];
      var pxMin = Infinity, pxMax = -Infinity, pyMin = Infinity, pyMax = -Infinity;
      var pvMembers = derived.members;
      for (var ri = 0; ri < pvMembers.length; ri++) {
        var pm = pvMembers[ri];
        var tk = trackById[keyOf(pm.scene_name, pm.track_idx)];
        if (!tk) continue;
        var xy = tk[frameKey] || [];
        var xs = [], ys = [];
        for (var pi = 0; pi < xy.length; pi++) {
          if (isFiniteNum(xy[pi][0]) && isFiniteNum(xy[pi][1])) {
            xs.push(xy[pi][0]); ys.push(xy[pi][1]);
          }
        }
        if (xs.length < 2) continue;
        if (previewNorm) { var nn = originNormalize(xs, ys); xs = nn[0]; ys = nn[1]; }
        for (var b = 0; b < xs.length; b++) {
          if (xs[b] < pxMin) pxMin = xs[b];
          if (xs[b] > pxMax) pxMax = xs[b];
          if (ys[b] < pyMin) pyMin = ys[b];
          if (ys[b] > pyMax) pyMax = ys[b];
        }
        paths.push({ xs: xs, ys: ys, cid: derived.labels[ri] });
      }
      var pvToggleBtn = function (norm, label) {
        var on = previewNorm === norm;
        return h("button", { key: "pv-" + label,
          onClick: function () { setPreviewNorm(norm); },
          style: { background: on ? V51_ORANGE : "transparent",
            color: on ? "#fff" : "#aaa",
            border: "1px solid " + (on ? V51_ORANGE : "#444"),
            borderRadius: 4, padding: "2px 8px", cursor: "pointer", fontSize: 11,
            fontFamily: "ui-sans-serif, system-ui" } }, label);
      };
      var pvHeader = h("div", { key: "pv-hd", style: { display: "flex", gap: 6,
        alignItems: "center" } }, [
        h("span", { key: "lbl", style: { fontSize: 11, color: "#888",
          fontFamily: "ui-sans-serif, system-ui", marginRight: 2 } }, "Preview:"),
        pvToggleBtn(true, "Normalized"), pvToggleBtn(false, "Raw"),
      ]);

      var pvSvg;
      if (paths.length && pxMax > pxMin && pyMax > pyMin) {
        var PW = 360, PH = 320, pad = 0.08;
        var spanX = pxMax - pxMin, spanY = pyMax - pyMin;
        var p1x = pxMax + pad * spanX, p1y = pyMax + pad * spanY;
        var dXp = (pxMax - pxMin) + 2 * pad * spanX;
        var dYp = (pyMax - pyMin) + 2 * pad * spanY;
        var sp = Math.min(PW / dYp, PH / dXp);
        var oxp = (PW - sp * dYp) / 2, oyp = (PH - sp * dXp) / 2;
        var projp = function (x, y) {
          // forward (x) → screen y inverted; left (y) → screen x inverted.
          return [oxp + sp * (p1y - y), oyp + sp * (p1x - x)];
        };
        var polylines = paths.map(function (pth, idx) {
          var on = !anySel || selectedClusterIds.has(pth.cid);
          var s = [];
          for (var q = 0; q < pth.xs.length; q++) {
            var pp = projp(pth.xs[q], pth.ys[q]);
            s.push(pp[0].toFixed(1) + "," + pp[1].toFixed(1));
          }
          return h("polyline", { key: "pp" + idx, points: s.join(" "),
            fill: "none", stroke: on ? clusterColor(pth.cid) : "#333",
            strokeWidth: on ? 1.5 : 0.8, strokeOpacity: on ? 0.9 : 0.4 });
        });
        // Origin marker in normalized mode (every path starts here).
        if (previewNorm) {
          var o = projp(0, 0);
          polylines = polylines.concat([h("circle", { key: "origin",
            cx: o[0].toFixed(1), cy: o[1].toFixed(1), r: 3, fill: "#2bff7f",
            stroke: "#0a0a0a", strokeWidth: 0.7 })]);
        }
        pvSvg = h("svg", { key: "preview-svg",
          viewBox: "0 0 " + PW + " " + PH, width: PW, height: PH,
          style: { background: "#0a0a0a", border: "1px solid #2c2c2c",
                   borderRadius: 4, display: "block" } }, polylines);
      } else {
        pvSvg = h("div", { key: "preview-empty", style: { width: 360,
          height: 320, display: "flex", alignItems: "center",
          justifyContent: "center", color: "#666", fontSize: 12,
          background: "#0a0a0a", border: "1px solid #2c2c2c", borderRadius: 4,
          fontFamily: "ui-sans-serif, system-ui" } },
          getTrajOp.isExecuting ? "Loading paths…" : "No paths to show");
      }
      preview = h("div", { key: "preview", style: { display: "flex",
        flexDirection: "column", gap: 6, flex: "0 0 auto" } },
        [pvHeader, pvSvg]);
    }

    // ----- cluster swatches (ctrl/⌘-click to select multiple) -----
    var swatches = null;
    if (derived) {
      var ids = Object.keys(derived.clusters).map(Number).sort(function (a, b) { return a - b; });
      var chips = ids.map(function (cid) {
        var mem = derived.clusters[cid];
        var on = selectedClusterIds.has(cid);
        return h("button", { key: "sw" + cid,
          onClick: function (e) { onClusterClick(cid, e); },
          title: "Click to select this cluster's " + mem.length
            + " trajectories · ctrl/⌘-click to add to the selection",
          style: { display: "flex", alignItems: "center", gap: 6,
            background: on ? "#1c1c1c" : "transparent",
            border: "1px solid " + (on ? V51_ORANGE : "#3a3a3a"),
            borderRadius: 4, padding: "4px 8px", cursor: "pointer",
            color: "#ddd", fontSize: 11, fontFamily: "ui-sans-serif, system-ui" } },
        [
          h("span", { key: "dot", style: { width: 10, height: 10,
            borderRadius: 5, background: clusterColor(cid),
            display: "inline-block" } }),
          "Cluster " + cid + " · " + mem.length,
        ]);
      });
      swatches = h("div", { key: "swatches", style: { display: "flex",
        gap: 6, flexWrap: "wrap", alignItems: "center" } },
        [
          h("span", { key: "lbl", style: { fontSize: 11, color: "#888",
            fontFamily: "ui-sans-serif, system-ui", marginRight: 2 } },
            "Drag the line to cut · click a cluster (ctrl/⌘-click for multiple):"),
        ].concat(chips));
    }

    // ----- selection actions: tag / export the selected cluster(s) in place -----
    var selActionBtn = function (key, label, enabled, title, onClick) {
      return h("button", { key: key, style: enabled ? BTN_STYLE : BTN_DISABLED,
        disabled: !enabled, title: title, onClick: onClick }, label);
    };
    var nClu = selectedClusterIds.size;
    var nSel = selMembers.length;
    var canTag = tagItems.length > 0;
    var canExport = exportItems.length > 0;
    var selectionActions = anySel
      ? h("div", { key: "cl-selactions", style: { display: "flex", gap: 8,
          alignItems: "center", flexWrap: "wrap", padding: "8px 10px",
          background: "#171717", border: "1px solid #2c2c2c",
          borderRadius: 4 } }, [
          h("span", { key: "lbl", style: { fontSize: 12, color: "#ddd",
            fontFamily: "ui-sans-serif, system-ui" } },
            nClu + " cluster" + (nClu === 1 ? "" : "s") + " · " + nSel
            + " trajector" + (nSel === 1 ? "y" : "ies") + " selected"),
          h("input", { key: "tag-input", value: tagText,
            placeholder: "tag (comma-sep)",
            onChange: function (e) { setTagText(e.target.value); },
            onKeyDown: function (e) {
              if (e.key === "Enter") applyClusterTags("add");
            },
            style: { background: "#222", color: "#eee", border: "1px solid #444",
              borderRadius: 4, padding: "4px 8px", fontSize: 12,
              fontFamily: "ui-sans-serif, system-ui", width: 150 } }),
          selActionBtn("tag-add", "Add tag",
            canTag && tagText.trim().length > 0,
            "Add tag(s) to the selected trajectories (written through to the "
            + "detection labels; ego has no labels to tag)",
            function () { applyClusterTags("add"); }),
          selActionBtn("tag-rm", "Remove tag",
            canTag && tagText.trim().length > 0,
            "Remove tag(s) from the selected trajectories",
            function () { applyClusterTags("remove"); }),
          selActionBtn("export", "Export (.json)", canExport,
            "Download the selected trajectories as JSON", exportSelected),
          selActionBtn("clear", "Clear", true,
            "Clear the cluster selection", clearClusterSelection),
          (tagTrajOp.isExecuting || exportTrajOp.isExecuting)
            ? h("span", { key: "busy", style: { fontSize: 11, color: "#888",
                fontFamily: "ui-sans-serif, system-ui" } }, "working…")
            : null,
        ])
      : null;

    var body = h("div", { key: "cl-body", style: { display: "flex",
      gap: 12, padding: 12, flexWrap: "wrap", alignItems: "flex-start" } }, [
      h("div", { key: "left", style: { flex: "1 1 480px", minWidth: 360,
        display: "flex", flexDirection: "column", gap: 10 } },
        [dendro, swatches, selectionActions]),
      preview,
    ]);

    return h("div", { style: { display: "flex", flexDirection: "column" } },
             [toolbar, runsBar, banner, body]);
  }


  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ModalTimelineSync — bridge to FiftyOne's native modal timeline.
  // Rendered (and its playback hooks therefore only called) on the modal
  // surface. Maps the native 1-based ordinal frame to this scene's frame_idx
  // to drive the panel, and exposes a seek fn (frame_idx → timeline %) so the
  // scrubber can move the native looker. Renders nothing.
  // ---------------------------------------------------------------------------
  function ModalTimelineSync(props) {
    var frameIndices = props.frameIndices || [];
    var onFrame = props.onFrame;
    var seekRef = props.seekRef;

    var name = "";
    try {
      name = fopb.useDefaultTimelineNameImperative
        ? fopb.useDefaultTimelineNameImperative().getName()
        : "";
    } catch (e) { /* no timeline context */ }

    var frameNumber = null;
    try {
      frameNumber = fopb.useFrameNumber ? fopb.useFrameNumber(name) : null;
    } catch (e) { /* not initialized */ }

    var viz = null;
    try {
      viz = fopb.useTimelineVizUtils ? fopb.useTimelineVizUtils(name) : null;
    } catch (e) { /* not initialized */ }

    // Native frame (1-based ordinal) → this scene's frame_idx → drive panel.
    useEffect(function () {
      if (frameNumber == null || !frameIndices.length) return;
      var pos = Math.max(0, Math.min(frameIndices.length - 1, frameNumber - 1));
      if (onFrame) onFrame(frameIndices[pos]);
    }, [frameNumber, frameIndices.length]);

    // Expose seek: frame_idx → ordinal % → seekTo (moves looker + native frame).
    useEffect(function () {
      if (!seekRef) return;
      seekRef.current = (viz && typeof viz.seekTo === "function")
        ? function (frameIdx) {
            var pos = frameIndices.indexOf(frameIdx);
            if (pos < 0) return;
            var len = frameIndices.length;
            viz.seekTo(len > 1 ? (pos / (len - 1)) * 100 : 0);
          }
        : null;
      return function () { if (seekRef) seekRef.current = null; };
    }, [viz, frameIndices]);

    return null;
  }


  // ---------------------------------------------------------------------------
  // Main panel
  // ---------------------------------------------------------------------------
  function BEVPanel(props) {
    var listScenesOp = foo.useOperatorExecutor(OP("list_tracking_scenes"));
    var resolveSceneOp = foo.useOperatorExecutor(OP("resolve_scene_for_sample"));
    var getPayloadOp = foo.useOperatorExecutor(OP("get_scene_track_payload"));
    var viewPatchesOp = foo.useOperatorExecutor(OP("view_track_patches"));
    var getCamUrlsOp = foo.useOperatorExecutor(OP("get_camera_frame_urls"));
    // Built-in operator: drives the modal looker to a given sample. Routes
    // through useSetExpandedSample, which resolves the correct group slice
    // for grouped datasets (unlike a raw modal-atom write).
    var openSampleOp = foo.useOperatorExecutor("@voxel51/operators/open_sample");

    // Surface detection. The `isModalPanel` prop is NOT passed to plugin
    // components by FiftyOne's Panel wrapper — the reliable signal is the
    // @fiftyone/spaces PanelContext scope ("modal" | "grid"), which Panel.tsx
    // always sets. Fall back to the prop if the hook is unavailable.
    var panelScope = null;
    try {
      panelScope = (fosp.usePanelContext && fosp.usePanelContext() || {}).scope;
    } catch (e) { /* spaces hook unavailable on this FOE version */ }
    var isModal = panelScope === "modal" || !!(props && props.isModalPanel);
    var bounds = (props && props.dimensions && props.dimensions.bounds) || {};

    // One-time log so we know which surface a given mount is on.
    var didLogSurfaceRef = useRef(false);
    if (!didLogSurfaceRef.current) {
      didLogSurfaceRef.current = true;
      console.log("[bev-panel] mounted on", isModal ? "modal" : "grid",
                  "surface; bounds:", bounds);
    }

    var [sceneInfo, setSceneInfo] = useState(null);    // listScenes result
    var [selectedScene, setSelectedScene] = useState(null);
    var [payloadCache, _setPayloadCache] = useState({});
    var payloadCacheRef = useRef(payloadCache);
    payloadCacheRef.current = payloadCache;
    var setPayloadCache = function (updater) {
      _setPayloadCache(function (prev) {
        var next = typeof updater === "function" ? updater(prev) : updater;
        payloadCacheRef.current = next;
        return next;
      });
    };

    var [activeTab, setActiveTab] = useState("scene");   // "scene" | "trajectories" | "clusters"
    var [viewMode, setViewMode] = useState("base");      // "base" | "world"
    var [scrubFrameIdx, setScrubFrameIdx] = useState(null);
    // Seek fn into FiftyOne's native modal timeline, populated by the modal-only
    // ModalTimelineSync child when a native timeline is present.
    var timelineSeekRef = useRef(null);
    var [hoveredInstanceId, setHoveredInstanceId] = useState(null);
    // Set of FO instance hexes the user has selected on the BEV
    // panel. Plain click → set-of-one (replace); Ctrl/Cmd/Shift-click
    // → toggle the clicked track in/out of the existing set.
    var [selectedInstanceIds, setSelectedInstanceIds] = useState(new Set());
    var handleSelectInstance = useCallback(function (id, additive) {
      if (id == null) { setSelectedInstanceIds(new Set()); return; }
      if (additive) {
        setSelectedInstanceIds(function (prev) {
          var next = new Set(prev);
          if (next.has(id)) next.delete(id); else next.add(id);
          return next;
        });
      } else {
        setSelectedInstanceIds(function (prev) {
          // Plain click on the already-sole-selected track toggles
          // it off; otherwise replace selection with just that track.
          if (prev && prev.size === 1 && prev.has(id)) return new Set();
          return new Set([id]);
        });
      }
    }, []);
    // Which group slice the inline camera thumbnail mirrors. Default
    // "image_02" for KITTI/multi-camera datasets; user can change in
    // the header dropdown. Set to null to hide the thumbnail.
    var [cameraMirrorSlice, setCameraMirrorSlice] = useState("image_02");
    // Cache: { "<scene>:<slice>": { "<frame_idx>": url } }
    var [cameraUrlsCache, _setCameraUrlsCache] = useState({});
    var cameraUrlsCacheRef = useRef(cameraUrlsCache);
    cameraUrlsCacheRef.current = cameraUrlsCache;
    var setCameraUrlsCache = function (updater) {
      _setCameraUrlsCache(function (prev) {
        var next = typeof updater === "function" ? updater(prev) : updater;
        cameraUrlsCacheRef.current = next;
        return next;
      });
    };
    var camCacheKey = (selectedScene && cameraMirrorSlice)
      ? (selectedScene + ":" + cameraMirrorSlice) : null;

    // ---- Recoil read sources ----
    // Modal-mode current sample id; only meaningful when isModal === true.
    var modalSampleId = null;
    try {
      var _msid = recoil.useRecoilValue(fos.modalSampleId);
      modalSampleId = _msid || null;
    } catch (e) { /* atom not exported on this FOE version */ }

    // Grid-mode multi-selection: Set<string>.
    var selectedSamples = new Set();
    try {
      selectedSamples =
        recoil.useRecoilValue(fos.selectedSamples) || new Set();
    } catch (e) { /* atom not exported on this FOE version */ }

    // ---- Recoil write sinks ----
    // Grid-mode: write into selectedSamples to highlight a sample without
    // opening the modal.
    var setSelectedSamples = null;
    try { setSelectedSamples = recoil.useSetRecoilState(fos.selectedSamples); }
    catch (e) { /* atom not exported */ }

    // FO 2.18 split the modal atom into multiple granular atoms, so the old
    // single-string-ID setter triggered a GraphQL failure (partial atom shape
    // can't resolve group + slice context). Instead of poking atoms directly,
    // commitJump drives the modal through the built-in `open_sample` operator
    // (openSampleOp), which routes through useSetExpandedSample and resolves
    // the group slice for grouped datasets — so scrubbing the panel now jumps
    // the modal looker to the scrubbed frame's group.

    // ---- Operator-call pattern. FOE's useOperatorExecutor uses fire-and-
    //      forget .execute() + a polled .result property; .then() chains do
    //      NOT resolve with the operator's return value. So we split each
    //      call into two effects: one fires, one consumes .result. ----

    // Fire list_tracking_scenes once on mount.
    useEffect(function () {
      try { listScenesOp.execute({}); }
      catch (e) { console.error("[bev-panel] list_tracking_scenes throw", e); }
    }, []);

    // Consume list_tracking_scenes result.
    useEffect(function () {
      var result = listScenesOp.result;
      if (!result) return;
      var out = result.result || result;
      if (out && out.error) {
        console.error("[bev-panel] list_tracking_scenes error:", out.error);
        return;
      }
      setSceneInfo(out);
      // Grid: default to the first scene. Modal: leave it for the inference
      // effect below to pick the scene that contains the open sample.
      if (!isModal && out && out.scenes && out.scenes.length && !selectedScene) {
        setSelectedScene(out.scenes[0].scene_name);
      }
    }, [listScenesOp.result]);

    // Modal: infer the scene from the open sample via a server lookup (the
    // modal has no scene dropdown). The modal's active id can be ANY slice's
    // sample (e.g. a camera) or the group id, so resolve it server-side rather
    // than guessing client-side. Re-runs as the modal navigates.
    useEffect(function () {
      if (!isModal || !modalSampleId) return;
      try { resolveSceneOp.execute({ sample_id: modalSampleId }); }
      catch (e) { console.error("[bev-panel] resolve_scene_for_sample throw", e); }
    }, [isModal, modalSampleId]);

    var lastResolvedSceneRef = useRef(null);
    useEffect(function () {
      var r = resolveSceneOp.result;
      if (!r || r === lastResolvedSceneRef.current) return;
      lastResolvedSceneRef.current = r;
      var out = r.result || r;
      if (out && out.scene_name && out.scene_name !== selectedScene) {
        setSelectedScene(out.scene_name);
      }
    }, [resolveSceneOp.result]);

    // Fire get_scene_track_payload when scene changes (deduped).
    var lastPayloadKeyRef = useRef(null);
    useEffect(function () {
      if (!selectedScene) return;
      var key = selectedScene;
      if (payloadCacheRef.current[key]) return;
      if (lastPayloadKeyRef.current === key) return;
      lastPayloadKeyRef.current = key;
      try {
        getPayloadOp.execute({ scene_name: selectedScene });
      } catch (e) {
        console.error("[bev-panel] get_scene_track_payload throw", e);
      }
    }, [selectedScene]);

    // Fire get_camera_frame_urls when (scene, slice) changes and
    // the cache doesn't already have that key. Deduped via a ref.
    var lastCamFetchKeyRef = useRef(null);
    useEffect(function () {
      if (!camCacheKey) return;
      if (cameraUrlsCacheRef.current[camCacheKey]) return;
      if (lastCamFetchKeyRef.current === camCacheKey) return;
      lastCamFetchKeyRef.current = camCacheKey;
      try {
        getCamUrlsOp.execute({
          scene_name: selectedScene,
          camera_slice: cameraMirrorSlice,
        });
      } catch (e) {
        console.error("[bev-panel] get_camera_frame_urls throw", e);
      }
    }, [camCacheKey]);

    // Consume get_camera_frame_urls result.
    var lastConsumedCamUrlsRef = useRef(null);
    useEffect(function () {
      var result = getCamUrlsOp.result;
      if (!result || result === lastConsumedCamUrlsRef.current) return;
      lastConsumedCamUrlsRef.current = result;
      var out = result.result || result;
      if (out && out.error) {
        console.warn("[bev-panel] get_camera_frame_urls:", out.error);
        return;
      }
      if (!out || !out.scene_name || !out.camera_slice) return;
      var key = out.scene_name + ":" + out.camera_slice;
      setCameraUrlsCache(function (prev) {
        var next = Object.assign({}, prev);
        next[key] = out.frame_urls || {};
        return next;
      });
    }, [getCamUrlsOp.result]);

    // Consume get_scene_track_payload result. Also surface executor-level
    // errors (the operator throwing server-side) by caching an error payload
    // so the chart shows the reason instead of hanging on "Loading scene…".
    var lastConsumedPayloadRef = useRef(null);
    useEffect(function () {
      if (getPayloadOp.error && selectedScene) {
        console.error("[bev-panel] get_scene_track_payload executor error ("
          + (isModal ? "modal" : "grid") + "):", getPayloadOp.error);
        var es = selectedScene;
        setPayloadCache(function (prev) {
          if (prev[es]) return prev;
          var next = Object.assign({}, prev);
          next[es] = { scene_name: es, error: String(getPayloadOp.error) };
          return next;
        });
        return;
      }
      var result = getPayloadOp.result;
      if (!result || result === lastConsumedPayloadRef.current) return;
      lastConsumedPayloadRef.current = result;
      var out = result.result || result;
      if (out && out.error) {
        console.error("[bev-panel] get_scene_track_payload error:", out.error);
        return;
      }
      if (!out || !out.scene_name) return;
      var key = out.scene_name;
      setPayloadCache(function (prev) {
        var next = Object.assign({}, prev); next[key] = out; return next;
      });
      if (scrubFrameIdx === null && out.frame_indices && out.frame_indices.length) {
        setScrubFrameIdx(out.frame_indices[0]);
      }
    }, [getPayloadOp.result, getPayloadOp.error]);

    var payload = payloadCache[selectedScene || ""] || null;

    // ---- Sync App → panel scrub position ----
    // Modal: drive the scrubber from the open modal sample.
    // Grid: if exactly one sample is selected and it's a lidar sample for
    // the current scene, jump the scrubber there. Multi-select is ignored
    // (size != 1 → no-op) for v1.
    useEffect(function () {
      if (!payload) return;

      if (isModal && modalSampleId) {
        var idx = (payload.lidar_sample_ids || []).indexOf(modalSampleId);
        if (idx >= 0 && payload.frame_indices[idx] !== scrubFrameIdx) {
          setScrubFrameIdx(payload.frame_indices[idx]);
        }
        return;
      }

      if (!isModal && selectedSamples && selectedSamples.size === 1) {
        var sid0 = selectedSamples.values().next().value;
        var i2 = (payload.lidar_sample_ids || []).indexOf(sid0);
        if (i2 >= 0 && payload.frame_indices[i2] !== scrubFrameIdx) {
          setScrubFrameIdx(payload.frame_indices[i2]);
        }
      }
    }, [modalSampleId, selectedSamples, payload, isModal]);

    // ---- Scrub handlers ----
    // commitJump pushes the scrubbed frame's sample back into App state,
    // via Recoil setters (no operator triggers — set_current_sample doesn't
    // exist as a built-in on this FOE version). Modal → drive modal sample;
    // grid → highlight in selectedSamples without opening the modal.
    var lastCommitted = useRef(null);
    var commitJump = useCallback(function (frameIdx) {
      if (!payload) return;
      var idx = payload.frame_indices.indexOf(frameIdx);
      if (idx < 0) return;
      var sid = payload.lidar_sample_ids[idx];
      if (!sid || sid === lastCommitted.current) return;
      lastCommitted.current = sid;

      // Grid mode: highlight the lidar sample at this frame so the user can
      // double-click into the modal manually. Modal mode: navigate the open
      // modal to this frame's group via the built-in open_sample operator
      // (mirrors a grid click on the lidar sample; group-slice aware).
      if (!isModal && setSelectedSamples) {
        setSelectedSamples(new Set([sid]));
      } else if (isModal) {
        try { openSampleOp.execute({ id: sid }); }
        catch (e) { console.error("[bev-panel] open_sample throw", e); }
      }
    }, [payload, isModal, setSelectedSamples, openSampleOp]);

    // Scrub handlers. In modal with a native timeline present, seek it (the
    // looker + our scrubber both follow, and native play/loop/speed continue).
    // Otherwise fall back to commitJump (grid highlight / modal open_sample).
    function onScrub(frameIdx) {
      setScrubFrameIdx(frameIdx);
      if (isModal && timelineSeekRef.current) timelineSeekRef.current(frameIdx);
    }
    function onCommit(frameIdx) {
      setScrubFrameIdx(frameIdx);
      if (isModal && timelineSeekRef.current) {
        timelineSeekRef.current(frameIdx);
      } else {
        commitJump(frameIdx);
      }
    }

    // ---- Header ----
    var sceneOptions = (sceneInfo && sceneInfo.scenes) || [];

    var header = h("div", {
      style: { display: "flex", gap: 12, padding: "8px 12px",
               alignItems: "center", borderBottom: "1px solid #2c2c2c",
               background: "#171717", color: "#ddd",
               fontFamily: "ui-sans-serif, system-ui", fontSize: 12 },
    }, [
      h("strong", { key: "ttl" }, "Scene"),

      // Modal infers its scene from the open sample → static label, no picker.
      (isModal || sceneOptions.length <= 1)
        ? h("span", { key: "scene-only" },
            selectedScene ? "  Scene: " + selectedScene : "")
        : h("label", { key: "scene-pick" }, [
            "  Scene: ",
            h("select", {
              key: "scene-sel",
              value: selectedScene || "",
              onChange: function (e) { setSelectedScene(e.target.value); },
              style: { background: "#222", color: "#eee", border: "1px solid #444" },
            }, sceneOptions.map(function (s) {
              return h("option", { key: s.scene_name, value: s.scene_name },
                       s.scene_name + " (" + s.n_frames + ")");
            })),
          ]),

      h("label", { key: "view-tog" }, [
        "View: ",
        h("select", {
          key: "view-sel",
          value: viewMode,
          onChange: function (e) { setViewMode(e.target.value); },
          style: { background: "#222", color: "#eee", border: "1px solid #444" },
        }, [
          h("option", { key: "base",  value: "base"  }, "Vehicle base"),
          h("option", { key: "world", value: "world" }, "World ENU"),
        ]),
      ]),

      // Class legend grows in the middle (flex: 1 to absorb space).
      h("div", {
        key: "legend-wrap",
        style: { flex: 1, display: "flex", justifyContent: "center" },
      }, h(ClassLegend, { payload: payload })),

      // Camera-mirror dropdown: picks which group slice the inline
      // thumbnail mirrors. Hidden until the scene payload has loaded so
      // we can populate the option list from the actual group slices.
      sceneInfo && sceneInfo.group_slices && sceneInfo.group_slices.all
        ? h("label", { key: "cam-pick" }, [
            "  Camera: ",
            h("select", {
              key: "cam-sel",
              value: cameraMirrorSlice || "",
              onChange: function (e) { setCameraMirrorSlice(e.target.value || null); },
              style: { background: "#222", color: "#eee", border: "1px solid #444" },
            }, [h("option", { key: "_none", value: "" }, "(none)")].concat(
              sceneInfo.group_slices.all
                .filter(function (s) { return s !== "lidar" && s !== "lidar_livox"; })
                .map(function (s) {
                  return h("option", { key: s, value: s }, s);
                })
            )),
          ])
        : null,

      // View patches button — fires view_track_patches against every
      // currently-selected track. Filters by FO Instance hex
      // (dataset-agnostic). Defaults to flattening across ALL
      // non-lidar group slices and ordering by frame_idx so the
      // resulting App grid shows temporally-ordered patches across
      // every camera. Ctrl/Cmd-click a track to add it to the
      // selection; the button caption shows the count.
      h("button", {
        key: "view-patches",
        onClick: function () {
          if (!selectedInstanceIds || selectedInstanceIds.size === 0
              || !selectedScene) return;
          try {
            viewPatchesOp.execute({
              scene_name: selectedScene,
              instance_ids: Array.from(selectedInstanceIds),
            });
          } catch (e) {
            console.error("[bev-panel] view_track_patches throw", e);
          }
        },
        disabled: !selectedInstanceIds || selectedInstanceIds.size === 0,
        title: (!selectedInstanceIds || selectedInstanceIds.size === 0)
          ? "Click a track on the BEV plot or timeline first "
            + "(Ctrl/Cmd-click to add more)"
          : "Switch the App view to per-patch crops of the "
            + (selectedInstanceIds.size === 1
                 ? "selected track"
                 : "selected " + selectedInstanceIds.size + " tracks")
            + " across all camera slices, ordered by frame_idx",
        style: {
          background: (selectedInstanceIds && selectedInstanceIds.size > 0)
            ? V51_ORANGE : "#333",
          color: "#eee", border: "1px solid #444", borderRadius: 4,
          padding: "4px 8px",
          cursor: (selectedInstanceIds && selectedInstanceIds.size > 0)
            ? "pointer" : "not-allowed",
          fontFamily: "ui-sans-serif, system-ui", fontSize: 11,
        },
      }, "View patches"),
    ]);

    // ---- Body ----
    // Dynamic sizing: full panel width, BEV chart takes the lion's share,
    // scrubber + track timeline below.
    var availW = bounds.width  || (isModal ? 1000 : 1280);
    var availH = bounds.height || (isModal ? 720  : 900);
    var contentW = Math.max(540, availW - 24);             // minus side padding
    var chartH   = Math.max(320, Math.min(640, Math.round(availH * 0.55)));
    var timelineMaxH = Math.max(160, Math.min(360, Math.round(availH * 0.3)));

    // Inline camera-mirror thumbnail: small image overlaid on the
    // top-right of the BEV chart. URL comes from the operator-fetched
    // cache, indexed by the scrubber's current frame_idx. The user
    // picks which slice via the header dropdown; "(none)" hides it.
    var camUrlMap = (camCacheKey && cameraUrlsCache[camCacheKey]) || null;
    var camFrameUrl = (camUrlMap && scrubFrameIdx != null)
      ? camUrlMap[String(scrubFrameIdx)] : null;
    var thumbW = 240, thumbH = 144;  // 5:3 aspect — generous for 1242×375 KITTI

    var bevBlock = h("div", {
      style: { padding: "12px 12px 8px", position: "relative" },
    }, [
      h(BEVChart, {
        key: "bev",
        payload: payload, viewMode: viewMode,
        currentFrameIdx: scrubFrameIdx,
        selectedInstanceIds: selectedInstanceIds,
        hoveredInstanceId: hoveredInstanceId,
        onHoverInstance: setHoveredInstanceId,
        onSelectInstance: handleSelectInstance,
        width: contentW, height: chartH,
      }),
      camFrameUrl
        ? h("div", {
            key: "cam-thumb",
            style: {
              position: "absolute",
              top: 14, right: 14,
              width: thumbW, height: thumbH,
              background: "#0a0a0a",
              border: "1px solid #2c2c2c",
              borderRadius: 3,
              overflow: "hidden",
              boxShadow: "0 2px 6px #0008",
              pointerEvents: "none",  // clicks pass through to the chart
            },
            title: cameraMirrorSlice + " @ frame " + scrubFrameIdx,
          }, [
            h("img", {
              key: "cam-img",
              src: resolveMediaSrc(camFrameUrl),
              style: { width: "100%", height: "100%", objectFit: "cover",
                       display: "block" },
            }),
            h("div", {
              key: "cam-lbl",
              style: {
                position: "absolute", left: 4, bottom: 2,
                color: "#ddd", fontSize: 9, fontFamily: "ui-monospace, monospace",
                textShadow: "0 0 3px #000, 0 0 3px #000",
              },
            }, cameraMirrorSlice + " · " + scrubFrameIdx),
          ])
        : null,
    ]);

    // Modal: bridge to FiftyOne's native timeline (the modal's own play / loop /
    // speed controls). Renders nothing — it follows the native frame to drive
    // the scrubber + BEV, and populates timelineSeekRef so dragging the scrubber
    // seeks the native looker. No custom transport controls in the panel.
    var timelineSync = isModal
      ? h(ModalTimelineSync, {
          key: "tl-sync",
          frameIndices: payload ? payload.frame_indices : [],
          onFrame: setScrubFrameIdx,
          seekRef: timelineSeekRef,
        })
      : null;

    var scrubberBlock = h("div", {
      style: { padding: "0 12px 6px" },
    }, h(Scrubber, {
      frameIndices: payload ? payload.frame_indices : [],
      currentFrameIdx: scrubFrameIdx,
      mFrameTimestamps: payload ? payload.m_frame_timestamps : [],
      onScrub: onScrub, onCommit: onCommit,
      width: contentW,
    }));

    var timelineBlock = h("div", {
      style: { padding: "0 12px 12px" },
    }, h(TrackTimeline, {
      payload: payload,
      currentFrameIdx: scrubFrameIdx,
      hoveredInstanceId: hoveredInstanceId,
      selectedInstanceIds: selectedInstanceIds,
      onHoverInstance: setHoveredInstanceId,
      onSelectInstance: handleSelectInstance,
      width: contentW,
      maxHeight: timelineMaxH,
    }));

    // ---- Tab bar ----
    // Generic Object Tracking panel: a "Scene" tab (the BEV scene viz) and a
    // "Trajectories" tab (ephemeral-tracklet workflow). Modeled on the
    // existing viewMode pattern: one state var + conditional content.
    function tabBtn(key, label) {
      var active = activeTab === key;
      return h("button", {
        key: "tab-" + key,
        onClick: function () { setActiveTab(key); },
        style: {
          background: active ? "#1c1c1c" : "transparent",
          color: active ? "#fff" : "#aaa",
          border: "none",
          borderBottom: active ? ("2px solid " + V51_ORANGE) : "2px solid transparent",
          padding: "8px 16px", cursor: "pointer",
          fontFamily: "ui-sans-serif, system-ui", fontSize: 13,
          fontWeight: active ? 600 : 400,
        },
      }, label);
    }

    var tabBar = h("div", {
      key: "tabbar",
      style: { display: "flex", gap: 4, padding: "0 8px",
               borderBottom: "1px solid #2c2c2c", background: "#141414" },
    }, [tabBtn("scene", "Scene"), tabBtn("trajectories", "Trajectories"),
        tabBtn("clusters", "Clusters")]);

    var sceneTab = h("div", { key: "scene-tab" },
                     [header, bevBlock, timelineSync, scrubberBlock,
                      timelineBlock]);

    return h("div", {
      style: {
        background: "#1c1c1c", color: "#eee", height: "100%", overflow: "auto",
      },
    }, [
      tabBar,
      activeTab === "scene"
        ? sceneTab
        : (activeTab === "clusters"
            ? h(ClustersTab, { key: "clusters-tab", selectedScene: selectedScene })
            : h(TrajectoriesTab, { key: "traj-tab", selectedScene: selectedScene })),
    ]);
  }


  // ---------------------------------------------------------------------------
  // TrajectoryRenderer — custom sample renderer for the trajectories dataset.
  //
  // Reads each sample's per-trajectory Parquet file via the
  // read_trajectory_payload operator (which caches per filepath on the
  // Python side), then renders an ego-relative BEV plot: robot rectangle
  // at origin, trajectory polyline in the class color, forward up, left
  // to the left, equal aspect, gap-aware dashing across fragments. Axis
  // convention matches makeProjector (data x = forward → screen y inv,
  // data y = left → screen x inv) so this renderer is visually
  // consistent with BEVTrackVisualization's scene view.
  //
  // Surface-conditional touches: grid is minimal (no labels); modal adds
  // axis tick labels every 5 m and a top-left legend.
  // ---------------------------------------------------------------------------
  function TrajectoryRenderer(props) {
    var ctx = (props && props.ctx) || props || {};
    var sample = ctx.sample || {};
    var surface = ctx.surface || "grid";
    var sampleId = sample.id || sample._id || null;
    var sampleKind = sample.kind || "object";
    var trackingName = sample.tracking_name || "unknown";

    var readOp = foo.useOperatorExecutor(OP("read_trajectory_payload"));
    var executedForRef = useRef(null);

    useEffect(function () {
      if (!sampleId) return;
      if (executedForRef.current === sampleId) return;
      executedForRef.current = sampleId;
      try {
        readOp.execute({ sample_id: sampleId });
      } catch (e) {
        console.error("[trajectory-renderer] execute throw", e);
      }
    }, [sampleId]);

    var payload = useMemo(function () {
      var r = readOp.result;
      if (!r) return null;
      return r.result || r;
    }, [readOp.result]);

    // Choose which coord arrays to plot. Prefer sample.kind, but if the
    // base-frame arrays are degenerate zeros (the ego case), fall back to
    // the scene-local arrays so the cell isn't blank.
    var traj = useMemo(function () {
      if (!payload || payload.error) return payload;
      var k = sampleKind;
      var xs, ys;
      if (k === "ego") {
        xs = payload.x_scene_local || [];
        ys = payload.y_scene_local || [];
      } else {
        xs = payload.x_base || [];
        ys = payload.y_base || [];
        if (xs.length && payload.x_scene_local && payload.x_scene_local.length) {
          var allZero = true;
          for (var ii = 0; ii < xs.length; ii++) {
            if (Math.abs(xs[ii]) > 1e-6 || Math.abs(ys[ii]) > 1e-6) {
              allZero = false; break;
            }
          }
          if (allZero) {
            xs = payload.x_scene_local;
            ys = payload.y_scene_local;
            k = "ego";
          }
        }
      }
      return {
        kind: k,
        xs: xs,
        ys: ys,
        fragments: payload.fragment_ids || [],
        color: payload.color_hex || "#cccccc",
        ego: payload.ego_size_lwh_m || [24, 2.9, 4],
      };
    }, [payload, sampleKind]);

    var containerStyle = {
      width: "100%",
      height: "100%",
      background: "#0a0a0a",
      position: "relative",
      overflow: "hidden",
    };

    if (!sampleId) {
      return h("div", { style: containerStyle }, [
        h("div", { style: { color: "#666", fontSize: 11, padding: 6 } }, ["no sample id"]),
      ]);
    }
    if (!payload) {
      return h("div", { style: containerStyle }, [
        h("div", { style: { color: "#666", fontSize: 11, padding: 6 } }, ["Loading trajectory…"]),
      ]);
    }
    if (payload.error) {
      return h("div", { style: containerStyle }, [
        h("div", { style: { color: "#c66", fontSize: 11, padding: 6 } }, [
          "Error: " + payload.error,
        ]),
      ]);
    }

    // Bounds: trajectory AABB + 1 m pad, with a 5 m minimum half-extent so
    // stationary tracks aren't a single pixel. For non-ego trajectories also
    // force-include the ego footprint so the robot at origin is always
    // visible relative to the path.
    var xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (var i = 0; i < traj.xs.length; i++) {
      var x = traj.xs[i], y = traj.ys[i];
      if (isFiniteNum(x) && isFiniteNum(y)) {
        if (x < xMin) xMin = x;
        if (x > xMax) xMax = x;
        if (y < yMin) yMin = y;
        if (y > yMax) yMax = y;
      }
    }
    if (!isFinite(xMin)) { xMin = -1; xMax = 1; yMin = -1; yMax = 1; }

    if (traj.kind !== "ego") {
      var halfL0 = traj.ego[0] / 2;
      var halfW0 = traj.ego[1] / 2;
      if (-halfL0 < xMin) xMin = -halfL0;
      if (halfL0  > xMax) xMax = halfL0;
      if (-halfW0 < yMin) yMin = -halfW0;
      if (halfW0  > yMax) yMax = halfW0;
    }

    var cxData = (xMin + xMax) / 2;
    var cyData = (yMin + yMax) / 2;
    var halfX  = Math.max(5, (xMax - xMin) / 2 + 1);
    var halfY  = Math.max(5, (yMax - yMin) / 2 + 1);
    var bounds = {
      xMin: cxData - halfX, xMax: cxData + halfX,
      yMin: cyData - halfY, yMax: cyData + halfY,
    };

    var W = 200, H = 200;
    var p = makeProjector(bounds, W, H, 0, 0, 1);
    var originPx = p.project(0, 0);

    var children = [
      h("line", { key: "ax", x1: 0, y1: originPx[1], x2: W, y2: originPx[1],
                  stroke: "#2a2a2a", strokeWidth: 1 }),
      h("line", { key: "ay", x1: originPx[0], y1: 0, x2: originPx[0], y2: H,
                  stroke: "#2a2a2a", strokeWidth: 1 }),
    ];

    // Ego rectangle (object kinds only — ego samples render the moving
    // path itself, not a rectangle at origin).
    if (traj.kind !== "ego") {
      var hl = traj.ego[0] / 2;
      var hw = traj.ego[1] / 2;
      var egoCorners = [
        p.project( hl,  hw), p.project( hl, -hw),
        p.project(-hl, -hw), p.project(-hl,  hw),
      ];
      var egoPts = egoCorners.map(function (c) {
        return c[0].toFixed(1) + "," + c[1].toFixed(1);
      }).join(" ");
      children.push(h("polygon", {
        key: "ego", points: egoPts,
        fill: "rgba(0, 255, 255, 0.18)",
        stroke: "#00ffff", strokeWidth: 1,
      }));
    }

    // Per-fragment solid polylines + dashed bridges across gaps.
    var n = traj.xs.length;
    var fragMap = {};
    for (var fi = 0; fi < n; fi++) {
      var fid = String(traj.fragments[fi] !== undefined ? traj.fragments[fi] : 0);
      if (!fragMap[fid]) fragMap[fid] = [];
      fragMap[fid].push(fi);
    }
    var fragKeys = Object.keys(fragMap).sort(function (a, b) { return (+a) - (+b); });
    for (var fk = 0; fk < fragKeys.length; fk++) {
      var idxs = fragMap[fragKeys[fk]];
      if (idxs.length === 1) {
        var sp = p.project(traj.xs[idxs[0]], traj.ys[idxs[0]]);
        children.push(h("circle", { key: "p" + fk, cx: sp[0], cy: sp[1], r: 1.5,
                                    fill: traj.color }));
      } else {
        var pts = idxs.map(function (i) {
          var pp = p.project(traj.xs[i], traj.ys[i]);
          return pp[0].toFixed(1) + "," + pp[1].toFixed(1);
        }).join(" ");
        children.push(h("polyline", {
          key: "f" + fk, points: pts, fill: "none",
          stroke: traj.color, strokeWidth: 1.6,
        }));
      }
    }
    for (var fk2 = 0; fk2 < fragKeys.length - 1; fk2++) {
      var aIdxs = fragMap[fragKeys[fk2]];
      var bIdxs = fragMap[fragKeys[fk2 + 1]];
      var a = aIdxs[aIdxs.length - 1];
      var b = bIdxs[0];
      var pa = p.project(traj.xs[a], traj.ys[a]);
      var pb = p.project(traj.xs[b], traj.ys[b]);
      children.push(h("line", {
        key: "br" + fk2,
        x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1],
        stroke: traj.color, strokeWidth: 1, opacity: 0.6,
        strokeDasharray: "3,2",
      }));
    }

    // Start (o) and end (x) markers at the first/last finite trajectory points.
    var startI = -1, endI = -1;
    for (var si = 0; si < traj.xs.length; si++) {
      if (isFiniteNum(traj.xs[si]) && isFiniteNum(traj.ys[si])) { startI = si; break; }
    }
    for (var ei = traj.xs.length - 1; ei >= 0; ei--) {
      if (isFiniteNum(traj.xs[ei]) && isFiniteNum(traj.ys[ei])) { endI = ei; break; }
    }
    if (startI >= 0) {
      var spx = p.project(traj.xs[startI], traj.ys[startI]);
      children.push(h("circle", {
        key: "start-o",
        cx: spx[0], cy: spx[1], r: 4,
        fill: "none", stroke: traj.color, strokeWidth: 1.6,
      }));
    }
    if (endI >= 0 && endI !== startI) {
      var epx = p.project(traj.xs[endI], traj.ys[endI]);
      var d = 3.5;
      children.push(h("line", {
        key: "end-x1",
        x1: epx[0] - d, y1: epx[1] - d, x2: epx[0] + d, y2: epx[1] + d,
        stroke: traj.color, strokeWidth: 1.6, strokeLinecap: "round",
      }));
      children.push(h("line", {
        key: "end-x2",
        x1: epx[0] - d, y1: epx[1] + d, x2: epx[0] + d, y2: epx[1] - d,
        stroke: traj.color, strokeWidth: 1.6, strokeLinecap: "round",
      }));
    }

    // Origin marker (ego center)
    children.push(h("circle", {
      key: "ori",
      cx: originPx[0], cy: originPx[1], r: 3,
      fill: "#2bff7f", stroke: "#0a0a0a", strokeWidth: 0.7,
    }));

    // Modal-only: 5 m tick labels along each axis + top-left legend.
    if (surface === "modal") {
      var step = 5;
      var xStart = Math.ceil(bounds.xMin / step) * step;
      for (var tx = xStart; tx <= bounds.xMax; tx += step) {
        if (tx === 0) continue;
        var pp = p.project(tx, 0);
        children.push(h("text", {
          key: "ttx" + tx, x: originPx[0] + 3, y: pp[1] - 2,
          fill: "#777", fontSize: 9,
        }, [tx + " m"]));
      }
      var yStart = Math.ceil(bounds.yMin / step) * step;
      for (var ty = yStart; ty <= bounds.yMax; ty += step) {
        if (ty === 0) continue;
        var pp2 = p.project(0, ty);
        children.push(h("text", {
          key: "tty" + ty, x: pp2[0] + 3, y: originPx[1] - 3,
          fill: "#777", fontSize: 9,
        }, [ty + " m"]));
      }
      var legend = trackingName +
        "  ·  " + (sample.n_frames != null ? sample.n_frames : "?") + " frames" +
        "  ·  " + (sample.duration_s != null
                   ? Number(sample.duration_s).toFixed(1)
                   : "?") + "s";
      children.push(h("text", {
        key: "lg", x: 6, y: 12, fill: "#ddd", fontSize: 10,
      }, [legend]));
    }

    var svg = h("svg", {
      viewBox: "0 0 " + W + " " + H,
      preserveAspectRatio: "xMidYMid meet",
      style: { width: "100%", height: "100%", display: "block" },
    }, children);

    return h("div", { style: containerStyle }, [svg]);
  }


  // ---------------------------------------------------------------------------
  // Registration
  // ---------------------------------------------------------------------------
  fop.registerComponent({
    name: "ObjectTracking",
    label: "Object Tracking",
    component: BEVPanel,
    type: fop.PluginComponentType.Panel,
    activator: function () { return true; },
    panelOptions: {
      surfaces: "grid modal",
      helpMarkdown:
        "Generic object-tracking panel.\n\n" +
        "**Scene tab:** per-scene bird's-eye-view of object trajectories " +
        "with a timeline scrubber.\n\n" +
        "**Trajectories tab:** build ephemeral tracklets from the tracking " +
        "dataset, filter them by one or more conditions, and browse them as " +
        "an in-panel grid.\n\n" +
        "**Grid surface (recommended):** scrubbing highlights the matching " +
        "lidar sample in the grid without opening the modal. Use the " +
        "**Open in modal** button to pop the looker open at the scrubbed " +
        "frame on demand.\n\n" +
        "**Modal surface:** the scrubber drives the modal's current sample " +
        "directly.\n\n" +
        "Toggle between vehicle-base and world-ENU views; hover or click " +
        "an instance to highlight its current cuboid footprint and its " +
        "2D bbox in the configured camera.",
    },
  });

  // TrajectoryRenderer has been removed — the trajectories dataset
  // now ships server-rendered PNGs as sample.filepath and FO's
  // built-in image renderer handles the grid + modal. The
  // TrajectoryRenderer function above is unused but kept for
  // reference (and in case interactive rendering is added back as an
  // opt-in later).
})();
