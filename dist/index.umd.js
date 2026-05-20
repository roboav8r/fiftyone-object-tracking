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
        background: atDefault ? "#222" : "#2a4a6a",
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
  // Main panel
  // ---------------------------------------------------------------------------
  function BEVPanel(props) {
    var listScenesOp = foo.useOperatorExecutor(OP("list_tracking_scenes"));
    var getPayloadOp = foo.useOperatorExecutor(OP("get_scene_track_payload"));
    var viewPatchesOp = foo.useOperatorExecutor(OP("view_track_patches"));
    var getCamUrlsOp = foo.useOperatorExecutor(OP("get_camera_frame_urls"));

    // FOE plugin SDK injects isModalPanel + dimensions on the panel props.
    // Default to grid behavior if the prop is missing.
    var isModal = !!(props && props.isModalPanel);
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

    var [viewMode, setViewMode] = useState("base");      // "base" | "world"
    var [scrubFrameIdx, setScrubFrameIdx] = useState(null);
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

    // FO 2.18 split the modal atom into multiple granular atoms
    // (currentModalSlice, modalGroupSlice, currentModalUniqueIdJotaiAtom,
    // useExpandSample, ...). The single-string-ID setter pattern that
    // worked in earlier versions now triggers a GraphQL failure because
    // the partial atom shape can't be resolved into a full modal state
    // (missing group + slice context). Rather than chase the right
    // multi-atom write sequence per FO version, the plugin treats modal
    // sync as App → panel only: opening the modal yourself + scrubbing
    // the panel updates the panel's BEV state, but the modal looker
    // does NOT auto-jump on scrub. For a per-frame camera view, see
    // the inline camera-mirror thumbnail (option-C).
    var setModalSample = null;

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
      if (out && out.scenes && out.scenes.length && !selectedScene) {
        setSelectedScene(out.scenes[0].scene_name);
      }
    }, [listScenesOp.result]);

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

    // Consume get_scene_track_payload result.
    var lastConsumedPayloadRef = useRef(null);
    useEffect(function () {
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
    }, [getPayloadOp.result]);

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

      // Grid mode: highlight the lidar sample at this frame so the user
      // can double-click into the modal manually. Modal mode: no-op
      // (FO 2.18 modal atom too fragile to write from a plugin; see the
      // "setModalSample = null" comment block above).
      if (!isModal && setSelectedSamples) {
        setSelectedSamples(new Set([sid]));
      }
    }, [payload, isModal, setSelectedSamples]);

    function onScrub(frameIdx) { setScrubFrameIdx(frameIdx); }
    function onCommit(frameIdx) {
      setScrubFrameIdx(frameIdx);
      commitJump(frameIdx);
    }

    // ---- Header ----
    var sceneOptions = (sceneInfo && sceneInfo.scenes) || [];

    var header = h("div", {
      style: { display: "flex", gap: 12, padding: "8px 12px",
               alignItems: "center", borderBottom: "1px solid #2c2c2c",
               background: "#171717", color: "#ddd",
               fontFamily: "ui-sans-serif, system-ui", fontSize: 12 },
    }, [
      h("strong", { key: "ttl" }, "BEV Track Visualization"),

      sceneOptions.length > 1
        ? h("label", { key: "scene-pick" }, [
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
          ])
        : h("span", { key: "scene-only" },
            selectedScene ? "  Scene: " + selectedScene : ""),

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
            ? "#2a4a6a" : "#333",
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
              src: camFrameUrl,
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

    return h("div", {
      style: {
        background: "#1c1c1c", color: "#eee", height: "100%", overflow: "auto",
      },
    }, [header, bevBlock, scrubberBlock, timelineBlock]);
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
    name: "BEVTrackVisualization",
    label: "BEV Track Visualization",
    component: BEVPanel,
    type: fop.PluginComponentType.Panel,
    activator: function () { return true; },
    panelOptions: {
      surfaces: "grid modal",
      helpMarkdown:
        "Per-scene bird's-eye-view of object trajectories with a timeline " +
        "scrubber.\n\n" +
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
