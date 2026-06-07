/**
 * ╔══════════════════════════════════════════════════════════════════════════════╗
 * ║  BRICK 5 — Asynchronous Telemetry Stream & Real-Time Orchestration Client  ║
 * ║  object-tracking-service/frontend/app.js                                   ║
 * ║                                                                            ║
 * ║  Architecture: Single-class OO orchestrator (VisionDashboardApp)           ║
 * ║  Initialised on: DOMContentLoaded                                          ║
 * ║                                                                            ║
 * ║  Calibrations: Synchronized baseline values to clear client-side freezing  ║
 * ╚══════════════════════════════════════════════════════════════════════════════╝
 */

'use strict';

// ══════════════════════════════════════════════════════════════════════════════
//  Constants
// ══════════════════════════════════════════════════════════════════════════════

const SSE_ENDPOINT        = '/stream/telemetry';
const WS_PROTOCOL         = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_ENDPOINT         = `${WS_PROTOCOL}//${window.location.host}/ws/control`;

const HEALTH_ENDPOINT     = '/api/health';
const CAMERA_ENDPOINT     = '/api/camera/status';
const MODEL_ENDPOINT      = '/api/model/info';

const HEALTH_INTERVAL_MS  = 5_000;
const SLIDER_DEBOUNCE_MS  = 100;

const BACKOFF_BASE_MS     = 500;
const BACKOFF_MULTIPLIER  = 1.8;
const BACKOFF_CEILING_MS  = 30_000;

const LOG_MAX_ENTRIES     = 120;
const REGISTRY_MAX_ROWS   = 60;

// ── FIXED SPEED TIER SYNC BOUNDS ─────────────────────────────────────────────
const FPS_TARGET          = 60;  // Synchronized to match the safe backend device pipeline profile!
const UI_SMOOTHING_FACTOR = 0.25; // Adjusted filter alpha for tighter rendering steps response
// ─────────────────────────────────────────────────────────────────────────────

const CONF_HIGH           = 0.75;
const CONF_MED            = 0.50;
const HISTOGRAM_MAX_BARS  = 8;

const WS_CONNECTING       = 0;
const WS_OPEN             = 1;
const WS_CLOSING          = 2;
const WS_CLOSED           = 3;


// ══════════════════════════════════════════════════════════════════════════════
//  Utility Functions
// ══════════════════════════════════════════════════════════════════════════════

function debounce(fn, delayMs) {
    let timer = null;
    return function debounced(...args) {
        if (timer !== null) clearTimeout(timer);
        timer = setTimeout(() => {
            timer = null;
            fn.apply(this, args);
        }, delayMs);
    };
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function nowTimestamp() {
    return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

function relativeTime(tsMs) {
    const diffSeconds = Math.floor((Date.now() - tsMs) / 1_000);
    if (diffSeconds < 2)  return 'now';
    if (diffSeconds < 60) return `${diffSeconds}s ago`;
    return `${Math.floor(diffSeconds / 60)}m ago`;
}

function confColour(conf) {
    if (conf >= CONF_HIGH) return 'text-emerald-400';
    if (conf >= CONF_MED)  return 'text-amber-400';
    return 'text-rose-400';
}


// ══════════════════════════════════════════════════════════════════════════════
//  Subsystem: EventLogger
// ══════════════════════════════════════════════════════════════════════════════

class EventLogger {
    constructor(container) {
        this._container = container;
        this._entryCount = 0;
    }

    log(message, variant = 'info') {
        if (!this._container) return;

        const colourMap = {
            info:    { ts: 'text-slate-600', msg: 'text-cyan-400/80',    prefix: '›' },
            success: { ts: 'text-slate-600', msg: 'text-emerald-400/90', prefix: '✓' },
            warn:    { ts: 'text-slate-600', msg: 'text-amber-400/80',   prefix: '!' },
            error:   { ts: 'text-slate-600', msg: 'text-rose-400',       prefix: '✗' },
        };
        const colour = colourMap[variant] ?? colourMap.info;

        const entry = document.createElement('div');
        entry.className = `flex items-start gap-2 px-4 py-1.5 log-new`;
        entry.innerHTML = `
            <span class="${colour.ts} flex-shrink-0 tabular-nums">${nowTimestamp()}</span>
            <span class="${colour.msg}">${colour.prefix} ${this._escapeHtml(message)}</span>
        `;

        this._container.insertBefore(entry, this._container.firstChild);
        this._entryCount++;

        while (this._container.children.length > LOG_MAX_ENTRIES) {
            this._container.removeChild(this._container.lastChild);
        }
    }

    clear() {
        if (!this._container) return;
        this._container.innerHTML = '';
        this._entryCount = 0;
        this.log('log cleared', 'info');
    }

    _escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  Subsystem: HistogramManager
// ══════════════════════════════════════════════════════════════════════════════

class HistogramManager {
    constructor(container) {
        this._container = container;
        this._counts = new Map();
        this._renderedClasses = new Set();
    }

    update(detections) {
        if (!this._container) return;

        const frameMap = new Map();
        for (const det of detections) {
            const label = det.class_label ?? 'unknown';
            frameMap.set(label, (frameMap.get(label) ?? 0) + 1);
            this._counts.set(label, (this._counts.get(label) ?? 0) + 1);
        }

        const sorted = [...frameMap.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, HISTOGRAM_MAX_BARS);

        const maxCount = sorted.length > 0 ? sorted[0][1] : 1;
        const newClassSet = new Set(sorted.map(([cls]) => cls));

        const classSetChanged =
            newClassSet.size !== this._renderedClasses.size ||
            [...newClassSet].some(c => !this._renderedClasses.has(c));

        if (classSetChanged) {
            this._rebuild(sorted, maxCount);
            this._renderedClasses = newClassSet;
        } else {
            this._updateBars(sorted, maxCount);
        }

        if (sorted.length === 0) {
            this._clearBars();
        }
    }

    _rebuild(sorted, maxCount) {
        this._container.innerHTML = '';
        for (const [label, count] of sorted) {
            const pct = Math.round((count / maxCount) * 100);
            const barColour = this._barColour(label);
            const row = document.createElement('div');
            row.className = 'flex items-center gap-3';
            row.dataset.classLabel = label;
            row.innerHTML = `
                <span class="text-slate-500 text-[10px] font-mono w-16 text-right truncate" title="${label}">${label}</span>
                <div class="flex-1 h-2 bg-slate-800/60 rounded-full overflow-hidden">
                    <div class="h-full rounded-full transition-all duration-300 ${barColour}"
                         data-bar style="width:${pct}%"></div>
                </div>
                <span class="text-slate-400 text-[10px] font-mono w-6 text-right" data-count>${count}</span>
            `;
            this._container.appendChild(row);
        }
    }

    _updateBars(sorted, maxCount) {
        for (const [label, count] of sorted) {
            const row = this._container.querySelector(`[data-class-label="${label}"]`);
            if (!row) continue;
            const bar   = row.querySelector('[data-bar]');
            const cnt   = row.querySelector('[data-count]');
            const pct   = Math.round((count / maxCount) * 100);
            if (bar) bar.style.width = `${pct}%`;
            if (cnt) cnt.textContent = count;
        }
    }

    _clearBars() {
        const bars = this._container.querySelectorAll('[data-bar]');
        const counts = this._container.querySelectorAll('[data-count]');
        bars.forEach(b => { b.style.width = '0%'; });
        counts.forEach(c => { c.textContent = '0'; });
    }

    _barColour(label) {
        const palette = [
            'bg-cyan-500', 'bg-emerald-500', 'bg-amber-500',
            'bg-violet-500', 'bg-rose-500', 'bg-sky-500',
            'bg-teal-500', 'bg-orange-500',
        ];
        let hash = 0;
        for (let i = 0; i < label.length; i++) {
            hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
        }
        return palette[hash % palette.length];
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  Subsystem: DOMRenderer (With Low-Pass Metric Smoothing Filters)
// ══════════════════════════════════════════════════════════════════════════════

class DOMRenderer {
    constructor() {
        this._pending = null;
        this._rafHandle = null;
        this._histogram = null;
        this._els = {};

        this._smoothedFPS = 30.0;
        this._smoothedLatency = 10.0;
        this._trackRegistry = new Map();
    }

    resolveElements() {
        const ids = [
            'metric-fps', 'metric-ids', 'metric-latency', 'metric-hw',
            'fps-bar', 'fps-target-label', 'hw-badge', 'hw-caption',
            'overlay-count', 'overlay-infer',
            'track-table-body', 'registry-count',
            'class-histogram',
            'diag-cam-idx', 'diag-cap-fps', 'diag-dropped',
            'diag-queue', 'diag-total-det', 'diag-avg-infer',
            'diag-ws', 'diag-sse',
            'hdr-model', 'hdr-device',
            'stream-res', 'stream-status-badge',
            'status-dot', 'status-label',
            'snapshot-gallery', 'snapshot-empty',
            'reg-total-seen', 'reg-active', 'reg-lost',
        ];
        for (const id of ids) {
            this._els[id] = document.getElementById(id);
        }

        if (this._els['class-histogram']) {
            this._histogram = new HistogramManager(this._els['class-histogram']);
        }
        
        if (this._els['fps-target-label']) {
            this._els['fps-target-label'].textContent = `/ ${FPS_TARGET}`;
        }
    }

    scheduleTelemetryUpdate(payload) {
        this._pending = payload;
        if (this._rafHandle === null) {
            this._rafHandle = requestAnimationFrame(() => this._flushTelemetry());
        }
    }

    writeHealthData(health, camera = null, model = null) {
        const { _els } = this;

        this._setText('hdr-model',  health.model  ?? '—');
        this._setText('hdr-device', health.device ?? '—');

        const device = (health.device ?? 'cpu').toUpperCase();
        this._setText('metric-hw', device);

        const isCpuOnly = device === 'CPU';
        if (_els['hw-badge']) {
            _els['hw-badge'].className = isCpuOnly ? 'badge badge-re' : 'badge badge-cy';
            _els['hw-badge'].textContent = device;
        }
        this._setText('hw-caption', health.camera_ready ? 'camera active' : 'camera offline');
        this._setConnectionStatus(health.camera_ready && health.tracker_ready);

        if (camera) {
            this._setText('diag-cam-idx', camera.index ?? '—');
            this._setText('stream-res',   camera.resolution ?? '—');
        }
        if (model) {
            this._setText('hdr-model', model.model_path ?? '—');
        }
    }

    writeConnectionState(channel, state) {
        const elementId = channel === 'ws' ? 'diag-ws' : 'diag-sse';
        const el = this._els[elementId];
        if (!el) return;

        const stateStyles = {
            connected:    'text-emerald-400',
            connecting:   'text-amber-400/80',
            disconnected: 'text-slate-700',
            error:        'text-rose-400',
        };
        el.className = stateStyles[state] ?? 'text-slate-600';
        el.textContent = state;
    }

    setOfflineState() {
        this._setConnectionStatus(false);
    }

    _flushTelemetry() {
        this._rafHandle = null;
        const payload = this._pending;
        this._pending = null;
        if (!payload) return;

        try {
            this._renderMetricCards(payload);
            this._renderTrackRegistry(payload);
            this._renderVideoOverlays(payload);
            this._renderSnapshots(payload);
            if (this._histogram) {
                this._histogram.update(payload.detections ?? []);
            }
        } catch (err) {
            console.warn('[DOMRenderer] Render error (suppressed):', err);
        }
    }

    _renderMetricCards(payload) {
        const rawFps     = payload.fps              ?? 0;
        const count      = payload.active_total_count ?? payload.count ?? 0;
        const rawLatency = payload.inference_latency_ms ?? 0;

        this._smoothedFPS = (rawFps * UI_SMOOTHING_FACTOR) + (this._smoothedFPS * (1 - UI_SMOOTHING_FACTOR));
        if (this._smoothedFPS < 0.1) this._smoothedFPS = 0.0;
        
        this._smoothedLatency = (rawLatency * UI_SMOOTHING_FACTOR) + (this._smoothedLatency * (1 - UI_SMOOTHING_FACTOR));

        this._setText('metric-fps', this._smoothedFPS.toFixed(1));
        
        const fpsPct = clamp(Math.round((this._smoothedFPS / FPS_TARGET) * 100), 0, 100);
        if (this._els['fps-bar']) {
            this._els['fps-bar'].style.width = `${fpsPct}%`;
            this._els['fps-bar'].className = [
                'h-full rounded-full transition-all duration-300',
                fpsPct >= 80 ? 'bg-emerald-500' : fpsPct >= 50 ? 'bg-amber-500' : 'bg-rose-500',
            ].join(' ');
        }

        this._setText('metric-ids', count);
        this._setText('metric-latency', rawLatency > 0 ? `${this._smoothedLatency.toFixed(1)}ms` : '—');
        this._setText('registry-count', count);
        
        this._setText('diag-avg-infer', rawLatency > 0 ? `${this._smoothedLatency.toFixed(1)} ms` : '—');
    }

    _renderTrackRegistry(payload) {
        const tbody = this._els['track-table-body'];
        if (!tbody) return;

        const detections = payload.detections ?? [];
        const currentTs = payload.ts ?? Date.now();
        const currentTsStr = new Date().toLocaleTimeString('en-GB', { hour12: false });

        for (const [id, track] of this._trackRegistry) {
            track.status = 'Lost';
        }

        for (const det of detections) {
            if (det.tracking_id == null) continue;
            const tid = det.tracking_id;
            const label = det.class_label ?? 'unknown';

            if (!this._trackRegistry.has(tid)) {
                this._trackRegistry.set(tid, {
                    class_label: label,
                    status: 'Active',
                    first_seen: currentTsStr,
                    last_seen: currentTsStr,
                    first_seen_ts: currentTs,
                    last_seen_ts: currentTs
                });
            } else {
                const t = this._trackRegistry.get(tid);
                t.status = 'Active';
                t.last_seen = currentTsStr;
                t.last_seen_ts = currentTs;
                t.class_label = label;
            }
        }

        if (this._trackRegistry.size > 1000) {
            const sortedEntries = [...this._trackRegistry.entries()].sort((a, b) => a[1].first_seen_ts - b[1].first_seen_ts);
            const toRemove = sortedEntries.slice(0, this._trackRegistry.size - 1000);
            for (const [tid, _] of toRemove) {
                this._trackRegistry.delete(tid);
            }
        }

        let activeCount = 0;
        let lostCount = 0;
        for (const track of this._trackRegistry.values()) {
            if (track.status === 'Active') activeCount++;
            else lostCount++;
        }

        this._setText('reg-total-seen', this._trackRegistry.size);
        this._setText('reg-active', activeCount);
        this._setText('reg-lost', lostCount);

        if (this._trackRegistry.size === 0) {
            tbody.innerHTML = `
                <div class="grid grid-cols-12 gap-x-2 px-4 py-3 text-[10px] font-mono text-slate-700 italic">
                    <span class="col-span-12 text-center">no objects detected yet</span>
                </div>`;
            return;
        }

        const sortedTracks = [...this._trackRegistry.entries()].sort((a, b) => {
            if (a[1].status !== b[1].status) {
                return a[1].status === 'Active' ? -1 : 1;
            }
            return b[1].last_seen_ts - a[1].last_seen_ts;
        });

        const fragment = document.createDocumentFragment();
        
        for (const [tid, track] of sortedTracks) {
            const rowClass = track.status === 'Active' 
                ? 'track-row grid grid-cols-12 gap-x-2 px-4 py-2 text-[10px] font-mono border-l-2 border-emerald-500/50 bg-emerald-500/5' 
                : 'track-row grid grid-cols-12 gap-x-2 px-4 py-2 text-[10px] font-mono border-l-2 border-transparent opacity-60';

            const idColours = ['text-cyan-400', 'text-emerald-400', 'text-amber-400', 'text-violet-400', 'text-rose-400', 'text-sky-400'];
            const idColour = idColours[tid % idColours.length];
            const statusColour = track.status === 'Active' ? 'text-emerald-400' : 'text-slate-500';
            
            const div = document.createElement('div');
            div.className = rowClass;
            div.innerHTML = `
                <span class="col-span-1 ${idColour} font-medium tabular-nums">${tid}</span>
                <span class="col-span-3 text-slate-300 truncate" title="${track.class_label}">${track.class_label}</span>
                <span class="col-span-2 text-center ${statusColour}">${track.status}</span>
                <span class="col-span-3 text-right text-slate-500 tabular-nums">${track.first_seen}</span>
                <span class="col-span-3 text-right text-slate-400 tabular-nums">${track.last_seen}</span>
            `;
            fragment.appendChild(div);
        }

        tbody.innerHTML = '';
        tbody.appendChild(fragment);
    }

    _renderVideoOverlays(payload) {
        const count   = payload.active_total_count ?? payload.count ?? 0;
        this._setText('overlay-count', count);
        
        const overlayInfer = this._els['overlay-infer'];
        if (overlayInfer) {
            overlayInfer.textContent = `${this._smoothedLatency.toFixed(1)}ms`;
            overlayInfer.className = [
                'font-mono text-xs tabular-nums',
                this._smoothedLatency < 20 ? 'text-emerald-400' : this._smoothedLatency < 50 ? 'text-amber-400' : 'text-rose-400',
            ].join(' ');
        }
    }

    _renderSnapshots(payload) {
        if (!payload.snapshots || payload.snapshots.length === 0) return;
        const gallery = this._els['snapshot-gallery'];
        if (!gallery) return;
        
        const emptyText = document.getElementById('snapshot-empty');
        if (emptyText) emptyText.style.display = 'none';
        
        if (!this._rendered_snapshot_ids) {
            this._rendered_snapshot_ids = new Set();
        }
        
        for (const snap of payload.snapshots) {
            if (this._rendered_snapshot_ids.has(snap.track_id)) continue;
            this._rendered_snapshot_ids.add(snap.track_id);
            
            const card = document.createElement('div');
            card.className = "relative rounded border border-rose-500/30 overflow-hidden group bg-slate-900 shadow-md shadow-rose-900/20 anim-d1";
            card.innerHTML = `
                <img src="data:image/jpeg;base64,${snap.image}" class="w-full h-auto object-cover opacity-80 group-hover:opacity-100 transition-opacity" style="aspect-ratio: 1/1;" />
                <div class="absolute bottom-0 left-0 right-0 bg-slate-950/80 px-1.5 py-1 border-t border-rose-500/30 flex justify-between items-center text-[9px] font-mono">
                    <span class="text-rose-400 font-medium">ID:${snap.track_id}</span>
                    <span class="text-slate-300 truncate ml-1">${snap.class_label}</span>
                </div>
                <div class="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <a href="data:image/jpeg;base64,${snap.image}" download="snapshot_${snap.class_label}_${snap.track_id}.jpg" class="bg-rose-600 hover:bg-rose-500 text-white p-1 rounded cursor-pointer flex items-center justify-center" title="Save Snapshot">
                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                    </a>
                </div>
            `;
            gallery.prepend(card);
        }
        
        // limit to 12 snapshots so UI doesn't lag
        while (gallery.children.length > 12) {
            gallery.removeChild(gallery.lastChild);
        }
        
        // Prevent unbounded memory growth of seen IDs
        if (this._rendered_snapshot_ids.size > 100) {
            const keep = new Set();
            for (const snap of payload.snapshots) keep.add(snap.track_id);
            this._rendered_snapshot_ids = keep;
        }
    }

    _setConnectionStatus(isActive) {
        const dot   = this._els['status-dot'];
        const label = this._els['status-label'];

        if (isActive) {
            if (dot) {
                dot.className = 'status-dot bg-emerald-500';
                dot.style.boxShadow = '0 0 6px rgba(16,185,129,0.8)';
            }
            if (label) {
                label.className = 'text-emerald-400 text-[10px] font-mono font-medium tracking-widest uppercase';
                label.textContent = 'EDGE ACTIVE';
            }
        } else {
            if (dot) {
                dot.className = 'status-dot bg-rose-500';
                dot.style.boxShadow = '0 0 6px rgba(239,68,68,0.8)';
            }
            if (label) {
                label.className = 'text-rose-400 text-[10px] font-mono font-medium tracking-widest uppercase';
                label.textContent = 'OFFLINE';
            }
        }
    }

    _setText(id, text) {
        const el = this._els[id];
        if (el) el.textContent = String(text);
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  Subsystem: SSEClient
// ══════════════════════════════════════════════════════════════════════════════

class SSEClient {
    constructor(endpoint, onPayload, onStatusChange) {
        this._endpoint       = endpoint;
        this._onPayload      = onPayload;
        this._onStatusChange = onStatusChange;
        this._source         = null;
        this._retryCount     = 0;
        this._retryTimer     = null;
        this._destroyed      = false;
    }

    connect() {
        if (this._destroyed) return;
        this._closeSource();
        this._onStatusChange('connecting', 'SSE connecting...');

        try {
            this._source = new EventSource(this._endpoint);
        } catch (err) {
            this._onStatusChange('error', `SSE init failed: ${err.message}`);
            this._scheduleReconnect();
            return;
        }

        this._source.onopen = () => {
            this._retryCount = 0;
            this._onStatusChange('connected', 'SSE stream connected');
        };

        this._source.onmessage = (event) => {
            if (!event.data) return;
            try {
                const payload = JSON.parse(event.data);
                this._onPayload(payload);
            } catch (parseError) {
                this._onStatusChange('warn', `SSE parse error: ${parseError.message}`);
            }
        };

        this._source.onerror = () => {
            const state = this._source ? this._source.readyState : -1;
            if (state === EventSource.CLOSED || state === 2) {
                this._onStatusChange('error', 'SSE connection closed — reconnecting');
                this._closeSource();
                this._scheduleReconnect();
            } else {
                this._onStatusChange('error', 'SSE error — browser retrying');
            }
        };
    }

    disconnect() {
        this._destroyed = true;
        if (this._retryTimer !== null) {
            clearTimeout(this._retryTimer);
            this._retryTimer = null;
        }
        this._closeSource();
        this._onStatusChange('disconnected', 'SSE disconnected');
    }

    _closeSource() {
        if (this._source) {
            this._source.onopen    = null;
            this._source.onmessage = null;
            this._source.onerror   = null;
            this._source.close();
            this._source = null;
        }
    }

    _scheduleReconnect() {
        if (this._destroyed) return;
        const delayMs = Math.min(
            BACKOFF_BASE_MS * Math.pow(BACKOFF_MULTIPLIER, this._retryCount),
            BACKOFF_CEILING_MS,
        );
        this._retryCount++;
        this._onStatusChange('warn', `SSE retry ${this._retryCount} in ${(delayMs / 1000).toFixed(1)}s`);
        this._retryTimer = setTimeout(() => {
            this._retryTimer = null;
            this.connect();
        }, delayMs);
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  Subsystem: WebSocketClient
// ══════════════════════════════════════════════════════════════════════════════

class WebSocketClient {
    constructor(endpoint, onMessage, onStatusChange) {
        this._endpoint       = endpoint;
        this._onMessage      = onMessage;
        this._onStatusChange = onStatusChange;
        this._socket         = null;
        this._retryCount     = 0;
        this._retryTimer     = null;
        this._destroyed      = false;
        this._sendQueue      = [];
    }

    connect() {
        if (this._destroyed) return;
        this._closeSocket();
        this._onStatusChange('connecting', 'WS connecting...');

        try {
            this._socket = new WebSocket(this._endpoint);
        } catch (err) {
            this._onStatusChange('error', `WS init failed: ${err.message}`);
            this._scheduleReconnect();
            return;
        }

        this._socket.onopen = () => {
            this._retryCount = 0;
            this._onStatusChange('connected', 'WS control channel open');
            while (this._sendQueue.length > 0) {
                const queued = this._sendQueue.shift();
                this._socket.send(queued);
            }
        };

        this._socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this._onMessage(data);
            } catch (parseError) {
                this._onStatusChange('warn', `WS parse error: ${parseError.message}`);
            }
        };

        this._socket.onclose = (event) => {
            const reason = event.reason ? ` (${event.reason})` : '';
            this._onStatusChange('error', `WS closed${reason} — reconnecting`);
            this._closeSocket();
            this._scheduleReconnect();
        };

        this._socket.onerror = () => {
            this._onStatusChange('error', 'WS transport error');
        };
    }

    send(action, value) {
        const payload = JSON.stringify({ action, value });
        if (this._socket && this._socket.readyState === WS_OPEN) {
            this._socket.send(payload);
        } else {
            if (this._sendQueue.length < 20) {
                this._sendQueue.push(payload);
            }
        }
    }

    ping() {
        this.send('ping', Date.now());
    }

    disconnect() {
        this._destroyed = true;
        if (this._retryTimer !== null) {
            clearTimeout(this._retryTimer);
            this._retryTimer = null;
        }
        this._closeSocket();
        this._onStatusChange('disconnected', 'WS disconnected');
    }

    isOpen() {
        return this._socket !== null && this._socket.readyState === WS_OPEN;
    }

    _closeSocket() {
        if (this._socket) {
            this._socket.onopen    = null;
            this._socket.onmessage = null;
            this._socket.onclose   = null;
            this._socket.onerror   = null;
            if (this._socket.readyState === WS_OPEN || this._socket.readyState === WS_CONNECTING) {
                this._socket.close();
            }
            this._socket = null;
        }
    }

    _scheduleReconnect() {
        if (this._destroyed) return;
        const delayMs = Math.min(
            BACKOFF_BASE_MS * Math.pow(BACKOFF_MULTIPLIER, this._retryCount),
            BACKOFF_CEILING_MS,
        );
        this._retryCount++;
        this._onStatusChange('warn', `WS retry ${this._retryCount} in ${(delayMs / 1000).toFixed(1)}s`);
        this._retryTimer = setTimeout(() => {
            this._retryTimer = null;
            this.connect();
        }, delayMs);
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  Subsystem: HealthPoller
// ══════════════════════════════════════════════════════════════════════════════

class HealthPoller {
    constructor(onSuccess, onFailure) {
        this._onSuccess  = onSuccess;
        this._onFailure  = onFailure;
        this._intervalId = null;
    }

    start() {
        this._poll();
        this._intervalId = setInterval(() => this._poll(), HEALTH_INTERVAL_MS);
    }

    stop() {
        if (this._intervalId !== null) {
            clearInterval(this._intervalId);
            this._intervalId = null;
        }
    }

    async _poll() {
        try {
            const [healthRes, cameraRes, modelRes] = await Promise.all([
                fetch(HEALTH_ENDPOINT,  { cache: 'no-store' }),
                fetch(CAMERA_ENDPOINT,  { cache: 'no-store' }),
                fetch(MODEL_ENDPOINT,   { cache: 'no-store' }),
            ]);

            if (!healthRes.ok) throw new Error(`Health endpoint returned HTTP ${healthRes.status}`);

            const health = await healthRes.json();
            const camera = cameraRes.ok  ? await cameraRes.json()  : null;
            const model  = modelRes.ok   ? await modelRes.json()   : null;

            this._onSuccess(health, camera, model);
        } catch (error) {
            this._onFailure(error);
        }
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  VisionDashboardApp — Main Orchestrator
// ══════════════════════════════════════════════════════════════════════════════

class VisionDashboardApp {
    constructor() {
        this._logger    = null;
        this._renderer  = null;
        this._sse       = null;
        this._ws        = null;
        this._poller    = null;

        this._frameCount          = 0;
        this._totalDetectionCount = 0;
    }

    init() {
        const logContainer = document.getElementById('event-log');
        this._logger = new EventLogger(logContainer);
        this._logger.log('VisionDashboardApp v1.0.0 initialising', 'info');

        this._renderer = new DOMRenderer();
        this._renderer.resolveElements();
        this._logger.log('DOM renderer ready — elements resolved', 'success');

        this._sse = new SSEClient(
            SSE_ENDPOINT,
            (payload) => this._onTelemetryPayload(payload),
            (state, message) => this._onSSEStatusChange(state, message),
        );
        this._sse.connect();

        this._ws = new WebSocketClient(
            WS_ENDPOINT,
            (data) => this._onWSMessage(data),
            (state, message) => this._onWSStatusChange(state, message),
        );
        this._ws.connect();

        this._poller = new HealthPoller(
            (health, camera, model) => this._onHealthSuccess(health, camera, model),
            (error) => this._onHealthFailure(error),
        );
        this._poller.start();

        this._bindSliders();
        this._bindButtons();
        this._setupVisibility();

        this._logger.log('All subsystems online', 'success');
    }

    _setupVisibility() {
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        
        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
            navigator.mediaDevices.enumerateDevices()
                .then(devices => {
                    const videoInputs = devices.filter(d => d.kind === 'videoinput');
                    this._totalCameras = videoInputs.length;
                    
                    const section = document.getElementById('camera-switcher-section');
                    if (section) {
                        // Show the switch button ONLY on Mobile Devices
                        if (isMobile) {
                            section.classList.remove('hidden');
                        } else {
                            section.classList.add('hidden');
                        }
                    }
                })
                .catch(err => {
                    this._logger.log('Could not enumerate cameras: ' + err, 'warning');
                });
        } else {
            const section = document.getElementById('camera-switcher-section');
            if (section) {
                if (isMobile) section.classList.remove('hidden');
                else section.classList.add('hidden');
            }
        }
    }

    _onTelemetryPayload(payload) {
        this._frameCount++;
        this._totalDetectionCount += (payload.active_total_count ?? payload.count ?? 0);

        const diagDet = document.getElementById('diag-total-det');
        if (diagDet) diagDet.textContent = this._totalDetectionCount;
        
        const diagQueue = document.getElementById('diag-queue');
        if (diagQueue && payload.queue_depth != null) {
            diagQueue.textContent = payload.queue_depth;
        }

        this._renderer.scheduleTelemetryUpdate(payload);
    }

    _onSSEStatusChange(state, message) {
        const logVariant = state === 'connected'  ? 'success'
                         : state === 'connecting' ? 'info'
                         : state === 'warn'        ? 'warn'
                         : 'error';

        this._logger.log(`[SSE] ${message}`, logVariant);

        const rendererState = state === 'connected'    ? 'connected'
                            : state === 'connecting'   ? 'connecting'
                            : state === 'disconnected' ? 'disconnected'
                            : 'error';

        this._renderer.writeConnectionState('sse', rendererState);

        if (state === 'error' || state === 'disconnected') {
            this._renderer.setOfflineState();
        }
    }

    _onWSMessage(data) {
        if (data.pong != null) {
            const rtt = Date.now() - data.pong;
            this._logger.log(`WS pong — RTT ${rtt}ms`, 'info');
            return;
        }

        if (data.ack != null) {
            const statusEl = document.getElementById('config-status');
            const valueStr = data.value != null ? ` → ${data.value}` : '';
            if (statusEl) {
                statusEl.innerHTML = `<span class="text-emerald-400/80">✓ ${data.ack}${valueStr} applied</span>`;
                setTimeout(() => {
                    statusEl.innerHTML = '<span class="text-slate-700 italic">config in sync</span>';
                }, 3_000);
            }
            this._logger.log(`WS ACK: ${data.ack}${valueStr}`, 'success');
            return;
        }

        if (data.event != null) {
            this._logger.log(`[SERVER] ${data.event}: ${JSON.stringify(data)}`, 'warn');
            return;
        }

        if (data.error != null) {
            this._logger.log(`[WS ERROR] ${data.error}`, 'error');
        }
    }

    _onWSStatusChange(state, message) {
        const logVariant = state === 'connected'  ? 'success'
                         : state === 'connecting' ? 'info'
                         : state === 'warn'        ? 'warn'
                         : 'error';

        this._logger.log(`[WS] ${message}`, logVariant);

        const rendererState = state === 'connected'    ? 'connected'
                            : state === 'connecting'   ? 'connecting'
                            : state === 'disconnected' ? 'disconnected'
                            : 'error';

        this._renderer.writeConnectionState('ws', rendererState);
    }

    _onHealthSuccess(health, camera, model) {
        this._renderer.writeHealthData(health, camera, model);

        if (camera) {
            const diagCapFps = document.getElementById('diag-cap-fps');
            if (diagCapFps) diagCapFps.textContent = camera.target_fps ?? '—';
        }
    }

    _onHealthFailure(error) {
        this._logger.log(`Health poll failed: ${error.message}`, 'error');
        this._renderer.setOfflineState();
    }

    _bindSliders() {
        const sliderConfig = [
            {
                sliderId:   'slider-conf',
                labelId:    'val-conf',
                wsAction:   'set_conf',
                parseValue: (v) => parseFloat(parseFloat(v).toFixed(2)),
                formatLabel:(v) => parseFloat(v).toFixed(2),
            },
            {
                sliderId:   'slider-brightness',
                labelId:    'val-brightness',
                wsAction:   'set_brightness',
                parseValue: (v) => parseFloat(parseFloat(v).toFixed(1)),
                formatLabel:(v) => parseFloat(v).toFixed(1).toString(),
            },
            {
                sliderId:   'slider-trails',
                labelId:    'val-trails',
                wsAction:   'set_trails',
                parseValue: (v) => parseInt(v, 10),
                formatLabel:(v) => parseInt(v, 10).toString(),
            },
            {
                sliderId:   'slider-tripwire',
                labelId:    'val-tripwire',
                wsAction:   'set_tripwire',
                parseValue: (v) => parseInt(v, 10),
                formatLabel:(v) => parseInt(v, 10).toString(),
            },
            {
                sliderId:   'slider-camera',
                labelId:    'val-camera',
                wsAction:   'switch_camera',
                parseValue: (v) => parseInt(v, 10),
                formatLabel:(v) => parseInt(v, 10).toString(),
            },
            {
                sliderId:   'slider-iou',
                labelId:    'val-iou',
                wsAction:   'set_iou',
                parseValue: (v) => parseFloat(parseFloat(v).toFixed(2)),
                formatLabel:(v) => parseFloat(v).toFixed(2),
            },
            {
                sliderId:   'slider-quality',
                labelId:    'val-quality',
                wsAction:   'set_quality',
                parseValue: (v) => parseInt(v, 10),
                formatLabel:(v) => parseInt(v, 10).toString(),
            },
        ];

        for (const config of sliderConfig) {
            const slider = document.getElementById(config.sliderId);
            const label  = document.getElementById(config.labelId);
            if (!slider) continue;

            slider.addEventListener('input', (event) => {
                const raw = event.target.value;
                if (label) label.textContent = config.formatLabel(raw);
                
                const statusEl = document.getElementById('config-status');
                if (statusEl) {
                    statusEl.innerHTML = `<span class="text-amber-400/80 animate-pulse">⟳ pending changes (click apply)</span>`;
                }
            });
        }

        const btnSwitchCamera = document.getElementById('btn-switch-camera');
        if (btnSwitchCamera) {
            btnSwitchCamera.addEventListener('click', () => {
                this._currentCamera = (this._currentCamera || 0) + 1;
                const maxCams = this._totalCameras || 2;
                if (this._currentCamera >= maxCams) this._currentCamera = 0;
                
                document.getElementById('val-camera').textContent = 'Cam ' + this._currentCamera;
                
                // Instantly apply camera switch without waiting for "APPLY" button
                this._ws.send('switch_camera', this._currentCamera);
                this._logger.log('Switched backend camera to index ' + this._currentCamera, 'info');
            });
        }
    }

    _bindButtons() {
        const btnApply = document.getElementById('btn-apply-config');
        if (btnApply) {
            btnApply.addEventListener('click', () => {
                const confSlider       = document.getElementById('slider-conf');
                const iouSlider        = document.getElementById('slider-iou');
                const qualitySlider    = document.getElementById('slider-quality');
                const brightnessSlider = document.getElementById('slider-brightness');
                const trailsSlider     = document.getElementById('slider-trails');
                const tripwireSlider   = document.getElementById('slider-tripwire');

                if (confSlider) this._ws.send('set_conf', parseFloat(parseFloat(confSlider.value).toFixed(2)));
                if (iouSlider) this._ws.send('set_iou', parseFloat(parseFloat(iouSlider.value).toFixed(2)));
                if (qualitySlider) this._ws.send('set_quality', parseInt(qualitySlider.value, 10));
                if (brightnessSlider) this._ws.send('set_brightness', parseFloat(parseFloat(brightnessSlider.value).toFixed(1)));
                if (trailsSlider) this._ws.send('set_trails', parseInt(trailsSlider.value, 10));
                if (tripwireSlider) this._ws.send('set_tripwire', parseInt(tripwireSlider.value, 10));

                const statusEl = document.getElementById('config-status');
                if (statusEl) {
                    statusEl.innerHTML = `<span class="text-green-400/80">✓ config applied</span>`;
                    setTimeout(() => { statusEl.innerHTML = `no pending changes`; }, 2000);
                }

                this._logger.log('Config batch applied via APPLY button', 'success');
            });
        }

        const btnExportCsv = document.getElementById('btn-export-csv');
        if (btnExportCsv) {
            btnExportCsv.addEventListener('click', () => {
                this._exportRegistryCSV();
            });
        }

        const btnClearLog = document.getElementById('btn-clear-log');
        if (btnClearLog) {
            btnClearLog.addEventListener('click', () => {
                if (this._renderer && this._renderer._trackRegistry) {
                    this._renderer._trackRegistry.clear();
                    this._renderer._setText('reg-total-seen', 0);
                    this._renderer._setText('reg-active', 0);
                    this._renderer._setText('reg-lost', 0);
                }
                const tbody = document.getElementById('track-table-body');
                if (tbody) {
                    tbody.innerHTML = `
                        <div class="grid grid-cols-12 gap-x-2 px-4 py-3 text-[10px] font-mono text-slate-700 italic">
                            <span class="col-span-12 text-center">registry cleared</span>
                        </div>`;
                }
                this._logger.log('Track registry cleared', 'info');
            });
        }

        const btnClearEvents = document.getElementById('btn-clear-events');
        if (btnClearEvents) {
            btnClearEvents.addEventListener('click', () => {
                this._logger.clear();
            });
        }
        
        const btnClearSnapshots = document.getElementById('btn-clear-snapshots');
        if (btnClearSnapshots) {
            btnClearSnapshots.addEventListener('click', () => {
                const gallery = document.getElementById('snapshot-gallery');
                if (gallery) {
                    gallery.innerHTML = '<div id="snapshot-empty" class="col-span-full text-center text-slate-700 text-[10px] font-mono italic py-4">awaiting snapshots...</div>';
                    if (this._rendered_snapshot_ids) this._rendered_snapshot_ids.clear();
                    this._logger.log('Snapshots gallery cleared', 'info');
                }
            });
        }

        const btnHealth = document.getElementById('btn-health-check');
        if (btnHealth) {
            btnHealth.addEventListener('click', async () => {
                btnHealth.textContent = '↻ Checking...';
                btnHealth.disabled = true;
                try {
                    const [healthRes, cameraRes, modelRes] = await Promise.all([
                        fetch(HEALTH_ENDPOINT,  { cache: 'no-store' }),
                        fetch(CAMERA_ENDPOINT,  { cache: 'no-store' }),
                        fetch(MODEL_ENDPOINT,   { cache: 'no-store' }),
                    ]);
                    if (!healthRes.ok) throw new Error(`HTTP ${healthRes.status}`);
                    const health = await healthRes.json();
                    const camera = cameraRes.ok ? await cameraRes.json() : null;
                    const model  = modelRes.ok  ? await modelRes.json()  : null;
                    this._onHealthSuccess(health, camera, model);
                    this._logger.log('Manual health check OK', 'success');
                } catch (err) {
                    this._logger.log(`Manual health check failed: ${err.message}`, 'error');
                    this._renderer.setOfflineState();
                } finally {
                    btnHealth.textContent = '↻ Refresh Health Check';
                    btnHealth.disabled = false;
                }
            });
        }
    }

    _exportRegistryCSV() {
        if (!this._renderer || !this._renderer._trackRegistry) return;
        let csvContent = "data:text/csv;charset=utf-8,ID,CLASS,STATUS,FIRST_SEEN,LAST_SEEN\n";
        
        for (const [tid, track] of this._renderer._trackRegistry.entries()) {
            csvContent += `${tid},${track.class_label},${track.status},${track.first_seen},${track.last_seen}\n`;
        }
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", `track_registry_${new Date().getTime()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        this._logger.log('Exported track registry to CSV', 'success');
    }

    _bindVisibilityAPI() {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this._logger.log('Tab hidden — health polling paused', 'info');
                this._poller.stop();
            } else {
                this._logger.log('Tab visible — resuming health polling', 'info');
                this._poller.start();
                if (this._ws.isOpen()) this._ws.ping();
            }
        });
    }

    destroy() {
        this._logger.log('VisionDashboardApp destroying...', 'warn');
        this._sse.disconnect();
        this._ws.disconnect();
        this._poller.stop();
    }
}


// ══════════════════════════════════════════════════════════════════════════════
//  Bootstrap
// ══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    const app = new VisionDashboardApp();
    app.init();
    window.app = app;
});
