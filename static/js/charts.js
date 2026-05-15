/**
 * SPECTER — Pure Canvas 2D metric charts.
 */

const Charts = (() => {
    'use strict';

    const MODEL_COLORS = [
        '#00ff88',  // green (primary)
        '#00f0ff',  // cyan
        '#ff00aa',  // magenta
        '#ffaa00',  // orange
        '#7b2dff',  // purple
        '#ff3366',  // red
        '#39ff14',  // lime
        '#00b4d8',  // blue
    ];

    class RingBuffer {
        constructor(maxLen) {
            this.data = [];
            this.max = maxLen;
        }
        push(v) {
            this.data.push(v);
            if (this.data.length > this.max) this.data.shift();
        }
        clear() { this.data = []; }
        toArray() { return this.data; }
        get length() { return this.data.length; }
        get last() { return this.data.length > 0 ? this.data[this.data.length - 1] : null; }
    }

    // ── Buffers ──────────────────────────────────────────────
    const tpsBuf = new RingBuffer(200);
    const memBuf = new RingBuffer(200);
    const ttftBuf = [];
    const utilBuf = new RingBuffer(200);

    const modelTpsBufs = {};
    let colorIndex = 0;
    const modelColors = {};

    function getModelColor(model) {
        if (!modelColors[model]) {
            modelColors[model] = MODEL_COLORS[colorIndex % MODEL_COLORS.length];
            colorIndex++;
        }
        return modelColors[model];
    }

    function getModelTpsBuf(model) {
        if (!modelTpsBufs[model]) {
            modelTpsBufs[model] = new RingBuffer(200);
        }
        return modelTpsBufs[model];
    }

    // ── Drawing helpers ──────────────────────────────────────
    function _prepCanvas(canvas) {
        const ctx = canvas.getContext('2d');
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        return { ctx, w, h };
    }

    function _drawEmpty(ctx, w, h, text) {
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(8, 8, 16, 0.8)';
        ctx.fillRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.font = "10px 'Share Tech Mono', monospace";
        ctx.textAlign = 'center';
        ctx.fillText(text || 'AWAITING DATA...', w / 2, h / 2);
    }

    function _drawGrid(ctx, w, h, pad) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            const gy = pad + (h - 2 * pad) * i / 4;
            ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
        }
    }

    function _drawLine(ctx, data, color, w, h, pad, yMin, yMax) {
        const toX = (i) => (i / (data.length - 1)) * w;
        const toY = (v) => pad + (1 - (v - yMin) / (yMax - yMin)) * (h - 2 * pad);

        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 3;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        data.forEach((v, i) => {
            const x = toX(i), y = toY(v);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        ctx.lineTo(toX(data.length - 1), h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.globalAlpha = 0.06;
        ctx.fillStyle = color;
        ctx.fill();
        ctx.restore();
    }

    // ── Tokens/sec chart (multi-model lines) ─────────────────
    function renderTps(canvas) {
        const { ctx, w, h } = _prepCanvas(canvas);
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(8, 8, 16, 0.8)';
        ctx.fillRect(0, 0, w, h);

        const entries = Object.entries(modelTpsBufs);
        const hasData = entries.some(([, buf]) => buf.length >= 2);

        if (!hasData) {
            _drawEmpty(ctx, w, h, 'AWAITING BENCHMARK...');
            return;
        }

        let yMin = 0, yMax = 10;
        entries.forEach(([, buf]) => {
            buf.toArray().forEach(v => { if (v > yMax) yMax = v; });
        });
        yMax = Math.ceil(yMax * 1.1);
        const pad = 4;

        _drawGrid(ctx, w, h, pad);
        entries.forEach(([model, buf]) => {
            if (buf.length >= 2) {
                _drawLine(ctx, buf.toArray(), getModelColor(model), w, h, pad, yMin, yMax);
            }
        });

        ctx.font = "9px 'Share Tech Mono', monospace";
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        let ly = 3;
        entries.forEach(([model, buf]) => {
            if (buf.last !== null) {
                ctx.fillStyle = getModelColor(model);
                const shortName = model.length > 20 ? model.substring(0, 18) + '..' : model;
                ctx.fillText(`${shortName}: ${buf.last.toFixed(1)}`, w - 4, ly);
                ly += 12;
            }
        });

        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.font = "8px 'Share Tech Mono', monospace";
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(yMax.toFixed(0), 2, 2);
        ctx.textBaseline = 'bottom';
        ctx.fillText('0', 2, h - 2);
    }

    // ── GPU Memory chart ─────────────────────────────────────
    function renderMem(canvas) {
        const { ctx, w, h } = _prepCanvas(canvas);
        const data = memBuf.toArray();

        if (data.length < 2) {
            _drawEmpty(ctx, w, h);
            return;
        }

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(8, 8, 16, 0.8)';
        ctx.fillRect(0, 0, w, h);

        const yMin = 0;
        const yMax = Math.max(Math.ceil(Math.max(...data) * 1.1 / 1024) * 1024, 8192);
        const pad = 4;

        _drawGrid(ctx, w, h, pad);
        _drawLine(ctx, data, '#00f0ff', w, h, pad, yMin, yMax);

        const last = data[data.length - 1];
        ctx.fillStyle = '#00f0ff';
        ctx.font = "10px 'Share Tech Mono', monospace";
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText((last / 1024).toFixed(1) + ' GB', w - 4, 3);

        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.font = "8px 'Share Tech Mono', monospace";
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText((yMax / 1024).toFixed(0) + 'G', 2, 2);
        ctx.textBaseline = 'bottom';
        ctx.fillText('0', 2, h - 2);
    }

    // ── TTFT bar chart ───────────────────────────────────────
    function renderTtft(canvas) {
        const { ctx, w, h } = _prepCanvas(canvas);
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(8, 8, 16, 0.8)';
        ctx.fillRect(0, 0, w, h);

        if (ttftBuf.length === 0) {
            _drawEmpty(ctx, w, h);
            return;
        }

        const pad = 4;
        const maxVal = Math.max(...ttftBuf.map(e => e.value), 100);
        const barH = Math.min(20, (h - 2 * pad) / ttftBuf.length - 2);
        const maxBarW = w - 100;

        ttftBuf.forEach((entry, i) => {
            const y = pad + i * (barH + 2);
            const barW = (entry.value / maxVal) * maxBarW;
            const color = getModelColor(entry.model);

            ctx.fillStyle = color;
            ctx.globalAlpha = 0.3;
            ctx.fillRect(80, y, barW, barH);
            ctx.globalAlpha = 1.0;

            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.strokeRect(80, y, barW, barH);

            ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
            ctx.font = "8px 'Share Tech Mono', monospace";
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            const shortName = entry.model.length > 12 ? entry.model.substring(0, 10) + '..' : entry.model;
            ctx.fillText(shortName, 76, y + barH / 2);

            ctx.fillStyle = color;
            ctx.textAlign = 'left';
            ctx.fillText(entry.value.toFixed(0) + 'ms', 80 + barW + 4, y + barH / 2);
        });
    }

    // ── GPU Utilization chart ────────────────────────────────
    function renderUtil(canvas) {
        const { ctx, w, h } = _prepCanvas(canvas);
        const data = utilBuf.toArray();

        if (data.length < 2) {
            _drawEmpty(ctx, w, h);
            return;
        }

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(8, 8, 16, 0.8)';
        ctx.fillRect(0, 0, w, h);

        const pad = 4;
        _drawGrid(ctx, w, h, pad);
        _drawLine(ctx, data, '#ffaa00', w, h, pad, 0, 100);

        const last = data[data.length - 1];
        ctx.fillStyle = '#ffaa00';
        ctx.font = "10px 'Share Tech Mono', monospace";
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText(last.toFixed(0) + '%', w - 4, 3);

        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.font = "8px 'Share Tech Mono', monospace";
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('100%', 2, 2);
        ctx.textBaseline = 'bottom';
        ctx.fillText('0%', 2, h - 2);
    }

    // ═══════════════════════════════════════════════════════════
    //  REPORT CHARTS (static, rendered on demand)
    // ═══════════════════════════════════════════════════════════

    /**
     * Horizontal bar chart: category breakdown (tok/s per category).
     * @param {HTMLCanvasElement} canvas
     * @param {Object} categories - {catName: {avg_tps, avg_ttft, count}}
     * @param {string} color - accent color
     */
    function renderCategoryBars(canvas, categories, color) {
        const { ctx, w, h } = _prepCanvas(canvas);
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(8, 8, 16, 0.8)';
        ctx.fillRect(0, 0, w, h);

        const cats = Object.entries(categories);
        if (cats.length === 0) {
            _drawEmpty(ctx, w, h, 'NO CATEGORY DATA');
            return;
        }

        const pad = 8;
        const labelW = 90;
        const maxVal = Math.max(...cats.map(([, d]) => d.avg_tps), 1);
        const barH = Math.min(22, (h - 2 * pad) / cats.length - 4);
        const maxBarW = w - labelW - 60;

        cats.sort((a, b) => b[1].avg_tps - a[1].avg_tps);

        cats.forEach(([name, data], i) => {
            const y = pad + i * (barH + 4);
            const barW = (data.avg_tps / maxVal) * maxBarW;
            const c = color || '#00ff88';

            // Bar fill
            const grad = ctx.createLinearGradient(labelW, y, labelW + barW, y);
            grad.addColorStop(0, c);
            grad.addColorStop(1, c + '44');
            ctx.fillStyle = grad;
            ctx.globalAlpha = 0.5;
            ctx.fillRect(labelW, y, barW, barH);
            ctx.globalAlpha = 1;

            // Bar border
            ctx.strokeStyle = c;
            ctx.lineWidth = 1;
            ctx.strokeRect(labelW, y, barW, barH);

            // Category label
            ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
            ctx.font = "9px 'Share Tech Mono', monospace";
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            const shortName = name.length > 12 ? name.substring(0, 11) + '.' : name;
            ctx.fillText(shortName, labelW - 6, y + barH / 2);

            // Value label
            ctx.fillStyle = c;
            ctx.textAlign = 'left';
            ctx.fillText(data.avg_tps.toFixed(1) + ' t/s', labelW + barW + 5, y + barH / 2);
        });
    }

    /**
     * Grouped vertical bar chart: compare this model vs leaderboard.
     * @param {HTMLCanvasElement} canvas
     * @param {Array} models - [{model, avg_tokens_per_sec}] sorted desc
     * @param {string} highlightModel - the model to highlight
     */
    function renderComparisonBars(canvas, models, highlightModel) {
        const { ctx, w, h } = _prepCanvas(canvas);
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(8, 8, 16, 0.8)';
        ctx.fillRect(0, 0, w, h);

        if (models.length === 0) {
            _drawEmpty(ctx, w, h, 'NO COMPARISON DATA');
            return;
        }

        const padTop = 8, padBottom = 30, padLeft = 35, padRight = 8;
        const chartW = w - padLeft - padRight;
        const chartH = h - padTop - padBottom;
        const maxVal = Math.max(...models.map(m => m.avg_tokens_per_sec), 1);
        const barW = Math.min(40, (chartW / models.length) - 6);
        const gap = (chartW - barW * models.length) / (models.length + 1);

        // Grid lines
        _drawGrid(ctx, w, h, padTop);

        // Y-axis
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.font = "8px 'Share Tech Mono', monospace";
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';
        ctx.fillText(maxVal.toFixed(0), padLeft - 4, padTop);
        ctx.textBaseline = 'bottom';
        ctx.fillText('0', padLeft - 4, padTop + chartH);

        models.forEach((m, i) => {
            const x = padLeft + gap + i * (barW + gap);
            const barH = (m.avg_tokens_per_sec / maxVal) * chartH;
            const y = padTop + chartH - barH;
            const isHighlight = m.model === highlightModel;
            const c = isHighlight ? '#00ff88' : '#00f0ff44';

            // Bar
            ctx.fillStyle = c;
            ctx.globalAlpha = isHighlight ? 0.8 : 0.4;
            ctx.fillRect(x, y, barW, barH);
            ctx.globalAlpha = 1;

            if (isHighlight) {
                ctx.strokeStyle = '#00ff88';
                ctx.lineWidth = 1;
                ctx.strokeRect(x, y, barW, barH);

                ctx.save();
                ctx.shadowColor = '#00ff88';
                ctx.shadowBlur = 8;
                ctx.strokeRect(x, y, barW, barH);
                ctx.restore();
            }

            // Value on top
            ctx.fillStyle = isHighlight ? '#00ff88' : 'rgba(255,255,255,0.4)';
            ctx.font = "8px 'Share Tech Mono', monospace";
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText(m.avg_tokens_per_sec.toFixed(1), x + barW / 2, y - 2);

            // Model name below (rotated)
            ctx.save();
            ctx.translate(x + barW / 2, padTop + chartH + 4);
            ctx.rotate(-Math.PI / 6);
            ctx.fillStyle = isHighlight ? '#00ff88' : 'rgba(255,255,255,0.3)';
            ctx.font = "7px 'Share Tech Mono', monospace";
            ctx.textAlign = 'right';
            ctx.textBaseline = 'top';
            const shortName = m.model.length > 14 ? m.model.substring(0, 12) + '..' : m.model;
            ctx.fillText(shortName, 0, 0);
            ctx.restore();
        });
    }

    // ── Public API ───────────────────────────────────────────
    function pushTps(model, value) {
        tpsBuf.push(value);
        getModelTpsBuf(model).push(value);
    }

    function pushMem(value) { memBuf.push(value); }
    function pushUtil(value) { utilBuf.push(value); }

    function addTtft(model, value) {
        const existing = ttftBuf.findIndex(e => e.model === model);
        if (existing >= 0) {
            ttftBuf[existing].value = value;
        } else {
            ttftBuf.push({ model, value });
        }
    }

    function clearAll() {
        tpsBuf.clear();
        memBuf.clear();
        utilBuf.clear();
        ttftBuf.length = 0;
        Object.values(modelTpsBufs).forEach(b => b.clear());
    }

    function render() {
        const tps = document.getElementById('chart-tps');
        const mem = document.getElementById('chart-mem');
        const ttft = document.getElementById('chart-ttft');
        const util = document.getElementById('chart-util');

        if (tps) renderTps(tps);
        if (mem) renderMem(mem);
        if (ttft) renderTtft(ttft);
        if (util) renderUtil(util);
    }

    // Render at 4fps
    setInterval(render, 250);

    return {
        pushTps,
        pushMem,
        pushUtil,
        addTtft,
        clearAll,
        getModelColor,
        renderCategoryBars,
        renderComparisonBars,
    };
})();
