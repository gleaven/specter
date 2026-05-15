/**
 * SPECTER — Main controller, WebSocket, state management, report & prompt UI.
 */

(() => {
    'use strict';

    // ── State ────────────────────────────────────────────────
    let ws = null;
    let selectedModels = new Set();
    let selectedPromptSets = new Set();
    let benchmarkRunning = false;
    let reconnectTimer = null;
    let currentResults = [];
    let leaderboardData = [];
    let modelInfoMap = {};  // model name -> {arch_type, family, ...}

    // ── DOM refs ─────────────────────────────────────────────
    const $ = id => document.getElementById(id);
    const $statusDot = $('status-dot');
    const $statusBadge = $('status-badge');
    const $gpuUtil = $('gpu-util');
    const $gpuMem = $('gpu-mem');
    const $gpuPower = $('gpu-power');
    const $modelList = $('model-list');
    const $promptSetList = $('prompt-set-list');
    const $btnRun = $('btn-run');
    const $btnStop = $('btn-stop');
    const $btnRefresh = $('btn-refresh-models');
    const $btnLoad = $('btn-load-models');
    const $btnUnload = $('btn-unload-models');
    const $progressSection = $('progress-section');
    const $progressLabel = $('progress-label');
    const $progressCount = $('progress-count');
    const $progressFill = $('progress-fill');
    const $progressDetail = $('progress-detail');
    const $resultsSection = $('results-section');
    const $resultsBody = $('results-body');
    const $leaderboardBody = $('leaderboard-body');
    const $tpsValue = $('tps-value');
    const $memValue = $('mem-value');
    const $ttftValue = $('ttft-value');
    const $utilValue = $('util-value');

    // Report modal refs
    const $reportOverlay = $('report-overlay');
    const $reportTitle = $('report-title');
    const $reportSubtitle = $('report-subtitle');
    const $reportCards = $('report-cards');
    const $reportOverview = $('report-overview');
    const $reportResultsBody = $('report-results-body');

    // Prompt editor refs
    const $promptEditorOverlay = $('prompt-editor-overlay');
    const $promptSetSelector = $('prompt-set-selector');
    const $promptList = $('prompt-list');
    const $btnAddPrompt = $('btn-add-prompt');

    // ── Tab Switching ────────────────────────────────────────
    document.querySelectorAll('.panel-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            $(tab.dataset.tab).classList.add('active');
        });
    });

    // ── WebSocket ────────────────────────────────────────────
    function connectWs() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/ws/metrics`;

        ws = new WebSocket(url);

        ws.onopen = () => {
            console.log('[SPECTER] WS connected');
            $statusDot.classList.add('active');
            $statusBadge.textContent = 'CONNECTED';
            $statusBadge.classList.add('active');
            if (reconnectTimer) {
                clearInterval(reconnectTimer);
                reconnectTimer = null;
            }
        };

        ws.onclose = () => {
            console.log('[SPECTER] WS disconnected');
            $statusDot.classList.remove('active', 'running');
            $statusBadge.textContent = 'DISCONNECTED';
            $statusBadge.classList.remove('active', 'running');
            if (!reconnectTimer) {
                reconnectTimer = setInterval(connectWs, 3000);
            }
        };

        ws.onerror = () => { ws.close(); };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            } catch (e) {
                console.warn('[SPECTER] Bad WS message:', e);
            }
        };
    }

    function handleMessage(msg) {
        switch (msg.type) {
            case 'gpu':
                updateGpuStats(msg);
                break;
            case 'status':
                if (msg.benchmark_running) {
                    setBenchmarkState(true);
                }
                break;
            case 'status_msg':
                if (msg.msg) {
                    $progressDetail.textContent = msg.msg;
                }
                break;
            case 'token':
                Charts.pushTps(msg.model, msg.tokens_per_sec);
                $tpsValue.textContent = msg.tokens_per_sec.toFixed(1);
                break;
            case 'benchmark_start':
                setBenchmarkState(true);
                $progressSection.style.display = 'block';
                $progressCount.textContent = `0 / ${msg.total}`;
                $progressFill.style.width = '0%';
                Charts.clearAll();
                currentResults = [];
                $resultsBody.innerHTML = '';
                $resultsSection.style.display = 'block';
                break;
            case 'benchmark_result':
                handleBenchmarkResult(msg);
                break;
            case 'benchmark_complete':
                setBenchmarkState(false);
                $progressLabel.textContent = 'COMPLETE';
                $progressFill.style.width = '100%';
                updateLeaderboardFromSummary(msg.summary);
                break;
            case 'benchmark_error':
                setBenchmarkState(false);
                $progressLabel.textContent = 'ERROR';
                $progressDetail.textContent = msg.error || '';
                break;
            case 'pong':
                break;
        }
    }

    function updateGpuStats(msg) {
        const isPartial = msg.partial;

        if (msg.gpu_util_pct >= 0) {
            const util = msg.gpu_util_pct;
            $gpuUtil.textContent = util.toFixed(0) + '%';
            $gpuUtil.className = util > 80 ? 'gpu-warn' : 'gpu-ok';
            Charts.pushUtil(util);
            $utilValue.textContent = util.toFixed(0) + '%';
        }

        if (msg.memory_used_mb > 0) {
            const memGb = (msg.memory_used_mb / 1024).toFixed(1);
            if (msg.memory_total_mb > 0) {
                const memTotalGb = (msg.memory_total_mb / 1024).toFixed(0);
                $gpuMem.textContent = `${memGb}/${memTotalGb}G`;
                $gpuMem.className = (msg.memory_used_mb / msg.memory_total_mb) > 0.85 ? 'gpu-warn' : 'gpu-ok';
            } else {
                $gpuMem.textContent = `${memGb}G`;
                $gpuMem.className = 'gpu-ok';
            }
            Charts.pushMem(msg.memory_used_mb);
            $memValue.textContent = memGb + ' GB';
        }

        if (msg.power_w > 0) {
            $gpuPower.textContent = msg.power_w.toFixed(0) + 'W';
        }
    }

    function handleBenchmarkResult(msg) {
        const r = msg.result;
        currentResults.push(r);

        const pct = (msg.completed / msg.total * 100).toFixed(0);
        $progressCount.textContent = `${msg.completed} / ${msg.total}`;
        $progressFill.style.width = pct + '%';
        $progressDetail.textContent = `${r.model} \u2014 ${r.prompt_name}`;

        if (r.ttft_ms > 0) {
            Charts.addTtft(r.model, r.ttft_ms);
            $ttftValue.textContent = r.ttft_ms.toFixed(0) + 'ms';
        }

        const thinkInfo = r.thinking_tokens > 0 ? ` (${r.thinking_tokens} think)` : '';
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="model-cell">${esc(r.model)}</td>
            <td>${esc(r.prompt_name)}</td>
            <td class="tps-cell">${r.tokens_per_sec.toFixed(1)}</td>
            <td class="ttft-cell">${r.ttft_ms > 0 ? r.ttft_ms.toFixed(0) + 'ms' : '\u2014'}</td>
            <td>${r.total_tokens}${thinkInfo}</td>
            <td>${(r.total_time_ms / 1000).toFixed(1)}s</td>
        `;
        $resultsBody.insertBefore(tr, $resultsBody.firstChild);
    }

    // ── Benchmark Control ────────────────────────────────────
    function setBenchmarkState(running) {
        benchmarkRunning = running;
        $btnRun.disabled = running || selectedModels.size === 0 || selectedPromptSets.size === 0;
        $btnStop.disabled = !running;

        if (running) {
            $statusDot.classList.remove('active');
            $statusDot.classList.add('running');
            $statusBadge.textContent = 'RUNNING';
            $statusBadge.classList.remove('active');
            $statusBadge.classList.add('running');
        } else {
            $statusDot.classList.remove('running');
            $statusDot.classList.add('active');
            $statusBadge.textContent = 'READY';
            $statusBadge.classList.remove('running');
            $statusBadge.classList.add('active');
        }
    }

    async function startBenchmark() {
        if (benchmarkRunning) return;
        if (selectedModels.size === 0 || selectedPromptSets.size === 0) return;

        const models = Array.from(selectedModels);
        const backend = $('benchmark-backend').value;

        // Auto-load vLLM models when running through LiteLLM
        if (backend === 'litellm') {
            const vllmModelNames = models.filter(m => m.startsWith('vllm/'));
            if (vllmModelNames.length > 0) {
                try {
                    // Look up the real HuggingFace model name from vLLM models API
                    const vllmResp = await fetch('/api/vllm/models');
                    const vllmData = await vllmResp.json();
                    const matchedModel = (vllmData.models || []).find(m => m.litellm_name === vllmModelNames[0]);

                    if (matchedModel) {
                        const realVllmModel = matchedModel.name;
                        // Check if vLLM already has this model loaded
                        if (vllmData.active_model !== realVllmModel) {
                            showToast(`Loading ${realVllmModel} in vLLM first...`, 'info');
                            $btnRun.disabled = true;
                            $btnRun.textContent = 'LOADING VLLM...';
                            const switchResp = await fetch('/api/vllm/switch', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ model: realVllmModel }),
                            });
                            const switchData = await switchResp.json();
                            $btnRun.textContent = 'RUN BENCHMARK';
                            if (!switchResp.ok) {
                                showToast(`Failed to load vLLM model: ${switchData.error}`, 'error');
                                $btnRun.disabled = false;
                                return;
                            }
                            showToast(`vLLM ready: ${realVllmModel}`, 'success');
                        } else {
                            showToast(`vLLM already has ${realVllmModel} loaded`, 'success');
                        }
                    }
                } catch (e) {
                    showToast(`vLLM check failed: ${e.message}`, 'error');
                    $btnRun.disabled = false;
                    return;
                }
            }
        }

        showToast(`Starting benchmark: ${models.join(', ')} via ${backend}`, 'info');

        try {
            const resp = await fetch('/api/benchmark/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    models,
                    prompt_sets: Array.from(selectedPromptSets),
                    runs_per_prompt: parseInt($('runs-per-prompt').value),
                    backend,
                }),
            });
            const data = await resp.json();
            if (resp.ok) {
                showToast(`Benchmark started (run ${data.run_id})`, 'success');
            } else {
                showToast(data.error || 'Failed to start benchmark', 'error');
            }
        } catch (e) {
            showToast('Failed to start benchmark', 'error');
        }
    }

    async function stopBenchmark() {
        showToast('Stopping benchmark...', 'warning');
        try {
            await fetch('/api/benchmark/stop', { method: 'POST' });
            showToast('Benchmark stop requested', 'success');
        } catch (e) {
            showToast('Failed to stop benchmark', 'error');
        }
    }

    async function loadUnloadModels(action) {
        if (selectedModels.size === 0) return;
        const models = Array.from(selectedModels);
        const backend = $('benchmark-backend').value;

        // vLLM: Load = switch model (works from vLLM backend or LiteLLM with vllm/* model)
        const isVllmLoad = action === 'load' && (backend === 'vllm' || (backend === 'litellm' && models.some(m => m.startsWith('vllm/'))));
        if (isVllmLoad) {
            // Resolve the real HuggingFace model name
            let model = models[0];
            if (backend === 'litellm' && model.startsWith('vllm/')) {
                try {
                    const vResp = await fetch('/api/vllm/models');
                    const vData = await vResp.json();
                    const matched = (vData.models || []).find(m => m.litellm_name === model);
                    if (matched) model = matched.name;
                } catch (e) { /* use as-is */ }
            }
            $btnLoad.disabled = true;
            $btnUnload.disabled = true;
            $btnRun.disabled = true;

            // Show persistent loading state
            $modelList.innerHTML = `<div class="loading-text" style="color:var(--accent-warning);">
                Loading ${model.split('/').pop()}...<br>
                <span style="color:var(--text-muted);font-size:0.5rem;">This takes 2-5 minutes. Downloading model (if first time) then loading into GPU.</span>
            </div>`;
            showToast(`Switching vLLM to ${model}...`, 'info');

            try {
                const resp = await fetch('/api/vllm/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model }),
                });
                const data = await resp.json();
                if (resp.ok) {
                    showToast(`vLLM ready: ${model}`, 'success');
                } else {
                    showToast(data.error || 'Load failed', 'error');
                }
            } catch (e) {
                showToast(`Load failed: ${e.message}`, 'error');
            } finally {
                $btnLoad.disabled = false;
                // Reload the right model list based on current backend
                if (backend === 'litellm') {
                    loadLitellmModels();
                } else {
                    loadVllmModels();
                }
            }
            return;
        }
        // vLLM unload is a no-op (container stays running)
        if (backend === 'vllm' && action === 'unload') {
            showToast('vLLM stays running — use Load to switch models', 'info');
            return;
        }

        const btn = action === 'load' ? $btnLoad : $btnUnload;
        const origText = btn.textContent;
        btn.disabled = true;
        btn.textContent = action === 'load' ? 'Loading...' : 'Unloading...';
        showToast(`${action === 'load' ? 'Loading' : 'Unloading'} ${models.join(', ')}...`, 'info');

        try {
            const resp = await fetch(`/api/models/${action}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ models }),
            });
            const data = await resp.json();
            if (resp.ok) {
                const results = data.results || {};
                Object.entries(results).forEach(([m, s]) => {
                    const ok = s === 'loaded' || s === 'unloaded';
                    showToast(`${m}: ${s}`, ok ? 'success' : 'error');
                });
                // Refresh model list to update LOADED badges
                setTimeout(loadModels, 1000);
            } else {
                showToast(data.error || `Failed to ${action}`, 'error');
            }
        } catch (e) {
            showToast(`Failed to ${action}: ${e.message}`, 'error');
        } finally {
            btn.textContent = origText;
            btn.disabled = selectedModels.size === 0;
        }
    }

    // ── Load Models ──────────────────────────────────────────
    async function loadModels() {
        const backend = $('benchmark-backend').value;
        if (backend === 'vllm') {
            return loadVllmModels();
        }
        if (backend === 'litellm') {
            return loadLitellmModels();
        }
        return loadOllamaModels();
    }

    async function loadVllmModels() {
        $modelList.innerHTML = '<div class="loading-text">Loading vLLM models...</div>';
        selectedModels.clear();
        try {
            const resp = await fetch('/api/vllm/models');
            const data = await resp.json();
            const models = data.models || [];
            const activeModel = data.active_model;
            const vllmUp = data.vllm_up;

            $modelList.innerHTML = '';
            if (!vllmUp) {
                $modelList.innerHTML += '<div class="loading-text" style="color:var(--accent-warning);">vLLM is starting up... <span style="color:var(--text-muted);font-size:0.5rem;">auto-retrying in 10s</span></div>';
                // Auto-retry until vLLM is up
                setTimeout(() => {
                    if ($('benchmark-backend').value === 'vllm') loadVllmModels();
                }, 10000);
            }

            models.forEach(m => {
                const div = document.createElement('div');
                div.className = 'model-item' + (m.active ? ' selected' : '');
                div.dataset.model = m.name;
                if (m.active) selectedModels.add(m.name);

                // Store in modelInfoMap for leaderboard
                modelInfoMap[m.name] = m;

                const activeBadge = m.active ? '<span class="model-loaded-badge">ACTIVE</span>' : '';
                const brokenBadge = m.broken ? '<span style="color:var(--accent-danger);font-size:0.5rem;">UNSUPPORTED</span>' : '';
                if (m.broken) div.style.opacity = '0.4';
                div.innerHTML = `
                    <div class="model-check"></div>
                    <div class="model-info">
                        <div class="model-name-text">${esc(m.name)}</div>
                        <div class="model-meta-text">
                            <span>${m.parameter_size}</span>
                            <span>${m.arch_type}</span>
                            <span>${m.size_gb}GB</span>
                            ${activeBadge}
                            ${brokenBadge}
                        </div>
                    </div>
                `;
                if (!m.broken) {
                    div.addEventListener('click', () => toggleModel(m.name, div));
                } else {
                    div.addEventListener('click', () => showToast(m.broken_reason || 'This model is unsupported', 'error'));
                    div.style.cursor = 'not-allowed';
                }
                $modelList.appendChild(div);
            });

            updateRunButton();
        } catch (e) {
            $modelList.innerHTML = '<div class="loading-text">Failed to load vLLM models</div>';
        }
    }

    async function loadLitellmModels() {
        $modelList.innerHTML = '<div class="loading-text">Loading models...</div>';
        selectedModels.clear();
        try {
            // Fetch Ollama models
            const ollamaResp = await fetch('/api/models');
            const ollamaData = await ollamaResp.json();
            const ollamaModels = (ollamaData.models || []).map(m => ({
                ...m,
                display_name: m.name,
                source: 'ollama',
            }));
            ollamaModels.forEach(m => { modelInfoMap[m.name] = m; });

            // Fetch vLLM models
            let vllmModels = [];
            try {
                const vllmResp = await fetch('/api/vllm/models');
                const vllmData = await vllmResp.json();
                vllmModels = (vllmData.models || []).filter(m => !m.broken).map(m => ({
                    ...m,
                    litellm_name: m.litellm_name || m.name,
                    display_name: m.name,
                    source: 'vllm',
                    active: vllmData.active_model === m.name,
                }));
                vllmModels.forEach(m => { modelInfoMap[m.litellm_name] = m; });
            } catch (e) { /* vLLM may not be available */ }

            $modelList.innerHTML = '';

            // Ollama section
            if (ollamaModels.length > 0) {
                const header = document.createElement('div');
                header.className = 'loading-text';
                header.style.cssText = 'text-align:left;padding:0.25rem 0.4rem;color:var(--accent-primary);font-size:0.5rem;letter-spacing:0.1em;';
                header.textContent = 'OLLAMA MODELS';
                $modelList.appendChild(header);
                ollamaModels.sort((a, b) => a.name.localeCompare(b.name));
                ollamaModels.forEach(m => {
                    const div = document.createElement('div');
                    div.className = 'model-item';
                    div.dataset.model = m.name;
                    const loadedBadge = m.loaded ? '<span class="model-loaded-badge">LOADED</span>' : '';
                    div.innerHTML = `
                        <div class="model-check"></div>
                        <div class="model-info">
                            <div class="model-name-text">${esc(m.name)}</div>
                            <div class="model-meta-text">
                                ${m.parameter_size ? `<span>${m.parameter_size}</span>` : ''}
                                <span>${m.size_gb}GB</span>
                                ${loadedBadge}
                            </div>
                        </div>
                    `;
                    div.addEventListener('click', () => toggleModel(m.name, div));
                    $modelList.appendChild(div);
                });
            }

            // vLLM section
            if (vllmModels.length > 0) {
                const header = document.createElement('div');
                header.className = 'loading-text';
                header.style.cssText = 'text-align:left;padding:0.25rem 0.4rem;color:var(--accent-tertiary);font-size:0.5rem;letter-spacing:0.1em;margin-top:0.5rem;';
                header.textContent = 'VLLM MODELS (auto-loaded when selected)';
                $modelList.appendChild(header);
                vllmModels.forEach(m => {
                    const div = document.createElement('div');
                    div.className = 'model-item' + (m.active ? ' selected' : '');
                    div.dataset.model = m.litellm_name;
                    div.dataset.vllmModel = m.name;  // Store real vLLM model name for auto-loading
                    if (m.active) selectedModels.add(m.litellm_name);
                    const activeBadge = m.active ? '<span class="model-loaded-badge">ACTIVE</span>' : '';
                    div.innerHTML = `
                        <div class="model-check"></div>
                        <div class="model-info">
                            <div class="model-name-text" style="color:var(--accent-tertiary);">${esc(m.litellm_name)}</div>
                            <div class="model-meta-text">
                                <span>${m.parameter_size}</span>
                                <span>${m.arch_type}</span>
                                <span>${m.size_gb}GB</span>
                                ${activeBadge}
                            </div>
                        </div>
                    `;
                    div.addEventListener('click', () => toggleModel(m.litellm_name, div));
                    $modelList.appendChild(div);
                });
            }

            updateRunButton();
        } catch (e) {
            $modelList.innerHTML = '<div class="loading-text">Failed to load models</div>';
        }
    }

    async function loadOllamaModels() {
        $modelList.innerHTML = '<div class="loading-text">Loading models...</div>';
        try {
            const resp = await fetch('/api/models');
            const data = await resp.json();
            const models = data.models || [];

            if (models.length === 0) {
                $modelList.innerHTML = '<div class="loading-text">No models found</div>';
                return;
            }

            // Clear selection state on refresh — prevents ghost selections
            selectedModels.clear();

            // Build model info lookup for leaderboard arch_type display
            models.forEach(m => { modelInfoMap[m.name] = m; });

            $modelList.innerHTML = '';
            models.sort((a, b) => a.name.localeCompare(b.name));

            models.forEach(m => {
                const div = document.createElement('div');
                div.className = 'model-item';
                div.dataset.model = m.name;

                const paramInfo = m.parameter_size || '';
                const quantInfo = m.quantization_level || '';
                const sizeInfo = m.size_gb + 'GB';
                const loadedBadge = m.loaded ? '<span class="model-loaded-badge">LOADED</span>' : '';

                div.innerHTML = `
                    <div class="model-check"></div>
                    <div class="model-info">
                        <div class="model-name-text">${esc(m.name)}</div>
                        <div class="model-meta-text">
                            ${paramInfo ? `<span>${paramInfo}</span>` : ''}
                            ${quantInfo ? `<span>${quantInfo}</span>` : ''}
                            <span>${sizeInfo}</span>
                            ${loadedBadge}
                        </div>
                    </div>
                `;

                div.addEventListener('click', () => toggleModel(m.name, div));
                $modelList.appendChild(div);
            });

            updateRunButton();
        } catch (e) {
            $modelList.innerHTML = '<div class="loading-text">Failed to load models</div>';
            console.error('Failed to load models:', e);
        }
    }

    function toggleModel(name, el) {
        if (selectedModels.has(name)) {
            selectedModels.delete(name);
            el.classList.remove('selected');
        } else {
            selectedModels.add(name);
            el.classList.add('selected');
        }
        updateRunButton();
    }

    // ── Load Prompt Sets ─────────────────────────────────────
    async function loadPromptSets() {
        try {
            const resp = await fetch('/api/prompts');
            const data = await resp.json();
            const sets = data.prompt_sets || [];

            $promptSetList.innerHTML = '';
            // Also populate the prompt editor selector
            $promptSetSelector.innerHTML = '<option value="">Select a set...</option>';

            sets.forEach(ps => {
                // Benchmark tab list
                const div = document.createElement('div');
                div.className = 'prompt-set-item';
                div.dataset.set = ps.id;
                div.innerHTML = `
                    <div class="prompt-set-check"></div>
                    <div class="prompt-set-info">
                        <div class="prompt-set-name">${esc(ps.name)}</div>
                        <div class="prompt-set-count">${ps.count} prompts</div>
                    </div>
                `;
                div.addEventListener('click', () => togglePromptSet(ps.id, div));
                $promptSetList.appendChild(div);

                // Prompt editor selector
                const opt = document.createElement('option');
                opt.value = ps.id;
                opt.textContent = `${ps.name} (${ps.count})`;
                $promptSetSelector.appendChild(opt);
            });
        } catch (e) {
            $promptSetList.innerHTML = '<div class="loading-text">Failed to load</div>';
        }
    }

    function togglePromptSet(id, el) {
        if (selectedPromptSets.has(id)) {
            selectedPromptSets.delete(id);
            el.classList.remove('selected');
        } else {
            selectedPromptSets.add(id);
            el.classList.add('selected');
        }
        updateRunButton();
    }

    function updateRunButton() {
        $btnRun.disabled = benchmarkRunning || selectedModels.size === 0 || selectedPromptSets.size === 0;
        const hasModels = selectedModels.size > 0;
        const isVllm = $('benchmark-backend').value === 'vllm';
        $btnLoad.disabled = !hasModels || benchmarkRunning;
        $btnUnload.disabled = !hasModels || benchmarkRunning;
        $btnUnload.style.display = isVllm ? 'none' : '';
    }

    // ── Leaderboard ──────────────────────────────────────────
    async function loadLeaderboard() {
        try {
            // Also fetch vLLM model info so arch badges work for vLLM entries
            try {
                const vResp = await fetch('/api/vllm/models');
                const vData = await vResp.json();
                (vData.models || []).forEach(m => {
                    modelInfoMap[m.name] = m;
                    if (m.litellm_name) modelInfoMap[m.litellm_name] = m;
                });
            } catch (e) { /* vLLM may not be available */ }

            const resp = await fetch('/api/leaderboard');
            const data = await resp.json();
            leaderboardData = data.leaderboard || [];
            renderLeaderboard(leaderboardData);
        } catch (e) {
            console.error('Failed to load leaderboard:', e);
        }
    }

    function renderLeaderboard(entries) {
        if (entries.length === 0) {
            $leaderboardBody.innerHTML = '<tr><td colspan="9" class="empty-row">No benchmark results yet</td></tr>';
            return;
        }

        $leaderboardBody.innerHTML = entries.map((e, i) => {
            const info = modelInfoMap[e.model] || {};
            const archType = info.arch_type || '';
            const archClass = archType === 'MoE' ? 'arch-moe' : 'arch-dense';
            const backend = e.backend || 'ollama';
            const engine = e.engine || backend;
            // Build engine display: "ollama", "vllm", or "litellm > ollama" / "litellm > vllm"
            let engineLabel, engineHtml;
            if (backend === 'litellm') {
                const targetClass = {'ollama': 'backend-ollama', 'vllm': 'backend-vllm'}[engine] || 'backend-ollama';
                engineHtml = `<span class="arch-badge backend-litellm">litellm</span> <span style="color:var(--text-muted);font-size:0.5rem;">\u25b8</span> <span class="arch-badge ${targetClass}">${engine}</span>`;
            } else {
                const cls = {'ollama': 'backend-ollama', 'vllm': 'backend-vllm'}[backend] || 'backend-ollama';
                engineHtml = `<span class="arch-badge ${cls}">${backend}</span>`;
            }
            return `
            <tr data-run-id="${esc(e.run_id)}" data-model="${esc(e.model)}">
                <td class="rank-cell">${i + 1}</td>
                <td class="model-cell">${esc(e.model)}</td>
                <td class="arch-cell"><span class="arch-badge ${archClass}">${archType || '\u2014'}</span></td>
                <td class="arch-cell" style="white-space:nowrap;">${engineHtml}</td>
                <td class="tps-cell">${e.avg_tokens_per_sec.toFixed(1)}</td>
                <td class="ttft-cell">${e.avg_ttft_ms.toFixed(0)}ms</td>
                <td class="accuracy-cell">${e.accuracy != null ? e.accuracy.toFixed(0) + '%' : '\u2014'}</td>
                <td>${e.prompts_completed}</td>
                <td class="delete-cell"><button class="btn-delete-row" data-run-id="${esc(e.run_id)}" title="Delete run">\u00d7</button></td>
            </tr>`;
        }).join('');

        // Attach click handlers
        $leaderboardBody.querySelectorAll('tr[data-run-id]').forEach(tr => {
            tr.addEventListener('click', (ev) => {
                // Don't open report if clicking delete button
                if (ev.target.closest('.btn-delete-row')) return;
                openReport(tr.dataset.runId, tr.dataset.model);
            });
        });

        // Delete handlers
        $leaderboardBody.querySelectorAll('.btn-delete-row').forEach(btn => {
            btn.addEventListener('click', async (ev) => {
                ev.stopPropagation();
                const runId = btn.dataset.runId;
                if (!confirm('Delete this benchmark run?')) return;
                try {
                    const resp = await fetch(`/api/run/${runId}`, { method: 'DELETE' });
                    if (resp.ok) {
                        loadLeaderboard();
                    }
                } catch (e) {
                    console.error('Delete failed:', e);
                }
            });
        });
    }

    function updateLeaderboardFromSummary(summary) {
        if (!summary || !summary.models) return;
        // Refresh from server (picks up latest-per-model logic)
        loadLeaderboard();
    }

    // ── Report Modal ─────────────────────────────────────────
    async function openReport(runId, modelName) {
        try {
            const resp = await fetch(`/api/run/${runId}`);
            if (!resp.ok) {
                alert('Failed to load run data');
                return;
            }
            const runData = await resp.json();
            renderReport(runData, modelName);
            $reportOverlay.classList.add('active');
        } catch (e) {
            console.error('Failed to load report:', e);
        }
    }

    function closeReport() {
        $reportOverlay.classList.remove('active');
    }

    function renderReport(runData, modelName) {
        const summary = runData.summary || {};
        const modelStats = (summary.models || {})[modelName] || {};
        const results = (runData.results || []).filter(r => r.model === modelName);
        const runDate = summary.timestamp ? new Date(summary.timestamp * 1000).toLocaleString() : 'Unknown';

        $reportTitle.textContent = `MODEL REPORT \u2014 ${modelName}`;
        $reportSubtitle.textContent = `Run ${runData.run_id} \u2022 ${runDate} \u2022 ${results.length} prompts`;

        // Summary cards
        const avgTps = modelStats.avg_tokens_per_sec || 0;
        const avgTtft = modelStats.avg_ttft_ms || 0;
        const prompts = modelStats.prompts_completed || 0;
        const accuracy = modelStats.accuracy;
        const thinkTok = modelStats.avg_thinking_tokens || 0;

        $reportCards.innerHTML = `
            <div class="report-card">
                <div class="report-card-value">${avgTps.toFixed(1)}</div>
                <div class="report-card-label">Avg tok/s</div>
            </div>
            <div class="report-card">
                <div class="report-card-value cyan">${avgTtft > 0 ? (avgTtft / 1000).toFixed(1) + 's' : '\u2014'}</div>
                <div class="report-card-label">Avg TTFT</div>
            </div>
            <div class="report-card">
                <div class="report-card-value orange">${accuracy != null ? accuracy.toFixed(0) + '%' : '\u2014'}</div>
                <div class="report-card-label">Accuracy</div>
            </div>
            <div class="report-card">
                <div class="report-card-value">${prompts}</div>
                <div class="report-card-label">Prompts</div>
            </div>
        `;

        // Technical overview
        const overview = generateOverview(modelName, modelStats, results);
        $reportOverview.innerHTML = overview;

        // Category chart
        const catCanvas = $('report-chart-category');
        if (catCanvas && modelStats.categories) {
            setTimeout(() => Charts.renderCategoryBars(catCanvas, modelStats.categories, '#00ff88'), 50);
        }

        // Comparison chart
        const cmpCanvas = $('report-chart-compare');
        if (cmpCanvas && leaderboardData.length > 0) {
            setTimeout(() => Charts.renderComparisonBars(cmpCanvas, leaderboardData, modelName), 50);
        }

        // Detailed results table
        renderReportResults(results);
    }

    function generateOverview(model, stats, results) {
        const lines = [];
        const tps = stats.avg_tokens_per_sec || 0;
        const ttft = stats.avg_ttft_ms || 0;
        const thinkAvg = stats.avg_thinking_tokens || 0;
        const n = stats.prompts_completed || 0;
        const errCount = results.filter(r => r.error).length;
        const zeroTokens = results.filter(r => !r.error && r.total_tokens === 0).length;

        lines.push(`<strong>${model}</strong> completed <strong>${n}</strong> prompts with an average throughput of <strong>${tps.toFixed(1)} tok/s</strong>.`);

        if (ttft > 0) {
            lines.push(`Average time-to-first-token was <strong>${(ttft / 1000).toFixed(1)}s</strong>.`);
        }

        if (stats.min_tokens_per_sec != null && stats.max_tokens_per_sec != null) {
            lines.push(`Throughput ranged from <strong>${stats.min_tokens_per_sec.toFixed(1)}</strong> to <strong>${stats.max_tokens_per_sec.toFixed(1)} tok/s</strong>.`);
        }

        if (thinkAvg > 0) {
            lines.push(`This model uses chain-of-thought reasoning, averaging <strong>${thinkAvg}</strong> thinking tokens per prompt. Thinking tokens are included in the throughput calculation.`);
        }

        if (zeroTokens > 0) {
            lines.push(`<strong>${zeroTokens}</strong> prompts produced zero visible output tokens (model may have exhausted token budget on reasoning).`);
        }

        if (errCount > 0) {
            lines.push(`<strong>${errCount}</strong> prompts resulted in errors.`);
        }

        if (stats.accuracy != null) {
            lines.push(`On prompts with expected answers, accuracy was <strong>${stats.accuracy.toFixed(0)}%</strong>.`);
        }

        // Category highlights
        const cats = stats.categories || {};
        const catEntries = Object.entries(cats).sort((a, b) => b[1].avg_tps - a[1].avg_tps);
        if (catEntries.length > 1) {
            const best = catEntries[0];
            const worst = catEntries[catEntries.length - 1];
            lines.push(`Strongest category: <strong>${best[0]}</strong> (${best[1].avg_tps.toFixed(1)} tok/s). ` +
                        `Weakest: <strong>${worst[0]}</strong> (${worst[1].avg_tps.toFixed(1)} tok/s).`);
        }

        // Compare to leaderboard
        if (leaderboardData.length > 1) {
            const rank = leaderboardData.findIndex(e => e.model === model);
            if (rank >= 0) {
                lines.push(`Leaderboard rank: <strong>#${rank + 1}</strong> of ${leaderboardData.length} models.`);
            }
        }

        return lines.join('<br><br>');
    }

    async function renderReportResults(results) {
        if (results.length === 0) {
            $reportResultsBody.innerHTML = '<tr><td colspan="7" class="empty-row">No results</td></tr>';
            return;
        }

        // Pre-load all prompt sets used in this run
        const promptCache = {};
        const setsNeeded = [...new Set(results.map(r => r.prompt_set).filter(Boolean))];
        await Promise.all(setsNeeded.map(async (setId) => {
            try {
                const resp = await fetch(`/api/prompts/${setId}/items`);
                if (resp.ok) {
                    const data = await resp.json();
                    promptCache[setId] = data.prompts || [];
                }
            } catch (e) {
                console.warn(`Failed to load prompts for ${setId}:`, e);
            }
        }));

        let html = '';
        results.forEach((r, idx) => {
            const scores = r.scores || {};
            let scoreBadge = '<span class="score-na">\u2014</span>';
            if ('correct' in scores) {
                scoreBadge = scores.correct
                    ? '<span class="score-badge score-pass">PASS</span>'
                    : '<span class="score-badge score-fail">FAIL</span>';
            } else if ('entity_pct' in scores) {
                scoreBadge = `<span class="score-badge ${scores.entity_pct >= 50 ? 'score-pass' : 'score-fail'}">${scores.entity_pct.toFixed(0)}%</span>`;
            }

            // Truncated answer preview (first 120 chars)
            const answerPreview = r.response_text
                ? (r.response_text.length > 120 ? r.response_text.substring(0, 120) + '...' : r.response_text)
                : (r.thinking_text ? '<em>(thinking only)</em>' : '<em>No output</em>');

            html += `
                <tr class="result-row" data-idx="${idx}">
                    <td>${esc(r.prompt_name)}</td>
                    <td>${esc(r.category || '')}</td>
                    <td class="tps-cell">${r.tokens_per_sec.toFixed(1)}</td>
                    <td class="ttft-cell">${r.ttft_ms > 0 ? r.ttft_ms.toFixed(0) + 'ms' : '\u2014'}</td>
                    <td>${scoreBadge}</td>
                    <td><button class="btn-expand" data-idx="${idx}" title="Show question & full answer">\u25bc</button></td>
                </tr>
                <tr class="answer-preview-row">
                    <td colspan="6" style="padding:0.25rem 0.5rem 0.5rem 0.5rem;">
                        <div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-secondary);line-height:1.4;">
                            ${r.response_text ? esc(answerPreview) : answerPreview}
                        </div>
                    </td>
                </tr>
                <tr class="response-row" id="resp-row-${idx}">
                    <td colspan="6">
                        <div class="response-label">Question <span style="font-weight:400;color:var(--text-muted);">(${esc(r.prompt_set)}/${esc(r.prompt_id)})</span></div>
                        <div class="response-text" id="question-text-${idx}"><em>Loading question...</em></div>
                        ${r.thinking_text ? `<div class="response-label" style="margin-top:0.5rem;">Thinking <span style="color:var(--accent-tertiary);">(${r.thinking_tokens} tokens)</span></div><div class="thinking-text">${esc(r.thinking_text)}</div>` : ''}
                        <div class="response-label" style="margin-top:0.5rem;">Full Response <span style="color:var(--text-muted);">(${r.total_tokens} tokens, ${(r.total_time_ms / 1000).toFixed(1)}s)</span></div>
                        <div class="response-text">${r.response_text ? esc(r.response_text) : '<em>No response text</em>'}</div>
                        ${r.error ? `<div class="response-label" style="color:var(--accent-danger);margin-top:0.5rem;">Error</div><div class="response-text" style="color:var(--accent-danger)">${esc(r.error)}</div>` : ''}
                        ${scores.expected_answer ? `<div class="response-label" style="margin-top:0.5rem;">Expected Answer</div><div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--accent-warning);padding:0.4rem;">${esc(scores.expected_answer)}</div>` : ''}
                    </td>
                </tr>
            `;
        });

        $reportResultsBody.innerHTML = html;

        // Pre-fill all question texts from cache (already loaded above)
        results.forEach((r, idx) => {
            const qEl = $('question-text-' + idx);
            if (qEl) {
                const prompts = promptCache[r.prompt_set] || [];
                const p = prompts.find(p => p.id === r.prompt_id);
                qEl.textContent = p ? p.prompt : '(prompt not found)';
            }
        });

        // Expand/collapse handlers
        $reportResultsBody.querySelectorAll('.btn-expand').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = btn.dataset.idx;
                const row = $('resp-row-' + idx);
                row.classList.toggle('active');
                btn.textContent = row.classList.contains('active') ? '\u25b2' : '\u25bc';
            });
        });
    }

    // ── Prompt Management ────────────────────────────────────
    let currentPromptSetId = null;
    let currentPrompts = [];
    let editingPrompt = null; // null = new, else = prompt object

    $promptSetSelector.addEventListener('change', () => {
        const setId = $promptSetSelector.value;
        currentPromptSetId = setId || null;
        $btnAddPrompt.disabled = !setId;
        if (setId) {
            loadPromptItems(setId);
        } else {
            $promptList.innerHTML = '';
        }
    });

    async function loadPromptItems(setId) {
        $promptList.innerHTML = '<div class="loading-text">Loading...</div>';
        try {
            const resp = await fetch(`/api/prompts/${setId}/items`);
            const data = await resp.json();
            currentPrompts = data.prompts || [];
            renderPromptList(currentPrompts);
        } catch (e) {
            $promptList.innerHTML = '<div class="loading-text">Failed to load</div>';
        }
    }

    function renderPromptList(prompts) {
        if (prompts.length === 0) {
            $promptList.innerHTML = '<div class="loading-text">No prompts</div>';
            return;
        }
        $promptList.innerHTML = '';
        prompts.forEach(p => {
            const div = document.createElement('div');
            div.className = 'prompt-item';
            div.innerHTML = `
                <span class="prompt-item-name">${esc(p.name)}</span>
                <span class="prompt-item-cat">${esc(p.category || '')}</span>
            `;
            div.addEventListener('click', () => openPromptEditor(p));
            $promptList.appendChild(div);
        });
    }

    function openPromptEditor(prompt) {
        editingPrompt = prompt || null;
        $('prompt-editor-title').textContent = prompt ? 'Edit Prompt' : 'New Prompt';
        $('pe-name').value = prompt ? (prompt.name || '') : '';
        $('pe-category').value = prompt ? (prompt.category || '') : '';
        $('pe-prompt').value = prompt ? (prompt.prompt || '') : '';
        $('pe-expected-answer').value = prompt ? (prompt.expected_answer || '') : '';
        $('pe-expected-entities').value = prompt ? (prompt.expected_entities || '') : '';
        $('pe-delete').style.display = prompt ? '' : 'none';
        $promptEditorOverlay.classList.add('active');
    }

    function closePromptEditor() {
        $promptEditorOverlay.classList.remove('active');
        editingPrompt = null;
    }

    async function savePrompt() {
        if (!currentPromptSetId) return;

        const body = {
            name: $('pe-name').value.trim(),
            category: $('pe-category').value.trim(),
            prompt: $('pe-prompt').value.trim(),
        };

        if (!body.name || !body.prompt) {
            alert('Name and prompt text are required');
            return;
        }

        const expectedAnswer = $('pe-expected-answer').value.trim();
        if (expectedAnswer) body.expected_answer = expectedAnswer;

        const expectedEntities = parseInt($('pe-expected-entities').value);
        if (expectedEntities > 0) body.expected_entities = expectedEntities;

        try {
            let resp;
            if (editingPrompt) {
                resp = await fetch(`/api/prompts/${currentPromptSetId}/items/${editingPrompt.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
            } else {
                resp = await fetch(`/api/prompts/${currentPromptSetId}/items`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });
            }

            if (resp.ok) {
                closePromptEditor();
                loadPromptItems(currentPromptSetId);
                loadPromptSets(); // refresh counts
            } else {
                const data = await resp.json();
                alert(data.error || 'Failed to save');
            }
        } catch (e) {
            console.error('Save prompt error:', e);
        }
    }

    async function deleteCurrentPrompt() {
        if (!editingPrompt || !currentPromptSetId) return;
        if (!confirm(`Delete prompt "${editingPrompt.name}"?`)) return;

        try {
            const resp = await fetch(`/api/prompts/${currentPromptSetId}/items/${editingPrompt.id}`, {
                method: 'DELETE',
            });
            if (resp.ok) {
                closePromptEditor();
                loadPromptItems(currentPromptSetId);
                loadPromptSets();
            }
        } catch (e) {
            console.error('Delete prompt error:', e);
        }
    }

    // ── Toast Feedback ─────────────────────────────────────
    function showToast(message, type) {
        // type: 'info', 'success', 'error', 'warning'
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.style.cssText = 'position:fixed;top:64px;right:16px;z-index:200;display:flex;flex-direction:column;gap:6px;pointer-events:none;';
            document.body.appendChild(container);
        }
        const colors = {
            info: 'var(--accent-secondary)',
            success: 'var(--accent-primary)',
            error: 'var(--accent-danger)',
            warning: 'var(--accent-warning)',
        };
        const color = colors[type] || colors.info;
        const toast = document.createElement('div');
        toast.style.cssText = `font-family:var(--font-mono);font-size:0.6rem;padding:0.4rem 0.8rem;background:var(--bg-card);border:1px solid ${color};color:${color};pointer-events:auto;opacity:0;transition:opacity 0.3s;`;
        toast.textContent = message;
        container.appendChild(toast);
        requestAnimationFrame(() => { toast.style.opacity = '1'; });
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // ── Helpers ──────────────────────────────────────────────
    function esc(s) {
        if (!s) return '';
        const div = document.createElement('div');
        div.textContent = String(s);
        return div.innerHTML;
    }

    // ── Keepalive ping ──────────────────────────────────────
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ cmd: 'ping' }));
        }
    }, 30000);

    // ── Event Bindings ──────────────────────────────────────
    $btnRun.addEventListener('click', startBenchmark);
    $btnStop.addEventListener('click', stopBenchmark);
    $btnRefresh.addEventListener('click', () => {
        showToast('Refreshing models...', 'info');
        loadModels().then(() => showToast('Models refreshed', 'success'));
    });
    $btnLoad.addEventListener('click', () => loadUnloadModels('load'));
    $btnUnload.addEventListener('click', () => loadUnloadModels('unload'));

    $('report-close').addEventListener('click', closeReport);
    $reportOverlay.addEventListener('click', (e) => {
        if (e.target === $reportOverlay) closeReport();
    });

    $('prompt-editor-close').addEventListener('click', closePromptEditor);
    $promptEditorOverlay.addEventListener('click', (e) => {
        if (e.target === $promptEditorOverlay) closePromptEditor();
    });
    $('pe-save').addEventListener('click', savePrompt);
    $('pe-cancel').addEventListener('click', closePromptEditor);
    $('pe-delete').addEventListener('click', deleteCurrentPrompt);
    $btnAddPrompt.addEventListener('click', () => openPromptEditor(null));

    // Backend change → re-load model list
    $('benchmark-backend').addEventListener('change', () => {
        selectedModels.clear();
        loadModels();
        updateRunButton();
    });

    // Escape key closes modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if ($reportOverlay.classList.contains('active')) closeReport();
            if ($promptEditorOverlay.classList.contains('active')) closePromptEditor();
        }
    });

    // ── Init ─────────────────────────────────────────────────
    connectWs();
    loadModels().then(() => loadLeaderboard());
    loadPromptSets();
})();
