/**
 * NEXUS AI - Synthesis Engine Controller
 * Optimized for local neural processing and high-fidelity research.
 * ============================================================================ */

const NexusApp = {
    // Neural State Matrix
    state: {
        activePage: 'home',
        isProcessing: false,
        nodes: [],
        memory: [],
        config: {
            api: '/api',
            topK: 5,
            strategy: 'hybrid'
        },
        abortController: null,
        stats: null,
        multiSynthesis: true
    },

    // --- SYSTEM INITIALIZATION ---
    async init() {
        console.log("Initializing Nexus Neural Link...");
        this.bindGlobalEvents();
        await this.synchronize();
        this.refreshUI();
        lucide.createIcons();
    },

    bindGlobalEvents() {
        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
            chatInput.addEventListener('input', () => {
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
            });
            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (!this.state.isProcessing) this.synthesize();
                }
            });
        }

        // Hero Query Input (Home Page)
        const heroInput = document.getElementById('hero-research-query');
        if (heroInput) {
            heroInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    if (!this.state.isProcessing) this.synthesize(heroInput.value);
                }
            });
        }

        // File Ingestion Logic
        const dropzone = document.getElementById('dropzone');
        const selector = document.getElementById('file-selector');
        if (dropzone && selector) {
            dropzone.onclick = () => selector.click();
            selector.onchange = (e) => this.ingestFiles(e.target.files);
            dropzone.ondragover = (e) => { 
                e.preventDefault(); 
                dropzone.classList.add('active-drop');
            };
            dropzone.ondragleave = () => dropzone.classList.remove('active-drop');
            dropzone.ondrop = (e) => {
                e.preventDefault();
                dropzone.classList.remove('active-drop');
                this.ingestFiles(e.dataTransfer.files);
            };
        }
    },

    async synchronize() {
        try {
            const [nodes, memory] = await Promise.all([
                fetch(`${this.state.config.api}/documents`).then(r => r.json()),
                fetch(`${this.state.config.api}/history`).then(r => r.json())
            ]);
            this.state.nodes = nodes;
            this.state.memory = memory;
            return true;
        } catch (e) {
            this.notify("Neural link desynchronized", "error");
            return false;
        }
    },

    // --- NAVIGATION & VIEWPORT ---
    navigate(pageId) {
        if (this.state.activePage === pageId) return;
        this.state.activePage = pageId;

        // Viewport Orchestration
        document.querySelectorAll('.page-content').forEach(p => p.classList.toggle('active', p.id === `page-${pageId}`));
        document.querySelectorAll('.nav-link').forEach(l => l.classList.toggle('active', l.id === `nav-${pageId}`));

        // Title Matrix
        const titles = {
            'home': 'Discovery Hub',
            'chat': 'Research Synthesis',
            'library': 'Knowledge Base',
            'settings': 'Logic Controls',
            'stats': 'Neural Metrics',
            'export': 'Synthesis Review'
        };
        const titleEl = document.getElementById('breadcrumb-active');
        if (titleEl) titleEl.innerText = titles[pageId] || 'Command Center';

        // Deferred Renders
        if (pageId === 'library') this.renderLibrary();
        if (pageId === 'stats') this.renderMetrics();
        if (pageId === 'export') this.renderSynthesisArchive();
        
        lucide.createIcons();

        // Auto-close overlay navigation
        const sidebar = document.getElementById('sidebar');
        if (sidebar) sidebar.classList.remove('open');
    },

    // --- SYNTHESIS CORE ---
    async synthesize(customQuery = null) {
        if (this.state.isProcessing) return;
        
        const inputEl = document.getElementById('chat-input');
        const query = (customQuery || (inputEl ? inputEl.value : "")).trim();
        if (!query) return;

        this.navigate('chat');
        this.appendMessage('user', query);
        if (inputEl) {
            inputEl.value = '';
            inputEl.style.height = 'auto';
        }
        const heroEl = document.getElementById('hero-research-query');
        if (heroEl) heroEl.value = '';

        // Prepare AI Response Channel
        const msgWrapper = this.appendMessage('ai', '');
        const contentArea = msgWrapper.querySelector('.bubble');
        contentArea.innerHTML = `<div class="rich-content"></div>`;
        const richEl = contentArea.querySelector('.rich-content');

        this.state.isProcessing = true;
        this.updateProcessingState(true);
        this.state.abortController = new AbortController();

        try {
            const res = await fetch(`${this.state.config.api}/query/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    top_k: this.state.config.topK,
                    search_type: this.state.config.strategy
                }),
                signal: this.state.abortController.signal
            });

            if (!res.ok) throw new Error("Link saturation failure");

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let synthesisBuffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                synthesisBuffer += decoder.decode(value, { stream: true });
                richEl.innerHTML = this.formatMarkdown(synthesisBuffer);
                this.scrollToBottom();
            }

            // Synthesis Complete
            this.notify("Synthesis complete");
            await this.synchronize();
            this.refreshUI();
        } catch (e) {
            if (e.name === 'AbortError') {
                richEl.innerHTML += `<div class="synthesis-halted">[Manual Halt Triggered]</div>`;
            } else {
                richEl.innerHTML = `<div class="neural-error">Synthesis Fault: ${e.message}</div>`;
            }
        } finally {
            this.state.isProcessing = false;
            this.updateProcessingState(false);
        }
    },

    halt() {
        if (this.state.abortController) this.state.abortController.abort();
    },

    // --- UI ATOMICS ---
    appendMessage(role, text) {
        const container = document.getElementById('chat-history');
        const row = document.createElement('div');
        row.className = `msg-row ${role} animate-in`;
        row.innerHTML = `<div class="bubble">${role === 'user' ? this.sanitize(text) : text}</div>`;
        container.appendChild(row);
        this.scrollToBottom();
        return row;
    },

    updateProcessingState(active) {
        const sendBtn = document.getElementById('chat-send-btn');
        const stopBtn = document.getElementById('chat-stop-btn');
        
        if (active) {
            if (sendBtn) sendBtn.style.display = 'none';
            if (stopBtn) stopBtn.style.display = 'flex';
        } else {
            if (sendBtn) sendBtn.style.display = 'flex';
            if (stopBtn) stopBtn.style.display = 'none';
        }
    },

    scrollToBottom() {
        const container = document.getElementById('chat-history');
        if (container) container.scrollTop = container.scrollHeight;
    },

    notify(msg, type = 'success') {
        const anchor = document.getElementById('toast-anchor');
        if (!anchor) return;
        const toast = document.createElement('div');
        toast.className = `toast-msg ${type}`;
        toast.innerText = msg;
        anchor.appendChild(toast);
        setTimeout(() => toast.classList.add('fade-out'), 3500);
        setTimeout(() => toast.remove(), 4000);
    },

    // --- RENDER ENGINES ---
    renderLibrary() {
        const grid = document.getElementById('library-grid');
        if (!this.state.nodes.length) {
            grid.innerHTML = `<div class="empty-state"><i data-lucide="database"></i><p>Neural Index Empty</p></div>`;
            lucide.createIcons();
            return;
        }

        grid.innerHTML = this.state.nodes.map(node => `
            <div class="doc-tile">
                <div class="doc-type-icon"><i data-lucide="${node.source_type === 'url' ? 'link' : 'file-text'}"></i></div>
                <div class="doc-name" title="${node.source}">${node.source}</div>
                <div class="doc-stats">${node.chunk_count} Fragments • ${new Date(node.created_at).toLocaleDateString()}</div>
                <div class="doc-footer">
                    <button class="tile-btn view" onclick="NexusApp.view('${node.source}')"><i data-lucide="eye"></i><span>View</span></button>
                    <button class="tile-btn del" onclick="NexusApp.purge('${node.source}')"><i data-lucide="trash-2"></i><span>Purge</span></button>
                </div>
            </div>`).join('');
        lucide.createIcons();
    },

    renderHistory() {
        const list = document.getElementById('sidebar-history-list');
        if (!this.state.memory.length) {
            list.innerHTML = `<div class="sidebar-empty">Zero Nodes Cached</div>`;
            return;
        }

        list.innerHTML = this.state.memory.slice(0, 15).map(item => `
            <div class="history-item-nav">
                <div class="nav-content" onclick="NexusApp.restoreNode('${item.id}')">
                    <div class="nav-query">${this.sanitize(item.query)}</div>
                    <div class="nav-ts">${new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                </div>
                <button class="nav-delete" onclick="NexusApp.deleteHistoryNode('${item.id}', event)"><i data-lucide="x"></i></button>
            </div>`).join('');
        lucide.createIcons();
    },

    renderMetrics() {
        const stats = this.state.stats;
        if (!stats) return;
        const grid = document.getElementById('stats-page-grid');
        const sizeMb = (stats.storage_size_bytes / (1024 * 1024)).toFixed(2);

        grid.innerHTML = `
            <div class="metric-card">
                <div class="metric-val">${stats.total_documents}</div>
                <div class="metric-label">Knowledge Nodes</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">${stats.total_chunks}</div>
                <div class="metric-label">Neural Fragments</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">${sizeMb} MB</div>
                <div class="metric-label">Index Volume</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">Ready</div>
                <div class="metric-label">Fabric Integrity</div>
            </div>`;
    },

    renderSynthesisArchive() {
        const container = document.getElementById('export-preview-container');
        if (!this.state.memory.length) {
            container.innerHTML = `<div class="empty-state"><p>Archive Empty</p></div>`;
            return;
        }

        container.innerHTML = this.state.memory.map((item, i) => `
            <div class="archive-block">
                <span class="block-tag">Thread ${i + 1}</span>
                <h2>${this.sanitize(item.query)}</h2>
                <div class="block-content">${this.formatMarkdown(item.answer)}</div>
            </div>`).join('');
    },

    // --- DATA MUTATION ---
    async ingestFiles(files) {
        if (!files.length) return;
        const wrap = document.getElementById('upload-progress-wrap');
        const bar = document.getElementById('upload-bar');
        wrap.style.display = 'block';

        for (let i = 0; i < files.length; i++) {
            const formData = new FormData();
            formData.append('file', files[i]);
            
            bar.style.width = `${((i + 1) / files.length) * 100}%`;
            try {
                await fetch(`${this.state.config.api}/upload`, { method: 'POST', body: formData });
                this.notify(`Ingested: ${files[i].name}`);
            } catch (e) {
                this.notify(`Fault: ${files[i].name}`, 'error');
            }
        }

        setTimeout(async () => {
            await this.synchronize();
            this.refreshUI();
            this.closeModal('upload');
            wrap.style.display = 'none';
        }, 800);
    },

    async purge(source) {
        if (!confirm(`Permanently excise node "${source}"?`)) return;
        try {
            const res = await fetch(`${this.state.config.api}/documents/${encodeURIComponent(source)}`, { method: 'DELETE' });
            if (res.ok) {
                this.notify("Node purged");
                await this.synchronize();
                this.refreshUI();
            }
        } catch (e) { this.notify("Purge failed", "error"); }
    },

    async purgeEverything() {
        if (!confirm("Nuclear Wipe: All neural fragments will be destroyed. Confirm?")) return;
        try {
            await fetch(`${this.state.config.api}/documents`, { method: 'DELETE' });
            this.notify("Neural fabric reset");
            await this.synchronize();
            this.refreshUI();
            this.closeModal('stats');
        } catch (e) { this.notify("Purge failed", "error"); }
    },

    // --- UTILITIES ---
    async refreshUI() {
        this.renderHistory();
        if (this.state.activePage === 'library') this.renderLibrary();
        const res = await fetch(`${this.state.config.api}/stats`);
        this.state.stats = await res.json();
        const statusEl = document.getElementById('stats-inline-val');
        if (statusEl) statusEl.innerText = `Index: ${this.state.stats.total_documents} Nodes`;
    },

    formatMarkdown(text) {
        if (!text) return "";
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/^- (.*)/gm, '<li>$1</li>')
            .replace(/### (.*)/g, '<h3>$1</h3>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .split('</p><p>').map(p => p.includes('<li>') ? `<ul>${p}</ul>` : `<p>${p}</p>`).join('');
    },

    sanitize(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    view(source) {
        const frame = document.getElementById('viewer-frame');
        document.getElementById('viewer-doc-title').innerText = source;
        frame.src = `${this.state.config.api}/documents/${encodeURIComponent(source)}`;
        this.openModal('viewer');
    },

    restoreNode(id) {
        const node = this.state.memory.find(m => m.id === id);
        if (!node) return;
        this.navigate('chat');
        const hist = document.getElementById('chat-history');
        hist.innerHTML = '';
        this.appendMessage('user', node.query);
        const ai = this.appendMessage('ai', '');
        ai.querySelector('.bubble').innerHTML = `<div class="rich-content">${this.formatMarkdown(node.answer)}</div>`;
        this.notify("Knowledge node restored");
    },

    openModal(id) { document.getElementById(`${id}-overlay`).style.display = 'flex'; },
    closeModal(id) { document.getElementById(`${id}-overlay`).style.display = 'none'; },
    
    export() {
        if (!this.state.memory.length) return this.notify("Memory empty", "error");
        
        let md = `# Nexus Synthesis Report - ${new Date().toLocaleDateString()}\n\n`;
        this.state.memory.forEach((n, idx) => {
            md += `### Thread ${idx + 1}\n**Query:** ${n.query}\n\n${n.answer}\n\n---\n\n`;
        });

        const blob = new Blob([md], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Synthesis_Export_${Date.now()}.md`;
        a.click();
        URL.revokeObjectURL(url);
        this.notify("Archive exported");
    },

    toggleSynthesis() {
        this.state.multiSynthesis = !this.state.multiSynthesis;
        const el = document.getElementById('toggle-synthesis');
        if (el) el.classList.toggle('active', this.state.multiSynthesis);
        this.notify(`Cross-Document Synthesis: ${this.state.multiSynthesis ? 'ON' : 'OFF'}`);
        
        // Strategy modulation
        this.state.config.strategy = this.state.multiSynthesis ? 'hybrid' : 'similarity';
        this.state.config.topK = this.state.multiSynthesis ? 5 : 2;
    },

    setModel(type) {
        this.state.config.model = type;
        const preciseEl = document.getElementById('model-precise');
        const fastEl = document.getElementById('model-fast');
        
        if (preciseEl) {
            preciseEl.style.borderColor = type === 'precise' ? 'var(--primary)' : 'var(--border-subtle)';
            preciseEl.style.background = type === 'precise' ? 'var(--primary-soft)' : 'var(--bg-surface)';
        }
        if (fastEl) {
            fastEl.style.borderColor = type === 'fast' ? 'var(--primary)' : 'var(--border-subtle)';
            fastEl.style.background = type === 'fast' ? 'var(--primary-soft)' : 'var(--bg-surface)';
        }
        
        this.notify(`Model: ${type === 'precise' ? 'Mistral Large' : 'Mistral Small'}`);
    }
};

// Global Bridge
window.NexusApp = NexusApp;
window.navigate = (p) => NexusApp.navigate(p);
window.openModal = (id) => NexusApp.openModal(id);
window.closeModal = (id) => NexusApp.closeModal(id);
window.triggerHeroQuery = () => {
    const q = document.getElementById('hero-research-query').value;
    NexusApp.synthesize(q);
};
window.addEventListener('load', () => NexusApp.init());
