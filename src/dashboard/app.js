const API_BASE = window.location.origin;

class ViraDashboard {
    constructor() {
        this.currentTab = 'overview';
        this.ws = null;
        this.init();
    }

    async init() {
        this.bindNavigation();
        this.bindModal();
        this.bindActions();
        this.loadStats();
        this.loadTab('overview');
        this.connectWebSocket();
    }

    bindNavigation() {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', () => {
                const tab = item.dataset.tab;
                this.switchTab(tab);
            });
        });
    }

    switchTab(tab) {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.tab === tab);
        });

        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === tab);
        });

        document.getElementById('page-title').textContent = this.formatTitle(tab);
        this.currentTab = tab;
        this.loadTab(tab);
    }

    formatTitle(tab) {
        return tab.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    }

    async loadTab(tab) {
        switch (tab) {
            case 'overview': await this.loadOverview(); break;
            case 'memories': await this.loadMemories(); break;
            case 'schedules': await this.loadSchedules(); break;
            case 'personas': await this.loadPersonas(); break;
            case 'entities': await this.loadEntities(); break;
            case 'triples': await this.loadTriples(); break;
            case 'chat-logs': await this.loadChatLogs(); break;
            case 'neural-events': await this.loadNeuralEvents(); break;
            case 'settings': await this.loadSettings(); break;
        }
    }

    async loadStats() {
        try {
            const stats = await this.api('/api/stats');
            document.getElementById('stat-memories').textContent = stats.memories || 0;
            document.getElementById('stat-schedules').textContent = stats.schedules || 0;
            document.getElementById('stat-personas').textContent = stats.personas || 0;
            document.getElementById('stat-entities').textContent = stats.entities || 0;
        } catch (e) {
            console.error('Failed to load stats:', e);
        }
    }

    async loadOverview() {
        await this.loadStats();

        try {
            const logs = await this.api('/api/chat-logs?limit=5');
            const container = document.getElementById('recent-activity');

            if (!logs || logs.length === 0) {
                container.innerHTML = '<p class="empty-state">No recent activity</p>';
                return;
            }

            container.innerHTML = logs.map(log => `
                <div class="chat-message">
                    <div class="chat-avatar ${log.role === 'user' ? 'user' : 'ai'}">
                        ${log.role === 'user' ? '👤' : '🧠'}
                    </div>
                    <div class="chat-content">
                        <div class="chat-meta">
                            <span class="chat-role">${log.role === 'user' ? 'User' : 'Vira'}</span>
                            <span class="chat-time">${this.formatDate(log.timestamp)}</span>
                        </div>
                        <div class="chat-text">${this.escapeHtml(log.content?.substring(0, 150) || '')}${log.content?.length > 150 ? '...' : ''}</div>
                    </div>
                </div>
            `).join('');
        } catch (e) {
            console.error('Failed to load recent activity:', e);
        }
    }

    async loadMemories() {
        try {
            const memories = await this.api('/api/memories');
            const tbody = document.getElementById('memories-list');

            if (!memories || memories.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No memories found</td></tr>';
                return;
            }

            tbody.innerHTML = memories.map(m => `
                <tr>
                    <td>${this.escapeHtml(m.summary?.substring(0, 80) || '')}${m.summary?.length > 80 ? '...' : ''}</td>
                    <td><span class="badge badge-info">${m.memory_type || 'general'}</span></td>
                    <td>${((m.confidence || 0) * 100).toFixed(0)}%</td>
                    <td>${this.formatDate(m.created_at)}</td>
                    <td>
                        <button class="btn btn-sm btn-secondary" onclick="dashboard.viewMemory('${m.id}')">View</button>
                        <button class="btn btn-sm btn-danger" onclick="dashboard.deleteMemory('${m.id}')">Delete</button>
                    </td>
                </tr>
            `).join('');
        } catch (e) {
            console.error('Failed to load memories:', e);
        }
    }

    async loadSchedules() {
        try {
            const schedules = await this.api('/api/schedules');
            const tbody = document.getElementById('schedules-list');

            if (!schedules || schedules.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No schedules found</td></tr>';
                return;
            }

            tbody.innerHTML = schedules.map(s => `
                <tr>
                    <td>${this.escapeHtml(s.context || '')}</td>
                    <td>${this.formatDate(s.scheduled_at || s.trigger_time)}</td>
                    <td><span class="badge ${this.getStatusBadge(s.status)}">${s.status || 'pending'}</span></td>
                    <td>${this.formatDate(s.created_at)}</td>
                    <td>
                        <button class="btn btn-sm btn-secondary" onclick="dashboard.editSchedule('${s.id}')">Edit</button>
                        <button class="btn btn-sm btn-danger" onclick="dashboard.deleteSchedule('${s.id}')">Delete</button>
                    </td>
                </tr>
            `).join('');
        } catch (e) {
            console.error('Failed to load schedules:', e);
        }
    }

    async loadPersonas() {
        try {
            const personas = await this.api('/api/personas');
            const container = document.getElementById('personas-list');

            if (!personas || personas.length === 0) {
                container.innerHTML = '<p class="empty-state">No personas found. Create one!</p>';
                return;
            }

            container.innerHTML = personas.map(p => `
                <div class="card ${p.is_active ? 'active' : ''}">
                    <div class="card-header">
                        <span class="card-title">${this.escapeHtml(p.name)}</span>
                        ${p.is_active ? '<span class="card-badge active">Active</span>' : ''}
                    </div>
                    <div class="card-body">
                        <p>${this.escapeHtml(p.description || 'No description')}</p>
                        <p style="margin-top: 8px; font-size: 12px; color: var(--text-muted);">
                            Temperature: ${p.temperature} | Tone: ${p.voice_tone}
                        </p>
                    </div>
                    <div class="card-footer">
                        ${!p.is_active ? `<button class="btn btn-sm btn-primary" onclick="dashboard.activatePersona('${p.id}')">Activate</button>` : ''}
                        <button class="btn btn-sm btn-secondary" onclick="dashboard.editPersona('${p.id}')">Edit</button>
                        <button class="btn btn-sm btn-danger" onclick="dashboard.deletePersona('${p.id}')">Delete</button>
                    </div>
                </div>
            `).join('');
        } catch (e) {
            console.error('Failed to load personas:', e);
        }
    }

    async loadEntities() {
        try {
            const entities = await this.api('/api/entities');
            const tbody = document.getElementById('entities-list');

            if (!entities || entities.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No entities found</td></tr>';
                return;
            }

            tbody.innerHTML = entities.map(e => `
                <tr>
                    <td><strong>${this.escapeHtml(e.name)}</strong></td>
                    <td><span class="badge badge-default">${e.entity_type || 'unknown'}</span></td>
                    <td>${(e.aliases || []).slice(0, 3).join(', ') || '-'}</td>
                    <td>${e.mention_count || 0}</td>
                    <td>
                        <button class="btn btn-sm btn-secondary" onclick="dashboard.viewEntity('${encodeURIComponent(e.name)}')">View</button>
                        <button class="btn btn-sm btn-danger" onclick="dashboard.deleteEntity('${encodeURIComponent(e.name)}')">Delete</button>
                    </td>
                </tr>
            `).join('');
        } catch (e) {
            console.error('Failed to load entities:', e);
        }
    }

    async loadTriples() {
        try {
            const triples = await this.api('/api/triples');
            const tbody = document.getElementById('triples-list');

            if (!triples || triples.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No triples found</td></tr>';
                return;
            }

            tbody.innerHTML = triples.map(t => `
                <tr>
                    <td>${this.escapeHtml(t.subject)}</td>
                    <td><span class="badge badge-info">${this.escapeHtml(t.predicate)}</span></td>
                    <td>${this.escapeHtml(t.object)}</td>
                    <td>${((t.confidence || 0) * 100).toFixed(0)}%</td>
                    <td>
                        <button class="btn btn-sm btn-danger" onclick="dashboard.deleteTriple('${t.id}')">Delete</button>
                    </td>
                </tr>
            `).join('');
        } catch (e) {
            console.error('Failed to load triples:', e);
        }
    }

    async loadChatLogs() {
        try {
            const logs = await this.api('/api/chat-logs?limit=50');
            const container = document.getElementById('chat-logs-list');

            if (!logs || logs.length === 0) {
                container.innerHTML = '<p class="empty-state">No chat logs found</p>';
                return;
            }

            container.innerHTML = logs.map(log => `
                <div class="chat-message">
                    <div class="chat-avatar ${log.role === 'user' ? 'user' : 'ai'}">
                        ${log.role === 'user' ? '👤' : '🧠'}
                    </div>
                    <div class="chat-content">
                        <div class="chat-meta">
                            <span class="chat-role">${log.role === 'user' ? 'User' : 'Vira'}</span>
                            <span class="chat-time">${this.formatDate(log.timestamp)}</span>
                        </div>
                        <div class="chat-text">${this.escapeHtml(log.content || '')}</div>
                    </div>
                </div>
            `).join('');
        } catch (e) {
            console.error('Failed to load chat logs:', e);
        }
    }

    async loadNeuralEvents() {
        try {
            const data = await this.api('/api/neural-events?limit=50');
            const tbody = document.getElementById('neural-events-list');

            if (!data.events || data.events.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No events found</td></tr>';
                return;
            }

            tbody.innerHTML = data.events.map(e => `
                <tr>
                    <td>${this.formatDate(e.timestamp)}</td>
                    <td><span class="badge ${this.getEventBadge(e.event_type)}">${e.event_type}</span></td>
                    <td>${this.escapeHtml(e.component || '-')}</td>
                    <td>${this.escapeHtml(JSON.stringify(e.data || {})).substring(0, 100)}</td>
                </tr>
            `).join('');
        } catch (e) {
            console.error('Failed to load neural events:', e);
        }
    }

    getEventBadge(type) {
        if (type.includes('error') || type.includes('failure')) return 'badge-danger';
        if (type.includes('warning')) return 'badge-warning';
        if (type.includes('success') || type.includes('complete')) return 'badge-success';
        return 'badge-info';
    }

    async loadModelHealth() {
        try {
            const status = await this.api('/api/openrouter/health');
            const indicator = document.getElementById('model-status-indicator');
            const tier = document.getElementById('model-active-tier');
            const list = document.getElementById('model-health-list');

            if (status.status === 'healthy') {
                indicator.className = 'badge badge-success';
                indicator.textContent = 'Healthy';
            } else {
                indicator.className = 'badge badge-danger';
                indicator.textContent = status.status;
            }

            tier.textContent = status.active_tier || 'None';

            if (status.model_health) {
                list.innerHTML = Object.entries(status.model_health).map(([model, info]) => `
                    <div style="margin-bottom: 5px; font-size: 13px;">
                        <strong>${model}:</strong> 
                        <span class="badge ${info.healthy ? 'badge-success' : 'badge-danger'}">${info.healthy ? 'OK' : 'Issues'}</span>
                        <span style="color: var(--text-muted)">(${info.errors || 0} errors)</span>
                    </div>
                `).join('');
            }
        } catch (e) {
            console.error('Failed to load model health:', e);
        }
    }

    async loadSettings() {
        try {
            const ctx = await this.api('/api/global-context');
            document.getElementById('global-context').value = ctx.context_text || '';
        } catch (e) { }

        try {
            const config = await this.api('/api/system-config');
            document.getElementById('temperature').value = config.temperature || 0.7;
            document.getElementById('temp-value').textContent = config.temperature || 0.7;
            document.getElementById('max-tokens').value = config.max_output_tokens || 512;
        } catch (e) { }

        this.loadModelHealth();

        document.getElementById('temperature').addEventListener('input', (e) => {
            document.getElementById('temp-value').textContent = e.target.value;
        });
    }

    bindModal() {
        const modal = document.getElementById('modal');
        modal.querySelector('.modal-backdrop').addEventListener('click', () => this.closeModal());
        modal.querySelector('.modal-close').addEventListener('click', () => this.closeModal());
        modal.querySelector('.modal-cancel').addEventListener('click', () => this.closeModal());
    }

    openModal(title, content, onConfirm) {
        const modal = document.getElementById('modal');
        document.getElementById('modal-title').textContent = title;
        document.getElementById('modal-body').innerHTML = content;
        modal.classList.add('active');

        const confirmBtn = modal.querySelector('.modal-confirm');
        confirmBtn.onclick = async () => {
            if (onConfirm) await onConfirm();
            this.closeModal();
        };
    }

    closeModal() {
        document.getElementById('modal').classList.remove('active');
    }

    bindActions() {
        document.getElementById('refresh-btn').addEventListener('click', () => {
            this.loadTab(this.currentTab);
            this.showToast('Refreshed!', 'success');
        });

        document.getElementById('add-persona-btn')?.addEventListener('click', () => this.showPersonaModal());
        document.getElementById('add-memory-btn')?.addEventListener('click', () => this.showMemoryModal());
        document.getElementById('add-schedule-btn')?.addEventListener('click', () => this.showScheduleModal());
        document.getElementById('add-triple-btn')?.addEventListener('click', () => this.showTripleModal());

        document.getElementById('save-context-btn')?.addEventListener('click', async () => {
            const text = document.getElementById('global-context').value;
            await this.api('/api/global-context', 'PUT', { context_text: text });
            this.showToast('Context saved!', 'success');
        });

        document.getElementById('save-config-btn')?.addEventListener('click', async () => {
            const temp = parseFloat(document.getElementById('temperature').value);
            const tokens = parseInt(document.getElementById('max-tokens').value);
            await this.api('/api/system-config', 'PUT', { temperature: temp, max_output_tokens: tokens });
            this.showToast('Configuration saved!', 'success');
        });

        document.getElementById('trigger-maintenance-btn')?.addEventListener('click', async () => {
            const res = await this.api('/api/maintenance/trigger', 'POST');
            this.showToast(`Maintenance ${res.status}`, 'success');
        });

        document.getElementById('compress-memories-btn')?.addEventListener('click', async () => {
            const res = await this.api('/api/maintenance/compress-memories', 'POST', { force: false });
            this.showToast(`Compressed ${res.compressed || 0} memories`, 'success');
        });

        document.getElementById('optimize-graph-btn')?.addEventListener('click', async () => {
            const res = await this.api('/api/maintenance/optimize-graph', 'POST');
            this.showToast(`Removed ${res.duplicates_removed || 0} duplicates`, 'success');
        });

        document.getElementById('clear-logs-btn')?.addEventListener('click', () => {
            this.openModal('Clear Chat Logs', '<p>Are you sure you want to clear all chat logs?</p>', async () => {
                await this.api('/api/chat-logs/sessions', 'DELETE');
                this.loadChatLogs();
                this.showToast('Chat logs cleared!', 'success');
            });
        });

        document.getElementById('reset-model-health-btn')?.addEventListener('click', async () => {
            await this.api('/api/openrouter/reset-health', 'POST');
            this.loadModelHealth();
            this.showToast('Model health metrics reset!', 'success');
        });

        document.getElementById('refresh-events-btn')?.addEventListener('click', () => {
            this.loadNeuralEvents();
            this.showToast('Events refreshed!', 'success');
        });

        document.getElementById('memory-search')?.addEventListener('input', (e) => {
            this.searchMemories(e.target.value);
        });
    }

    async searchMemories(query) {
        if (!query || query.length < 2) {
            this.loadMemories();
            return;
        }
        try {
            const results = await this.api(`/api/search/memories?q=${encodeURIComponent(query)}`);
            const tbody = document.getElementById('memories-list');
            if (!results || results.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No results</td></tr>';
                return;
            }
            tbody.innerHTML = results.map(m => `
                <tr>
                    <td>${this.escapeHtml(m.summary?.substring(0, 80) || '')}</td>
                    <td><span class="badge badge-info">${m.memory_type || 'general'}</span></td>
                    <td>${((m.confidence || 0) * 100).toFixed(0)}%</td>
                    <td>${this.formatDate(m.created_at)}</td>
                    <td>
                        <button class="btn btn-sm btn-secondary" onclick="dashboard.viewMemory('${m.id}')">View</button>
                        <button class="btn btn-sm btn-danger" onclick="dashboard.deleteMemory('${m.id}')">Delete</button>
                    </td>
                </tr>
            `).join('');
        } catch (e) {
            console.error('Search failed:', e);
        }
    }

    showPersonaModal(persona = null) {
        const isEdit = !!persona;
        const content = `
            <div class="form-group">
                <label>Name</label>
                <input type="text" id="persona-name" value="${persona?.name || ''}" placeholder="Persona name">
            </div>
            <div class="form-group">
                <label>Description</label>
                <input type="text" id="persona-desc" value="${persona?.description || ''}" placeholder="Brief description">
            </div>
            <div class="form-group">
                <label>Instruction</label>
                <textarea id="persona-instruction" style="width:100%;min-height:100px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;padding:12px;color:var(--text-primary)">${persona?.instruction || ''}</textarea>
            </div>
            <div class="form-group">
                <label>Temperature</label>
                <input type="range" id="persona-temp" min="0" max="1" step="0.1" value="${persona?.temperature || 0.7}" style="width:80%">
                <span id="persona-temp-val">${persona?.temperature || 0.7}</span>
            </div>
            <div class="form-group">
                <label>Voice Tone</label>
                <select id="persona-tone" style="width:100%;padding:12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary)">
                    <option value="friendly" ${persona?.voice_tone === 'friendly' ? 'selected' : ''}>Friendly</option>
                    <option value="professional" ${persona?.voice_tone === 'professional' ? 'selected' : ''}>Professional</option>
                    <option value="casual" ${persona?.voice_tone === 'casual' ? 'selected' : ''}>Casual</option>
                    <option value="formal" ${persona?.voice_tone === 'formal' ? 'selected' : ''}>Formal</option>
                </select>
            </div>
        `;

        this.openModal(isEdit ? 'Edit Persona' : 'Create Persona', content, async () => {
            const data = {
                name: document.getElementById('persona-name').value,
                description: document.getElementById('persona-desc').value,
                instruction: document.getElementById('persona-instruction').value,
                temperature: parseFloat(document.getElementById('persona-temp').value),
                voice_tone: document.getElementById('persona-tone').value
            };

            if (isEdit) {
                await this.api(`/api/personas/${persona.id}`, 'PUT', data);
                this.showToast('Persona updated!', 'success');
            } else {
                await this.api('/api/personas', 'POST', data);
                this.showToast('Persona created!', 'success');
            }
            this.loadPersonas();
        });

        setTimeout(() => {
            document.getElementById('persona-temp')?.addEventListener('input', (e) => {
                document.getElementById('persona-temp-val').textContent = e.target.value;
            });
        }, 100);
    }

    showMemoryModal() {
        const content = `
            <div class="form-group">
                <label>Summary</label>
                <textarea id="memory-summary" style="width:100%;min-height:100px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;padding:12px;color:var(--text-primary)" placeholder="Memory content..."></textarea>
            </div>
            <div class="form-group">
                <label>Type</label>
                <select id="memory-type" style="width:100%;padding:12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary)">
                    <option value="general">General</option>
                    <option value="preference">Preference</option>
                    <option value="fact">Fact</option>
                    <option value="event">Event</option>
                </select>
            </div>
            <div class="form-group">
                <label>Priority (0-1)</label>
                <input type="number" id="memory-priority" value="0.5" min="0" max="1" step="0.1" style="width:100%;padding:12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary)">
            </div>
        `;

        this.openModal('Add Memory', content, async () => {
            const data = {
                summary: document.getElementById('memory-summary').value,
                memory_type: document.getElementById('memory-type').value,
                priority: parseFloat(document.getElementById('memory-priority').value)
            };
            await this.api('/api/memories', 'POST', data);
            this.showToast('Memory created!', 'success');
            this.loadMemories();
        });
    }

    showScheduleModal() {
        const now = new Date().toISOString().slice(0, 16);
        const content = `
            <div class="form-group">
                <label>Context/Task</label>
                <textarea id="schedule-context" style="width:100%;min-height:80px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;padding:12px;color:var(--text-primary)" placeholder="What should be done..."></textarea>
            </div>
            <div class="form-group">
                <label>Scheduled Time</label>
                <input type="datetime-local" id="schedule-time" value="${now}" style="width:100%;padding:12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary)">
            </div>
            <div class="form-group">
                <label>Priority (0-10)</label>
                <input type="number" id="schedule-priority" value="5" min="0" max="10" style="width:100%;padding:12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary)">
            </div>
        `;

        this.openModal('Add Schedule', content, async () => {
            const data = {
                context: document.getElementById('schedule-context').value,
                trigger_time: document.getElementById('schedule-time').value,
                priority: parseInt(document.getElementById('schedule-priority').value)
            };
            await this.api('/api/schedules', 'POST', data);
            this.showToast('Schedule created!', 'success');
            this.loadSchedules();
        });
    }

    showTripleModal() {
        const content = `
            <div class="form-group">
                <label>Subject</label>
                <input type="text" id="triple-subject" style="width:100%;padding:12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary)" placeholder="e.g., John">
            </div>
            <div class="form-group">
                <label>Predicate</label>
                <input type="text" id="triple-predicate" style="width:100%;padding:12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary)" placeholder="e.g., likes">
            </div>
            <div class="form-group">
                <label>Object</label>
                <input type="text" id="triple-object" style="width:100%;padding:12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary)" placeholder="e.g., coffee">
            </div>
            <div class="form-group">
                <label>Confidence (0-1)</label>
                <input type="number" id="triple-confidence" value="0.8" min="0" max="1" step="0.1" style="width:100%;padding:12px;background:var(--bg-tertiary);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary)">
            </div>
        `;

        this.openModal('Add Triple (Knowledge)', content, async () => {
            const data = {
                subject: document.getElementById('triple-subject').value,
                predicate: document.getElementById('triple-predicate').value,
                object: document.getElementById('triple-object').value,
                confidence: parseFloat(document.getElementById('triple-confidence').value)
            };
            await this.api('/api/triples', 'POST', data);
            this.showToast('Triple created!', 'success');
            this.loadTriples();
        });
    }

    async viewMemory(id) {
        try {
            const memory = await this.api(`/api/memories/${id}`);
            const content = `
                <div style="color:var(--text-secondary)">
                    <p><strong>Summary:</strong> ${this.escapeHtml(memory.summary || '')}</p>
                    <p><strong>Type:</strong> ${memory.memory_type || 'general'}</p>
                    <p><strong>Priority:</strong> ${memory.priority || 0}</p>
                    <p><strong>Confidence:</strong> ${((memory.confidence || 0) * 100).toFixed(0)}%</p>
                    <p><strong>Created:</strong> ${this.formatDate(memory.created_at)}</p>
                    <p><strong>Tags:</strong> ${(memory.tags || []).join(', ') || 'none'}</p>
                </div>
            `;
            this.openModal('Memory Details', content, () => { });
        } catch (e) {
            this.showToast('Failed to load memory', 'error');
        }
    }

    async viewEntity(name) {
        try {
            const entity = await this.api(`/api/entities/${name}`);
            const content = `
                <div style="color:var(--text-secondary)">
                    <p><strong>Name:</strong> ${this.escapeHtml(entity.name || '')}</p>
                    <p><strong>Type:</strong> ${entity.entity_type || 'unknown'}</p>
                    <p><strong>Aliases:</strong> ${(entity.aliases || []).join(', ') || 'none'}</p>
                    <p><strong>Mentions:</strong> ${entity.mention_count || 0}</p>
                </div>
            `;
            this.openModal('Entity Details', content, () => { });
        } catch (e) {
            this.showToast('Failed to load entity', 'error');
        }
    }

    async editPersona(id) {
        try {
            const personas = await this.api('/api/personas');
            const persona = personas.find(p => p.id === id);
            if (persona) this.showPersonaModal(persona);
        } catch (e) {
            this.showToast('Failed to load persona', 'error');
        }
    }

    async activatePersona(id) {
        try {
            await this.api(`/api/personas/${id}/activate`, 'POST');
            this.showToast('Persona activated!', 'success');
            this.loadPersonas();
        } catch (e) {
            this.showToast('Failed to activate persona', 'error');
        }
    }

    async deletePersona(id) {
        this.openModal('Delete Persona', '<p>Are you sure you want to delete this persona?</p>', async () => {
            try {
                await this.api(`/api/personas/${id}`, 'DELETE');
                this.showToast('Persona deleted!', 'success');
                this.loadPersonas();
            } catch (e) {
                this.showToast('Failed to delete persona', 'error');
            }
        });
    }

    async deleteMemory(id) {
        this.openModal('Delete Memory', '<p>Are you sure you want to delete this memory?</p>', async () => {
            try {
                await this.api(`/api/memories/${id}`, 'DELETE');
                this.showToast('Memory deleted!', 'success');
                this.loadMemories();
            } catch (e) {
                this.showToast('Failed to delete memory', 'error');
            }
        });
    }

    async deleteSchedule(id) {
        this.openModal('Delete Schedule', '<p>Are you sure you want to delete this schedule?</p>', async () => {
            try {
                await this.api(`/api/schedules/${id}`, 'DELETE');
                this.showToast('Schedule deleted!', 'success');
                this.loadSchedules();
            } catch (e) {
                this.showToast('Failed to delete schedule', 'error');
            }
        });
    }

    async deleteEntity(name) {
        this.openModal('Delete Entity', '<p>Are you sure you want to delete this entity?</p>', async () => {
            try {
                await this.api(`/api/entities/${name}`, 'DELETE');
                this.showToast('Entity deleted!', 'success');
                this.loadEntities();
            } catch (e) {
                this.showToast('Failed to delete entity', 'error');
            }
        });
    }

    async deleteTriple(id) {
        this.openModal('Delete Triple', '<p>Are you sure you want to delete this triple?</p>', async () => {
            try {
                await this.api(`/api/triples/${id}`, 'DELETE');
                this.showToast('Triple deleted!', 'success');
                this.loadTriples();
            } catch (e) {
                this.showToast('Failed to delete triple', 'error');
            }
        });
    }

    connectWebSocket() {
        const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                this.updateConnectionStatus(true);
            };

            this.ws.onclose = () => {
                this.updateConnectionStatus(false);
                setTimeout(() => this.connectWebSocket(), 5000);
            };

            this.ws.onerror = () => {
                this.updateConnectionStatus(false);
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (e) { }
            };
        } catch (e) {
            console.error('WebSocket connection failed:', e);
        }
    }

    handleWebSocketMessage(data) {
        if (data.type === 'memory_update') {
            if (this.currentTab === 'memories') this.loadMemories();
            this.loadStats();
        } else if (data.type === 'schedule_update') {
            if (this.currentTab === 'schedules') this.loadSchedules();
            this.loadStats();
        } else if (data.type === 'entity_update') {
            if (this.currentTab === 'entities') this.loadEntities();
            this.loadStats();
        }
    }

    updateConnectionStatus(connected) {
        const status = document.getElementById('connection-status');
        if (status) {
            status.querySelector('.status-dot').style.background = connected ? 'var(--success)' : 'var(--danger)';
            status.querySelector('.status-text').textContent = connected ? 'Connected' : 'Disconnected';
        }

        const sidebar = document.querySelector('.status-indicator');
        if (sidebar) {
            sidebar.classList.toggle('online', connected);
        }
    }

    async api(endpoint, method = 'GET', body = null) {
        const options = {
            method,
            headers: { 'Content-Type': 'application/json' }
        };
        if (body) options.body = JSON.stringify(body);

        const response = await fetch(`${API_BASE}${endpoint}`, options);
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || 'API request failed');
        }
        return response.json();
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span>${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.animation = 'toastIn 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatDate(dateStr) {
        if (!dateStr) return '-';
        const date = new Date(dateStr);
        if (isNaN(date.getTime())) return '-';
        return date.toLocaleDateString('id-ID', {
            day: 'numeric',
            month: 'short',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    getStatusBadge(status) {
        switch (status) {
            case 'pending': return 'badge-warning';
            case 'executed': case 'completed': return 'badge-success';
            case 'failed': return 'badge-danger';
            default: return 'badge-default';
        }
    }

    async loadNeuralEvents() {
        try {
            const data = await this.api('/api/neural-events?limit=50');
            const tbody = document.getElementById('neural-events-list');

            if (!data.events || data.events.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="empty-state">No events found</td></tr>';
                return;
            }

            tbody.innerHTML = data.events.map(e => `
                <tr>
                    <td>${this.formatDate(e.timestamp)}</td>
                    <td><span class="badge ${this.getEventBadge(e.event_type)}">${e.event_type}</span></td>
                    <td>${this.escapeHtml(e.component || '-')}</td>
                    <td>${this.escapeHtml(JSON.stringify(e.data || {})).substring(0, 100)}</td>
                </tr>
            `).join('');
        } catch (e) {
            console.error('Failed to load neural events:', e);
        }
    }

    getEventBadge(type) {
        if (type.includes('error') || type.includes('failure')) return 'badge-danger';
        if (type.includes('warning')) return 'badge-warning';
        if (type.includes('success') || type.includes('complete')) return 'badge-success';
        return 'badge-info';
    }

    async loadModelHealth() {
        try {
            const status = await this.api('/api/openrouter/health');
            const indicator = document.getElementById('model-status-indicator');
            const tier = document.getElementById('model-active-tier');
            const list = document.getElementById('model-health-list');

            if (status.status === 'healthy') {
                indicator.className = 'badge badge-success';
                indicator.textContent = 'Healthy';
            } else {
                indicator.className = 'badge badge-danger';
                indicator.textContent = status.status;
            }

            tier.textContent = status.active_tier || 'None';

            if (status.model_health) {
                list.innerHTML = Object.entries(status.model_health).map(([model, info]) => `
                    <div style="margin-bottom: 5px; font-size: 13px;">
                        <strong>${model}:</strong> 
                        <span class="badge ${info.healthy ? 'badge-success' : 'badge-danger'}">${info.healthy ? 'OK' : 'Issues'}</span>
                        <span style="color: var(--text-muted)">(${info.errors || 0} errors)</span>
                    </div>
                `).join('');
            }
        } catch (e) {
            console.error('Failed to load model health:', e);
        }
    }
}

const dashboard = new ViraDashboard();
