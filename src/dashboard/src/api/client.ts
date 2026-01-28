import type { Memory, Schedule, Triple, Persona, Entity, ChatLog, SystemStats, ApiStatus, NeuralEvent, GenericConfig } from './types';

const API_BASE = '/api';

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${endpoint}`, options);
    if (!response.ok) {
        throw new Error(`API Error ${response.status}: ${response.statusText}`);
    }
    return response.json();
}

export const api = {
    // Stats & Health
    getHealth: () => fetchJson<{ status: string; uptime: number }>('/health'),
    getStats: () => fetchJson<SystemStats>('/stats'),
    getLogs: (lines = 100) => fetchJson<{ logs: string[] }>(`/logs/history?lines=${lines}`),

    // Memories
    getMemories: (query = '', limit = 100) => fetchJson<Memory[]>(`/memories?query=${encodeURIComponent(query)}&limit=${limit}`),
    createMemory: (data: Partial<Memory>) => fetchJson<Memory>('/memories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }),
    updateMemory: (id: string, data: Partial<Memory>) => fetchJson<Memory>(`/memories/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }),
    deleteMemory: (id: string) => fetchJson<{ success: boolean }>(`/memories/${id}`, {
        method: 'DELETE'
    }),
    deleteMemories: (ids: string[]) => fetchJson<{ count: number }>('/memories/bulk-delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ memory_ids: ids })
    }),

    // Schedules
    getSchedules: (upcoming = true, limit = 50) => fetchJson<Schedule[]>(`/schedules?upcoming=${upcoming}&limit=${limit}`),
    createSchedule: (data: Partial<Schedule>) => fetchJson<Schedule>('/schedules', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }),
    updateSchedule: (id: string, data: Partial<Schedule>) => fetchJson<Schedule>(`/schedules/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }),
    deleteSchedule: (id: string) => fetchJson<{ success: boolean }>(`/schedules/${id}`, {
        method: 'DELETE'
    }),

    // Knowledge Graph (Triples)
    getTriples: (limit = 100) => fetchJson<Triple[]>(`/triples?limit=${limit}`),
    queryEntity: (entity: string) => fetchJson<unknown>(`/triples/query?entity=${encodeURIComponent(entity)}`),
    createTriple: (data: Partial<Triple>) => fetchJson<Triple>('/triples', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }),
    deleteTriple: (id: string) => fetchJson<{ success: boolean }>(`/triples/${id}`, {
        method: 'DELETE'
    }),

    // Entities
    getEntities: (limit = 100) => fetchJson<Entity[]>(`/entities?limit=${limit}`),
    getEntity: (name: string) => fetchJson<Entity>(`/entities/${encodeURIComponent(name)}`),

    // Chat Logs
    getChatLogs: (limit = 100, sessionId?: string) => {
        const url = sessionId
            ? `/chat-logs?limit=${limit}&session_id=${sessionId}`
            : `/chat-logs?limit=${limit}`;
        return fetchJson<ChatLog[]>(url);
    },
    getSessions: () => fetchJson<{ sessions: string[], count: number }>(`/chat-logs/sessions`),
    deleteSession: (sessionId: string) => fetchJson<{ status: string; count: number }>(`/chat-logs/session/${sessionId}`, {
        method: 'DELETE'
    }),

    // Personas
    getPersonas: () => fetchJson<Persona[]>('/personas'),
    getActivePersona: () => fetchJson<{ status: string, persona: Persona | null }>('/personas/active'),
    createPersona: (data: Partial<Persona>) => fetchJson<Persona>('/personas', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }),
    updatePersona: (id: string, data: Partial<Persona>) => fetchJson<Persona>(`/personas/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }),
    deletePersona: (id: string) => fetchJson<{ status: string }>(`/personas/${id}`, {
        method: 'DELETE'
    }),
    activatePersona: (id: string) => fetchJson<{ status: string, persona: Persona }>(`/personas/${id}/activate`, {
        method: 'POST'
    }),

    // System & Config
    getGlobalContext: () => fetchJson<{ context_text: string; metadata: any; last_updated: string }>('/global-context'),
    updateGlobalContext: (context_text: string) => fetchJson<{ status: string }>('/global-context', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ context_text })
    }),
    getConfig: () => fetchJson<GenericConfig>('/system-config'),
    updateConfig: (data: GenericConfig) => fetchJson<{ status: string }>('/system-config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }),

    // Maintenance & Models
    triggerMaintenance: () => fetchJson<{ status: string; results?: any }>('/maintenance/trigger', { method: 'POST' }),
    compressMemories: (force = false) => fetchJson<{ status: string; compressed: number }>('/maintenance/compress-memories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force })
    }),
    optimizeGraph: () => fetchJson<{ status: string }>('/maintenance/optimize-graph', { method: 'POST' }),

    // OpenRouter
    getOpenRouterModels: () => fetchJson<any>('/openrouter/models'),
    getOpenRouterStatus: () => fetchJson<ApiStatus>('/openrouter/health'),
    resetOpenRouter: () => fetchJson<{ status: string }>('/openrouter/reset-health', { method: 'POST' }),

    // Neural Events
    getNeuralEvents: (limit = 50) => fetchJson<{ events: NeuralEvent[], count: number }>(`/neural-events?limit=${limit}`),

    // Search
    searchMemories: (query: string, limit = 10) => fetchJson<Memory[]>(`/search/memories?query=${encodeURIComponent(query)}&limit=${limit}`),
    searchEntities: (query: string, limit = 10) => fetchJson<Entity[]>(`/search/entities?query=${encodeURIComponent(query)}&limit=${limit}`),
};
