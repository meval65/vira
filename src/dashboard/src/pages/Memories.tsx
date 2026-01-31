import { useState, useEffect } from 'react';
import { api, formatDate, escapeHtml } from '../api';
import type { Memory } from '../types';
import type { WsMessage } from '../types';

interface MemoriesProps {
  showToast: (msg: string, type: 'success' | 'error' | 'info') => void;
  lastWsEvent: WsMessage | null;
}

export function Memories({ showToast, lastWsEvent }: MemoriesProps): React.ReactNode {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [modal, setModal] = useState<{ mode: 'create' | 'view'; id?: string; summary?: string; memory_type?: string; priority?: number } | null>(null);

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const data = search.length >= 2
        ? await api<Memory[]>(`/api/search/memories?query=${encodeURIComponent(search)}&limit=100`).catch(() => [])
        : await api<Memory[]>('/api/memories?limit=200');
      setMemories(Array.isArray(data) ? data : []);
    } catch {
      setMemories([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [search]);
  useEffect(() => { if (lastWsEvent?.type === 'memory_update') load(); }, [lastWsEvent?.type]);

  const deleteMemory = async (id: string): Promise<void> => {
    if (!window.confirm('Delete this memory?')) return;
    try {
      await api(`/api/memories/${id}`, 'DELETE');
      showToast('Memory deleted!', 'success');
      load();
      setModal(null);
    } catch (e) {
      showToast((e as Error).message ?? 'Failed to delete memory', 'error');
    }
  };

  const bulkDelete = async (): Promise<void> => {
    const ids = [...selectedIds];
    if (!ids.length || !window.confirm(`Archive ${ids.length} memories?`)) return;
    try {
      await api('/api/memories/bulk-delete', 'POST', { memory_ids: ids, hard_delete: false });
      showToast(`${ids.length} memories archived`, 'success');
      setSelectedIds(new Set());
      load();
    } catch (e) {
      showToast((e as Error).message ?? 'Bulk delete failed', 'error');
    }
  };

  const openCreate = (): void => setModal({ mode: 'create', summary: '', memory_type: 'general', priority: 0.5 });
  const openView = async (m: Memory): Promise<void> => {
    try {
      const full = await api<Memory>(`/api/memories/${m.id}`);
      setModal({ mode: 'view', ...full });
    } catch {
      showToast('Failed to load memory', 'error');
    }
  };

  const saveCreate = async (): Promise<void> => {
    if (!modal?.summary?.trim()) { showToast('Summary required', 'error'); return; }
    try {
      await api('/api/memories', 'POST', {
        summary: modal.summary.trim(),
        memory_type: modal.memory_type ?? 'general',
        priority: Number(modal.priority) ?? 0.5,
      });
      showToast('Memory created!', 'success');
      setModal(null);
      load();
    } catch (e) {
      showToast((e as Error).message ?? 'Create failed', 'error');
    }
  };

  const saveEdit = async (): Promise<void> => {
    if (!modal?.id) return;
    try {
      await api(`/api/memories/${modal.id}`, 'PUT', {
        summary: modal.summary,
        memory_type: modal.memory_type,
        priority: modal.priority != null ? Number(modal.priority) : undefined,
      });
      showToast('Memory updated!', 'success');
      setModal(null);
      load();
    } catch (e) {
      showToast((e as Error).message ?? 'Update failed', 'error');
    }
  };

  const toggleSelect = (id: string): void => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  return (
    <section className="tab-content active">
      <div className="toolbar">
        <input type="text" className="search-input" placeholder="Search (min 2 chars) or list all..." value={search} onChange={(e) => setSearch(e.target.value)} />
        <button type="button" className="btn btn-primary" onClick={openCreate}>+ Add Memory</button>
        {selectedIds.size > 0 && <button type="button" className="btn btn-danger" onClick={bulkDelete}>Archive selected ({selectedIds.size})</button>}
      </div>
      <div className="panel">
        <div className="panel-body">
          {loading ? <p className="empty-state">Loading...</p> : !memories.length ? <p className="empty-state">No memories found</p> : (
            <table className="data-table">
              <thead>
                <tr>
                  <th><input type="checkbox" checked={memories.length > 0 && selectedIds.size === memories.length} onChange={(e) => e.target.checked ? setSelectedIds(new Set(memories.map((m) => m.id))) : setSelectedIds(new Set())} /></th>
                  <th>Summary</th>
                  <th>Type</th>
                  <th>Confidence</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {memories.map((m) => (
                  <tr key={m.id}>
                    <td><input type="checkbox" checked={selectedIds.has(m.id)} onChange={() => toggleSelect(m.id)} /></td>
                    <td>{escapeHtml((m.summary ?? '').substring(0, 80))}{(m.summary ?? '').length > 80 ? '...' : ''}</td>
                    <td><span className="badge badge-info">{m.memory_type ?? 'general'}</span></td>
                    <td>{((m.confidence ?? 0) * 100).toFixed(0)}%</td>
                    <td>{formatDate(m.created_at)}</td>
                    <td>
                      <button type="button" className="btn btn-sm btn-secondary" onClick={() => openView(m)}>View</button>
                      <button type="button" className="btn btn-sm btn-danger" onClick={() => deleteMemory(m.id)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {modal && (
        <div className="modal-overlay" onClick={() => setModal(null)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <h3>{modal.mode === 'create' ? 'Add Memory' : 'View / Edit Memory'}</h3>
            <div className="form-group">
              <label>Summary</label>
              <textarea value={modal.summary ?? ''} onChange={(e) => setModal({ ...modal, summary: e.target.value })} rows={4} placeholder="Summary..." />
            </div>
            <div className="form-group">
              <label>Type</label>
              <select value={modal.memory_type ?? 'general'} onChange={(e) => setModal({ ...modal, memory_type: e.target.value })}>
                <option value="general">general</option>
                <option value="emotion">emotion</option>
                <option value="fact">fact</option>
                <option value="preference">preference</option>
                <option value="event">event</option>
              </select>
            </div>
            <div className="form-group">
              <label>Priority (0–1)</label>
              <input type="number" min={0} max={1} step={0.1} value={modal.priority ?? 0.5} onChange={(e) => setModal({ ...modal, priority: Number(e.target.value) })} />
            </div>
            <div className="modal-actions">
              {modal.mode === 'create' && <button type="button" className="btn btn-primary" onClick={saveCreate}>Create</button>}
              {modal.mode === 'view' && <button type="button" className="btn btn-primary" onClick={saveEdit}>Save</button>}
              <button type="button" className="btn btn-secondary" onClick={() => setModal(null)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
