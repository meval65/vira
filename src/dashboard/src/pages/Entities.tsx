import { useState, useEffect } from 'react';
import { api, escapeHtml } from '../api';
import type { Entity } from '../types';
import type { WsMessage } from '../types';

interface EntitiesProps {
  showToast: (msg: string, type: 'success' | 'error' | 'info') => void;
  lastWsEvent: WsMessage | null;
}

export function Entities({ showToast, lastWsEvent }: EntitiesProps): React.ReactNode {
  const [entities, setEntities] = useState<Entity[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState<{ mode: 'create'; name?: string; entity_type?: string } | null>(null);

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const data = search.length >= 2
        ? await api<Entity[]>(`/api/search/entities?query=${encodeURIComponent(search)}&limit=100`).catch(() => [])
        : await api<Entity[]>('/api/entities?limit=200');
      setEntities(Array.isArray(data) ? data : []);
    } catch {
      setEntities([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [search]);
  useEffect(() => { if (lastWsEvent?.type === 'entity_update') load(); }, [lastWsEvent?.type]);

  const openCreate = (): void => setModal({ mode: 'create', name: '', entity_type: 'unknown' });
  const deleteEntity = async (name: string): Promise<void> => {
    if (!window.confirm(`Delete entity "${name}"?`)) return;
    try {
      await api(`/api/entities/${encodeURIComponent(name)}`, 'DELETE');
      showToast('Entity deleted!', 'success');
      load();
      setModal(null);
    } catch (e) {
      showToast((e as Error).message ?? 'Delete failed', 'error');
    }
  };

  const saveCreate = async (): Promise<void> => {
    if (!modal?.name?.trim()) { showToast('Name required', 'error'); return; }
    try {
      await api('/api/entities', 'POST', {
        name: modal.name.trim(),
        entity_type: (modal.entity_type ?? 'unknown').trim(),
      });
      showToast('Entity created!', 'success');
      setModal(null);
      load();
    } catch (e) {
      showToast((e as Error).message ?? 'Create failed', 'error');
    }
  };

  return (
    <section className="tab-content active">
      <div className="toolbar">
        <input type="text" className="search-input" placeholder="Search (min 2 chars) or list all..." value={search} onChange={(e) => setSearch(e.target.value)} />
        <button type="button" className="btn btn-primary" onClick={openCreate}>+ Add Entity</button>
      </div>
      <div className="panel">
        <div className="panel-body">
          {loading ? <p className="empty-state">Loading...</p> : !entities.length ? <p className="empty-state">No entities found</p> : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Type</th>
                  <th>Mentions</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {entities.map((e) => (
                  <tr key={e.name ?? e.id}>
                    <td><strong>{escapeHtml(e.name ?? '')}</strong></td>
                    <td><span className="badge badge-info">{e.entity_type ?? 'unknown'}</span></td>
                    <td>{e.mention_count ?? 0}</td>
                    <td>
                      <button type="button" className="btn btn-sm btn-danger" onClick={() => deleteEntity(e.name)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {modal && modal.mode === 'create' && (
        <div className="modal-overlay" onClick={() => setModal(null)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <h3>Add Entity</h3>
            <div className="form-group">
              <label>Name</label>
              <input type="text" value={modal.name ?? ''} onChange={(e) => setModal({ ...modal, name: e.target.value })} placeholder="Entity name" />
            </div>
            <div className="form-group">
              <label>Type</label>
              <input type="text" value={modal.entity_type ?? ''} onChange={(e) => setModal({ ...modal, entity_type: e.target.value })} placeholder="e.g. person, place" />
            </div>
            <div className="modal-actions">
              <button type="button" className="btn btn-primary" onClick={saveCreate}>Create</button>
              <button type="button" className="btn btn-secondary" onClick={() => setModal(null)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
