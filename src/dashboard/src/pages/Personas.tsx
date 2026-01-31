import { useState, useEffect } from 'react';
import { api, escapeHtml } from '../api';
import type { Persona } from '../types';
import type { WsMessage } from '../types';

interface PersonasProps {
  showToast: (msg: string, type: 'success' | 'error' | 'info') => void;
  lastWsEvent: WsMessage | null;
}

export function Personas({ showToast, lastWsEvent }: PersonasProps): React.ReactNode {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState<{ mode: 'create' | 'edit'; id?: string; name?: string; description?: string; instruction?: string; temperature?: number; voice_tone?: string } | null>(null);

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const data = await api<Persona[]>('/api/personas');
      setPersonas(Array.isArray(data) ? data : []);
    } catch {
      setPersonas([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);
  useEffect(() => { if (lastWsEvent?.type === 'persona_update') load(); }, [lastWsEvent?.type]);

  const openCreate = (): void => setModal({ mode: 'create', name: '', description: '', instruction: '', temperature: 0.7, voice_tone: 'friendly' });
  const openEdit = (p: Persona): void => setModal({ mode: 'edit', id: p.id, name: p.name ?? '', description: p.description ?? '', instruction: p.instruction ?? '', temperature: p.temperature ?? 0.7, voice_tone: p.voice_tone ?? 'friendly' });

  const saveCreate = async (): Promise<void> => {
    if (!modal?.name?.trim()) { showToast('Name required', 'error'); return; }
    if (!modal?.instruction?.trim()) { showToast('Instruction required', 'error'); return; }
    try {
      await api('/api/personas', 'POST', {
        name: modal.name.trim(),
        description: (modal.description ?? '').trim(),
        instruction: modal.instruction.trim(),
        temperature: Number(modal.temperature) ?? 0.7,
        voice_tone: modal.voice_tone ?? 'friendly',
      });
      showToast('Persona created!', 'success');
      setModal(null);
      load();
    } catch (e) {
      showToast((e as Error).message ?? 'Create failed', 'error');
    }
  };

  const saveEdit = async (): Promise<void> => {
    if (!modal?.id) return;
    try {
      await api(`/api/personas/${modal.id}`, 'PUT', {
        name: modal.name?.trim(),
        description: modal.description?.trim(),
        instruction: modal.instruction?.trim(),
        temperature: Number(modal.temperature),
        voice_tone: modal.voice_tone,
      });
      showToast('Persona updated!', 'success');
      setModal(null);
      load();
    } catch (e) {
      showToast((e as Error).message ?? 'Update failed', 'error');
    }
  };

  const activate = async (id: string): Promise<void> => {
    try {
      await api(`/api/personas/${id}/activate`, 'POST');
      showToast('Persona activated!', 'success');
      load();
    } catch (e) {
      showToast((e as Error).message ?? 'Activate failed', 'error');
    }
  };

  const deletePersona = async (id: string): Promise<void> => {
    if (!window.confirm('Delete this persona?')) return;
    try {
      await api(`/api/personas/${id}`, 'DELETE');
      showToast('Persona deleted!', 'success');
      load();
      setModal(null);
    } catch (e) {
      showToast((e as Error).message ?? 'Delete failed', 'error');
    }
  };

  return (
    <section className="tab-content active">
      <div className="toolbar">
        <button type="button" className="btn btn-primary" onClick={openCreate}>+ Add Persona</button>
      </div>
      <div className="panel">
        <div className="panel-body">
          {loading && <p className="empty-state">Loading...</p>}
          {!loading && !personas.length && <p className="empty-state">No personas found.</p>}
          {!loading && personas.length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '24px' }}>
              {personas.map((p) => (
                <div key={p.id} className="stat-card" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                    <strong>{escapeHtml(p.name ?? '')}</strong>
                    {p.is_active && <span className="badge badge-success">Active</span>}
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '12px' }}>
                    {escapeHtml((p.description ?? 'No description').substring(0, 120))}{(p.description ?? '').length > 120 ? '...' : ''}
                  </p>
                  <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                    Temperature: {p.temperature} | Tone: {p.voice_tone}
                  </p>
                  <div style={{ display: 'flex', gap: '8px', marginTop: '12px', flexWrap: 'wrap' }}>
                    {!p.is_active && <button type="button" className="btn btn-sm btn-primary" onClick={() => activate(p.id)}>Activate</button>}
                    <button type="button" className="btn btn-sm btn-secondary" onClick={() => openEdit(p)}>Edit</button>
                    <button type="button" className="btn btn-sm btn-danger" onClick={() => deletePersona(p.id)}>Delete</button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {modal && (
        <div className="modal-overlay" onClick={() => setModal(null)}>
          <div className="modal-box" onClick={(e) => e.stopPropagation()}>
            <h3>{modal.mode === 'create' ? 'Add Persona' : 'Edit Persona'}</h3>
            <div className="form-group">
              <label>Name</label>
              <input type="text" value={modal.name ?? ''} onChange={(e) => setModal({ ...modal, name: e.target.value })} placeholder="Name" />
            </div>
            <div className="form-group">
              <label>Description</label>
              <input type="text" value={modal.description ?? ''} onChange={(e) => setModal({ ...modal, description: e.target.value })} placeholder="Short description" />
            </div>
            <div className="form-group">
              <label>Instruction</label>
              <textarea value={modal.instruction ?? ''} onChange={(e) => setModal({ ...modal, instruction: e.target.value })} rows={4} placeholder="System instruction" />
            </div>
            <div className="form-group">
              <label>Temperature</label>
              <input type="number" min={0} max={2} step={0.1} value={modal.temperature ?? 0.7} onChange={(e) => setModal({ ...modal, temperature: Number(e.target.value) })} />
            </div>
            <div className="form-group">
              <label>Voice tone</label>
              <input type="text" value={modal.voice_tone ?? ''} onChange={(e) => setModal({ ...modal, voice_tone: e.target.value })} placeholder="e.g. friendly" />
            </div>
            <div className="modal-actions">
              <button type="button" className="btn btn-primary" onClick={modal.mode === 'create' ? saveCreate : saveEdit}>{modal.mode === 'create' ? 'Create' : 'Save'}</button>
              <button type="button" className="btn btn-secondary" onClick={() => setModal(null)}>Cancel</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
