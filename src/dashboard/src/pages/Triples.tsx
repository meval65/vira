import { useState, useEffect } from 'react';
import { api, escapeHtml } from '../api';
import type { Triple } from '../types';
import type { WsMessage } from '../types';

interface TriplesProps {
  showToast: (msg: string, type: 'success' | 'error' | 'info') => void;
  lastWsEvent: WsMessage | null;
}

export function Triples({ showToast, lastWsEvent }: TriplesProps): React.ReactNode {
  const [triples, setTriples] = useState<Triple[]>([]);
  const [loading, setLoading] = useState(true);
  const [subjectFilter, setSubjectFilter] = useState('');
  const [predicateFilter, setPredicateFilter] = useState('');
  const [modal, setModal] = useState<{ mode: 'create'; subject?: string; predicate?: string; object?: string; confidence?: number } | null>(null);

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: '200' });
      if (subjectFilter.trim()) params.set('subject', subjectFilter.trim());
      if (predicateFilter.trim()) params.set('predicate', predicateFilter.trim());
      const data = await api<Triple[]>(`/api/triples?${params}`);
      setTriples(Array.isArray(data) ? data : []);
    } catch {
      setTriples([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [subjectFilter, predicateFilter]);
  useEffect(() => { if (lastWsEvent?.type === 'triple_update') load(); }, [lastWsEvent?.type]);

  const deleteTriple = async (id: string): Promise<void> => {
    if (!window.confirm('Delete this triple?')) return;
    try {
      await api(`/api/triples/${id}`, 'DELETE');
      showToast('Triple deleted!', 'success');
      load();
      setModal(null);
    } catch (e) {
      showToast((e as Error).message ?? 'Failed to delete triple', 'error');
    }
  };

  const openCreate = (): void => setModal({ mode: 'create', subject: '', predicate: '', object: '', confidence: 0.8 });
  const saveCreate = async (): Promise<void> => {
    if (!modal?.subject?.trim() || !modal?.predicate?.trim() || !modal?.object?.trim()) {
      showToast('Subject, predicate, and object required', 'error');
      return;
    }
    try {
      await api('/api/triples', 'POST', {
        subject: modal.subject.trim(),
        predicate: modal.predicate.trim(),
        object: modal.object.trim(),
        confidence: Number(modal.confidence) ?? 0.8,
      });
      showToast('Triple created!', 'success');
      setModal(null);
      load();
    } catch (e) {
      showToast((e as Error).message ?? 'Create failed', 'error');
    }
  };

  return (
    <section className="tab-content active">
      <div className="toolbar">
        <input type="text" className="search-input" placeholder="Filter by subject..." value={subjectFilter} onChange={(e) => setSubjectFilter(e.target.value)} style={{ maxWidth: '180px' }} />
        <input type="text" className="search-input" placeholder="Filter by predicate..." value={predicateFilter} onChange={(e) => setPredicateFilter(e.target.value)} style={{ maxWidth: '180px' }} />
        <button type="button" className="btn btn-primary" onClick={openCreate}>+ Add Triple</button>
      </div>
      <div className="panel">
        <div className="panel-body">
          {loading ? <p className="empty-state">Loading...</p> : !triples.length ? <p className="empty-state">No triples found</p> : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Subject</th>
                  <th>Predicate</th>
                  <th>Object</th>
                  <th>Confidence</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {triples.map((t) => (
                  <tr key={t.id}>
                    <td>{escapeHtml(t.subject)}</td>
                    <td><span className="badge badge-info">{escapeHtml(t.predicate)}</span></td>
                    <td>{escapeHtml(t.object)}</td>
                    <td>{((t.confidence ?? 0) * 100).toFixed(0)}%</td>
                    <td>
                      <button type="button" className="btn btn-sm btn-danger" onClick={() => deleteTriple(t.id)}>Delete</button>
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
            <h3>Add Triple</h3>
            <div className="form-group">
              <label>Subject</label>
              <input type="text" value={modal.subject ?? ''} onChange={(e) => setModal({ ...modal, subject: e.target.value })} placeholder="Subject" />
            </div>
            <div className="form-group">
              <label>Predicate</label>
              <input type="text" value={modal.predicate ?? ''} onChange={(e) => setModal({ ...modal, predicate: e.target.value })} placeholder="Predicate" />
            </div>
            <div className="form-group">
              <label>Object</label>
              <input type="text" value={modal.object ?? ''} onChange={(e) => setModal({ ...modal, object: e.target.value })} placeholder="Object" />
            </div>
            <div className="form-group">
              <label>Confidence (0–1)</label>
              <input type="number" min={0} max={1} step={0.1} value={modal.confidence ?? 0.8} onChange={(e) => setModal({ ...modal, confidence: Number(e.target.value) })} />
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
