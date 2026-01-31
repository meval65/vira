import { useState, useEffect } from 'react';
import { api, formatDate } from '../api';
import type { Schedule } from '../types';
import type { WsMessage } from '../types';

interface SchedulesProps {
  showToast: (msg: string, type: 'success' | 'error' | 'info') => void;
  lastWsEvent: WsMessage | null;
}

const modalStyle = { position: 'fixed' as const, inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 };
const boxStyle = { background: 'var(--bg-secondary)', padding: '24px', borderRadius: 'var(--radius-lg)', maxWidth: '450px', width: '100%', maxHeight: '90vh', overflowY: 'auto' as const };

export function Schedules({ showToast, lastWsEvent }: SchedulesProps): React.ReactNode {
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('');
  const [modal, setModal] = useState<{ mode: 'create' | 'edit'; id?: string; context?: string; scheduled_at?: string; priority?: number; status?: string } | null>(null);

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: '200', upcoming: 'false' });
      if (statusFilter) params.set('status', statusFilter);
      const data = await api<Schedule[]>(`/api/schedules?${params}`);
      setSchedules(Array.isArray(data) ? data : []);
    } catch {
      setSchedules([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [statusFilter]);
  useEffect(() => { if (lastWsEvent?.type === 'schedule_update') load(); }, [lastWsEvent?.type]);

  const deleteSchedule = async (id: string): Promise<void> => {
    if (!window.confirm('Delete this schedule?')) return;
    try {
      await api(`/api/schedules/${id}`, 'DELETE');
      showToast('Schedule deleted!', 'success');
      load();
      setModal(null);
    } catch (e) {
      showToast((e as Error).message ?? 'Failed to delete schedule', 'error');
    }
  };

  const openCreate = (): void => {
    const d = new Date();
    d.setHours(d.getHours() + 1, 0, 0, 0);
    setModal({ mode: 'create', context: '', scheduled_at: d.toISOString().slice(0, 16), priority: 0, status: 'pending' });
  };

  const openEdit = (s: Schedule): void => setModal({
    mode: 'edit', id: s.id, context: s.context ?? '', scheduled_at: (s.scheduled_at ?? s.trigger_time ?? '').toString().slice(0, 16), priority: s.priority ?? 0, status: s.status ?? 'pending',
  });

  const saveCreate = async (): Promise<void> => {
    if (!modal?.context?.trim()) { showToast('Context required', 'error'); return; }
    try {
      await api('/api/schedules', 'POST', {
        context: modal.context.trim(),
        scheduled_at: new Date(modal.scheduled_at ?? 0).toISOString(),
        priority: Number(modal.priority) ?? 0,
        status: modal.status ?? 'pending',
      });
      showToast('Schedule created!', 'success');
      setModal(null);
      load();
    } catch (e) {
      showToast((e as Error).message ?? 'Create failed', 'error');
    }
  };

  const saveEdit = async (): Promise<void> => {
    if (!modal?.id) return;
    try {
      await api(`/api/schedules/${modal.id}`, 'PUT', {
        context: modal.context,
        scheduled_at: modal.scheduled_at ? new Date(modal.scheduled_at).toISOString() : undefined,
        priority: Number(modal.priority),
        status: modal.status,
      });
      showToast('Schedule updated!', 'success');
      setModal(null);
      load();
    } catch (e) {
      showToast((e as Error).message ?? 'Update failed', 'error');
    }
  };

  return (
    <section className="tab-content active">
      <div className="toolbar">
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="">All statuses</option>
          <option value="pending">Pending</option>
          <option value="done">Done</option>
          <option value="cancelled">Cancelled</option>
        </select>
        <button type="button" className="btn btn-primary" onClick={openCreate}>+ Add Schedule</button>
      </div>
      <div className="panel">
        <div className="panel-body">
          {loading ? <p className="empty-state">Loading...</p> : !schedules.length ? <p className="empty-state">No schedules found</p> : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Context</th>
                  <th>Scheduled At</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {schedules.map((s) => (
                  <tr key={s.id}>
                    <td>{String(s.context ?? '')}</td>
                    <td>{formatDate(s.scheduled_at ?? s.trigger_time)}</td>
                    <td><span className="badge badge-info">{s.status ?? 'pending'}</span></td>
                    <td>{formatDate(s.created_at)}</td>
                    <td>
                      <button type="button" className="btn btn-sm btn-secondary" onClick={() => openEdit(s)}>Edit</button>
                      <button type="button" className="btn btn-sm btn-danger" onClick={() => deleteSchedule(s.id)}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {modal && (
        <div style={modalStyle} onClick={() => setModal(null)}>
          <div style={boxStyle} onClick={(e) => e.stopPropagation()}>
            <h3>{modal.mode === 'create' ? 'Add Schedule' : 'Edit Schedule'}</h3>
            <div className="form-group">
              <label>Context</label>
              <textarea value={modal.context ?? ''} onChange={(e) => setModal({ ...modal, context: e.target.value })} rows={3} placeholder="Context..." />
            </div>
            <div className="form-group">
              <label>Scheduled at</label>
              <input type="datetime-local" value={modal.scheduled_at ?? ''} onChange={(e) => setModal({ ...modal, scheduled_at: e.target.value })} />
            </div>
            <div className="form-group">
              <label>Priority</label>
              <input type="number" value={modal.priority ?? 0} onChange={(e) => setModal({ ...modal, priority: Number(e.target.value) })} />
            </div>
            <div className="form-group">
              <label>Status</label>
              <select value={modal.status ?? 'pending'} onChange={(e) => setModal({ ...modal, status: e.target.value })}>
                <option value="pending">Pending</option>
                <option value="done">Done</option>
                <option value="cancelled">Cancelled</option>
              </select>
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
