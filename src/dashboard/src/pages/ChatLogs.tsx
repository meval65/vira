import { useState, useEffect } from 'react';
import { api, formatDate, escapeHtml } from '../api';
import type { ChatLog } from '../types';
import type { WsMessage } from '../types';

interface ChatLogsProps {
  showToast: (msg: string, type: 'success' | 'error' | 'info') => void;
  lastWsEvent: WsMessage | null;
}

interface SessionsResponse {
  sessions: string[];
  count: number;
}

export function ChatLogs({ showToast, lastWsEvent }: ChatLogsProps): React.ReactNode {
  const [logs, setLogs] = useState<ChatLog[]>([]);
  const [sessions, setSessions] = useState<string[]>([]);
  const [sessionFilter, setSessionFilter] = useState('');
  const [loading, setLoading] = useState(true);

  const loadLogs = async (): Promise<void> => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: '100' });
      if (sessionFilter) params.set('session_id', sessionFilter);
      const data = await api<ChatLog[]>(`/api/chat-logs?${params}`);
      setLogs(Array.isArray(data) ? data : []);
    } catch {
      setLogs([]);
    } finally {
      setLoading(false);
    }
  };

  const loadSessions = async (): Promise<void> => {
    try {
      const res = await api<SessionsResponse>('/api/chat-logs/sessions');
      setSessions(res?.sessions ?? []);
    } catch {
      setSessions([]);
    }
  };

  useEffect(() => {
    loadLogs();
    loadSessions();
  }, [sessionFilter]);
  useEffect(() => { if (lastWsEvent?.type === 'chat_log_update') { loadLogs(); loadSessions(); } }, [lastWsEvent?.type]);

  const deleteSession = async (sessionId: string): Promise<void> => {
    if (!sessionId || !window.confirm(`Delete all logs in session "${sessionId.slice(0, 20)}..."?`)) return;
    try {
      await api(`/api/chat-logs/session/${encodeURIComponent(sessionId)}`, 'DELETE');
      showToast('Session deleted!', 'success');
      loadLogs();
      loadSessions();
      if (sessionFilter === sessionId) setSessionFilter('');
    } catch (e) {
      showToast((e as Error).message ?? 'Failed to delete session', 'error');
    }
  };

  const deleteLog = async (logId: string): Promise<void> => {
    if (!window.confirm('Delete this log entry?')) return;
    try {
      await api(`/api/chat-logs/${logId}`, 'DELETE');
      showToast('Log deleted!', 'success');
      loadLogs();
    } catch (e) {
      showToast((e as Error).message ?? 'Failed to delete', 'error');
    }
  };

  return (
    <section className="tab-content active">
      <div className="toolbar">
        <select value={sessionFilter} onChange={(e) => setSessionFilter(e.target.value)} style={{ maxWidth: '280px' }}>
          <option value="">All sessions</option>
          {sessions.map((s) => (
            <option key={s} value={s}>{String(s).slice(0, 36)}{String(s).length > 36 ? '...' : ''}</option>
          ))}
        </select>
        <button type="button" className="btn btn-secondary" onClick={() => { loadLogs(); loadSessions(); }}>Refresh</button>
      </div>
      <div className="panel" style={{ marginBottom: '16px' }}>
        <div className="panel-header"><h3>Sessions ({sessions.length})</h3></div>
        <div className="panel-body" style={{ maxHeight: '120px', overflowY: 'auto' }}>
          {!sessions.length ? <p className="empty-state">No sessions</p> : (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {sessions.slice(0, 20).map((s) => (
                <span key={s} style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', padding: '6px 10px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm)', fontSize: '13px' }}>
                  <span style={{ maxWidth: '180px', overflow: 'hidden', textOverflow: 'ellipsis' }}>{String(s)}</span>
                  <button type="button" className="btn btn-sm btn-danger" onClick={() => deleteSession(s)} title="Delete session">×</button>
                </span>
              ))}
              {sessions.length > 20 && <span style={{ color: 'var(--text-muted)' }}>+{sessions.length - 20} more</span>}
            </div>
          )}
        </div>
      </div>
      <div className="panel">
        <div className="panel-header"><h3>Chat Logs</h3></div>
        <div className="panel-body" style={{ maxHeight: '500px', overflowY: 'auto' }}>
          {loading && <p className="empty-state">Loading...</p>}
          {!loading && !logs.length && <p className="empty-state">No chat logs found</p>}
          {!loading && logs.length > 0 && logs.map((log) => (
            <div key={log.id} className="chat-message" style={{ position: 'relative' }}>
              <div className={`chat-avatar ${log.role === 'user' ? 'user' : 'ai'}`}>{log.role === 'user' ? '👤' : '🧠'}</div>
              <div className="chat-content">
                <div className="chat-meta">
                  <span>{log.role === 'user' ? 'User' : 'Vira'}</span>
                  <span className="chat-time">{formatDate(log.timestamp)}</span>
                  {log.session_id && <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}> · {String(log.session_id).slice(0, 8)}</span>}
                  <button type="button" className="btn btn-sm btn-danger" style={{ marginLeft: '8px', padding: '2px 6px' }} onClick={() => deleteLog(log.id)} title="Delete">×</button>
                </div>
                <div className="chat-text">{escapeHtml(log.content ?? '')}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
