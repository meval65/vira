import { useState, useEffect } from 'react';
import { api } from '../api';
import type { Health, LogEntry, OpenRouterHealth } from '../types';

interface SystemProps {
  showToast: (msg: string, type: 'success' | 'error' | 'info') => void;
}

export function System({ showToast }: SystemProps): React.ReactNode {
  const [health, setHealth] = useState<Health | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [openRouter, setOpenRouter] = useState<OpenRouterHealth | null>(null);
  const [loading, setLoading] = useState(true);

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const [h, l, or] = await Promise.all([
        api<Health>('/api/health').catch(() => null),
        api<LogEntry[]>('/api/logs/history').catch(() => []),
        api<OpenRouterHealth>('/api/openrouter/health').catch(() => null),
      ]);
      setHealth(h ?? null);
      setLogs(Array.isArray(l) ? l : []);
      setOpenRouter(or ?? null);
    } catch {
      setHealth(null);
      setLogs([]);
      setOpenRouter(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const resetOpenRouterHealth = async (): Promise<void> => {
    try {
      await api('/api/openrouter/reset-health', 'POST');
      showToast('OpenRouter health reset', 'success');
      load();
    } catch {
      showToast('Reset failed', 'error');
    }
  };

  if (loading) {
    return <section className="tab-content active"><p className="empty-state">Loading...</p></section>;
  }

  return (
    <section className="tab-content active">
      <div className="panel" style={{ marginBottom: '24px' }}>
        <div className="panel-header">
          <h3>API Health</h3>
          <button type="button" className="btn btn-sm btn-secondary" onClick={load}>Refresh</button>
        </div>
        <div className="panel-body">
          {health ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '24px' }}>
              <div>
                <strong>Status:</strong>{' '}
                <span className={`badge ${health.status === 'healthy' ? 'badge-success' : 'badge-danger'}`}>{health.status}</span>
              </div>
              <div><strong>Database:</strong> {health.database ?? 'N/A'}</div>
              {health.error && <div style={{ color: 'var(--danger)' }}>{health.error}</div>}
            </div>
          ) : (
            <p className="empty-state">Health check failed</p>
          )}
        </div>
      </div>

      <div className="panel" style={{ marginBottom: '24px' }}>
        <div className="panel-header">
          <h3>OpenRouter</h3>
          {openRouter && (
            <span className={`badge ${openRouter.status === 'healthy' ? 'badge-success' : 'badge-warning'}`}>{openRouter.status}</span>
          )}
        </div>
        <div className="panel-body">
          {openRouter ? (
            <>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '16px', marginBottom: '12px' }}>
                <span>API: {openRouter.api_configured ? 'Configured' : 'Not configured'}</span>
                {openRouter.active_tier && <span>Tier: {openRouter.active_tier}</span>}
                {openRouter.total_requests != null && <span>Requests: {openRouter.total_requests}</span>}
              </div>
              <button type="button" className="btn btn-secondary" onClick={resetOpenRouterHealth}>Reset health counters</button>
            </>
          ) : (
            <p className="empty-state">OpenRouter status unavailable</p>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-header"><h3>Log History</h3></div>
        <div className="panel-body" style={{ maxHeight: '400px', overflowY: 'auto' }}>
          {!logs.length ? <p className="empty-state">No logs</p> : (
            [...logs].reverse().map((log, i) => (
              <div key={i} style={{ fontSize: '12px', marginBottom: '10px', padding: '8px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm)' }}>
                <span style={{ color: 'var(--text-muted)' }}>{log.timestamp}</span>
                <span className={`badge badge-${log.level === 'ERROR' ? 'danger' : log.level === 'WARNING' ? 'warning' : 'info'}`} style={{ marginLeft: '8px' }}>{log.level}</span>
                <span style={{ marginLeft: '8px' }}>{log.source}</span>
                <div style={{ marginTop: '4px', wordBreak: 'break-word' }}>{log.message}</div>
              </div>
            ))
          )}
        </div>
      </div>
    </section>
  );
}
