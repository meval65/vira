import { useState, useEffect } from 'react';
import { api, formatDate } from '../api';
import type { Stats, Health, LogEntry, ChatLog } from '../types';
import type { WsMessage } from '../types';
import { NeuralBrain } from '../components/NeuralBrain';
import { ToolsPanel } from '../components/ToolsPanel';

interface OverviewProps {
  lastWsEvent: WsMessage | null;
}

export function Overview({ lastWsEvent }: OverviewProps): React.ReactNode {
  const [stats, setStats] = useState<Partial<Stats>>({});
  const [health, setHealth] = useState<Health | null>(null);
  const [chatLogs, setChatLogs] = useState<ChatLog[]>([]);
  const [systemLogs, setSystemLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load(): Promise<void> {
      try {
        const [s, h, chat, sysLogs] = await Promise.all([
          api<Stats>('/api/stats'),
          api<Health>('/api/health').catch(() => null),
          api<ChatLog[]>('/api/chat-logs?limit=5').catch(() => []),
          api<LogEntry[]>('/api/logs/history').catch(() => []),
        ]);
        if (!cancelled) {
          setStats(s ?? {});
          setHealth(h ?? null);
          setChatLogs(Array.isArray(chat) ? chat : []);
          setSystemLogs(Array.isArray(sysLogs) ? sysLogs.slice(-20) : []);
        }
      } catch {
        if (!cancelled) {
          setChatLogs([]);
          setSystemLogs([]);
          setStats({});
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [lastWsEvent?.type]);

  if (loading) {
    return <section className="tab-content active"><p className="empty-state">Loading...</p></section>;
  }

  return (
    <section className="tab-content active">
      {/* Neural Brain Visualization */}
      <div className="panel overview-brain-section">
        <div className="panel-header">
          <h3>🧠 Neural Activity</h3>
          {health && (
            <span className={`badge ${health.status === 'healthy' ? 'badge-success' : 'badge-danger'}`}>
              {health.status === 'healthy' ? 'System Healthy' : 'System Issue'}
            </span>
          )}
        </div>
        <div className="panel-body">
          <NeuralBrain lastWsEvent={lastWsEvent} />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="stats-grid">
        {[
          { icon: '💭', value: stats.memories ?? '--', label: 'Memories' },
          { icon: '📅', value: stats.schedules ?? '--', label: 'Schedules' },
          { icon: '🎭', value: stats.personas ?? '--', label: 'Personas' },
          { icon: '👤', value: stats.entities ?? '--', label: 'Entities' },
          { icon: '🔗', value: stats.triples ?? '--', label: 'Triples' },
          { icon: '💬', value: stats.chat_logs ?? '--', label: 'Chat Logs' },
        ].map(({ icon, value, label }) => (
          <div key={label} className="stat-card">
            <div className="stat-icon">{icon}</div>
            <div className="stat-info">
              <span className="stat-value">{value}</span>
              <span className="stat-label">{label}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Chat & Logs Grid */}
      <div className="overview-grid">
        <div className="panel">
          <div className="panel-header"><h3>💬 Recent Chat</h3></div>
          <div className="panel-body">
            {!chatLogs.length ? <p className="empty-state">No recent chat</p> : (
              chatLogs.map((log) => (
                <div key={log.id} className="chat-message">
                  <div className={`chat-avatar ${log.role === 'user' ? 'user' : 'ai'}`}>{log.role === 'user' ? '👤' : '🧠'}</div>
                  <div className="chat-content">
                    <div className="chat-meta">
                      <span>{log.role === 'user' ? 'User' : 'Vira'}</span>
                      <span className="chat-time">{formatDate(log.timestamp)}</span>
                    </div>
                    <div className="chat-text">{(log.content ?? '').substring(0, 120)}{(log.content ?? '').length > 120 ? '...' : ''}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
        <div className="panel">
          <div className="panel-header"><h3>📋 System Logs</h3></div>
          <div className="panel-body" style={{ maxHeight: '320px', overflowY: 'auto' }}>
            {!systemLogs.length ? <p className="empty-state">No system logs</p> : (
              [...systemLogs].reverse().map((log, i) => (
                <div key={i} style={{ fontSize: '12px', marginBottom: '8px', fontFamily: 'monospace' }}>
                  <span style={{ color: 'var(--text-muted)' }}>{log.timestamp}</span>
                  <span className={`badge badge-${log.level === 'ERROR' ? 'danger' : log.level === 'WARNING' ? 'warning' : 'info'}`} style={{ marginLeft: '8px' }}>{log.level}</span>
                  {(log.message ?? '').substring(0, 80)}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Tools Panel */}
      <ToolsPanel lastWsEvent={lastWsEvent} />
    </section>
  );
}
