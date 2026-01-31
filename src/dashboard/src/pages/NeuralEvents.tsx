import { useState, useEffect } from 'react';
import { api, formatDate, escapeHtml } from '../api';
import type { NeuralEvent } from '../types';
import type { WsMessage } from '../types';

interface NeuralEventsResponse {
  events: NeuralEvent[];
  count: number;
}

interface NeuralEventsProps {
  lastWsEvent: WsMessage | null;
}

function getEventBadge(type: string | undefined): string {
  if (type?.includes('error') || type?.includes('failure')) return 'badge-danger';
  if (type?.includes('warning')) return 'badge-warning';
  if (type?.includes('success') || type?.includes('complete')) return 'badge-success';
  return 'badge-info';
}

export function NeuralEvents({ lastWsEvent }: NeuralEventsProps): React.ReactNode {
  const [events, setEvents] = useState<NeuralEvent[]>([]);
  const [loading, setLoading] = useState(true);

  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const res = await api<NeuralEventsResponse>('/api/neural-events?limit=100');
      setEvents(res?.events ?? []);
    } catch {
      setEvents([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);
  useEffect(() => { if (lastWsEvent?.type && lastWsEvent.type !== 'pong') load(); }, [lastWsEvent?.type]);

  return (
    <section className="tab-content active">
      <div className="toolbar">
        <button type="button" className="btn btn-secondary" onClick={load}>Refresh</button>
      </div>
      <div className="panel">
        <div className="panel-body">
          {loading && <p className="empty-state">Loading...</p>}
          {!loading && !events.length && <p className="empty-state">No events found</p>}
          {!loading && events.length > 0 && (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Event Type</th>
                  <th>Component</th>
                  <th>Details</th>
                </tr>
              </thead>
              <tbody>
                {events.map((e, i) => (
                  <tr key={i}>
                    <td>{formatDate(e.timestamp)}</td>
                    <td><span className={`badge ${getEventBadge(e.event_type)}`}>{e.event_type ?? '-'}</span></td>
                    <td>{escapeHtml(e.component ?? '-')}</td>
                    <td>{escapeHtml(JSON.stringify(e.data ?? {})).substring(0, 100)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </section>
  );
}
