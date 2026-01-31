import { useState } from 'react';
import { api, escapeHtml } from '../api';
import type { Memory, Entity } from '../types';

export function Search(): React.ReactNode {
  const [query, setQuery] = useState('');
  const [memories, setMemories] = useState<Memory[]>([]);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const search = async (): Promise<void> => {
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setSearched(true);
    try {
      const [mem, ent] = await Promise.all([
        api<Memory[]>(`/api/search/memories?query=${encodeURIComponent(q)}&limit=20`).catch(() => []),
        api<Entity[]>(`/api/search/entities?query=${encodeURIComponent(q)}&limit=20`).catch(() => []),
      ]);
      setMemories(Array.isArray(mem) ? mem : []);
      setEntities(Array.isArray(ent) ? ent : []);
    } catch {
      setMemories([]);
      setEntities([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="tab-content active">
      <div className="toolbar">
        <input
          type="text"
          className="search-input"
          placeholder="Search memories and entities..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && search()}
          style={{ flex: 1, maxWidth: '400px' }}
        />
        <button type="button" className="btn btn-primary" onClick={search} disabled={loading || !query.trim()}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {!searched && <p className="empty-state">Enter a query and click Search to find memories and entities.</p>}

      {searched && !loading && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginTop: '24px' }}>
          <div className="panel">
            <div className="panel-header">
              <h3>Memories</h3>
              <span className="badge badge-info">{memories.length}</span>
            </div>
            <div className="panel-body">
              {!memories.length ? <p className="empty-state">No memories found</p> : (
                <ul style={{ listStyle: 'none' }}>
                  {memories.map((m) => (
                    <li key={m.id} style={{ marginBottom: '12px', padding: '12px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-sm)' }}>
                      <div style={{ fontWeight: 600 }}>{escapeHtml((m.summary ?? '').substring(0, 100))}{(m.summary ?? '').length > 100 ? '...' : ''}</div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                        {m.memory_type ?? 'general'} · {((m.confidence ?? 0) * 100).toFixed(0)}%
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
          <div className="panel">
            <div className="panel-header">
              <h3>Entities</h3>
              <span className="badge badge-info">{entities.length}</span>
            </div>
            <div className="panel-body">
              {!entities.length ? <p className="empty-state">No entities found</p> : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Type</th>
                      <th>Mentions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {entities.map((e) => (
                      <tr key={e.name}>
                        <td>{escapeHtml(e.name ?? '')}</td>
                        <td><span className="badge badge-info">{e.entity_type ?? 'unknown'}</span></td>
                        <td>{e.mention_count ?? 0}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
