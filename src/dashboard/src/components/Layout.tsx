import type { ReactNode } from 'react';

export const TABS = [
  { id: 'overview', label: 'Overview', icon: '📊' },
  { id: 'memories', label: 'Memories', icon: '💭' },
  { id: 'schedules', label: 'Schedules', icon: '📅' },
  { id: 'personas', label: 'Personas', icon: '🎭' },
  { id: 'entities', label: 'Entities', icon: '👤' },
  { id: 'triples', label: 'Knowledge', icon: '🔗' },
  { id: 'chat-logs', label: 'Chat Logs', icon: '💬' },
  { id: 'search', label: 'Search', icon: '🔍' },
  { id: 'neural-events', label: 'Neural Events', icon: '⚡' },
  { id: 'system', label: 'System', icon: '🖥️' },
  { id: 'settings', label: 'Settings', icon: '⚙️' },
] as const;

export type TabId = (typeof TABS)[number]['id'];

interface LayoutProps {
  currentTab: TabId;
  onTabChange: (tab: TabId) => void;
  connected: boolean;
  children: ReactNode;
}

export function Layout({ currentTab, onTabChange, connected, children }: LayoutProps): ReactNode {
  const pageTitle = TABS.find((t) => t.id === currentTab)?.label ?? 'Overview';

  return (
    <div className="app">
      <nav className="sidebar">
        <div className="logo">
          <span className="logo-icon">🧠</span>
          <span className="logo-text">Vira</span>
        </div>
        <ul className="nav-menu">
          {TABS.map((tab) => (
            <li key={tab.id}>
              <button
                type="button"
                className={`nav-item ${currentTab === tab.id ? 'active' : ''}`}
                onClick={() => onTabChange(tab.id)}
              >
                <span className="nav-icon">{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            </li>
          ))}
        </ul>
        <div className="sidebar-footer">
          <div className={`status-indicator ${connected ? 'online' : ''}`} />
          <span>{connected ? 'System Online' : 'Disconnected'}</span>
        </div>
      </nav>

      <main className="main-content">
        <header className="header">
          <h1>{pageTitle}</h1>
          <div className="connection-status">
            <span
              className="status-dot"
              style={{ background: connected ? 'var(--success)' : 'var(--danger)' }}
            />
            <span>{connected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </header>

        <div className="content-wrapper">{children}</div>
      </main>
    </div>
  );
}
