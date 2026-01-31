import { useState, useCallback } from 'react';
import { Layout, type TabId } from './components/Layout';
import { Toast } from './components/Toast';
import { useWebSocket } from './hooks/useWebSocket';
import { Overview } from './pages/Overview';
import { Memories } from './pages/Memories';
import { Schedules } from './pages/Schedules';
import { Personas } from './pages/Personas';
import { Entities } from './pages/Entities';
import { Triples } from './pages/Triples';
import { ChatLogs } from './pages/ChatLogs';
import { Search } from './pages/Search';
import { NeuralEvents } from './pages/NeuralEvents';
import { System } from './pages/System';
import { Settings } from './pages/Settings';
import type { WsMessage } from './types';

interface ToastItem {
  id: number;
  message: string;
  type: 'success' | 'error' | 'info';
}

export default function App(): React.ReactNode {
  const [currentTab, setCurrentTab] = useState<TabId>('overview');
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const showToast = useCallback((message: string, type: 'success' | 'error' | 'info' = 'info') => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3000);
  }, []);

  const { connected, lastEvent } = useWebSocket((msg: WsMessage) => {
    if (msg.type === 'log' && (msg.data as { level?: string })?.level === 'ERROR') {
      showToast((msg.data as { message?: string })?.message ?? 'System error', 'error');
    }
  });

  const renderContent = (): React.ReactNode => {
    switch (currentTab) {
      case 'overview':
        return <Overview lastWsEvent={lastEvent} />;
      case 'memories':
        return <Memories showToast={showToast} lastWsEvent={lastEvent} />;
      case 'schedules':
        return <Schedules showToast={showToast} lastWsEvent={lastEvent} />;
      case 'personas':
        return <Personas showToast={showToast} lastWsEvent={lastEvent} />;
      case 'entities':
        return <Entities showToast={showToast} lastWsEvent={lastEvent} />;
      case 'triples':
        return <Triples showToast={showToast} lastWsEvent={lastEvent} />;
      case 'chat-logs':
        return <ChatLogs showToast={showToast} lastWsEvent={lastEvent} />;
      case 'search':
        return <Search />;
      case 'neural-events':
        return <NeuralEvents lastWsEvent={lastEvent} />;
      case 'system':
        return <System showToast={showToast} />;
      case 'settings':
        return <Settings showToast={showToast} />;
      default:
        return <Overview lastWsEvent={lastEvent} />;
    }
  };

  return (
    <>
      <Layout currentTab={currentTab} onTabChange={setCurrentTab} connected={connected}>
        {renderContent()}
      </Layout>

      <div className="toast-container">
        {toasts.map((t) => (
          <Toast key={t.id} message={t.message} type={t.type} />
        ))}
      </div>
    </>
  );
}
