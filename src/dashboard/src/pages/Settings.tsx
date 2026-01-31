import { useState, useEffect } from 'react';
import { api } from '../api';
import type { AdminProfile, SystemConfig, GlobalContext, OpenRouterModels } from '../types';

interface SettingsProps {
  showToast: (msg: string, type: 'success' | 'error' | 'info') => void;
}

export function Settings({ showToast }: SettingsProps): React.ReactNode {
  const [contextText, setContextText] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.95);
  const [maxTokens, setMaxTokens] = useState(512);
  const [chatModel, setChatModel] = useState('');
  const [models, setModels] = useState<OpenRouterModels['models']>({});
  const [telegramName, setTelegramName] = useState('');
  const [additionalInfo, setAdditionalInfo] = useState('');

  useEffect(() => {
    api<GlobalContext & { context_text?: string }>('/api/global-context').then((r) => setContextText(r?.context_text ?? '')).catch(() => {});
    api<SystemConfig>('/api/system-config').then((r) => {
      setTemperature(r?.temperature ?? 0.7);
      setTopP(r?.top_p ?? 0.95);
      setMaxTokens(r?.max_output_tokens ?? 512);
      setChatModel(r?.chat_model ?? '');
    }).catch(() => {});
    api<OpenRouterModels>('/api/openrouter/models').then((r) => setModels(r?.models ?? {})).catch(() => {});
    api<AdminProfile>('/api/admin/profile').then((r) => {
      setTelegramName(r?.telegram_name ?? '');
      setAdditionalInfo(r?.additional_info ?? '');
    }).catch(() => {});
  }, []);

  const saveContext = async (): Promise<void> => {
    try {
      await api('/api/global-context', 'PUT', { context_text: contextText });
      showToast('Context saved!', 'success');
    } catch {
      showToast('Failed to save context', 'error');
    }
  };

  const saveConfig = async (): Promise<void> => {
    try {
      await api('/api/system-config', 'PUT', {
        temperature: Number(temperature),
        top_p: Number(topP),
        max_output_tokens: Number(maxTokens),
        ...(chatModel ? { chat_model: chatModel } : {}),
      });
      showToast('Configuration saved!', 'success');
    } catch {
      showToast('Failed to save config', 'error');
    }
  };

  const saveAdminProfile = async (): Promise<void> => {
    try {
      await api('/api/admin/profile', 'PUT', {
        telegram_name: telegramName,
        additional_info: additionalInfo,
      });
      showToast('Admin profile saved!', 'success');
    } catch {
      showToast('Failed to save profile', 'error');
    }
  };

  const triggerMaintenance = async (): Promise<void> => {
    try {
      const res = await api<{ status?: string }>('/api/maintenance/trigger', 'POST');
      showToast(`Maintenance ${res?.status ?? 'done'}`, 'success');
    } catch {
      showToast('Maintenance failed', 'error');
    }
  };

  const compressMemories = async (): Promise<void> => {
    try {
      const res = await api<{ compressed?: number }>('/api/maintenance/compress-memories', 'POST', { force: false });
      showToast(`Compressed ${res?.compressed ?? 0} memories`, 'success');
    } catch {
      showToast('Compress failed', 'error');
    }
  };

  const optimizeGraph = async (): Promise<void> => {
    try {
      const res = await api<{ duplicates_removed?: number }>('/api/maintenance/optimize-graph', 'POST');
      showToast(`Removed ${res?.duplicates_removed ?? 0} duplicates`, 'success');
    } catch {
      showToast('Optimize failed', 'error');
    }
  };

  const chatModels: { id: string; name: string }[] = [
    ...(models.chat_model ?? []),
    ...(models.analysis_model ?? []),
    ...(models.utility_model ?? []),
  ];

  return (
    <section className="tab-content active">
      <div className="settings-section">
        <h3>Admin Profile</h3>
        <div className="form-group">
          <label>Telegram Name</label>
          <input type="text" value={telegramName} onChange={(e) => setTelegramName(e.target.value)} placeholder="Display name" />
        </div>
        <div className="form-group">
          <label>Additional Info</label>
          <textarea value={additionalInfo} onChange={(e) => setAdditionalInfo(e.target.value)} placeholder="Extra profile info" style={{ minHeight: '80px' }} />
        </div>
        <button type="button" className="btn btn-primary" onClick={saveAdminProfile}>Save Profile</button>
      </div>

      <div className="settings-section">
        <h3>Global Context</h3>
        <textarea value={contextText} onChange={(e) => setContextText(e.target.value)} placeholder="Enter global context..." style={{ minHeight: '150px', marginBottom: '16px' }} />
        <button type="button" className="btn btn-primary" onClick={saveContext}>Save Context</button>
      </div>

      <div className="settings-section">
        <h3>System Configuration</h3>
        <div className="form-group">
          <label>Chat Model</label>
          <select value={chatModel} onChange={(e) => setChatModel(e.target.value)}>
            <option value="">— Default —</option>
            {chatModels.map((m) => (
              <option key={m.id} value={m.id}>{m.name ?? m.id}</option>
            ))}
          </select>
        </div>
        <div className="form-group">
          <label>Temperature</label>
          <input type="range" min={0} max={1} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
          <span style={{ marginLeft: '8px' }}>{temperature}</span>
        </div>
        <div className="form-group">
          <label>Top P</label>
          <input type="range" min={0} max={1} step={0.01} value={topP} onChange={(e) => setTopP(Number(e.target.value))} />
          <span style={{ marginLeft: '8px' }}>{topP}</span>
        </div>
        <div className="form-group">
          <label>Max Output Tokens</label>
          <input type="number" value={maxTokens} min={64} max={4096} onChange={(e) => setMaxTokens(Number(e.target.value))} style={{ width: '120px' }} />
        </div>
        <button type="button" className="btn btn-primary" onClick={saveConfig}>Save Configuration</button>
      </div>

      <div className="settings-section">
        <h3>Maintenance</h3>
        <div className="maintenance-actions">
          <button type="button" className="btn btn-secondary" onClick={triggerMaintenance}>Run Maintenance</button>
          <button type="button" className="btn btn-secondary" onClick={compressMemories}>Compress Memories</button>
          <button type="button" className="btn btn-secondary" onClick={optimizeGraph}>Optimize Graph</button>
        </div>
      </div>
    </section>
  );
}
