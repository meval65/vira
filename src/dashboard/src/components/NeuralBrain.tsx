import { useEffect, useState, useCallback } from 'react';
import type { WsMessage } from '../types';

interface BrainModule {
  id: string;
  name: string;
  description: string;
  color: string;
  status: 'active' | 'idle';
  activity: string | null;
  data?: Record<string, unknown>;
}

interface BrainConnection {
  source: string;
  target: string;
  type: string;
}

interface BrainStateResponse {
  modules: Record<string, BrainModule>;
  connections: BrainConnection[];
  recent_events: Array<{
    source: string;
    target: string;
    type: string;
    payload: Record<string, unknown>;
    timestamp: string;
  }>;
  current_activity: Record<string, string>;
  system_health: {
    status: string;
    brain_initialized: boolean;
  };
}

interface NeuralBrainProps {
  lastWsEvent: WsMessage | null;
}

// Module positions for SVG layout
const modulePositions: Record<string, { x: number; y: number }> = {
  brainstem: { x: 200, y: 180 },
  hippocampus: { x: 95, y: 110 },
  amygdala: { x: 305, y: 110 },
  prefrontal_cortex: { x: 200, y: 40 },
  thalamus: { x: 55, y: 180 },
  motor_cortex: { x: 345, y: 180 },
  parietal_lobe: { x: 100, y: 265 },
  occipital_lobe: { x: 300, y: 265 },
  cerebellum: { x: 200, y: 320 },
};

export function NeuralBrain({ lastWsEvent }: NeuralBrainProps): React.ReactNode {
  const [brainState, setBrainState] = useState<BrainStateResponse | null>(null);
  const [activeConnections, setActiveConnections] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);

  // Fetch brain state from API
  const fetchBrainState = useCallback(async () => {
    try {
      const response = await fetch('/api/brain-state');
      if (response.ok) {
        const data = await response.json();
        setBrainState(data);
      }
    } catch (error) {
      console.error('Failed to fetch brain state:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial fetch and polling
  useEffect(() => {
    fetchBrainState();
    const interval = setInterval(fetchBrainState, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, [fetchBrainState]);

  // Handle WebSocket events for real-time updates
  useEffect(() => {
    if (lastWsEvent?.data) {
      const eventData = lastWsEvent.data as { source?: string; target?: string };
      const source = eventData.source;
      const target = eventData.target;

      if (source && target) {
        const connectionKey = `${source}-${target}`;
        setActiveConnections(prev => new Set([...prev, connectionKey]));

        // Clear after animation
        setTimeout(() => {
          setActiveConnections(prev => {
            const next = new Set(prev);
            next.delete(connectionKey);
            return next;
          });
        }, 1500);
      }

      // Refetch on event
      fetchBrainState();
    }
  }, [lastWsEvent, fetchBrainState]);

  if (loading) {
    return (
      <div className="neural-brain-container">
        <div className="neural-loading">Loading neural state...</div>
      </div>
    );
  }

  const modules = brainState?.modules || {};
  const connections = brainState?.connections || [];

  return (
    <div className="neural-brain-container">
      <div className="neural-header">
        <h4>🧠 Neural Network Status</h4>
        <span className={`health-badge ${brainState?.system_health?.status === 'healthy' ? 'healthy' : 'warning'}`}>
          {brainState?.system_health?.status || 'unknown'}
        </span>
      </div>

      <svg viewBox="0 0 400 360" className="neural-brain-svg">
        <defs>
          <radialGradient id="brainGlow">
            <stop offset="0%" stopColor="rgba(139, 92, 246, 0.15)" />
            <stop offset="100%" stopColor="transparent" />
          </radialGradient>

          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          <linearGradient id="activeConnection" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#6366f1">
              <animate attributeName="stop-color" values="#6366f1;#8b5cf6;#6366f1" dur="1s" repeatCount="indefinite" />
            </stop>
            <stop offset="100%" stopColor="#8b5cf6">
              <animate attributeName="stop-color" values="#8b5cf6;#6366f1;#8b5cf6" dur="1s" repeatCount="indefinite" />
            </stop>
          </linearGradient>
        </defs>

        {/* Background glow */}
        <circle cx="200" cy="180" r="160" fill="url(#brainGlow)" />

        {/* Connections */}
        <g className="connections">
          {connections.map((conn) => {
            const from = modulePositions[conn.source];
            const to = modulePositions[conn.target];
            if (!from || !to) return null;

            const key = `${conn.source}-${conn.target}`;
            const isActive = activeConnections.has(key) || activeConnections.has(`${conn.target}-${conn.source}`);

            return (
              <line
                key={key}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke={isActive ? 'url(#activeConnection)' : 'rgba(99, 102, 241, 0.15)'}
                strokeWidth={isActive ? 3 : 1}
                className={isActive ? 'connection-active' : ''}
              />
            );
          })}
        </g>

        {/* Module nodes */}
        <g className="modules">
          {Object.values(modules).map((module) => {
            const pos = modulePositions[module.id];
            if (!pos) return null;

            const isActive = module.status === 'active' || module.activity;

            return (
              <g key={module.id} className={`module-node ${isActive ? 'module-active' : 'module-idle'}`}>
                {/* Glow ring for active modules */}
                {isActive && (
                  <circle
                    cx={pos.x}
                    cy={pos.y}
                    r={25}
                    fill={module.color}
                    opacity={0.25}
                    className="module-glow-ring"
                  />
                )}

                {/* Main circle */}
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={18}
                  fill={isActive ? module.color : 'var(--bg-tertiary)'}
                  stroke={module.color}
                  strokeWidth={2}
                  filter={isActive ? 'url(#glow)' : undefined}
                />

                {/* Module label */}
                <text
                  x={pos.x}
                  y={pos.y + 32}
                  textAnchor="middle"
                  fill="var(--text-primary)"
                  fontSize="9"
                  fontWeight="600"
                >
                  {module.name}
                </text>

                {/* Activity indicator */}
                {module.activity && (
                  <text
                    x={pos.x}
                    y={pos.y + 42}
                    textAnchor="middle"
                    fill={module.color}
                    fontSize="7"
                    className="activity-label"
                  >
                    {module.activity.length > 20 ? module.activity.substring(0, 20) + '...' : module.activity}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      {/* Activity feed */}
      <div className="neural-activity-feed">
        <h5>Recent Activity</h5>
        <div className="activity-list">
          {brainState?.recent_events?.slice(-5).reverse().map((event, i) => (
            <div key={i} className="activity-item">
              <span className="activity-dot" style={{ background: modules[event.source]?.color || '#6366f1' }} />
              <span className="activity-source">{event.source}</span>
              <span className="activity-arrow">→</span>
              <span className="activity-target">{event.target}</span>
              <span className="activity-type">{event.type}</span>
            </div>
          ))}
          {(!brainState?.recent_events?.length) && (
            <div className="activity-empty">Waiting for activity...</div>
          )}
        </div>
      </div>

      {/* Module data cards */}
      <div className="neural-data-grid">
        {Object.values(modules).filter(m => m.data).slice(0, 4).map((module) => (
          <div key={module.id} className="neural-data-card" style={{ borderColor: module.color }}>
            <div className="data-card-header">
              <span className="data-dot" style={{ background: module.color }} />
              <span>{module.name}</span>
            </div>
            <div className="data-card-content">
              {Object.entries(module.data || {}).map(([key, value]) => (
                <div key={key} className="data-row">
                  <span className="data-key">{key.replace(/_/g, ' ')}</span>
                  <span className="data-value">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
