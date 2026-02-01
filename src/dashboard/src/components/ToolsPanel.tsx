import { useEffect, useState, useCallback } from 'react';
import type { WsMessage } from '../types';

interface Tool {
    name: string;
    description: string;
    enabled: boolean;
    usage_count: number;
    last_used: string | null;
}

interface ToolsResponse {
    tools: Tool[];
    total: number;
    total_executions: number;
}

interface ToolsPanelProps {
    lastWsEvent: WsMessage | null;
}

export function ToolsPanel({ lastWsEvent }: ToolsPanelProps): React.ReactNode {
    const [toolsData, setToolsData] = useState<ToolsResponse | null>(null);
    const [loading, setLoading] = useState(true);

    const fetchTools = useCallback(async () => {
        try {
            const response = await fetch('/api/tools');
            if (response.ok) {
                const data = await response.json();
                setToolsData(data);
            }
        } catch (error) {
            console.error('Failed to fetch tools:', error);
        } finally {
            setLoading(false);
        }
    }, []);

    // Initial fetch and polling
    useEffect(() => {
        fetchTools();
        const interval = setInterval(fetchTools, 5000);
        return () => clearInterval(interval);
    }, [fetchTools]);

    // Refetch on tool execution events
    useEffect(() => {
        if (lastWsEvent?.type === 'parietal_lobe_event' || lastWsEvent?.data?.type === 'tool_executed') {
            fetchTools();
        }
    }, [lastWsEvent, fetchTools]);

    if (loading) {
        return (
            <div className="tools-panel">
                <div className="panel-header">
                    <h3>🔧 Tools</h3>
                </div>
                <div className="tools-loading">Loading tools...</div>
            </div>
        );
    }

    const tools = toolsData?.tools || [];

    return (
        <div className="tools-panel">
            <div className="panel-header">
                <h3>🔧 Tools</h3>
                <span className="tools-badge">
                    {toolsData?.total_executions || 0} executions
                </span>
            </div>

            <div className="tools-grid">
                {tools.map((tool) => (
                    <div
                        key={tool.name}
                        className={`tool-card ${tool.usage_count > 0 ? 'tool-used' : ''} ${!tool.enabled ? 'tool-disabled' : ''}`}
                    >
                        <div className="tool-header">
                            <span className="tool-name">{tool.name.replace(/_/g, ' ')}</span>
                            <span className="tool-count">{tool.usage_count}</span>
                        </div>
                        <div className="tool-description">{tool.description}</div>
                        {tool.last_used && (
                            <div className="tool-last-used">
                                Last: {new Date(tool.last_used).toLocaleTimeString()}
                            </div>
                        )}
                    </div>
                ))}
            </div>

            {tools.length === 0 && (
                <div className="tools-empty">No tools available</div>
            )}
        </div>
    );
}
