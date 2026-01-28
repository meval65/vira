import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import type { Entity } from '../api/types';
import { Search, Users, Hash } from 'lucide-react';


import { useWebSocket } from '../hooks/useWebSocket';

export default function EntitiesPanel() {
    const [entities, setEntities] = useState<Entity[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');

    const { lastMessage } = useWebSocket('/ws');

    const loadEntities = useCallback(async () => {
        setLoading(true);
        try {
            const data = await api.getEntities(100);
            setEntities(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadEntities();
    }, [loadEntities]);

    useEffect(() => {
        if (lastMessage && (lastMessage as any).type === 'entity_update') {
            loadEntities();
        }
    }, [lastMessage, loadEntities]);

    const filteredEntities = entities.filter(e =>
        e.name.toLowerCase().includes(search.toLowerCase()) ||
        e.entity_type.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div className="space-y-6 fade-in-up">
            <div className="flex justify-between items-center gap-4">
                <div>
                    <h2 className="text-xl font-bold text-white">Recognized Entities</h2>
                    <p className="text-xs text-slate-400">Named entities extracted from conversations</p>
                </div>
                <div className="flex gap-4">
                    <div className="relative w-64">
                        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Filter entities..."
                            className="input-neural w-full pl-10 py-2 rounded-full text-sm bg-black/20 focus:bg-black/40"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                        />
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 max-h-[60vh] overflow-y-auto scrollbar pr-2">
                {filteredEntities.map((entity, idx) => (
                    <div key={entity.id || idx} className="glass-card p-4 rounded-xl flex flex-col gap-3 group animate-fade-in-up" style={{ animationDelay: `${idx * 30}ms` }}>
                        <div className="flex items-start justify-between">
                            <div className="p-2 bg-pink-500/10 rounded-lg text-pink-400">
                                {entity.entity_type === 'person' ? <Users size={18} /> : <Hash size={18} />}
                            </div>
                            <span className="text-[10px] bg-white/5 px-2 py-0.5 rounded text-slate-400 font-mono">
                                {entity.mention_count} refs
                            </span>
                        </div>
                        <div>
                            <h3 className="font-bold text-white text-lg">{entity.name}</h3>
                            <span className="text-xs text-slate-400 uppercase tracking-wider">{entity.entity_type}</span>
                        </div>
                    </div>
                ))}

                {!loading && filteredEntities.length === 0 && (
                    <div className="col-span-full flex flex-col items-center justify-center py-20 text-slate-500 border border-dashed border-white/10 rounded-2xl">
                        <Users size={40} className="text-slate-600 mb-2 opacity-50" />
                        <p className="text-sm font-medium text-slate-400">No entities found.</p>
                    </div>
                )}
            </div>
        </div>
    );
}
