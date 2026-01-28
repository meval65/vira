import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import type { Memory } from '../api/types';
import { Search, Plus, Trash2, Edit2, Brain } from 'lucide-react';
import { format } from 'date-fns';
import Modal from './Modal';
import { useWebSocket } from '../hooks/useWebSocket';

export default function MemoriesPanel() {
    const [memories, setMemories] = useState<Memory[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');
    const [showCreate, setShowCreate] = useState(false);
    const [newMemory, setNewMemory] = useState({ summary: '', type: 'general', priority: 0.5 });

    // WS connection
    const { lastMessage } = useWebSocket('/ws');

    const loadMemories = useCallback(async () => {
        setLoading(true);
        try {
            const data = await api.getMemories(search, 50);
            const filtered = data.filter(m => m.summary.toLowerCase().includes(search.toLowerCase()));
            setMemories(filtered);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }, [search]);

    useEffect(() => {
        loadMemories();
    }, [loadMemories]);

    useEffect(() => {
        if (lastMessage && (lastMessage as any).type === 'memory_update') {
            loadMemories();
        }
    }, [lastMessage, loadMemories]);

    const handleCreate = async () => {
        try {
            await api.createMemory({
                summary: newMemory.summary,
                type: newMemory.type,
                priority: Number(newMemory.priority)
            });
            setShowCreate(false);
            setNewMemory({ summary: '', type: 'general', priority: 0.5 });
            loadMemories();
        } catch (error) {
            console.error(error);
            alert('Failed to create memory');
        }
    };

    const handleDelete = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!confirm('Are you sure you want to delete this memory?')) return;
        try {
            await api.deleteMemory(id);
            loadMemories();
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div className="space-y-6 fade-in-up">
            <div className="flex justify-between items-center gap-4">
                <div>
                    <h2 className="text-xl font-bold text-white">Core Memories</h2>
                    <p className="text-xs text-slate-400">Long-term storage and retrieval system</p>
                </div>
                <div className="flex gap-4">
                    <div className="relative w-64">
                        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search traces..."
                            className="input-neural w-full pl-10 py-2 rounded-full text-sm bg-black/20 focus:bg-black/40"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                        />
                    </div>
                    <button
                        onClick={() => setShowCreate(true)}
                        className="px-4 py-2 bg-gradient-to-r from-violet-600 to-indigo-600 hover:opacity-90 rounded-lg text-sm font-medium shadow-lg shadow-violet-500/20 transition-all flex items-center gap-2"
                    >
                        <Plus size={16} /> Add Memory
                    </button>
                </div>
            </div>

            {showCreate && (
                <Modal title="Initialize Memory Node" onClose={() => setShowCreate(false)}>
                    <div className="space-y-5">
                        <textarea
                            className="input-neural w-full h-32 rounded-xl p-4 text-sm text-white placeholder-slate-500"
                            placeholder="Enter memory data content..."
                            value={newMemory.summary}
                            onChange={(e) => setNewMemory({ ...newMemory, summary: e.target.value })}
                            autoFocus
                        />

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Category</label>
                                <div className="relative">
                                    <select
                                        value={newMemory.type}
                                        onChange={e => setNewMemory({ ...newMemory, type: e.target.value })}
                                        className="w-full input-neural rounded-xl p-3 text-sm text-white appearance-none cursor-pointer"
                                    >
                                        <option value="general">General</option>
                                        <option value="fact">Fact</option>
                                        <option value="preference">Preference</option>
                                        <option value="event">Event</option>
                                        <option value="bio">Bio</option>
                                    </select>
                                </div>
                            </div>

                            <div>
                                <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Neural Priority</label>
                                <div className="flex items-center gap-3 bg-black/20 p-2 rounded-xl border border-white/5">
                                    <input
                                        type="range"
                                        min="0" max="1" step="0.1"
                                        className="flex-1 accent-violet-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                                        value={newMemory.priority}
                                        onChange={(e) => setNewMemory({ ...newMemory, priority: parseFloat(e.target.value) || 0 })}
                                    />
                                    <span className="text-xs font-mono text-violet-400 w-8 text-right">{(newMemory.priority * 100).toFixed(0)}%</span>
                                </div>
                            </div>
                        </div>

                        <button onClick={handleCreate} className="w-full py-3 bg-violet-600 hover:bg-violet-500 rounded-xl font-bold text-white shadow-lg shadow-violet-900/20 transition-all transform active:scale-95">
                            Write to Core
                        </button>
                    </div>
                </Modal>
            )}

            <div className="grid gap-3 max-h-[60vh] overflow-y-auto scrollbar pr-2">
                {memories.map((memory, idx) => (
                    <div key={memory.id} className="glass-card p-4 rounded-xl flex gap-4 group animate-fade-in-up" style={{ animationDelay: `${idx * 50}ms` }}>
                        <div className={`w-1 rounded-full ${memory.type === 'fact' ? 'bg-emerald-500' : memory.type === 'preference' ? 'bg-blue-500' : 'bg-slate-500'}`}></div>
                        <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                                <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-md bg-white/5 border border-white/5 ${memory.type === 'fact' ? 'text-emerald-400' :
                                    memory.type === 'preference' ? 'text-blue-400' : 'text-slate-400'
                                    }`}>{memory.type}</span>
                                <div className="h-1 flex-1 bg-white/5 rounded-full overflow-hidden max-w-[100px]">
                                    <div className="h-full bg-violet-500" style={{ width: `${memory.priority * 100}%` }}></div>
                                </div>
                                <span className="text-[10px] text-slate-600 font-mono ml-auto">{format(new Date(memory.created_at || new Date()), 'MMM d')}</span>
                            </div>
                            <p className="text-sm text-slate-200 leading-relaxed font-medium">{memory.summary}</p>
                        </div>
                        <div className="flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-all transform translate-x-2 group-hover:translate-x-0">
                            <button className="p-2 rounded-lg bg-blue-500/10 text-blue-400 hover:bg-blue-500 hover:text-white transition-colors"><Edit2 size={14} /></button>
                            <button onClick={(e) => handleDelete(memory.id, e)} className="p-2 rounded-lg bg-rose-500/10 text-rose-400 hover:bg-rose-500 hover:text-white transition-colors"><Trash2 size={14} /></button>
                        </div>
                    </div>
                ))}
            </div>

            {!loading && memories.length === 0 && (
                <div className="flex flex-col items-center justify-center py-20 text-slate-500 border border-dashed border-white/10 rounded-2xl">
                    <Brain size={40} className="text-slate-600 mb-2 opacity-50" />
                    <p className="text-sm font-medium text-slate-400">No memory traces found.</p>
                </div>
            )}
        </div>
    );
}
