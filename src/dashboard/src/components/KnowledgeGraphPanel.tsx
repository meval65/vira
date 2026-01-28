import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import type { Triple } from '../api/types';
import { Search, Plus, Trash2, Share2, RefreshCw, Database } from 'lucide-react';
import { format } from 'date-fns';
import Modal from './Modal';

import { useWebSocket } from '../hooks/useWebSocket';

export default function KnowledgeGraphPanel() {
    const [triples, setTriples] = useState<Triple[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');
    const [showCreate, setShowCreate] = useState(false);
    const [newTriple, setNewTriple] = useState({ subject: '', predicate: '', object: '' });

    // WS
    const { lastMessage } = useWebSocket('/ws');

    const loadTriples = useCallback(async () => {
        setLoading(true);
        try {
            const data = await api.getTriples(100);
            setTriples(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadTriples();
    }, [loadTriples]);

    useEffect(() => {
        if (lastMessage && (lastMessage as any).type === 'triple_update') {
            loadTriples();
        }
    }, [lastMessage, loadTriples]);

    const handleCreate = async () => {
        try {
            await api.createTriple({
                subject: newTriple.subject,
                predicate: newTriple.predicate,
                object: newTriple.object,
                confidence: 0.8
            });
            setShowCreate(false);
            setNewTriple({ subject: '', predicate: '', object: '' });
            loadTriples();
        } catch (error) {
            console.error(error);
        }
    };

    const handleDelete = async (id: string) => {
        if (!confirm('Delete this triple?')) return;
        try {
            await api.deleteTriple(id);
            loadTriples();
        } catch (error) {
            console.error(error);
        }
    };

    const filteredTriples = triples.filter(t =>
        t.subject.toLowerCase().includes(search.toLowerCase()) ||
        t.object.toLowerCase().includes(search.toLowerCase()) ||
        t.predicate.toLowerCase().includes(search.toLowerCase())
    );

    return (
        <div className="space-y-6 fade-in-up">
            <div className="flex justify-between items-center gap-4">
                <div>
                    <h2 className="text-xl font-bold text-white">Neural Connections</h2>
                    <p className="text-xs text-slate-400">Knowledge graph triples and associations</p>
                </div>
                <div className="flex gap-4">
                    <div className="relative w-64">
                        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search nodes..."
                            className="input-neural w-full pl-10 py-2 rounded-full text-sm bg-black/20 focus:bg-black/40"
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                        />
                    </div>
                    <button onClick={loadTriples} className="p-2 text-slate-400 hover:text-white transition-colors">
                        <RefreshCw size={18} />
                    </button>
                    <button
                        onClick={() => setShowCreate(true)}
                        className="px-4 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:opacity-90 rounded-lg text-sm font-medium shadow-lg shadow-cyan-500/20 transition-all flex items-center gap-2"
                    >
                        <Plus size={16} /> Add Triple
                    </button>
                </div>
            </div>

            {showCreate && (
                <Modal title="Form New Connection" onClose={() => setShowCreate(false)}>
                    <div className="space-y-4">
                        <div className="grid grid-cols-3 gap-2 items-end">
                            <div>
                                <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Subject</label>
                                <input
                                    className="input-neural w-full p-3 rounded-xl text-sm text-white"
                                    placeholder="Entity A"
                                    value={newTriple.subject}
                                    onChange={e => setNewTriple({ ...newTriple, subject: e.target.value })}
                                    autoFocus
                                />
                            </div>
                            <div className="pb-3 text-center text-slate-500 font-mono text-xs">──[predicate]──▶</div>
                            <div>
                                <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Object</label>
                                <input
                                    className="input-neural w-full p-3 rounded-xl text-sm text-white"
                                    placeholder="Entity B"
                                    value={newTriple.object}
                                    onChange={e => setNewTriple({ ...newTriple, object: e.target.value })}
                                />
                            </div>
                        </div>
                        <div>
                            <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Predicate (Relationship)</label>
                            <input
                                className="input-neural w-full p-3 rounded-xl text-sm text-white"
                                placeholder="e.g. loves, is_located_in, has_part"
                                value={newTriple.predicate}
                                onChange={e => setNewTriple({ ...newTriple, predicate: e.target.value })}
                            />
                        </div>
                        <button onClick={handleCreate} className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 rounded-xl font-bold text-white shadow-lg shadow-cyan-900/20 transition-all mt-4">
                            Forge Connection
                        </button>
                    </div>
                </Modal>
            )}

            <div className="grid gap-3 max-h-[60vh] overflow-y-auto scrollbar pr-2">
                {filteredTriples.map((triple, idx) => (
                    <div key={triple.id} className="glass-card p-4 rounded-xl flex items-center gap-4 group animate-fade-in-up" style={{ animationDelay: `${idx * 50}ms` }}>
                        <div className="p-3 bg-cyan-500/10 rounded-lg text-cyan-400">
                            <Share2 size={20} />
                        </div>
                        <div className="flex-1 flex items-center gap-2 font-mono text-sm">
                            <span className="text-white font-bold bg-white/5 px-2 py-1 rounded">{triple.subject}</span>
                            <span className="text-slate-500 text-xs">── {triple.predicate} ──▶</span>
                            <span className="text-white font-bold bg-white/5 px-2 py-1 rounded">{triple.object}</span>
                        </div>
                        <div className="flex flex-col items-end gap-1">
                            <span className="text-[10px] text-slate-500">{format(new Date(triple.created_at || Date.now()), 'MMM d')}</span>
                            <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-1.5 rounded">{(triple.confidence * 100).toFixed(0)}% CONF</span>
                        </div>
                        <button onClick={() => handleDelete(triple.id)} className="opacity-0 group-hover:opacity-100 p-2 text-rose-400 hover:bg-rose-500/10 rounded-lg transition-all">
                            <Trash2 size={16} />
                        </button>
                    </div>
                ))}

                {!loading && filteredTriples.length === 0 && (
                    <div className="flex flex-col items-center justify-center py-20 text-slate-500 border border-dashed border-white/10 rounded-2xl">
                        <Database size={40} className="text-slate-600 mb-2 opacity-50" />
                        <p className="text-sm font-medium text-slate-400">No connections found.</p>
                    </div>
                )}
            </div>
        </div>
    );
}
