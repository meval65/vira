import { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import type { Persona } from '../api/types';
import { Plus, User, Trash2, Smartphone } from 'lucide-react';
import Modal from './Modal';

export default function PersonasPanel() {
    const [personas, setPersonas] = useState<Persona[]>([]);
    const [activePersona, setActivePersona] = useState<Persona | null>(null);

    const [showCreate, setShowCreate] = useState(false);
    const [newPersona, setNewPersona] = useState({ name: '', instruction: '', temperature: 0.7, description: '' });

    const loadData = useCallback(async () => {
        try {
            const [all, active] = await Promise.all([
                api.getPersonas(),
                api.getActivePersona()
            ]);
            setPersonas(all);
            setActivePersona(active.persona);
        } catch (error) {
            console.error(error);
            if (personas.length === 0) {
                const mock: Persona[] = [
                    { id: '1', name: 'Default Assistant', instruction: 'You are helpful.', temperature: 0.7, is_active: true, created_at: new Date().toISOString() },
                    { id: '2', name: 'Creative Writer', instruction: 'You are a poet.', temperature: 0.9, is_active: false, created_at: new Date().toISOString() }
                ];
                setPersonas(mock);
                setActivePersona(mock[0]);
            }
        } finally {
            // setLoading(false);
        }
    }, [personas.length]);

    useEffect(() => {
        loadData();
    }, [loadData]);

    const handleCreate = async () => {
        try {
            await api.createPersona(newPersona);
            setShowCreate(false);
            setNewPersona({ name: '', instruction: '', temperature: 0.7, description: '' });
            loadData();
        } catch (error) {
            console.error(error);
        }
    };

    const handleActivate = async (id: string) => {
        try {
            await api.activatePersona(id);
            loadData();
        } catch (error) {
            console.error(error);
        }
    };

    const handleDelete = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!confirm('Delete this persona?')) return;
        try {
            await api.deletePersona(id);
            loadData();
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div className="space-y-6 fade-in-up">
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-xl font-bold text-white">Personality Matrix</h2>
                    <p className="text-xs text-slate-400">Manage AI behaviors and instruction sets</p>
                </div>
                <button
                    onClick={() => setShowCreate(true)}
                    className="px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:opacity-90 rounded-lg text-sm font-medium shadow-lg shadow-purple-500/20 transition-all flex items-center gap-2"
                >
                    <Plus size={16} /> New Persona
                </button>
            </div>

            {showCreate && (
                <Modal title="Design New Persona" onClose={() => setShowCreate(false)}>
                    <div className="space-y-4">
                        <div>
                            <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Name</label>
                            <input
                                className="input-neural w-full p-3 rounded-xl text-sm text-white"
                                placeholder="e.g. Coder, Poet, Cynic"
                                value={newPersona.name}
                                onChange={e => setNewPersona({ ...newPersona, name: e.target.value })}
                                autoFocus
                            />
                        </div>
                        <div>
                            <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">System Instruction</label>
                            <textarea
                                className="input-neural w-full h-32 p-3 rounded-xl text-sm text-white font-mono"
                                placeholder="You are a helpful assistant..."
                                value={newPersona.instruction}
                                onChange={e => setNewPersona({ ...newPersona, instruction: e.target.value })}
                            />
                        </div>
                        <div>
                            <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Creativity (Temperature): {newPersona.temperature}</label>
                            <input
                                type="range"
                                min="0" max="1" step="0.1"
                                value={newPersona.temperature}
                                onChange={e => setNewPersona({ ...newPersona, temperature: parseFloat(e.target.value) })}
                                className="w-full accent-purple-500"
                            />
                        </div>
                        <button onClick={handleCreate} className="w-full py-3 bg-purple-600 hover:bg-purple-500 rounded-xl font-bold text-white shadow-lg shadow-purple-900/20 mt-2">
                            Initialize Persona
                        </button>
                    </div>
                </Modal>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {personas.map(persona => {
                    const isActive = activePersona?.id === persona.id;
                    return (
                        <div
                            key={persona.id}
                            className={`glass-card p-6 rounded-2xl transition-all relative group ${isActive ? 'border-purple-500/50 bg-purple-500/5' : 'hover:border-white/20'}`}
                        >
                            <div className="flex justify-between items-start mb-4">
                                <div className="flex items-center gap-3">
                                    <div className={`p-3 rounded-xl ${isActive ? 'bg-purple-500 text-white shadow-lg shadow-purple-500/30' : 'bg-white/5 text-slate-400'}`}>
                                        <User size={24} />
                                    </div>
                                    <div>
                                        <h3 className="text-lg font-bold text-white">{persona.name}</h3>
                                        <span className="text-xs font-mono text-slate-500">Temp: {persona.temperature}</span>
                                    </div>
                                </div>
                                {isActive && <div className="px-3 py-1 bg-purple-500/20 text-purple-200 text-xs font-bold rounded-full border border-purple-500/20">ACTIVE</div>}
                            </div>

                            <p className="text-sm text-slate-300 line-clamp-3 mb-6 font-mono bg-black/20 p-3 rounded-lg border border-white/5">
                                {persona.instruction}
                            </p>

                            <div className="flex gap-3">
                                {!isActive && (
                                    <button
                                        onClick={() => handleActivate(persona.id)}
                                        className="flex-1 py-2 bg-white/5 hover:bg-white/10 rounded-lg text-sm font-medium text-white transition-colors flex items-center justify-center gap-2"
                                    >
                                        <Smartphone size={16} /> Activate
                                    </button>
                                )}
                                <button
                                    onClick={(e) => handleDelete(persona.id, e)}
                                    className="p-2 bg-rose-500/10 text-rose-400 hover:bg-rose-500 hover:text-white rounded-lg transition-colors"
                                >
                                    <Trash2 size={18} />
                                </button>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
