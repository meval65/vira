import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { GenericConfig } from '../api/types';
import { Save, RefreshCw, Cpu, Database, Activity, Terminal } from 'lucide-react';

export default function SettingsPanel() {
    const [config, setConfig] = useState<GenericConfig>({});
    const [globalContext, setGlobalContext] = useState('');

    const [saving, setSaving] = useState(false);

    useEffect(() => {
        loadSettings();
    }, []);

    const loadSettings = async () => {
        try {
            const [conf, ctx] = await Promise.all([
                api.getConfig(),
                api.getGlobalContext()
            ]);
            setConfig(conf);
            setGlobalContext(ctx.context_text || '');
        } catch (error) {
            console.error(error);
            // Mock
            if (!config.chat_model) {
                setConfig({ chat_model: 'openai/gpt-4o', temperature: 0.7, max_tokens: 512 });
            }
        } finally {

        }
    };

    const handleSaveConfig = async () => {
        setSaving(true);
        try {
            await api.updateConfig(config);
            await api.updateGlobalContext(globalContext);
            alert('Settings saved successfully');
        } catch (error) {
            console.error(error);
        } finally {
            setSaving(false);
        }
    };

    const triggerMaintenance = async () => {
        if (!confirm('Run system maintenance?')) return;
        try {
            await api.triggerMaintenance();
            alert('Maintenance triggered');
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div className="space-y-6 fade-in-up">
            <div className="flex justify-between items-center">
                <div>
                    <h2 className="text-xl font-bold text-white">System Configuration</h2>
                    <p className="text-xs text-slate-400">Global parameters and cortex settings</p>
                </div>
                <button
                    onClick={handleSaveConfig}
                    disabled={saving}
                    className="px-6 py-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:opacity-90 rounded-lg text-sm font-bold shadow-lg shadow-emerald-500/20 transition-all flex items-center gap-2 disabled:opacity-50"
                >
                    <Save size={18} /> {saving ? 'Saving...' : 'Save Changes'}
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="glass-panel p-6 rounded-2xl space-y-4">
                    <div className="flex items-center gap-3 mb-2">
                        <Cpu className="text-violet-400" size={20} />
                        <h3 className="font-bold text-white">Model Parameters</h3>
                    </div>

                    <div className="space-y-3">
                        <div>
                            <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Primary Model</label>
                            <input
                                className="input-neural w-full p-3 rounded-xl text-sm text-white font-mono"
                                value={String(config.chat_model || '')}
                                onChange={e => setConfig({ ...config, chat_model: e.target.value })}
                            />
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Temperature: {config.temperature as number}</label>
                                <input
                                    type="range" min="0" max="1" step="0.05"
                                    className="w-full accent-violet-500"
                                    value={Number(config.temperature || 0)}
                                    onChange={e => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
                                />
                            </div>
                            <div>
                                <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Max Tokens</label>
                                <input
                                    type="number"
                                    className="input-neural w-full p-2 rounded-xl text-sm text-white"
                                    value={Number(config.max_output_tokens || 512)}
                                    onChange={e => setConfig({ ...config, max_output_tokens: parseInt(e.target.value) })}
                                />
                            </div>
                        </div>
                    </div>
                </div>

                <div className="glass-panel p-6 rounded-2xl space-y-4">
                    <div className="flex items-center gap-3 mb-2">
                        <Database className="text-blue-400" size={20} />
                        <h3 className="font-bold text-white">Global Context</h3>
                    </div>
                    <p className="text-xs text-slate-400 mb-2">Persistent facts injected into every prompt</p>
                    <textarea
                        className="input-neural w-full h-40 p-4 rounded-xl text-sm text-white font-mono leading-relaxed"
                        placeholder="e.g. User is a software engineer living in Jakarta..."
                        value={globalContext}
                        onChange={e => setGlobalContext(e.target.value)}
                    />
                </div>

                <div className="glass-panel p-6 rounded-2xl space-y-4 lg:col-span-2">
                    <div className="flex items-center gap-3 mb-2">
                        <Activity className="text-rose-400" size={20} />
                        <h3 className="font-bold text-white">Maintenance & Diagnostics</h3>
                    </div>
                    <div className="flex gap-4">
                        <button onClick={triggerMaintenance} className="btn bg-white/5 hover:bg-white/10 text-white border border-white/10 flex items-center gap-2 px-4 py-2 rounded-lg">
                            <RefreshCw size={16} /> Run Maintenance Tasks
                        </button>
                        <button className="btn bg-white/5 hover:bg-white/10 text-white border border-white/10 flex items-center gap-2 px-4 py-2 rounded-lg">
                            <Terminal size={16} /> View System Logs
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
