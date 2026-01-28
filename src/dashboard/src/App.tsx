import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import {
  Brain, Calendar, MessageSquare,
  Users, Settings, Terminal as TerminalIcon,
  RefreshCw, LayoutDashboard, Database, Cpu, Heart, Share2, Clock
} from 'lucide-react';
import './index.css';

// Components
import Stat from './components/Stat';
import Tab from './components/Tab';
// import Modal from './components/Modal';

// Panels
import MemoriesPanel from './components/MemoriesPanel';
import SchedulesPanel from './components/SchedulesPanel';
import ChatLogsPanel from './components/ChatLogsPanel';
import KnowledgeGraphPanel from './components/KnowledgeGraphPanel';
import EntitiesPanel from './components/EntitiesPanel';
import PersonasPanel from './components/PersonasPanel';
import SettingsPanel from './components/SettingsPanel';
import TerminalPanel from './components/TerminalPanel';
import NeuralViz from './components/NeuralViz';

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const { isConnected } = useWebSocket('/ws'); // Removed lastMessage
  const [loading, setLoading] = useState(false);
  const [data] = useState({
    stats: { memories: 124, pending_schedules: 3, triples: 845, entities: 42 },
    memories: [],
    chatLogs: [],
    schedules: [],
    triples: [],
    entities: [],
    personas: [{ id: '1', name: 'Vira Core', is_active: true, temperature: 0.7, instruction: 'You are Vira, a helpful assistant.', created_at: new Date().toISOString() }],
    profile: { telegram_name: 'user123', full_name: 'Administrator' },
    emotional: { mood: 'Curious', empathy: 0.8, satisfaction: 0.9 },
    system: { status: 'Operational', database: 'MongoDB', api: { health: 'healthy', current_model: 'models/gemini-2.0-flash-exp' }, memory_usage: 45 },
    instruction: { is_custom: false, name: 'Default' },
    moduleStates: {},
    activities: {},
    neuralEvents: []
  });

  // Mock Refresh
  const refresh = useCallback(async () => {
    setLoading(true);
    // Simulate fetch
    setTimeout(() => setLoading(false), 800);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);


  const InfoBlock = ({ title, value, sub, color = "white" }: any) => (
    <div className="glass-card p-4 rounded-xl">
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">{title}</p>
      <p className={`text-lg font-bold text-${color} truncate`}>{value}</p>
      {sub && <p className="text-[10px] text-slate-500">{sub}</p>}
    </div>
  );

  return (
    <div className="relative min-h-screen text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Background Elements */}
      <div className="fixed inset-0 z-[-1] overflow-hidden pointer-events-none">
        <div className="absolute inset-0 bg-grid"></div>
        <div className="absolute top-[-10%] left-[-10%] w-96 h-96 bg-primary/20 rounded-full mix-blend-screen filter blur-[100px] opacity-30 animate-blob"></div>
        <div className="absolute top-[-10%] right-[-10%] w-96 h-96 bg-accent/20 rounded-full mix-blend-screen filter blur-[100px] opacity-30 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-32 left-1/2 w-96 h-96 bg-blue-500/20 rounded-full mix-blend-screen filter blur-[100px] opacity-30 animate-blob animation-delay-4000"></div>
      </div>

      <div className="p-4 md:p-8 flex flex-col max-w-[1600px] mx-auto min-h-screen">
        {/* Top Bar */}
        <header className="flex flex-col md:flex-row justify-between items-center gap-6 mb-8 fade-in-up">
          <div className="flex items-center gap-4">
            <div className="relative w-14 h-14 bg-gradient-to-tr from-violet-600 to-indigo-600 rounded-2xl flex items-center justify-center shadow-lg shadow-violet-500/30">
              <Brain size={32} className="text-white" />
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-black rounded-full flex items-center justify-center">
                <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`}></div>
              </div>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">VIRA <span className="font-light text-violet-300">NEURAL INTERFACE</span></h1>
              <p className="text-xs text-slate-400 uppercase tracking-widest">System v2.4.0 • {isConnected ? 'Online' : 'Offline'}</p>
            </div>
          </div>

          {/* Floating Navigation */}
          <div className="glass-panel p-1.5 rounded-full flex gap-1 overflow-x-auto max-w-full no-scrollbar shadow-2xl">
            <Tab active={activeTab === 'overview'} icon={LayoutDashboard} label="Overview" onClick={() => setActiveTab('overview')} />
            <Tab active={activeTab === 'memories'} icon={Brain} label="Memory" onClick={() => setActiveTab('memories')} badge={data.memories.length} />
            <Tab active={activeTab === 'schedules'} icon={Calendar} label="Tasks" onClick={() => setActiveTab('schedules')} badge={data.schedules.length} />
            <Tab active={activeTab === 'chatlogs'} icon={MessageSquare} label="Logs" onClick={() => setActiveTab('chatlogs')} />
            <Tab active={activeTab === 'knowledge'} icon={Database} label="Graph" onClick={() => setActiveTab('knowledge')} />
            <Tab active={activeTab === 'entities'} icon={Users} label="Entities" onClick={() => setActiveTab('entities')} />
            <Tab active={activeTab === 'models'} icon={Cpu} label="Models" onClick={() => setActiveTab('models')} />
            <Tab active={activeTab === 'personas'} icon={Users} label="Persona" onClick={() => setActiveTab('personas')} />
            <Tab active={activeTab === 'terminal'} icon={TerminalIcon} label="Terminal" onClick={() => setActiveTab('terminal')} />
            <Tab active={activeTab === 'settings'} icon={Settings} label="Config" onClick={() => setActiveTab('settings')} />
          </div>

          <button onClick={refresh} className={`p-3 rounded-xl glass-panel hover:bg-white/10 transition-all ${loading ? 'animate-spin text-violet-400' : 'text-slate-400 hover:text-white'}`}>
            <RefreshCw size={20} />
          </button>
        </header>

        {/* Stats Row */}
        {activeTab === 'overview' && (
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-6 fade-in-up" style={{ animationDelay: '100ms' }}>
            <Stat icon={Brain} value={data.stats.memories || 0} label="Memory Nodes" color="violet" />
            <Stat icon={MessageSquare} value={data.chatLogs.length} label="Interactions" color="blue" />
            <Stat icon={Clock} value={data.stats.pending_schedules || 0} label="Pending Tasks" color="amber" />
            <Stat icon={Share2} value={data.stats.triples || 0} label="Knowledge Links" color="emerald" />
            <Stat icon={Cpu} value={`${((data.system?.memory_usage || 45)).toFixed(1)}%`} label="System Load" color="rose" />
          </div>
        )}

        {/* Main Content Area */}
        <div className="flex-1 glass-panel rounded-3xl p-6 md:p-8 fade-in-up shadow-2xl relative overflow-hidden min-h-[500px]" style={{ animationDelay: '200ms' }}>

          {/* Decorative background in panel */}
          <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-gradient-to-b from-violet-500/5 to-transparent rounded-full blur-3xl -z-10 pointer-events-none"></div>

          {activeTab === 'overview' && (
            <div className="grid lg:grid-cols-12 gap-8">
              <div className="lg:col-span-8 h-[500px]">
                <NeuralViz />
              </div>
              <div className="lg:col-span-4 space-y-4">
                <div className="bg-gradient-to-br from-white/5 to-white/0 rounded-2xl p-5 border border-white/5">
                  <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2"><Cpu size={16} /> System Health</h3>
                  <div className="space-y-3">
                    <InfoBlock title="Operational Status" value={data.system?.status || 'OPTIMAL'} color="emerald-400" />
                    <div className="grid grid-cols-2 gap-3">
                      <InfoBlock title="Database" value={data.system?.database || 'Mongo'} />
                      <InfoBlock title="Model" value={data.system?.api?.current_model?.split('/').pop() || 'Gemini 2.0'} />
                    </div>
                    <InfoBlock title="Active Persona" value={data.personas.find(p => p.is_active)?.name || 'Default'} color="violet-400" sub="Temperature: 0.7" />
                  </div>
                </div>

                <div className="glass-card p-5 rounded-2xl">
                  <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2"><Heart size={16} /> Emotional Core</h3>
                  <div className="flex justify-between items-end mb-2">
                    <span className="text-2xl font-bold text-white capitalize">{data.emotional.mood || 'Neutral'}</span>
                    <span className="text-xs text-slate-400 mb-1">Current State</span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-2 mb-4">
                    <div className="bg-gradient-to-r from-violet-500 to-pink-500 h-2 rounded-full" style={{ width: `${(data.emotional.satisfaction || 0.7) * 100}%` }}></div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-black/20 rounded p-2 text-center text-slate-300">Emp: {((data.emotional.empathy || 0) * 100).toFixed(0)}%</div>
                    <div className="bg-black/20 rounded p-2 text-center text-slate-300">Sts: {((data.emotional.satisfaction || 0) * 100).toFixed(0)}%</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'memories' && <MemoriesPanel />}
          {activeTab === 'schedules' && <SchedulesPanel />}
          {activeTab === 'chatlogs' && <ChatLogsPanel />}
          {activeTab === 'terminal' && <TerminalPanel />}
          {activeTab === 'knowledge' && <KnowledgeGraphPanel />}
          {activeTab === 'entities' && <EntitiesPanel />}
          {activeTab === 'personas' && <PersonasPanel />}
          {activeTab === 'settings' && <SettingsPanel />}
          {activeTab === 'models' && <div className="text-center p-10 text-slate-500">Models Panel Placeholder</div>}

        </div>

        <footer className="mt-8 text-center text-[10px] text-slate-600 font-mono tracking-widest uppercase">
          VIRA NEURAL OS • SECURE CONNECTION • LATENCY: 24ms
        </footer>
      </div>
    </div>
  );
}

export default App;
