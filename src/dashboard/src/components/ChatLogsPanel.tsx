import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '../api/client';
import type { ChatLog } from '../api/types';
import { MessageSquare, User, Bot, Trash2, Search } from 'lucide-react';
import { format } from 'date-fns';

export default function ChatLogsPanel() {
    const [logs, setLogs] = useState<ChatLog[]>([]);
    const [loading, setLoading] = useState(true);
    const [sessions, setSessions] = useState<string[]>([]);
    const [selectedSession, setSelectedSession] = useState<string>('');
    const scrollRef = useRef<HTMLDivElement>(null);

    // Mock Sessions if API fails
    const mockSessions = ['session-2024-05-15', 'session-2024-05-14', 'debug-trace-001'];

    const loadSessions = useCallback(async () => {
        try {
            // const data = await api.getSessions();
            // setSessions(data);
            // MOCK
            setSessions(mockSessions);
            if (mockSessions.length > 0 && !selectedSession) {
                setSelectedSession(mockSessions[0]);
            }
        } catch (error) {
            console.error(error);
        }
    }, [selectedSession]);

    const loadLogs = useCallback(async () => {
        if (!selectedSession) return;
        setLoading(true);
        try {
            // const data = await api.getChatLogs(100, selectedSession);
            // setLogs(data.reverse());
            // MOCK DATA
            const mockLogs: ChatLog[] = [
                { id: '1', role: 'user', content: 'What is the current system status?', timestamp: new Date(Date.now() - 100000).toISOString(), session_id: selectedSession },
                { id: '2', role: 'assistant', content: 'All systems operational. CPU load at 45%. Memory usage normal.', timestamp: new Date(Date.now() - 90000).toISOString(), session_id: selectedSession },
                { id: '3', role: 'user', content: 'check memory banks', timestamp: new Date(Date.now() - 80000).toISOString(), session_id: selectedSession },
                { id: '4', role: 'assistant', content: 'Memory banks scanned. 156 nodes active. No corruption detected.', timestamp: new Date(Date.now() - 70000).toISOString(), session_id: selectedSession }
            ];
            setLogs(mockLogs);

            setTimeout(() => {
                if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
            }, 100);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }, [selectedSession]);

    useEffect(() => {
        loadSessions();
    }, [loadSessions]);

    useEffect(() => {
        loadLogs();
    }, [loadLogs]);

    const handleDeleteSession = async (sessionId: string) => {
        if (!confirm('Delete this chat session history?')) return;
        try {
            await api.deleteSession(sessionId);
            loadSessions();
            if (selectedSession === sessionId) setSelectedSession('');
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-250px)] fade-in-up">
            {/* Session List */}
            <div className="lg:col-span-1 glass-panel flex flex-col overflow-hidden">
                <div className="p-4 border-b border-white/5 font-bold text-white flex justify-between items-center">
                    <span>Recent Sessions</span>
                    <span className="text-xs bg-white/10 px-2 py-1 rounded-full text-slate-300">{sessions.length}</span>
                </div>
                <div className="overflow-y-auto flex-1 p-2 space-y-1 scrollbar">
                    {sessions.map(session => (
                        <div
                            key={session}
                            onClick={() => setSelectedSession(session)}
                            className={`p-3 rounded-xl cursor-pointer flex justify-between items-center group transition-all duration-200 ${selectedSession === session
                                ? 'bg-gradient-to-r from-violet-600/50 to-indigo-600/50 text-white border border-violet-500/30'
                                : 'hover:bg-white/5 text-slate-400 hover:text-white border border-transparent'
                                }`}
                        >
                            <div className="truncate text-xs font-mono flex-1">
                                {session}
                            </div>
                            <button
                                onClick={(e) => { e.stopPropagation(); handleDeleteSession(session); }}
                                className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-rose-500/20 rounded-lg text-rose-400 transition-all"
                            >
                                <Trash2 size={12} />
                            </button>
                        </div>
                    ))}
                </div>
            </div>

            {/* Chat Area */}
            <div className="lg:col-span-3 glass-panel flex flex-col overflow-hidden relative">
                <div className="p-4 border-b border-white/5 flex justify-between items-center bg-black/20 backdrop-blur-md z-10">
                    <h3 className="font-bold text-slate-200 flex items-center gap-2">
                        <MessageSquare size={16} className="text-violet-400" />
                        {loading ? 'Accessing Archives...' : selectedSession || 'Select Session'}
                    </h3>
                    <div className="flex gap-2">
                        <div className="relative">
                            <input placeholder="Search logs..." className="bg-black/20 border border-white/10 rounded-full px-3 py-1 text-xs text-white w-48 focus:w-64 transition-all outline-none" />
                            <Search size={12} className="absolute right-3 top-1.5 text-slate-500" />
                        </div>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar bg-black/10" ref={scrollRef}>
                    {logs.map((log) => (
                        <div key={log.id} className={`flex gap-4 ${log.role === 'user' ? 'flex-row-reverse' : ''} animate-fade-in-up`}>
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 shadow-lg ${log.role === 'user' ? 'bg-gradient-to-br from-blue-500 to-cyan-500 text-white' : 'bg-gradient-to-br from-emerald-500 to-teal-500 text-white'
                                }`}>
                                {log.role === 'user' ? <User size={14} /> : <Bot size={14} />}
                            </div>

                            <div className={`max-w-[80%] rounded-2xl p-4 shadow-xl backdrop-blur-sm border ${log.role === 'user'
                                ? 'bg-blue-500/10 border-blue-500/20 text-blue-100 rounded-tr-none'
                                : 'bg-slate-800/60 border-white/10 text-slate-200 rounded-tl-none'
                                }`}>
                                <div className="text-sm whitespace-pre-wrap leading-relaxed">
                                    {log.content}
                                </div>
                                <div className={`text-[10px] mt-2 opacity-50 font-mono ${log.role === 'user' ? 'text-right' : ''}`}>
                                    {format(new Date(log.timestamp), 'h:mm a')}
                                </div>
                            </div>
                        </div>
                    ))}

                    {logs.length === 0 && !loading && (
                        <div className="h-full flex flex-col items-center justify-center text-slate-500">
                            <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center mb-4">
                                <MessageSquare size={32} className="opacity-30" />
                            </div>
                            <p>Select a session to view logs</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
