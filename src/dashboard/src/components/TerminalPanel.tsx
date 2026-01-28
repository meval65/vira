import { useState, useEffect, useRef } from 'react';
import { api } from '../api/client';
import { Terminal as TerminalIcon, Trash2 } from 'lucide-react';

export default function TerminalPanel() {
    const [logs, setLogs] = useState<string[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Initial fetch
        api.getLogs().then(data => {
            // Handle mock vs real data structure if api isn't perfect yet
            // @ts-ignore
            if (Array.isArray(data)) setLogs(data.map(l => typeof l === 'string' ? l : JSON.stringify(l)));
            // @ts-ignore
            else if (data?.logs) setLogs(data.logs);
        }).catch(console.error);

        // Connect to websocket for real-time logs
        const ws = new WebSocket('ws://localhost:5000/ws');

        ws.onopen = () => setIsConnected(true);
        ws.onclose = () => setIsConnected(false);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'log') {
                const logLine = `[${data.data.timestamp}] [${data.data.level}] ${data.data.message}`;
                setLogs(prev => [...prev.slice(-199), logLine]);
            }
        };

        return () => ws.close();
    }, []);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="h-[70vh] flex flex-col glass-panel rounded-2xl overflow-hidden fade-in-up">
            <div className="bg-black/40 px-4 py-2 flex justify-between items-center border-b border-white/10">
                <div className="flex items-center gap-2">
                    <TerminalIcon size={16} className="text-emerald-500" />
                    <span className="text-xs font-mono font-bold text-slate-300">SYSTEM.LOG</span>
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`}></div>
                        <span className="text-[10px] uppercase text-slate-500 font-bold">{isConnected ? 'LIVE' : 'OFFLINE'}</span>
                    </div>
                    <button onClick={() => setLogs([])} className="text-slate-500 hover:text-white transition-colors">
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 font-mono text-xs space-y-1 bg-black/80 text-emerald-500/90 leading-relaxed scrollbar">
                {logs.length === 0 && <span className="opacity-50">Initialize system monitoring...</span>}
                {logs.map((log, i) => (
                    <div key={i} className="break-all border-l-2 border-transparent hover:border-emerald-500/50 pl-2 hover:bg-white/5">
                        {log}
                    </div>
                ))}
            </div>
        </div>
    );
}
