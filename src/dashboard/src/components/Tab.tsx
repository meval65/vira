import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface TabProps {
    active: boolean;
    icon: LucideIcon;
    label: string;
    onClick: () => void;
    badge?: number;
}

const Tab: React.FC<TabProps> = ({ active, icon: Icon, label, onClick, badge }) => (
    <button
        onClick={onClick}
        className={`relative flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-300 shrink-0 ${active
            ? 'text-white shadow-[0_0_20px_rgba(139,92,246,0.3)]'
            : 'text-slate-400 hover:text-white hover:bg-white/5'
            }`}
    >
        {active && (
            <span className="absolute inset-0 bg-gradient-to-r from-violet-600 to-indigo-600 rounded-full -z-10 animate-fade-in-up"></span>
        )}
        {/* Handle both component and string icons if needed, but prefer Component for React */}
        {typeof Icon !== 'string' ? <Icon size={16} /> : <i data-lucide={Icon} className="w-4 h-4"></i>}
        <span>{label}</span>
        {badge !== undefined && badge > 0 && (
            <span className={`ml-1 px-1.5 py-0.5 text-[10px] font-bold rounded-full ${active ? 'bg-white text-violet-600' : 'bg-white/10 text-slate-300'}`}>
                {badge}
            </span>
        )}
    </button>
);

export default Tab;
