import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface StatProps {
    icon: LucideIcon;
    value: string | number;
    label: string;
    color?: 'violet' | 'blue' | 'amber' | 'emerald' | 'rose';
}

const Stat: React.FC<StatProps> = ({ icon: Icon, value, label, color = "violet" }) => {
    const colors = {
        violet: "from-violet-500 to-purple-500",
        blue: "from-blue-500 to-cyan-500",
        amber: "from-amber-500 to-orange-500",
        emerald: "from-emerald-500 to-teal-500",
        rose: "from-rose-500 to-pink-500"
    };
    const bgGradient = colors[color] || colors.violet;

    return (
        <div className="glass-card rounded-2xl p-4 flex items-center gap-4 relative overflow-hidden group">
            <div className={`absolute inset-0 bg-gradient-to-r ${bgGradient} opacity-0 group-hover:opacity-5 transition-opacity duration-500`}></div>
            <div className={`p-3 rounded-xl bg-gradient-to-br ${bgGradient} bg-opacity-10 shadow-lg shadow-${color}-500/20`}>
                {/* If Icon is a component */}
                {typeof Icon !== 'string' ? <Icon size={24} className="text-white" /> : <i data-lucide={Icon} className="text-white w-6 h-6"></i>}
            </div>
            <div>
                <h4 className="text-2xl font-bold text-white tracking-tight">{value}</h4>
                <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</p>
            </div>
        </div>
    );
};

export default Stat;
