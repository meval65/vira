import React from 'react';
import { X } from 'lucide-react';

interface ModalProps {
    title: string;
    onClose: () => void;
    children: React.ReactNode;
    wide?: boolean;
}

const Modal: React.FC<ModalProps> = ({ title, onClose, children, wide }) => (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in-up" onClick={onClose}>
        <div className={`glass-panel rounded-2xl p-0 ${wide ? 'w-full max-w-4xl' : 'w-full max-w-lg'} max-h-[90vh] overflow-hidden flex flex-col shadow-2xl border border-white/10`} onClick={e => e.stopPropagation()}>
            <div className="px-6 py-4 border-b border-white/5 flex justify-between items-center bg-white/5">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <span className="w-1 h-5 bg-accent rounded-full"></span>
                    {title}
                </h3>
                <button onClick={onClose} className="p-1 rounded-full hover:bg-white/10 text-slate-400 hover:text-white transition-colors">
                    <X size={20} />
                </button>
            </div>
            <div className="p-6 overflow-y-auto scrollbar">
                {children}
            </div>
        </div>
    </div>
);

export default Modal;
