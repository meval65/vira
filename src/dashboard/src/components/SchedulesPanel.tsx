import { useState, useEffect } from 'react';
import { api } from '../api/client';
import type { Schedule } from '../api/types';
import { Plus, Clock, CheckCircle, XCircle, RefreshCw } from 'lucide-react';
import { format } from 'date-fns';
import Modal from './Modal';

import { useWebSocket } from '../hooks/useWebSocket';

export default function SchedulesPanel() {
    const [schedules, setSchedules] = useState<Schedule[]>([]);
    const [loading, setLoading] = useState(true);
    const [showCreate, setShowCreate] = useState(false);
    const [newSchedule, setNewSchedule] = useState({
        context: '',
        date: format(new Date(), 'yyyy-MM-dd'),
        time: format(new Date(), 'HH:mm')
    });

    const { lastMessage } = useWebSocket('/ws');

    const loadSchedules = async () => {
        setLoading(true);
        try {
            const data = await api.getSchedules(true);
            setSchedules(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadSchedules();
    }, []);

    useEffect(() => {
        if (lastMessage && (lastMessage as any).type === 'schedule_update') {
            loadSchedules();
        }
    }, [lastMessage]);

    const handleCreate = async () => {
        try {
            const scheduledAt = new Date(`${newSchedule.date}T${newSchedule.time}`).toISOString();
            await api.createSchedule({
                context: newSchedule.context,
                scheduled_at: scheduledAt
            });
            setShowCreate(false);
            setNewSchedule({
                context: '',
                date: format(new Date(), 'yyyy-MM-dd'),
                time: format(new Date(), 'HH:mm')
            });
            loadSchedules();
        } catch (error) {
            console.error(error);
            // alert('Failed to create schedule');
            setShowCreate(false);
        }
    };

    const handleDelete = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!confirm('Cancel this schedule?')) return;
        try {
            await api.deleteSchedule(id);
            loadSchedules();
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div className="space-y-6 fade-in-up">
            <div className="flex justify-between items-center gap-4">
                <div>
                    <h2 className="text-xl font-bold text-white">Temporal Queue</h2>
                    <p className="text-xs text-slate-400">Scheduled tasks and automated triggers</p>
                </div>
                <div className="flex gap-2">
                    <button onClick={loadSchedules} className="btn btn-ghost hover:bg-white/10 text-slate-400 hover:text-white">
                        <RefreshCw size={18} />
                    </button>
                    <button
                        onClick={() => setShowCreate(true)}
                        className="btn btn-primary shadow-lg shadow-violet-500/20"
                    >
                        <Plus size={18} /> Add Task
                    </button>
                </div>
            </div>

            {showCreate && (
                <Modal title="Schedule New Task" onClose={() => setShowCreate(false)}>
                    <div className="space-y-4">
                        <textarea
                            className="input-neural w-full h-24 rounded-xl p-4 text-sm text-white"
                            placeholder="Task description..."
                            value={newSchedule.context}
                            onChange={(e) => setNewSchedule({ ...newSchedule, context: e.target.value })}
                        />
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Date</label>
                                <input
                                    type="date"
                                    className="input-neural w-full p-3 rounded-xl text-sm text-white"
                                    value={newSchedule.date}
                                    onChange={(e) => setNewSchedule({ ...newSchedule, date: e.target.value })}
                                />
                            </div>
                            <div>
                                <label className="text-xs font-semibold text-slate-400 uppercase mb-2 block">Time</label>
                                <input
                                    type="time"
                                    className="input-neural w-full p-3 rounded-xl text-sm text-white"
                                    value={newSchedule.time}
                                    onChange={(e) => setNewSchedule({ ...newSchedule, time: e.target.value })}
                                />
                            </div>
                        </div>

                        <div className="flex justify-end gap-2 pt-2">
                            <button onClick={() => setShowCreate(false)} className="btn btn-ghost hover:bg-white/5">Cancel</button>
                            <button onClick={handleCreate} className="btn btn-primary px-6">Add to Queue</button>
                        </div>
                    </div>
                </Modal>
            )}

            <div className="space-y-3 max-h-[60vh] overflow-y-auto scrollbar pr-2">
                {schedules.map(schedule => {
                    const scheduledDate = new Date(schedule.scheduled_at);
                    const isOverdue = scheduledDate < new Date() && schedule.status === 'pending';
                    const isExecuted = schedule.status === 'executed';

                    return (
                        <div key={schedule.id} className={`glass-card p-4 rounded-xl flex items-center justify-between group animate-fade-in-up border-l-4 ${isExecuted ? 'border-l-emerald-500' : isOverdue ? 'border-l-amber-500' : 'border-l-slate-500'}`}>
                            <div className="flex items-start gap-4">
                                <div className="flex flex-col items-center justify-center w-12 h-12 rounded-lg bg-white/5 border border-white/5">
                                    <span className="text-xs text-slate-400 font-bold uppercase">{format(scheduledDate, 'MMM')}</span>
                                    <span className="text-lg font-bold text-white">{format(scheduledDate, 'd')}</span>
                                </div>
                                <div>
                                    <div className="flex items-center gap-2 mb-1">
                                        <span className={`w-2 h-2 rounded-full ${schedule.status === 'pending' ? 'bg-amber-500 animate-pulse' : schedule.status === 'executed' ? 'bg-emerald-500' : 'bg-slate-500'}`}></span>
                                        <span className="text-xs font-mono text-slate-400">{format(scheduledDate, 'HH:mm')}</span>
                                        <span className={`text-[9px] font-bold uppercase px-1.5 py-0.5 rounded border border-white/5 ${schedule.status === 'pending' ? 'text-amber-400' : schedule.status === 'executed' ? 'text-emerald-400' : 'text-slate-400'}`}>
                                            {isOverdue ? 'OVERDUE' : schedule.status}
                                        </span>
                                    </div>
                                    <p className="text-sm font-medium text-slate-200">{schedule.context}</p>
                                </div>
                            </div>

                            <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                {schedule.status === 'pending' && (
                                    <button
                                        onClick={(e) => handleDelete(schedule.id, e)}
                                        className="p-2 hover:bg-rose-500/10 text-slate-400 hover:text-rose-400 rounded-lg transition-colors"
                                    >
                                        <XCircle size={18} />
                                    </button>
                                )}
                                {schedule.status === 'executed' && <CheckCircle size={18} className="text-emerald-500" />}
                            </div>
                        </div>
                    );
                })}

                {!loading && schedules.length === 0 && (
                    <div className="text-center py-12 border border-dashed border-white/10 rounded-xl">
                        <Clock size={40} className="mx-auto mb-2 opacity-50 text-slate-600" />
                        <p className="text-slate-500">Queue is empty.</p>
                    </div>
                )}
            </div>
        </div>
    );
}
