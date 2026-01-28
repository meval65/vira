export interface Memory {
    id: string;
    summary: string;
    type: string;
    priority: number;
    confidence: number;
    created_at: string;
    last_used: string;
    use_count: number;
    status: string;
    is_compressed: boolean;
}

export interface Schedule {
    id: string;
    scheduled_at: string;
    context: string;
    priority: number;
    status: 'pending' | 'executed' | 'cancelled' | 'failed';
    created_at: string;
    executed_at?: string;
    recurrence?: string;
}

export interface Triple {
    id: string;
    subject: string;
    predicate: string;
    object: string;
    confidence: number;
    source_memory_id?: string;
    created_at: string;
}

export interface Persona {
    id: string;
    name: string;
    instruction: string;
    temperature: number;
    is_active: boolean;
    created_at: string;
}

export interface Entity {
    id: string;
    name: string;
    entity_type: string;
    mention_count: number;
    first_seen: string;
    last_seen: string;
}

export interface ChatLog {
    id: string;
    role: string;
    content: string;
    timestamp: string;
    session_id: string;
    metadata?: Record<string, any>;
}

export interface SystemStats {
    memories: number;
    triples: number;
    pending_schedules: number;
    entities: number;
    chat_logs: number;
    uncompressed_memories: number;
    personas: number;
    has_global_context: boolean;
}

export interface ModelHealth {
    health_score: number;
    success_count: number;
    failure_count: number;
    avg_latency_ms: number;
    is_available: boolean;
}

export interface ApiStatus {
    api_configured: boolean;
    failed_models: string[];
    model_health: Record<string, ModelHealth>;
    total_models: number;
}

export interface NeuralEvent {
    id?: string;
    timestamp?: string;
    source?: string;
    target?: string;
    type?: string;
    payload?: Record<string, unknown>;
    [key: string]: unknown;
}

export interface GenericConfig {
    [key: string]: unknown;
}
