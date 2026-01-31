/** API response types aligned with occipital_lobe */

export interface Stats {
  memories: number;
  schedules: number;
  personas: number;
  entities: number;
  triples: number;
  chat_logs: number;
}

export interface Health {
  status: 'healthy' | 'unhealthy';
  database?: string;
  error?: string;
  timestamp?: string;
}

export interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  source: string;
}

export interface Memory {
  id: string;
  summary: string;
  memory_type?: string;
  confidence?: number;
  priority?: number;
  created_at?: string;
  status?: string;
}

export interface Schedule {
  id: string;
  context: string;
  scheduled_at?: string;
  trigger_time?: string;
  priority?: number;
  status?: string;
  created_at?: string;
}

export interface Persona {
  id: string;
  name: string;
  description?: string;
  instruction?: string;
  temperature?: number;
  voice_tone?: string;
  is_active?: boolean;
}

export interface Entity {
  id?: string;
  name: string;
  entity_type?: string;
  mention_count?: number;
  aliases?: string[];
}

export interface Triple {
  id: string;
  subject: string;
  predicate: string;
  object: string;
  confidence?: number;
}

export interface ChatLog {
  id: string;
  session_id?: string;
  role: string;
  content: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

export interface NeuralEvent {
  timestamp?: string;
  event_type?: string;
  component?: string;
  data?: Record<string, unknown>;
}

export interface OpenRouterModels {
  models: Record<string, { id: string; name: string }[]>;
  tiers: string[];
}

export interface OpenRouterHealth {
  api_configured: boolean;
  status: string;
  model_health?: Record<string, unknown>;
  active_tier?: string;
  total_requests?: number;
}

export interface AdminProfile {
  telegram_name?: string;
  additional_info?: string;
}

export interface SystemConfig {
  chat_model?: string;
  temperature?: number;
  top_p?: number;
  max_output_tokens?: number;
}

export interface GlobalContext {
  context_text: string;
  last_updated?: string;
}

export type WsMessage = { type: string; data?: Record<string, unknown>; timestamp?: string };
