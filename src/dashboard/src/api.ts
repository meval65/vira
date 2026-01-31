const API_BASE = window.location.origin;

export async function api<T = unknown>(
  endpoint: string,
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' = 'GET',
  body?: unknown
): Promise<T> {
  const options: RequestInit = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body != null) options.body = JSON.stringify(body);
  const response = await fetch(`${API_BASE}${endpoint}`, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const detail = (error as { detail?: string | { msg?: string }[] }).detail;
    const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail[0]?.msg : 'API request failed';
    throw new Error(message ?? 'API request failed');
  }
  return response.json() as Promise<T>;
}

export function formatDate(dateStr: string | undefined): string {
  if (!dateStr) return '–';
  const date = new Date(dateStr);
  if (Number.isNaN(date.getTime())) return '–';
  return date.toLocaleDateString('id-ID', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function escapeHtml(text: string): string {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
