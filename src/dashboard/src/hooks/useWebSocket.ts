import { useEffect, useRef, useState, useCallback } from 'react';
import type { WsMessage } from '../types';

export function useWebSocket(onMessage?: (msg: WsMessage) => void): {
  connected: boolean;
  lastEvent: WsMessage | null;
} {
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<WsMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const handleMessage = useCallback((ev: MessageEvent) => {
    try {
      const msg = JSON.parse(ev.data as string) as WsMessage;
      setLastEvent(msg);
      onMessageRef.current?.(msg);
    } catch {
      // ignore non-JSON
    }
  }, []);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.onopen = () => {
        setConnected(true);
        pingRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) ws.send('ping');
        }, 25000);
      };
      ws.onclose = () => {
        setConnected(false);
        if (pingRef.current) {
          clearInterval(pingRef.current);
          pingRef.current = null;
        }
        setTimeout(() => window.location.reload(), 5000);
      };
      ws.onerror = () => setConnected(false);
      ws.onmessage = handleMessage;
    } catch {
      setConnected(false);
    }
    return () => {
      if (pingRef.current) clearInterval(pingRef.current);
      wsRef.current?.close();
    };
  }, [handleMessage]);

  return { connected, lastEvent };
}
