import { useEffect, useRef, useState, useCallback } from 'react';

export const useWebSocket = (url: string) => {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState<unknown>(null);
    const ws = useRef<WebSocket | null>(null);

    useEffect(() => {
        let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

        const connect = () => {
            const socketUrl = url.startsWith('/')
                ? `ws://${window.location.host}${url}`
                : url;

            const socket = new WebSocket(socketUrl);
            ws.current = socket;

            socket.onopen = () => {
                console.log('WS Connected');
                setIsConnected(true);
            };

            socket.onclose = () => {
                console.log('WS Disconnected');
                setIsConnected(false);
                reconnectTimeout = setTimeout(connect, 3000);
            };

            socket.onerror = (error: Event) => {
                console.error('WS Error:', error);
                setIsConnected(false);
            };

            socket.onmessage = (event: MessageEvent) => {
                try {
                    const data = JSON.parse(event.data);
                    setLastMessage(data);
                } catch (e) {
                    console.error('Failed to parse WS message', e);
                }
            };
        };

        connect();

        return () => {
            if (ws.current) ws.current.close();
            if (reconnectTimeout) clearTimeout(reconnectTimeout);
        };
    }, [url]);

    const sendMessage = useCallback((msg: unknown) => {
        if (ws.current?.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify(msg));
        }
    }, []);

    return { isConnected, lastMessage, sendMessage };
};
