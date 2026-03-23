import { useEffect, useRef, useState, useCallback } from 'react';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';
const RECONNECT_DELAY_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 10;

export function useWebSocket({ onMessage } = {}) {
    const [connectionState, setConnectionState] = useState('connecting');
    const wsRef = useRef(null);
    const attemptsRef = useRef(0);
    const reconnectTimerRef = useRef(null);
    const unmountedRef = useRef(false);

    const onMessageRef = useRef(onMessage);
    useEffect(() => {
        onMessageRef.current = onMessage;
    }, [onMessage]);

    useEffect(() => {
        unmountedRef.current = false;
        
        function connect() {
            if (unmountedRef.current) return;
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

            setConnectionState('connecting');

            try {
                const ws = new WebSocket(WS_URL);
                wsRef.current = ws;

                ws.onopen = () => {
                    if (unmountedRef.current) return;
                    attemptsRef.current = 0;
                    setConnectionState('connected');
                };

                ws.onmessage = (event) => {
                    if (unmountedRef.current) return;
                    try {
                        const data = JSON.parse(event.data);
                        if (onMessageRef.current) {
                            onMessageRef.current(data);
                        }
                    } catch {
                        // ignore malformed messages
                    }
                };

                ws.onerror = () => {
                    if (unmountedRef.current) return;
                    setConnectionState('error');
                };

                ws.onclose = () => {
                    if (unmountedRef.current) return;
                    setConnectionState('disconnected');

                    if (attemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
                        attemptsRef.current += 1;
                        const delay = RECONNECT_DELAY_MS * Math.min(attemptsRef.current, 4);
                        reconnectTimerRef.current = setTimeout(connect, delay);
                    }
                };
            } catch {
                setConnectionState('error');
            }
        }

        connect();

        return () => {
            unmountedRef.current = true;
            clearTimeout(reconnectTimerRef.current);
            if (wsRef.current) {
                wsRef.current.onclose = null; // prevent reconnect triggers
                wsRef.current.close();
            }
        };
    }, []);

    const sendMessage = useCallback((data) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(data));
        }
    }, []);

    return {
        connected: connectionState === 'connected',
        connectionState,
        sendMessage,
    };
}
