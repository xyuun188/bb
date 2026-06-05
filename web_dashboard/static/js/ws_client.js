/**
 * WebSocket client for real-time dashboard updates.
 */
class WSClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.reconnectDelay = 2000;
        this.maxReconnectDelay = 30000;
        this.handlers = {};
        this.connected = false;
        this.pingInterval = null;
    }

    connect() {
        try {
            this.ws = new WebSocket(this.url);
        } catch (e) {
            console.error('WS connect error:', e);
            this.scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            console.log('WS connected');
            this.connected = true;
            this.reconnectDelay = 2000;
            this._startPing();
            this._emit('connected');
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this._emit('message', data);
                if (data.type) {
                    this._emit(data.type, data);
                }
            } catch (e) {
                console.error('WS parse error:', e);
            }
        };

        this.ws.onclose = () => {
            console.log('WS disconnected');
            this.connected = false;
            this._stopPing();
            this._emit('disconnected');
            this.scheduleReconnect();
        };

        this.ws.onerror = (err) => {
            console.error('WS error:', err);
        };
    }

    on(event, handler) {
        if (!this.handlers[event]) {
            this.handlers[event] = [];
        }
        this.handlers[event].push(handler);
        return () => {
            this.handlers[event] = this.handlers[event].filter(h => h !== handler);
        };
    }

    send(data) {
        if (this.ws && this.connected) {
            this.ws.send(JSON.stringify(data));
        }
    }

    scheduleReconnect() {
        setTimeout(() => {
            if (!this.connected) {
                console.log('WS reconnecting...');
                this.connect();
                this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnectDelay);
            }
        }, this.reconnectDelay);
    }

    _emit(event, data) {
        (this.handlers[event] || []).forEach(h => h(data));
    }

    _startPing() {
        this._stopPing();
        this.pingInterval = setInterval(() => this.send({ type: 'ping' }), 30000);
    }

    _stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
}
