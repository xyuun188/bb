/**
 * Dashboard charts using Chart.js (loaded from CDN).
 * Handles: PnL curves, performance comparison, market price charts.
 */

if (typeof window.SimpleLineChart === 'undefined') {
    window.SimpleLineChart = class SimpleLineChart {
        constructor(ctx, config) {
            this.ctx = ctx;
            this.canvas = ctx.canvas;
            this.type = config.type;
            this.data = config.data || { labels: [], datasets: [] };
            this.options = config.options || {};
            this.update();
        }

        update() {
            const ctx = this.ctx;
            const width = this.canvas.clientWidth || this.canvas.width || 600;
            const height = this.canvas.clientHeight || this.canvas.height || 260;
            const themeColors = this.options.themeColors || {};
            const textColor = themeColors.textMuted || '#8b949e';
            const gridColor = themeColors.border || '#21262d';
            const zeroColor = themeColors.borderLight || '#30363d';
            const dpr = window.devicePixelRatio || 1;
            this.canvas.width = Math.max(1, Math.floor(width * dpr));
            this.canvas.height = Math.max(1, Math.floor(height * dpr));
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, width, height);

            const datasets = (this.data.datasets || [])
                .map(ds => ({ ...ds, data: (ds.data || []).map(Number).filter(Number.isFinite) }))
                .filter(ds => ds.data.length);
            const pad = { left: 52, right: 18, top: 18, bottom: 34 };
            const plotW = Math.max(1, width - pad.left - pad.right);
            const plotH = Math.max(1, height - pad.top - pad.bottom);

            if (!datasets.length) {
                ctx.font = '12px system-ui, -apple-system, Segoe UI, sans-serif';
                ctx.fillStyle = textColor;
                ctx.fillText('暂无收益曲线数据', pad.left, pad.top + 18);
                return;
            }

            const values = datasets.flatMap(ds => ds.data);
            let min = Math.min(...values, 0);
            let max = Math.max(...values, 0);
            const yOptions = this.options?.scales?.y || {};
            if (Number.isFinite(yOptions.suggestedMin)) min = Math.min(min, yOptions.suggestedMin);
            if (Number.isFinite(yOptions.suggestedMax)) max = Math.max(max, yOptions.suggestedMax);
            if (min === max) {
                const span = Math.max(Math.abs(min) * 0.2, 0.05);
                min -= span;
                max += span;
            }

            const yOf = value => pad.top + (max - value) / (max - min) * plotH;

            ctx.font = '11px system-ui, -apple-system, Segoe UI, sans-serif';
            ctx.lineWidth = 1;
            ctx.strokeStyle = gridColor;
            ctx.fillStyle = textColor;
            for (let i = 0; i <= 4; i++) {
                const y = pad.top + (plotH / 4) * i;
                const value = max - ((max - min) / 4) * i;
                ctx.beginPath();
                ctx.moveTo(pad.left, y);
                ctx.lineTo(width - pad.right, y);
                ctx.stroke();
                ctx.fillText(this.options?.formatTick ? this.options.formatTick(value) : value.toFixed(Math.abs(max - min) < 10 ? 4 : 1), 6, y + 4);
            }
            if (min < 0 && max > 0) {
                const y = yOf(0);
                ctx.strokeStyle = zeroColor;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(pad.left, y);
                ctx.lineTo(width - pad.right, y);
                ctx.stroke();
            }

            const labels = this.data.labels || [];
            const ticks = Math.min(5, labels.length);
            for (let i = 0; i < ticks; i++) {
                const idx = ticks === 1 ? 0 : Math.round(i * (labels.length - 1) / (ticks - 1));
                const x = pad.left + (plotW * idx / Math.max(labels.length - 1, 1));
                const label = String(labels[idx] || '');
                ctx.fillText(label.slice(0, 8), Math.min(x, width - 62), height - 10);
            }

            datasets.forEach(ds => {
                const points = ds.data;
                ctx.strokeStyle = ds.borderColor || '#58a6ff';
                ctx.lineWidth = ds.borderWidth || 2;
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                ctx.beginPath();
                if (points.length === 1) {
                    const y = yOf(points[0]);
                    ctx.moveTo(pad.left, y);
                    ctx.lineTo(width - pad.right, y);
                } else {
                    points.forEach((value, idx) => {
                        const x = pad.left + (plotW * idx / Math.max(points.length - 1, 1));
                        const y = yOf(value);
                        if (idx === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);
                    });
                }
                ctx.stroke();

                ctx.fillStyle = ds.borderColor || '#58a6ff';
                points.forEach((value, idx) => {
                    const x = points.length === 1
                        ? pad.left + plotW / 2
                        : pad.left + (plotW * idx / Math.max(points.length - 1, 1));
                    const y = yOf(value);
                    ctx.beginPath();
                    ctx.arc(x, y, points.length <= 20 ? 2.8 : 1.8, 0, Math.PI * 2);
                    ctx.fill();
                });
            });
        }
    };
}
if (typeof window.Chart === 'undefined') {
    window.Chart = window.SimpleLineChart;
}

class DashboardCharts {
    constructor() {
        this.charts = {};
    }

    getThemePalette() {
        const styles = getComputedStyle(document.documentElement);
        const read = (name, fallback) => styles.getPropertyValue(name).trim() || fallback;
        return {
            textMuted: read('--text-muted', '#8b949e'),
            border: read('--border', '#21262d'),
            borderLight: read('--border-light', '#30363d'),
            blue: read('--blue', '#58a6ff'),
            blueDim: read('--blue-dim', 'rgba(88, 166, 255, 0.1)'),
        };
    }

    applyTheme() {
        const palette = this.getThemePalette();
        const pnlChart = this.charts.pnl;
        if (pnlChart) {
            pnlChart.options.themeColors = palette;
            pnlChart.update();
        }

        const priceChart = this.charts.price;
        if (!priceChart) return;

        if (priceChart.options?.scales?.x) {
            priceChart.options.scales.x.ticks.color = palette.textMuted;
            priceChart.options.scales.x.grid.color = palette.border;
        }
        if (priceChart.options?.scales?.y) {
            priceChart.options.scales.y.ticks.color = palette.textMuted;
            priceChart.options.scales.y.grid.color = palette.border;
        }
        if (priceChart.data?.datasets?.[0]) {
            priceChart.data.datasets[0].borderColor = palette.blue;
            priceChart.data.datasets[0].backgroundColor = palette.blueDim;
        }
        priceChart.update('none');
    }

    formatTimeLabel(value) {
        if (!value) return '';
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return String(value);
        return date.toLocaleString('zh-CN', {
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
        });
    }

    formatClockLabel(value) {
        if (!value) return '';
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return String(value);
        return date.toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false,
        });
    }

    formatPercentValue(value) {
        const n = Number(value);
        if (!Number.isFinite(n)) return '0.0000%';
        const abs = Math.abs(n);
        const digits = abs < 0.1 ? 4 : abs < 1 ? 3 : 2;
        return `${n.toFixed(digits)}%`;
    }

    setBalancedYAxis(chart, values, padding = 1) {
        const nums = values.map(Number).filter(Number.isFinite);
        if (!nums.length || !chart.options.scales || !chart.options.scales.y) return;

        const min = Math.min(...nums);
        const max = Math.max(...nums);
        if (min === max) {
            const span = Math.max(Math.abs(min) * 0.2, padding);
            chart.options.scales.y.suggestedMin = min - span;
            chart.options.scales.y.suggestedMax = max + span;
        } else {
            const range = max - min;
            const pad = Math.max(range * 0.2, padding);
            chart.options.scales.y.suggestedMin = min - pad;
            chart.options.scales.y.suggestedMax = max + pad;
        }
    }

    initPnLChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.charts.pnl = new window.SimpleLineChart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => `${ctx.dataset.label}: ${this.formatPercentValue(ctx.parsed.y)}`,
                        },
                    },
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#8b949e',
                            autoSkip: true,
                            maxTicksLimit: 8,
                            maxRotation: 0,
                        },
                        grid: { color: '#21262d' },
                    },
                    y: {
                        ticks: {
                            color: '#8b949e',
                            callback: v => this.formatPercentValue(v),
                        },
                        grid: { color: '#21262d' },
                    },
                },
            },
        });
        this.charts.pnl.options.formatTick = value => this.formatPercentValue(value);
    }

    updatePnLChart(modelHistory) {
        const chart = this.charts.pnl;
        if (!chart) return;
        chart.options.plugins = chart.options.plugins || {};
        chart.options.plugins.legend = { display: false };

        const colors = ['#58a6ff', '#3fb950', '#d2991d', '#f85149', '#db6d28'];
        const entries = Object.entries(modelHistory || {})
            .map(([name, data]) => {
                const rawValues = Array.isArray(data?.pnl_curve) ? data.pnl_curve : [];
                const rawLabels = Array.isArray(data?.labels) ? data.labels : [];
                const points = rawValues
                    .map((value, idx) => ({
                        value: Number(value),
                        label: rawLabels[idx] ?? idx,
                    }))
                    .filter(point => Number.isFinite(point.value));
                return [name, points];
            })
            .filter(([, points]) => points.length > 0);
        if (entries.length === 0) {
            chart.data.labels = [];
            chart.data.datasets = [];
            chart.update();
            return;
        }

        let sharedLabels = null;
        chart.data.datasets = entries.map(([name, points], i) => {
            if (!sharedLabels || points.length > sharedLabels.length) {
                sharedLabels = points.map(point => point.label);
            }
            return {
                label: name,
                data: points.map(point => point.value),
                borderColor: colors[i % colors.length],
                backgroundColor: 'transparent',
                tension: 0.3,
                pointRadius: points.length <= 1 ? 3 : 2,
                pointHoverRadius: 4,
                borderWidth: 2,
                fill: false,
                spanGaps: true,
            };
        });

        const labels = sharedLabels || (
            chart.data.datasets.length > 0
                ? Array.from({ length: chart.data.datasets[0].data.length }, (_, i) => i)
                : []
        );
        chart.data.labels = labels.map(label => (
            typeof label === 'number' ? label : this.formatClockLabel(label)
        ));
        this.setBalancedYAxis(chart, chart.data.datasets.flatMap(ds => ds.data), 0.005);
        chart.update();
    }

    initPriceChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.charts.price = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '价格',
                    data: [],
                    borderColor: '#58a6ff',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    fill: true,
                    tension: 0.2,
                    pointRadius: 0,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        ticks: {
                            color: '#8b949e',
                            autoSkip: true,
                            maxTicksLimit: 8,
                            maxRotation: 0,
                        },
                        grid: { color: '#21262d' },
                    },
                    y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
                },
            },
        });
    }

    updatePriceChart(klines) {
        const chart = this.charts.price;
        if (!chart || !klines) return;

        const points = klines
            .filter(k => k && k.time && Number.isFinite(Number(k.close)))
            .map(k => ({ label: this.formatTimeLabel(k.time), close: Number(k.close) }));

        chart.data.labels = points.map(p => p.label);
        chart.data.datasets[0].data = points.map(p => p.close);
        this.setBalancedYAxis(chart, chart.data.datasets[0].data, 0.5);
        chart.update();
    }
}
