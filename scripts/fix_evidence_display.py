#!/usr/bin/env python3
"""修复盈利归因证据显示问题 - 隐藏非关键的证据缺失提示"""

import sys
from pathlib import Path

dashboard_js = Path("E:/code/bb/web_dashboard/static/js/dashboard.js")

if not dashboard_js.exists():
    print(f"文件不存在: {dashboard_js}")
    sys.exit(1)

content = dashboard_js.read_text(encoding='utf-8')

# 要替换的代码段
old_code = """    if (!available) {
        const reason = sourceStatus.missing_reason || `${label}证据未匹配`;
        const missingLabel = profitAttributionMissingLabel(reason);
        return {
            tone: 'missing',
            text: `${label} ${reason}`,
            html: `<span class="profit-attribution-evidence-chip${typeClass} missing" title="${escHtml(reason)}"><b>${escHtml(label)}</b><em>${escHtml(missingLabel)}</em></span>`,
        };
    }"""

new_code = """    if (!available) {
        const reason = sourceStatus.missing_reason || `${label}证据未匹配`;
        // 优化：非关键证据缺失时静默隐藏，减少视觉噪音
        const isCritical = reason.includes('未匹配到开仓 AI 决策');
        if (!isCritical) {
            return { tone: 'hidden', text: '', html: '' };
        }
        // 关键证据缺失，显示灰色低调提示
        const missingLabel = profitAttributionMissingLabel(reason);
        return {
            tone: 'info',
            text: `${label} ${reason}`,
            html: `<span class="profit-attribution-evidence-chip${typeClass} info" title="${escHtml(reason)}" style="opacity: 0.5;"><b>${escHtml(label)}</b><em style="color: var(--text-muted);">${escHtml(missingLabel)}</em></span>`,
        };
    }"""

if old_code in content:
    content = content.replace(old_code, new_code)
    dashboard_js.write_text(content, encoding='utf-8')
    print("✓ 修改成功：非关键证据缺失将不再显示")
    print("  - ML信号、盈利预测、时序、情绪等证据缺失时会静默隐藏")
    print("  - 只有未匹配到AI决策这种关键问题才会显示灰色提示")
else:
    print("✗ 未找到匹配的代码段，可能已经被修改过")
    sys.exit(1)
