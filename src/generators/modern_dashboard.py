"""Modern dashboard generator for Claude Conversation Analyzer."""

import json
from pathlib import Path
from typing import Dict, Any
import difflib
import html


def _generate_claude_md_diff_section(analysis_results: Dict[str, Any]) -> str:
    """Generate CLAUDE.md diff section if both original and proposed content are available."""
    original = analysis_results.get('original_claude_md_content', '')
    proposed = analysis_results.get('proposed_claude_md_content', '')
    
    if not original or not proposed:
        return ""
    
    # Split content into lines for diffing
    original_lines = original.splitlines(keepends=True)
    proposed_lines = proposed.splitlines(keepends=True)
    
    # Generate unified diff
    diff = difflib.unified_diff(
        original_lines,
        proposed_lines,
        fromfile='Original CLAUDE.md',
        tofile='Proposed CLAUDE.md',
        lineterm=''
    )
    
    # Convert diff to HTML with syntax highlighting
    diff_html = []
    for line in diff:
        escaped_line = html.escape(line)
        if line.startswith('+++') or line.startswith('---'):
            diff_html.append(f'<div class="diff-header">{escaped_line}</div>')
        elif line.startswith('@@'):
            diff_html.append(f'<div class="diff-hunk">{escaped_line}</div>')
        elif line.startswith('+'):
            diff_html.append(f'<div class="diff-add">{escaped_line}</div>')
        elif line.startswith('-'):
            diff_html.append(f'<div class="diff-remove">{escaped_line}</div>')
        else:
            diff_html.append(f'<div class="diff-context">{escaped_line}</div>')
    
    # Generate side-by-side view
    side_by_side_html = _generate_side_by_side_diff(original_lines, proposed_lines)
    
    return f"""
        <h2>CLAUDE.md Changes</h2>
        
        <div class="diff-view-toggle">
            <button class="toggle-btn active" onclick="showDiffView('unified')">Unified Diff</button>
            <button class="toggle-btn" onclick="showDiffView('side-by-side')">Side-by-Side</button>
        </div>
        
        <div id="unified-diff" class="diff-container">
            <div class="diff-content">
                {''.join(diff_html) if diff_html else '<p class="no-changes">No differences found</p>'}
            </div>
        </div>
        
        <div id="side-by-side-diff" class="diff-container" style="display: none;">
            {side_by_side_html}
        </div>
        
        <script>
            function showDiffView(view) {{
                document.querySelectorAll('.diff-container').forEach(el => el.style.display = 'none');
                document.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
                
                if (view === 'unified') {{
                    document.getElementById('unified-diff').style.display = 'block';
                    document.querySelector('[onclick="showDiffView(\'unified\')"]').classList.add('active');
                }} else {{
                    document.getElementById('side-by-side-diff').style.display = 'block';
                    document.querySelector('[onclick="showDiffView(\'side-by-side\')"]').classList.add('active');
                }}
            }}
        </script>
    """


def _generate_side_by_side_diff(original_lines, proposed_lines):
    """Generate side-by-side diff view."""
    differ = difflib.Differ()
    diff = list(differ.compare(original_lines, proposed_lines))
    
    left_lines = []
    right_lines = []
    
    i = 0
    while i < len(diff):
        line = diff[i]
        if line.startswith('  '):  # Common line
            left_lines.append((line[2:], 'common'))
            right_lines.append((line[2:], 'common'))
        elif line.startswith('- '):  # Removed line
            # Check if next line is an addition (changed line)
            if i + 1 < len(diff) and diff[i + 1].startswith('+ '):
                left_lines.append((line[2:], 'changed'))
                right_lines.append((diff[i + 1][2:], 'changed'))
                i += 1  # Skip the addition line
            else:
                left_lines.append((line[2:], 'removed'))
                right_lines.append(('', 'empty'))
        elif line.startswith('+ '):  # Added line
            left_lines.append(('', 'empty'))
            right_lines.append((line[2:], 'added'))
        i += 1
    
    # Generate HTML table
    html_rows = []
    for idx, (left, right) in enumerate(zip(left_lines, right_lines), 1):
        left_text, left_class = left
        right_text, right_class = right
        
        html_rows.append(f'''
            <tr>
                <td class="line-number">{idx if left_text else ''}</td>
                <td class="diff-{left_class}">{html.escape(left_text)}</td>
                <td class="line-number">{idx if right_text else ''}</td>
                <td class="diff-{right_class}">{html.escape(right_text)}</td>
            </tr>
        ''')
    
    return f'''
        <table class="side-by-side-diff">
            <thead>
                <tr>
                    <th colspan="2">Original CLAUDE.md</th>
                    <th colspan="2">Proposed CLAUDE.md</th>
                </tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    '''


def _generate_performance_metrics_section(analysis_results: Dict[str, Any]) -> str:
    """Generate performance metrics section if metrics are available."""
    if 'metrics' not in analysis_results:
        return ""
    
    metrics = analysis_results['metrics']
    costs = analysis_results.get('costs', {})
    
    # Generate phase metrics cards
    phase_cards = ""
    for phase_name, phase_data in metrics.get('phase_metrics', {}).items():
        phase_display = phase_name.replace('_', ' ').title()
        phase_cards += f"""
            <div class="metric-card">
                <div class="metric-label">{phase_display}</div>
                <div class="metric-value" style="font-size: 1.5rem;">
                    {phase_data['total_tokens']:,} tokens
                    <div style="font-size: 0.875rem; color: #6b7280; margin-top: 0.5rem;">
                        {phase_data['call_count']} calls • {phase_data['avg_response_time']:.2f}s avg
                    </div>
                </div>
            </div>
        """
    
    return f"""
        <h2>Performance & Cost Metrics</h2>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Duration</div>
                <div class="metric-value">{metrics['run_duration_formatted']}</div>
            </div>
            <div class="metric-card {'success' if costs.get('total_cost', 0) < 2.0 else 'warning' if costs.get('total_cost', 0) < 5.0 else 'error'}">
                <div class="metric-label">Total Cost</div>
                <div class="metric-value">${costs.get('total_cost', 0):.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Processing Rate</div>
                <div class="metric-value" style="font-size: 2rem;">{metrics['conversations_per_minute']:.1f}</div>
                <div style="font-size: 0.875rem; color: #6b7280;">convs/min</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Tokens</div>
                <div class="metric-value">{metrics['total_tokens']:,}</div>
            </div>
        </div>
        
        <h3 style="margin-top: 2rem;">Phase Breakdown</h3>
        <div class="metrics-grid">
            {phase_cards}
        </div>
        
        <!-- Token Usage Chart -->
        <div class="chart-container" style="margin-top: 2rem;">
            <h3 class="chart-title">Token Usage by Phase</h3>
            <canvas id="tokenChart"></canvas>
        </div>
        
        <script>
        // Token usage chart
        const tokenData = {json.dumps({phase: data['total_tokens'] for phase, data in metrics.get('phase_metrics', {}).items()})};
        new Chart(document.getElementById('tokenChart'), {{
            type: 'bar',
            data: {{
                labels: Object.keys(tokenData).map(k => k.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())),
                datasets: [{{
                    label: 'Tokens',
                    data: Object.values(tokenData),
                    backgroundColor: '#8b5cf6',
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            callback: function(value) {{
                                return value.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});
        </script>
    """


def generate_modern_dashboard(output_dir: Path, analysis_results: Dict[str, Any]) -> Path:
    """Generate a modern interactive HTML dashboard."""
    stats = analysis_results.get('summary_statistics', {})
    
    # Generate HTML with modern styling
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Claude Conversation Analysis Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #f8f9fa;
            color: #2c3e50;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 2rem;
            color: #1a202c;
            font-weight: 700;
        }}
        
        h2 {{
            font-size: 1.75rem;
            margin: 2rem 0 1rem;
            color: #2d3748;
            font-weight: 600;
        }}
        
        /* Metrics Cards */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }}
        
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}
        
        .metric-label {{
            font-size: 0.875rem;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #2563eb;
        }}
        
        .metric-card.success .metric-value {{
            color: #10b981;
        }}
        
        .metric-card.warning .metric-value {{
            color: #f59e0b;
        }}
        
        .metric-card.error .metric-value {{
            color: #ef4444;
        }}
        
        /* Charts Section */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }}
        
        .chart-container {{
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        
        .chart-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #2d3748;
        }}
        
        canvas {{
            max-height: 300px;
        }}
        
        /* Insights Section */
        .insights-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }}
        
        .insight-card {{
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        
        .insight-card h3 {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #2d3748;
        }}
        
        .insight-list {{
            list-style: none;
        }}
        
        .insight-item {{
            padding: 0.75rem 0;
            border-bottom: 1px solid #e5e7eb;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .insight-item:last-child {{
            border-bottom: none;
        }}
        
        .insight-text {{
            flex: 1;
            margin-right: 1rem;
            color: #4a5568;
        }}
        
        .insight-count {{
            background: #e0e7ff;
            color: #4338ca;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            min-width: 2rem;
            text-align: center;
        }}
        
        /* Recommendations */
        .recommendations {{
            background: #fef3c7;
            border: 1px solid #fbbf24;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        
        .recommendations h3 {{
            color: #92400e;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .recommendations ul {{
            list-style: none;
            color: #78350f;
        }}
        
        .recommendations li {{
            padding: 0.5rem 0;
            padding-left: 1.5rem;
            position: relative;
        }}
        
        .recommendations li:before {{
            content: "→";
            position: absolute;
            left: 0;
            font-weight: bold;
        }}
        
        /* Success Patterns */
        .success-patterns {{
            background: #d1fae5;
            border: 1px solid #34d399;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        
        .success-patterns h3 {{
            color: #064e3b;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .success-patterns ul {{
            list-style: none;
            color: #065f46;
        }}
        
        .success-patterns li {{
            padding: 0.5rem 0;
            padding-left: 1.5rem;
            position: relative;
        }}
        
        .success-patterns li:before {{
            content: "✓";
            position: absolute;
            left: 0;
            font-weight: bold;
        }}
        
        /* Footer */
        .footer {{
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid #e5e7eb;
            text-align: center;
            color: #6b7280;
            font-size: 0.875rem;
        }}
        /* Diff View Styles */
        .diff-view-toggle {{
            margin-bottom: 1rem;
            display: flex;
            gap: 0.5rem;
        }}
        
        .toggle-btn {{
            padding: 0.5rem 1rem;
            border: 1px solid #e5e7eb;
            background: white;
            color: #4b5563;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
        }}
        
        .toggle-btn:hover {{
            background: #f3f4f6;
        }}
        
        .toggle-btn.active {{
            background: #2563eb;
            color: white;
            border-color: #2563eb;
        }}
        
        .diff-container {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            overflow: hidden;
            margin-bottom: 2rem;
        }}
        
        .diff-content {{
            padding: 1rem;
            font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
            font-size: 0.875rem;
            overflow-x: auto;
        }}
        
        .diff-header {{
            color: #6b7280;
            background: #f9fafb;
            padding: 0.5rem;
            margin: -0.5rem -0.5rem 0.5rem -0.5rem;
        }}
        
        .diff-hunk {{
            color: #8b5cf6;
            background: #f3f4f6;
            padding: 0.25rem 0.5rem;
            margin: 0.5rem -0.5rem;
        }}
        
        .diff-add {{
            background: #d1fae5;
            color: #065f46;
            padding: 0.125rem 0.5rem;
        }}
        
        .diff-remove {{
            background: #fee2e2;
            color: #991b1b;
            padding: 0.125rem 0.5rem;
        }}
        
        .diff-context {{
            color: #4b5563;
            padding: 0.125rem 0.5rem;
        }}
        
        /* Side-by-side diff */
        .side-by-side-diff {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'Monaco', 'Consolas', 'Courier New', monospace;
            font-size: 0.875rem;
        }}
        
        .side-by-side-diff thead th {{
            background: #f3f4f6;
            padding: 1rem;
            font-weight: 600;
            text-align: left;
            border-bottom: 2px solid #e5e7eb;
        }}
        
        .side-by-side-diff td {{
            padding: 0.25rem 0.5rem;
            border-bottom: 1px solid #f3f4f6;
            vertical-align: top;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        
        .side-by-side-diff .line-number {{
            background: #f9fafb;
            color: #9ca3af;
            text-align: right;
            width: 50px;
            padding-right: 1rem;
            user-select: none;
        }}
        
        .side-by-side-diff .diff-common {{
            background: white;
        }}
        
        .side-by-side-diff .diff-changed {{
            background: #fef3c7;
        }}
        
        .side-by-side-diff .diff-removed {{
            background: #fee2e2;
        }}
        
        .side-by-side-diff .diff-added {{
            background: #d1fae5;
        }}
        
        .side-by-side-diff .diff-empty {{
            background: #f9fafb;
        }}
        
        .no-changes {{
            text-align: center;
            color: #6b7280;
            padding: 2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Claude Conversation Analysis Dashboard</h1>
        
        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Conversations</div>
                <div class="metric-value">{stats.get('total_conversations', 0)}</div>
            </div>
            <div class="metric-card {'success' if stats.get('success_rate', 0) > 0.7 else 'warning' if stats.get('success_rate', 0) > 0.4 else 'error'}">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value">{stats.get('success_rate', 0) * 100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Interventions</div>
                <div class="metric-value">{stats.get('total_interventions', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Conversations with Issues</div>
                <div class="metric-value">{stats.get('conversations_with_interventions', 0)}</div>
            </div>
        </div>
        
        <!-- Performance Metrics -->
        {_generate_performance_metrics_section(analysis_results)}
        
        <!-- CLAUDE.md Diff View -->
        {_generate_claude_md_diff_section(analysis_results)}
        
        <!-- Top Recommendations -->
        <div class="recommendations">
            <h3>💡 Top Recommendations for CLAUDE.md</h3>
            <ul>
"""
    
    # Add top recommendations
    top_recs = stats.get('top_recommendations', [])
    if top_recs:
        for rec, count in top_recs[:5]:
            html_content += f"                <li>{rec}</li>\n"
    else:
        html_content += "                <li>No recommendations generated yet</li>\n"
    
    html_content += """            </ul>
        </div>
        
        <!-- Success Patterns -->
        <div class="success-patterns">
            <h3>✅ Successful Patterns to Reinforce</h3>
            <ul>
"""
    
    # Add successful patterns from deep analyses
    success_patterns = {}
    deep_analyses = analysis_results.get('deep_analyses', [])
    
    # If we have deep analyses, extract patterns
    if deep_analyses:
        for da in deep_analyses:
            # Handle both object and dict formats
            patterns = getattr(da, 'successful_patterns', None) or da.get('successful_patterns', [])
            for pattern in patterns:
                success_patterns[pattern] = success_patterns.get(pattern, 0) + 1
        
        sorted_patterns = sorted(success_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        if sorted_patterns:
            for pattern, count in sorted_patterns:
                html_content += f"                <li>{pattern}</li>\n"
        else:
            html_content += "                <li>No successful patterns identified in deep analysis</li>\n"
    else:
        # No deep analyses available - show informative message
        html_content += "                <li>Deep analysis not performed - run with --depth deep to identify successful patterns</li>\n"
    
    html_content += """            </ul>
        </div>
        
        <!-- Charts -->
        <div class="charts-grid">
            <div class="chart-container">
                <h3 class="chart-title">Success Distribution</h3>
                <canvas id="successChart"></canvas>
            </div>
            <div class="chart-container">
                <h3 class="chart-title">Intervention Categories</h3>
                <canvas id="interventionChart"></canvas>
            </div>
            <div class="chart-container">
                <h3 class="chart-title">Task Complexity</h3>
                <canvas id="complexityChart"></canvas>
            </div>
            <div class="chart-container">
                <h3 class="chart-title">Conversation Tone</h3>
                <canvas id="toneChart"></canvas>
            </div>
        </div>
        
        <!-- Insights -->
        <div class="insights-grid">
            <div class="insight-card">
                <h3>🔍 Top Systemic Issues</h3>
                <ul class="insight-list">
"""
    
    # Add top issues
    for issue, count in stats.get('top_systemic_issues', [])[:5]:
        html_content += f"""                    <li class="insight-item">
                        <span class="insight-text">{issue}</span>
                        <span class="insight-count">{count}</span>
                    </li>
"""
    
    html_content += """                </ul>
            </div>
            
            <div class="insight-card">
                <h3>📚 Most Common Intervention Types</h3>
                <ul class="insight-list">
"""
    
    # Add intervention categories
    intervention_cats = stats.get('intervention_categories', {})
    sorted_cats = sorted(intervention_cats.items(), key=lambda x: x[1], reverse=True)[:5]
    for cat, count in sorted_cats:
        # Make category names more readable
        readable_cat = cat.replace('_', ' ').title()
        html_content += f"""                    <li class="insight-item">
                        <span class="insight-text">{readable_cat}</span>
                        <span class="insight-count">{count}</span>
                    </li>
"""
    
    html_content += f"""                </ul>
            </div>
        </div>
        
        <div class="footer">
            Generated on {analysis_results.get('timestamp', 'Unknown')} • Claude Conversation Analyzer
        </div>
    </div>
    
    <script>
        // Chart.js configuration
        Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif';
        Chart.defaults.font.size = 14;
        
        // Success Distribution Chart
        const successData = {json.dumps(stats.get('success_distribution', {}))};
        new Chart(document.getElementById('successChart'), {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(successData),
                datasets: [{{
                    data: Object.values(successData),
                    backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        
        // Intervention Categories Chart
        const interventionData = {json.dumps(dict(sorted_cats[:8]))} || {{}};
        new Chart(document.getElementById('interventionChart'), {{
            type: 'bar',
            data: {{
                labels: Object.keys(interventionData).map(k => k.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())),
                datasets: [{{
                    data: Object.values(interventionData),
                    backgroundColor: '#6366f1',
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            stepSize: 1
                        }}
                    }}
                }}
            }}
        }});
        
        // Complexity Chart
        const complexityData = {json.dumps(stats.get('complexity_distribution', {}))};
        new Chart(document.getElementById('complexityChart'), {{
            type: 'bar',
            data: {{
                labels: Object.keys(complexityData),
                datasets: [{{
                    data: Object.values(complexityData),
                    backgroundColor: ['#34d399', '#fbbf24', '#f87171'],
                    borderRadius: 8
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        ticks: {{
                            stepSize: 1
                        }}
                    }}
                }}
            }}
        }});
        
        // Tone Chart
        const toneData = {json.dumps(stats.get('tone_distribution', {}))};
        new Chart(document.getElementById('toneChart'), {{
            type: 'pie',
            data: {{
                labels: Object.keys(toneData),
                datasets: [{{
                    data: Object.values(toneData),
                    backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ef4444'],
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Save dashboard
    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(html_content)
    return dashboard_path