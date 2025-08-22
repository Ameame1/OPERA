#!/usr/bin/env python3
"""
Create an interactive visualization of OPERA's three-agent judge system
showing how each agent is evaluated by different reward criteria
"""

import json

def create_judge_visualization_html():
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OPERA Agent Judge System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 1400px;
            width: 100%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            text-align: center;
            color: #2d3748;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }
        
        .subtitle {
            text-align: center;
            color: #718096;
            margin-bottom: 40px;
            font-size: 1.1rem;
        }
        
        .judges-arena {
            display: flex;
            justify-content: space-around;
            margin-bottom: 50px;
            gap: 30px;
        }
        
        .agent-card {
            flex: 1;
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .agent-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        .agent-card.plan {
            border-top: 5px solid #10b981;
        }
        
        .agent-card.analysis {
            border-top: 5px solid #f59e0b;
        }
        
        .agent-card.rewrite {
            border-top: 5px solid #8b5cf6;
        }
        
        .agent-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .agent-icon {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            margin-right: 15px;
        }
        
        .plan .agent-icon {
            background: #10b98120;
            color: #10b981;
        }
        
        .analysis .agent-icon {
            background: #f59e0b20;
            color: #f59e0b;
        }
        
        .rewrite .agent-icon {
            background: #8b5cf620;
            color: #8b5cf6;
        }
        
        .agent-title {
            font-weight: 700;
            font-size: 1.3rem;
            color: #2d3748;
        }
        
        .agent-role {
            color: #718096;
            font-size: 0.9rem;
        }
        
        .judge-panel {
            background: #f7fafc;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        .judge-title {
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .judge-icon {
            width: 25px;
            height: 25px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
        }
        
        .criteria-list {
            list-style: none;
            padding: 0;
        }
        
        .criteria-item {
            display: flex;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .criteria-item:last-child {
            border-bottom: none;
        }
        
        .criteria-score {
            min-width: 50px;
            height: 25px;
            background: linear-gradient(90deg, #10b981 0%, #10b981 var(--score));
            border-radius: 12px;
            margin-right: 15px;
            position: relative;
            background-color: #e2e8f0;
        }
        
        .criteria-score::after {
            content: attr(data-score);
            position: absolute;
            right: -35px;
            top: 50%;
            transform: translateY(-50%);
            font-weight: 600;
            color: #2d3748;
            font-size: 0.9rem;
        }
        
        .criteria-name {
            flex: 1;
            color: #4a5568;
        }
        
        .reward-formula {
            background: #fff;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px;
            margin-top: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            color: #2d3748;
            text-align: center;
        }
        
        .deepseek-judge {
            background: linear-gradient(135deg, #3b82f6, #06b6d4);
            color: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .deepseek-judge::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 3s ease-in-out infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }
        
        .deepseek-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }
        
        .deepseek-subtitle {
            font-size: 1.1rem;
            opacity: 0.95;
            position: relative;
            z-index: 1;
        }
        
        .training-flow {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 30px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 15px;
        }
        
        .flow-step {
            flex: 1;
            text-align: center;
            position: relative;
        }
        
        .flow-step::after {
            content: '→';
            position: absolute;
            right: -20px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.5rem;
            color: #cbd5e0;
        }
        
        .flow-step:last-child::after {
            display: none;
        }
        
        .flow-number {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 10px;
            font-weight: 700;
        }
        
        .flow-text {
            color: #4a5568;
            font-size: 0.9rem;
        }
        
        .score-animation {
            animation: scoreGrow 1.5s ease-out forwards;
        }
        
        @keyframes scoreGrow {
            from {
                width: 0%;
                opacity: 0;
            }
            to {
                width: var(--score);
                opacity: 1;
            }
        }
        
        .agent-performance {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255, 255, 255, 0.9);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .plan .agent-performance {
            color: #10b981;
            border: 2px solid #10b981;
        }
        
        .analysis .agent-performance {
            color: #f59e0b;
            border: 2px solid #f59e0b;
        }
        
        .rewrite .agent-performance {
            color: #8b5cf6;
            border: 2px solid #8b5cf6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚖️ OPERA Multi-Agent Judge System</h1>
        <p class="subtitle">How DeepSeek R1 Evaluates Each Agent's Performance in MAPGRPO Training</p>
        
        <!-- DeepSeek Judge Panel -->
        <div class="deepseek-judge">
            <div class="deepseek-title">🎯 DeepSeek R1 - The Supreme Judge</div>
            <div class="deepseek-subtitle">Provides fine-grained scalar rewards for each agent's actions</div>
        </div>
        
        <!-- Three Agent Cards -->
        <div class="judges-arena">
            <!-- Plan Agent -->
            <div class="agent-card plan" onclick="animateScores('plan')">
                <div class="agent-performance">Score: 87%</div>
                <div class="agent-header">
                    <div class="agent-icon">📋</div>
                    <div>
                        <div class="agent-title">Plan Agent</div>
                        <div class="agent-role">Strategic Decomposer</div>
                    </div>
                </div>
                
                <div class="judge-panel">
                    <div class="judge-title">
                        <div class="judge-icon">⚖️</div>
                        Evaluation Criteria
                    </div>
                    <ul class="criteria-list">
                        <li class="criteria-item">
                            <div class="criteria-score" style="--score: 90%" data-score="90%"></div>
                            <span class="criteria-name">Logical Correctness (λ₁ = 0.5)</span>
                        </li>
                        <li class="criteria-item">
                            <div class="criteria-score" style="--score: 85%" data-score="85%"></div>
                            <span class="criteria-name">Structural Validity (λ₂ = 0.3)</span>
                        </li>
                        <li class="criteria-item">
                            <div class="criteria-score" style="--score: 82%" data-score="82%"></div>
                            <span class="criteria-name">Execution Success (λ₃ = 0.2)</span>
                        </li>
                    </ul>
                    <div class="reward-formula">
                        R_plan = 0.5 × Logic + 0.3 × Structure + 0.2 × Execution
                    </div>
                </div>
            </div>
            
            <!-- Analysis-Answer Agent -->
            <div class="agent-card analysis" onclick="animateScores('analysis')">
                <div class="agent-performance">Score: 92%</div>
                <div class="agent-header">
                    <div class="agent-icon">🔍</div>
                    <div>
                        <div class="agent-title">Analysis-Answer Agent</div>
                        <div class="agent-role">Information Extractor</div>
                    </div>
                </div>
                
                <div class="judge-panel">
                    <div class="judge-title">
                        <div class="judge-icon">⚖️</div>
                        Evaluation Criteria
                    </div>
                    <ul class="criteria-list">
                        <li class="criteria-item">
                            <div class="criteria-score" style="--score: 95%" data-score="95%"></div>
                            <span class="criteria-name">Answer Correctness (α₁ = 0.6)</span>
                        </li>
                        <li class="criteria-item">
                            <div class="criteria-score" style="--score: 88%" data-score="88%"></div>
                            <span class="criteria-name">Evidence Quality (α₂ = 0.25)</span>
                        </li>
                        <li class="criteria-item">
                            <div class="criteria-score" style="--score: 90%" data-score="90%"></div>
                            <span class="criteria-name">Format Compliance (α₃ = 0.15)</span>
                        </li>
                    </ul>
                    <div class="reward-formula">
                        R_analysis = 0.6 × Correct + 0.25 × Evidence + 0.15 × Format
                    </div>
                </div>
            </div>
            
            <!-- Rewrite Agent -->
            <div class="agent-card rewrite" onclick="animateScores('rewrite')">
                <div class="agent-performance">Score: 79%</div>
                <div class="agent-header">
                    <div class="agent-icon">✏️</div>
                    <div>
                        <div class="agent-title">Rewrite Agent</div>
                        <div class="agent-role">Query Optimizer</div>
                    </div>
                </div>
                
                <div class="judge-panel">
                    <div class="judge-title">
                        <div class="judge-icon">⚖️</div>
                        Evaluation Criteria
                    </div>
                    <ul class="criteria-list">
                        <li class="criteria-item">
                            <div class="criteria-score" style="--score: 78%" data-score="78%"></div>
                            <span class="criteria-name">Retrieval Improvement (β₁ = 0.7)</span>
                        </li>
                        <li class="criteria-item">
                            <div class="criteria-score" style="--score: 82%" data-score="82%"></div>
                            <span class="criteria-name">Query Clarity (β₂ = 0.3)</span>
                        </li>
                    </ul>
                    <div class="reward-formula">
                        R_rewrite = 0.7 × Retrieval + 0.3 × Clarity
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Training Flow -->
        <div class="training-flow">
            <div class="flow-step">
                <div class="flow-number">1</div>
                <div class="flow-text">Generate Candidates<br>(Group Size = 5)</div>
            </div>
            <div class="flow-step">
                <div class="flow-number">2</div>
                <div class="flow-text">DeepSeek R1 Scoring<br>(Fine-grained Rewards)</div>
            </div>
            <div class="flow-step">
                <div class="flow-number">3</div>
                <div class="flow-text">Compute Advantages<br>(Group Relative)</div>
            </div>
            <div class="flow-step">
                <div class="flow-number">4</div>
                <div class="flow-text">Update Policy<br>(GRPO Loss)</div>
            </div>
        </div>
    </div>
    
    <script>
        function animateScores(agent) {
            const card = document.querySelector(`.agent-card.${agent}`);
            const scores = card.querySelectorAll('.criteria-score');
            
            scores.forEach(score => {
                score.style.animation = 'none';
                setTimeout(() => {
                    score.style.animation = 'scoreGrow 1.5s ease-out forwards';
                }, 10);
            });
        }
        
        // Initial animation on load
        window.addEventListener('load', () => {
            setTimeout(() => {
                ['plan', 'analysis', 'rewrite'].forEach((agent, index) => {
                    setTimeout(() => animateScores(agent), index * 500);
                });
            }, 500);
        });
    </script>
</body>
</html>'''
    
    # Save the HTML file
    with open('docs/opera_judge_system.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Judge visualization created: docs/opera_judge_system.html")

if __name__ == "__main__":
    create_judge_visualization_html()