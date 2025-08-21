#!/usr/bin/env python3
"""
Create a simple animated GIF demonstration of OPERA workflow using matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_opera_animation():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'query': '#667eea',
        'plan': '#10b981',
        'analysis': '#f59e0b',
        'rewrite': '#8b5cf6',
        'retriever': '#06b6d4',
        'answer': '#22c55e'
    }
    
    # Title (always visible)
    title = ax.text(50, 95, 'OPERA: Orchestrated Planner-Executor Reasoning Architecture', 
                   fontsize=18, fontweight='bold', ha='center', va='top')
    subtitle = ax.text(50, 90, 'Multi-Agent Progressive Workflow Demonstration', 
                      fontsize=12, ha='center', va='top', color='gray')
    
    # Initialize components (will be animated)
    components = []
    arrows = []
    
    # Query box
    query_rect = FancyBboxPatch((15, 70), 70, 12, 
                                boxstyle="round,pad=0.1",
                                facecolor=colors['query'], 
                                edgecolor='white',
                                alpha=0, linewidth=2)
    ax.add_patch(query_rect)
    query_text = ax.text(50, 76, '', fontsize=11, ha='center', va='center', 
                         color='white', fontweight='bold', alpha=0)
    
    # Plan Agent
    plan_rect = FancyBboxPatch((5, 45), 28, 18,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['plan'],
                               edgecolor='white',
                               alpha=0, linewidth=2)
    ax.add_patch(plan_rect)
    plan_text = ax.text(19, 54, '', fontsize=10, ha='center', va='center',
                       color='white', alpha=0)
    
    # Analysis Agent
    analysis_rect = FancyBboxPatch((36, 45), 28, 18,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['analysis'],
                                   edgecolor='white',
                                   alpha=0, linewidth=2)
    ax.add_patch(analysis_rect)
    analysis_text = ax.text(50, 54, '', fontsize=10, ha='center', va='center',
                           color='white', alpha=0)
    
    # Rewrite Agent
    rewrite_rect = FancyBboxPatch((67, 45), 28, 18,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['rewrite'],
                                  edgecolor='white',
                                  alpha=0, linewidth=2)
    ax.add_patch(rewrite_rect)
    rewrite_text = ax.text(81, 54, '', fontsize=10, ha='center', va='center',
                          color='white', alpha=0)
    
    # Retriever
    retriever_rect = FancyBboxPatch((25, 25), 50, 10,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['retriever'],
                                    edgecolor='white',
                                    alpha=0, linewidth=2)
    ax.add_patch(retriever_rect)
    retriever_text = ax.text(50, 30, '', fontsize=11, ha='center', va='center',
                            color='white', fontweight='bold', alpha=0)
    
    # Final Answer
    answer_rect = FancyBboxPatch((20, 8), 60, 10,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['answer'],
                                edgecolor='white',
                                alpha=0, linewidth=2)
    ax.add_patch(answer_rect)
    answer_text = ax.text(50, 13, '', fontsize=12, ha='center', va='center',
                         color='white', fontweight='bold', alpha=0)
    
    # Create arrows (initially invisible)
    arrow1 = FancyArrowPatch((50, 70), (19, 63), 
                            connectionstyle="arc3,rad=.2",
                            arrowstyle='->', mutation_scale=20,
                            color='gray', alpha=0, linewidth=2)
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((50, 70), (50, 63),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='->', mutation_scale=20,
                            color='gray', alpha=0, linewidth=2)
    ax.add_patch(arrow2)
    
    arrow3 = FancyArrowPatch((50, 70), (81, 63),
                            connectionstyle="arc3,rad=-.2",
                            arrowstyle='->', mutation_scale=20,
                            color='gray', alpha=0, linewidth=2)
    ax.add_patch(arrow3)
    
    arrow4 = FancyArrowPatch((50, 45), (50, 35),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='<->', mutation_scale=20,
                            color='gray', alpha=0, linewidth=2)
    ax.add_patch(arrow4)
    
    arrow5 = FancyArrowPatch((50, 25), (50, 18),
                            connectionstyle="arc3,rad=0",
                            arrowstyle='->', mutation_scale=20,
                            color='gray', alpha=0, linewidth=2)
    ax.add_patch(arrow5)
    
    def update(frame):
        # Timeline of animations
        fade_speed = 0.05
        
        # Frame 0-20: Show query
        if 0 <= frame <= 20:
            alpha = min(1, frame * fade_speed)
            query_rect.set_alpha(alpha * 0.9)
            query_text.set_alpha(alpha)
            if frame == 10:
                query_text.set_text('Complex Query:\n"What is the GDP per capita of the country\nwhere GitHub acquirer\'s HQ is located?"')
        
        # Frame 15-35: Show agents
        if 15 <= frame <= 35:
            alpha = min(1, (frame - 15) * fade_speed)
            plan_rect.set_alpha(alpha * 0.9)
            plan_text.set_alpha(alpha)
            arrow1.set_alpha(alpha * 0.7)
            if frame == 20:
                plan_text.set_text('📋 Plan Agent\n\n1. Which company?\n2. HQ location?\n3. Country?\n4. GDP per capita?')
        
        if 20 <= frame <= 40:
            alpha = min(1, (frame - 20) * fade_speed)
            analysis_rect.set_alpha(alpha * 0.9)
            analysis_text.set_alpha(alpha)
            arrow2.set_alpha(alpha * 0.7)
            if frame == 25:
                analysis_text.set_text('🔍 Analysis Agent\n\n✓ Microsoft\n✓ Redmond, WA\n✓ United States\n✓ $76,000')
        
        if 25 <= frame <= 45:
            alpha = min(1, (frame - 25) * fade_speed)
            rewrite_rect.set_alpha(alpha * 0.9)
            rewrite_text.set_alpha(alpha)
            arrow3.set_alpha(alpha * 0.7)
            if frame == 30:
                rewrite_text.set_text('✏️ Rewrite Agent\n\nQuery optimization:\n• "Microsoft HQ"\n• "Redmond location"\n• "US GDP per capita"')
        
        # Frame 35-50: Show retriever
        if 35 <= frame <= 50:
            alpha = min(1, (frame - 35) * fade_speed)
            retriever_rect.set_alpha(alpha * 0.9)
            retriever_text.set_alpha(alpha)
            arrow4.set_alpha(alpha * 0.7)
            if frame == 40:
                retriever_text.set_text('🔎 BGE-M3 Dense Retriever\nFetching relevant documents from corpus...')
        
        # Frame 45-60: Show final answer
        if 45 <= frame <= 60:
            alpha = min(1, (frame - 45) * fade_speed)
            answer_rect.set_alpha(alpha * 0.9)
            answer_text.set_alpha(alpha)
            arrow5.set_alpha(alpha * 0.7)
            if frame == 50:
                answer_text.set_text('✨ Final Answer: The GDP per capita is approximately $76,000')
        
        # Frame 60-70: Highlight complete workflow
        if 60 <= frame <= 70:
            # Add subtle pulsing effect to show completion
            pulse = 0.1 * np.sin((frame - 60) * 0.5) + 0.9
            answer_rect.set_alpha(pulse)
        
        return [query_rect, query_text, plan_rect, plan_text, 
                analysis_rect, analysis_text, rewrite_rect, rewrite_text,
                retriever_rect, retriever_text, answer_rect, answer_text,
                arrow1, arrow2, arrow3, arrow4, arrow5]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=80, interval=100, blit=True)
    
    # Save as GIF
    writer = PillowWriter(fps=10)
    anim.save('opera_demo.gif', writer=writer, dpi=100)
    plt.close()
    
    print("✅ Animation created successfully: opera_demo.gif")
    
    # Also save individual frames as images
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor='white')
    stages = [
        ("Initial Query", "Complex multi-hop question requiring reasoning"),
        ("Plan Agent", "Decomposes into logical sub-questions"),
        ("Analysis Agent", "Executes plan and checks sufficiency"),
        ("Rewrite Agent", "Optimizes queries for retrieval"),
        ("Retriever", "Fetches relevant documents"),
        ("Final Answer", "Synthesizes complete response")
    ]
    
    for idx, (ax, (stage_title, stage_desc)) in enumerate(zip(axes.flat, stages)):
        ax.text(0.5, 0.7, stage_title, fontsize=16, fontweight='bold', 
                ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.3, stage_desc, fontsize=12, ha='center', va='center',
                transform=ax.transAxes, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add colored border
        stage_colors = [colors['query'], colors['plan'], colors['analysis'],
                       colors['rewrite'], colors['retriever'], colors['answer']]
        rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, 
                            edgecolor=stage_colors[idx], linewidth=3)
        ax.add_patch(rect)
    
    plt.suptitle('OPERA Workflow Stages', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('opera_stages.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Stage diagram saved: opera_stages.png")

if __name__ == "__main__":
    create_opera_animation()