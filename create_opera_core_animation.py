#!/usr/bin/env python3
"""
Create an animated GIF that highlights OPERA's core innovations:
1. Hierarchical multi-agent architecture with placeholder mechanism
2. MAPGRPO training with multi-dimensional rewards
3. Progressive training pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_opera_core_animation():
    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 10), facecolor='#f8f9fa')
    
    # Create grid for layout
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1.5, 1], 
                          hspace=0.15, wspace=0.15)
    
    # Main workflow subplot (top left)
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_xlim(0, 100)
    ax_main.set_ylim(0, 100)
    ax_main.axis('off')
    
    # Reward function subplot (bottom left)
    ax_reward = fig.add_subplot(gs[1, 0])
    ax_reward.set_xlim(0, 100)
    ax_reward.set_ylim(0, 100)
    ax_reward.axis('off')
    
    # Training pipeline subplot (bottom right)
    ax_train = fig.add_subplot(gs[1, 1])
    ax_train.set_xlim(0, 100)
    ax_train.set_ylim(0, 100)
    ax_train.axis('off')
    
    # Title
    fig.suptitle('OPERA: Core Innovations in Multi-Agent RAG', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Color scheme
    colors = {
        'plan': '#10b981',      # Green
        'analysis': '#f59e0b',  # Orange
        'rewrite': '#8b5cf6',   # Purple
        'placeholder': '#ef4444', # Red for placeholders
        'reward': '#3b82f6',    # Blue
        'frozen': '#94a3b8',    # Gray for frozen agents
        'active': '#22c55e'     # Bright green for active training
    }
    
    # === Main Workflow Animation ===
    
    # Query box
    query_rect = FancyBboxPatch((10, 85), 80, 10,
                                boxstyle="round,pad=0.1",
                                facecolor='#1e40af', edgecolor='none',
                                alpha=0)
    ax_main.add_patch(query_rect)
    query_text = ax_main.text(50, 90, '', fontsize=11, ha='center', 
                              va='center', color='white', fontweight='bold', alpha=0)
    
    # Plan Agent with steps
    plan_rect = FancyBboxPatch((5, 50), 28, 25,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['plan'], edgecolor='white',
                               alpha=0, linewidth=2)
    ax_main.add_patch(plan_rect)
    plan_title = ax_main.text(19, 70, '', fontsize=11, ha='center',
                              color='white', fontweight='bold', alpha=0)
    plan_steps = []
    for i in range(4):
        step_text = ax_main.text(19, 65-i*4, '', fontsize=9, ha='center',
                                 color='white', alpha=0)
        plan_steps.append(step_text)
    
    # Analysis Agent with execution
    analysis_rect = FancyBboxPatch((36, 50), 28, 25,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['analysis'], edgecolor='white',
                                   alpha=0, linewidth=2)
    ax_main.add_patch(analysis_rect)
    analysis_title = ax_main.text(50, 70, '', fontsize=11, ha='center',
                                  color='white', fontweight='bold', alpha=0)
    analysis_steps = []
    for i in range(4):
        step_text = ax_main.text(50, 65-i*4, '', fontsize=9, ha='center',
                                 color='white', alpha=0)
        analysis_steps.append(step_text)
    
    # Rewrite Agent
    rewrite_rect = FancyBboxPatch((67, 50), 28, 25,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['rewrite'], edgecolor='white',
                                  alpha=0, linewidth=2)
    ax_main.add_patch(rewrite_rect)
    rewrite_title = ax_main.text(81, 70, '', fontsize=11, ha='center',
                                 color='white', fontweight='bold', alpha=0)
    rewrite_content = ax_main.text(81, 62, '', fontsize=9, ha='center',
                                   color='white', alpha=0, multialignment='center')
    
    # Placeholder connections (curved arrows showing dependencies)
    placeholder_arrows = []
    for i in range(3):
        arrow = FancyArrowPatch((33, 65-i*4), (36, 65-i*4),
                               connectionstyle="arc3,rad=0.3",
                               arrowstyle='->', mutation_scale=15,
                               color=colors['placeholder'], alpha=0, 
                               linewidth=2.5, linestyle='--')
        ax_main.add_patch(arrow)
        placeholder_arrows.append(arrow)
    
    # Retriever box
    retriever_rect = FancyBboxPatch((25, 30), 50, 8,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#06b6d4', edgecolor='white',
                                    alpha=0, linewidth=2)
    ax_main.add_patch(retriever_rect)
    retriever_text = ax_main.text(50, 34, '', fontsize=10, ha='center',
                                  color='white', fontweight='bold', alpha=0)
    
    # Final answer
    answer_rect = FancyBboxPatch((20, 15), 60, 8,
                                boxstyle="round,pad=0.1",
                                facecolor='#22c55e', edgecolor='white',
                                alpha=0, linewidth=2)
    ax_main.add_patch(answer_rect)
    answer_text = ax_main.text(50, 19, '', fontsize=11, ha='center',
                               color='white', fontweight='bold', alpha=0)
    
    # === Reward Function Visualization ===
    
    reward_title = ax_reward.text(50, 90, 'Multi-Dimensional Rewards', 
                                  fontsize=13, ha='center', fontweight='bold', alpha=0)
    
    # Reward components bars
    reward_bars = []
    reward_labels = []
    bar_positions = [20, 40, 60, 80]
    bar_names = ['Correctness', 'Coherence', 'Efficiency', 'Role-Specific']
    bar_colors = ['#10b981', '#f59e0b', '#3b82f6', '#8b5cf6']
    
    for i, (pos, name, color) in enumerate(zip(bar_positions, bar_names, bar_colors)):
        # Bar background
        bg_rect = FancyBboxPatch((pos-8, 20), 16, 50,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#e5e7eb', edgecolor='none',
                                 alpha=0)
        ax_reward.add_patch(bg_rect)
        
        # Animated bar
        bar = FancyBboxPatch((pos-8, 20), 16, 0,
                             boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='none',
                             alpha=0)
        ax_reward.add_patch(bar)
        reward_bars.append(bar)
        
        # Label
        label = ax_reward.text(pos, 10, name, fontsize=9, ha='center',
                               fontweight='bold', alpha=0)
        reward_labels.append(label)
    
    # Reward formula
    formula_text = ax_reward.text(50, 75, '', fontsize=10, ha='center',
                                  style='italic', alpha=0)
    
    # === Training Pipeline Visualization ===
    
    train_title = ax_train.text(50, 90, 'Progressive Training', 
                               fontsize=13, ha='center', fontweight='bold', alpha=0)
    
    # Stage indicators
    stages = []
    stage_names = ['Stage 1\nPlan Agent', 'Stage 2\nAnalysis Agent', 'Stage 3\nRewrite Agent']
    stage_positions = [(25, 50), (50, 50), (75, 50)]
    
    for i, (pos, name) in enumerate(zip(stage_positions, stage_names)):
        # Circle for stage
        circle = Circle(pos, 12, facecolor=colors['frozen'], 
                       edgecolor='white', linewidth=2, alpha=0)
        ax_train.add_patch(circle)
        stages.append(circle)
        
        # Stage text
        stage_text = ax_train.text(pos[0], pos[1], name, fontsize=9,
                                   ha='center', va='center', color='white',
                                   fontweight='bold', alpha=0)
        stages.append(stage_text)
    
    # Progress arrow
    progress_arrow = FancyArrowPatch((10, 30), (10, 30),
                                    arrowstyle='->', mutation_scale=20,
                                    color=colors['active'], alpha=0, linewidth=3)
    ax_train.add_patch(progress_arrow)
    
    # Animation function
    def update(frame):
        fade_speed = 0.05
        
        # Phase 1: Show query (frames 0-15)
        if 0 <= frame <= 15:
            alpha = min(1, frame * fade_speed)
            query_rect.set_alpha(alpha * 0.9)
            query_text.set_alpha(alpha)
            if frame == 5:
                query_text.set_text('Complex Query: "GDP per capita of country where\nGitHub acquirer\'s headquarters is located?"')
        
        # Phase 2: Plan Agent creates plan with placeholders (frames 10-30)
        if 10 <= frame <= 30:
            alpha = min(1, (frame - 10) * fade_speed)
            plan_rect.set_alpha(alpha * 0.9)
            plan_title.set_alpha(alpha)
            if frame == 15:
                plan_title.set_text('📋 Plan Agent')
            
            # Show steps progressively
            step_delays = [18, 21, 24, 27]
            step_texts = [
                '1. Find company that acquired GitHub',
                '2. Find HQ of [company from step 1]',
                '3. Find country of [location from step 2]',
                '4. Find GDP per capita of [country from step 3]'
            ]
            for i, (delay, text) in enumerate(zip(step_delays, step_texts)):
                if frame >= delay:
                    step_alpha = min(1, (frame - delay) * 0.2)
                    plan_steps[i].set_alpha(step_alpha)
                    if frame == delay:
                        plan_steps[i].set_text(text)
                        # Highlight placeholders in red
                        if '[' in text:
                            plan_steps[i].set_color('#ffcccc')
        
        # Phase 3: Analysis Agent executes with placeholder resolution (frames 25-50)
        if 25 <= frame <= 50:
            alpha = min(1, (frame - 25) * fade_speed)
            analysis_rect.set_alpha(alpha * 0.9)
            analysis_title.set_alpha(alpha)
            if frame == 30:
                analysis_title.set_text('🔍 Analysis Agent')
            
            # Show execution with placeholder resolution
            exec_delays = [33, 36, 39, 42]
            exec_texts = [
                '✓ Microsoft acquired GitHub',
                '✓ Microsoft HQ: Redmond, WA',
                '✓ Redmond is in United States',
                '✓ US GDP per capita: $76,000'
            ]
            for i, (delay, text) in enumerate(zip(exec_delays, exec_texts)):
                if frame >= delay:
                    step_alpha = min(1, (frame - delay) * 0.2)
                    analysis_steps[i].set_alpha(step_alpha)
                    if frame == delay:
                        analysis_steps[i].set_text(text)
                    
                    # Show placeholder arrows
                    if i < 3 and frame >= delay + 1:
                        placeholder_arrows[i].set_alpha(step_alpha * 0.8)
        
        # Phase 4: Rewrite Agent optimizes (frames 40-55)
        if 40 <= frame <= 55:
            alpha = min(1, (frame - 40) * fade_speed)
            rewrite_rect.set_alpha(alpha * 0.9)
            rewrite_title.set_alpha(alpha)
            rewrite_content.set_alpha(alpha)
            if frame == 45:
                rewrite_title.set_text('✏️ Rewrite Agent')
                rewrite_content.set_text('Optimized Queries:\n"Microsoft headquarters"\n"US GDP per capita 2024"')
        
        # Phase 5: Retriever and Answer (frames 50-65)
        if 50 <= frame <= 65:
            alpha = min(1, (frame - 50) * fade_speed)
            retriever_rect.set_alpha(alpha * 0.9)
            retriever_text.set_alpha(alpha)
            if frame == 52:
                retriever_text.set_text('🔎 BGE-M3 Dense Retriever')
        
        if 55 <= frame <= 70:
            alpha = min(1, (frame - 55) * fade_speed)
            answer_rect.set_alpha(alpha * 0.9)
            answer_text.set_alpha(alpha)
            if frame == 58:
                answer_text.set_text('✨ Final Answer: $76,000 (United States)')
        
        # Phase 6: Show reward function (frames 60-80)
        if 60 <= frame <= 80:
            alpha = min(1, (frame - 60) * fade_speed)
            reward_title.set_alpha(alpha)
            formula_text.set_alpha(alpha)
            if frame == 62:
                formula_text.set_text('R = λ₁R_correct + λ₂R_coherent + λ₃R_efficient + λ₄R_role')
            
            # Animate bars
            for i, (bar, label) in enumerate(zip(reward_bars, reward_labels)):
                if frame >= 65 + i*2:
                    bar_alpha = min(1, (frame - 65 - i*2) * 0.1)
                    bar.set_alpha(bar_alpha * 0.9)
                    label.set_alpha(bar_alpha)
                    # Animate bar height
                    heights = [45, 40, 35, 42]  # Different heights for visual interest
                    current_height = min(heights[i], (frame - 65 - i*2) * 5)
                    bar.set_height(current_height)
        
        # Phase 7: Show training pipeline (frames 70-90)
        if 70 <= frame <= 90:
            alpha = min(1, (frame - 70) * fade_speed)
            train_title.set_alpha(alpha)
            
            # Animate stages
            for i, stage in enumerate(stages[::2]):  # Every other element is a circle
                if frame >= 73 + i*3:
                    stage_alpha = min(1, (frame - 73 - i*3) * 0.1)
                    stage.set_alpha(stage_alpha * 0.9)
                    stages[i*2+1].set_alpha(stage_alpha)  # Text
                    
                    # Change color to active when training
                    if i == 0 and 75 <= frame <= 78:
                        stage.set_facecolor(colors['active'])
                    elif i == 1 and 78 <= frame <= 81:
                        stage.set_facecolor(colors['active'])
                        stages[0].set_facecolor(colors['frozen'])  # Freeze previous
                    elif i == 2 and 81 <= frame <= 84:
                        stage.set_facecolor(colors['active'])
                        stages[2].set_facecolor(colors['frozen'])  # Freeze previous
            
            # Animate progress arrow
            if frame >= 75:
                arrow_alpha = min(1, (frame - 75) * 0.1)
                progress_arrow.set_alpha(arrow_alpha)
                progress = min(80, (frame - 75) * 2)
                progress_arrow.set_positions((10, 30), (10 + progress, 30))
        
        # Phase 8: Highlight innovations (frames 85-100)
        if 85 <= frame <= 100:
            # Pulse effect on key innovations
            pulse = 0.3 * np.sin((frame - 85) * 0.3) + 0.7
            
            # Pulse placeholders
            for arrow in placeholder_arrows:
                if arrow.get_alpha() > 0:
                    arrow.set_alpha(pulse)
            
            # Pulse reward bars
            for bar in reward_bars:
                if bar.get_alpha() > 0:
                    bar.set_alpha(pulse * 0.9)
            
            # Pulse training stages
            if stages[4].get_facecolor() == tuple(np.array([c/255 for c in (34, 197, 94)] + [1.0])):
                stages[4].set_alpha(pulse * 0.9)
        
        return ([query_rect, query_text, plan_rect, plan_title] + plan_steps +
                [analysis_rect, analysis_title] + analysis_steps +
                [rewrite_rect, rewrite_title, rewrite_content] +
                placeholder_arrows + [retriever_rect, retriever_text,
                answer_rect, answer_text, reward_title, formula_text] +
                reward_bars + reward_labels + stages + [train_title, progress_arrow])
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=105, interval=80, blit=True)
    
    # Save as GIF
    writer = PillowWriter(fps=12)
    anim.save('docs/opera_core_demo.gif', writer=writer, dpi=100)
    plt.close()
    
    print("✅ Core innovation animation created: opera_core_demo.gif")

if __name__ == "__main__":
    create_opera_core_animation()