#!/usr/bin/env python3
"""
Create an animated GIF demonstration of OPERA workflow
"""

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import os
import imageio

def create_opera_demo_gif():
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1400,900')
    
    # Initialize driver
    print("Initializing browser...")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Load the HTML file
        html_path = os.path.abspath('docs/opera_demo_animation.html')
        driver.get(f'file://{html_path}')
        
        # Create screenshots directory
        os.makedirs('screenshots', exist_ok=True)
        
        # Capture screenshots at different stages
        print("Capturing animation frames...")
        timestamps = [
            (0.5, "Initial state"),
            (1.5, "Query appears"),
            (2.5, "Plan Agent appears"),
            (3.0, "Plan step 1"),
            (3.5, "Plan step 2"),
            (4.0, "Plan step 3"),
            (4.5, "Plan step 4"),
            (5.0, "Analysis Agent starts"),
            (5.5, "Analysis step 1"),
            (6.0, "Analysis step 2"),
            (6.5, "Analysis step 3 & Rewrite Agent"),
            (7.0, "Analysis step 4"),
            (7.5, "Retriever appears"),
            (8.0, "Final answer"),
            (8.5, "Complete")
        ]
        
        screenshots = []
        for i, (delay, description) in enumerate(timestamps):
            time.sleep(delay if i == 0 else timestamps[i][0] - timestamps[i-1][0])
            screenshot_path = f'screenshots/frame_{i:02d}.png'
            driver.save_screenshot(screenshot_path)
            print(f"  Frame {i+1}/{len(timestamps)}: {description}")
            
            # Load and crop the screenshot
            img = Image.open(screenshot_path)
            # Crop to content area (adjust these values based on actual rendering)
            img_cropped = img.crop((100, 50, 1300, 850))
            screenshots.append(img_cropped)
        
        # Create GIF
        print("Creating GIF...")
        # Adjust duration for each frame (in seconds)
        durations = [1.0] + [0.5] * (len(screenshots) - 2) + [2.0]  # First frame longer, last frame longest
        
        imageio.mimsave(
            'opera_workflow_demo.gif',
            screenshots,
            duration=durations,
            loop=0
        )
        
        print("✅ GIF created successfully: opera_workflow_demo.gif")
        
        # Clean up screenshots
        for i in range(len(timestamps)):
            os.remove(f'screenshots/frame_{i:02d}.png')
        os.rmdir('screenshots')
        
    finally:
        driver.quit()

def create_static_demo_animation():
    """
    Alternative: Create a static animation using matplotlib
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation, PillowWriter
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    title = ax.text(5, 5.5, 'OPERA Workflow Demo', fontsize=20, fontweight='bold', 
                    ha='center', va='center')
    
    # Components (initially hidden)
    components = {
        'query': {'pos': (5, 4.5), 'text': 'Complex Query:\n"GDP per capita of country where\nGitHub acquirer is headquartered?"', 
                  'color': '#667eea', 'visible': False},
        'plan': {'pos': (2, 3), 'text': 'Plan Agent\n1. Company?\n2. HQ location?\n3. Country?\n4. GDP?', 
                 'color': '#10b981', 'visible': False},
        'analysis': {'pos': (5, 3), 'text': 'Analysis Agent\n✓ Microsoft\n✓ Redmond, WA\n✓ United States\n✓ $76,000', 
                     'color': '#f59e0b', 'visible': False},
        'rewrite': {'pos': (8, 3), 'text': 'Rewrite Agent\nOptimized queries\nfor retrieval', 
                    'color': '#8b5cf6', 'visible': False},
        'retriever': {'pos': (5, 1.5), 'text': 'BGE-M3 Retriever\nFetching documents...', 
                      'color': '#06b6d4', 'visible': False},
        'answer': {'pos': (5, 0.5), 'text': 'Final Answer: $76,000', 
                   'color': '#22c55e', 'visible': False}
    }
    
    # Create text objects
    text_objects = {}
    for name, comp in components.items():
        text_objects[name] = ax.text(comp['pos'][0], comp['pos'][1], '', 
                                     fontsize=11, ha='center', va='center',
                                     bbox=dict(boxstyle="round,pad=0.3", 
                                              facecolor=comp['color'], alpha=0))
    
    def update(frame):
        # Animation timeline
        if frame < 20:
            # Show title
            title.set_alpha(min(1, frame/10))
        
        if 10 < frame < 30:
            # Show query
            alpha = min(1, (frame-10)/10)
            text_objects['query'].set_text(components['query']['text'])
            text_objects['query'].get_bbox_patch().set_alpha(alpha * 0.8)
            text_objects['query'].set_alpha(alpha)
        
        if 20 < frame < 40:
            # Show plan agent
            alpha = min(1, (frame-20)/10)
            text_objects['plan'].set_text(components['plan']['text'])
            text_objects['plan'].get_bbox_patch().set_alpha(alpha * 0.8)
            text_objects['plan'].set_alpha(alpha)
        
        if 30 < frame < 50:
            # Show analysis agent
            alpha = min(1, (frame-30)/10)
            text_objects['analysis'].set_text(components['analysis']['text'])
            text_objects['analysis'].get_bbox_patch().set_alpha(alpha * 0.8)
            text_objects['analysis'].set_alpha(alpha)
        
        if 35 < frame < 55:
            # Show rewrite agent
            alpha = min(1, (frame-35)/10)
            text_objects['rewrite'].set_text(components['rewrite']['text'])
            text_objects['rewrite'].get_bbox_patch().set_alpha(alpha * 0.8)
            text_objects['rewrite'].set_alpha(alpha)
        
        if 40 < frame < 60:
            # Show retriever
            alpha = min(1, (frame-40)/10)
            text_objects['retriever'].set_text(components['retriever']['text'])
            text_objects['retriever'].get_bbox_patch().set_alpha(alpha * 0.8)
            text_objects['retriever'].set_alpha(alpha)
        
        if 50 < frame:
            # Show final answer
            alpha = min(1, (frame-50)/10)
            text_objects['answer'].set_text(components['answer']['text'])
            text_objects['answer'].get_bbox_patch().set_alpha(alpha * 0.8)
            text_objects['answer'].set_alpha(alpha)
        
        return list(text_objects.values()) + [title]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=70, interval=100, blit=True)
    
    # Save as GIF
    writer = PillowWriter(fps=10)
    anim.save('opera_workflow_demo_matplotlib.gif', writer=writer)
    plt.close()
    
    print("✅ Matplotlib GIF created: opera_workflow_demo_matplotlib.gif")

if __name__ == "__main__":
    # Try selenium approach first
    try:
        create_opera_demo_gif()
    except Exception as e:
        print(f"Selenium approach failed: {e}")
        print("Trying matplotlib approach...")
        create_static_demo_animation()