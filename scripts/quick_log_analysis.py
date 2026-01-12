#!/usr/bin/env python3
"""
Quick analysis of ROS driver logs - show velocities, accelerations, and trajectory stats
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def analyze_log(csv_file):
    """Analyze a single driver command log"""
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {Path(csv_file).name}")
    print(f"{'='*70}\n")
    
    # Load data
    df = pd.read_csv(csv_file)
    
    if len(df) == 0:
        print("⚠️  Log file is empty (no commands sent)")
        return
    
    print(f"📊 **Summary Statistics**\n")
    print(f"Total Commands: {len(df)}")
    print(f"Time Span: {(df['timestamp_pc_us'].iloc[-1] - df['timestamp_pc_us'].iloc[0]) / 1e6:.2f} seconds")
    print(f"First command: {df['timestamp_pc_iso'].iloc[0]}")
    print(f"Last command:  {df['timestamp_pc_iso'].iloc[-1]}")
    
    # Calculate command rate
    if len(df) > 1:
        time_diffs = np.diff(df['timestamp_pc_us'].values) / 1000.0  # Convert to ms
        avg_interval = np.mean(time_diffs)
        command_rate = 1000.0 / avg_interval  # Hz
        
        print(f"\n⏱️  **Timing Analysis**\n")
        print(f"Average command interval: {avg_interval:.2f} ms")
        print(f"Command rate: {command_rate:.1f} Hz")
        print(f"Min interval: {np.min(time_diffs):.2f} ms")
        print(f"Max interval: {np.max(time_diffs):.2f} ms")
        print(f"Jitter (std dev): {np.std(time_diffs):.2f} ms")
    
    # Analyze joint data
    print(f"\n🤖 **Joint Position Analysis** (radians)\n")
    pos_cols = ['j1_pos', 'j2_pos', 'j3_pos', 'j4_pos', 'j5_pos', 'j6_pos']
    
    for col in pos_cols:
        if col in df.columns:
            joint_num = col[1]  # Extract joint number
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            print(f"  J{joint_num}: Min={min_val:+.3f} rad ({np.degrees(min_val):+7.2f}°), "
                  f"Max={max_val:+.3f} rad ({np.degrees(max_val):+7.2f}°), "
                  f"Range={range_val:.3f} rad ({np.degrees(range_val):.2f}°)")
    
    # Analyze velocities
    print(f"\n⚡ **Velocity Analysis** (rad/s)\n")
    vel_cols = ['j1_vel', 'j2_vel', 'j3_vel', 'j4_vel', 'j5_vel', 'j6_vel']
    
    for col in vel_cols:
        if col in df.columns and df[col].abs().max() > 0:
            joint_num = col[1]
            max_abs_vel = df[col].abs().max()
            avg_abs_vel = df[col].abs().mean()
            print(f"  J{joint_num}: Max={max_abs_vel:.3f} rad/s ({np.degrees(max_abs_vel):.2f} °/s), "
                  f"Avg={avg_abs_vel:.3f} rad/s")
    
    # Analyze accelerations  
    print(f"\n🚀 **Acceleration Analysis** (rad/s²)\n")
    acc_cols = ['j1_acc', 'j2_acc', 'j3_acc', 'j4_acc', 'j5_acc', 'j6_acc']
    
    for col in acc_cols:
        if col in df.columns and df[col].abs().max() > 0:
            joint_num = col[1]
            max_abs_acc = df[col].abs().max()
            avg_abs_acc = df[col].abs().mean()
            print(f"  J{joint_num}: Max={max_abs_acc:.3f} rad/s², "
                  f"Avg={avg_abs_acc:.3f} rad/s²")
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'Trajectory Analysis: {Path(csv_file).name}', fontsize=14, fontweight='bold')
    
    # Use command numbers for x-axis
    x_axis = np.arange(len(df))
    
    # Plot 1: Positions
    ax = axes[0]
    for col in pos_cols:
        if col in df.columns:
            ax.plot(x_axis, np.degrees(df[col].values), label=col.upper(), linewidth=1.5)
    ax.set_ylabel('Position (degrees)', fontsize=11)
    ax.set_title('Joint Positions Over Time')
    ax.legend(loc='best', ncol=6)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Velocities
    ax = axes[1]
    for col in vel_cols:
        if col in df.columns:
            ax.plot(x_axis, np.degrees(df[col].values), label=col.upper(), linewidth=1.5)
    ax.set_ylabel('Velocity (°/s)', fontsize=11)
    ax.set_title('Joint Velocities Over Time')
    ax.legend(loc='best', ncol=6)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Accelerations
    ax = axes[2]
    for col in acc_cols:
        if col in df.columns:
            ax.plot(x_axis, np.degrees(df[col].values), label=col.upper(), linewidth=1.5)
    ax.set_xlabel('Command Number', fontsize=11)
    ax.set_ylabel('Acceleration (°/s²)', fontsize=11)
    ax.set_title('Joint Accelerations Over Time')
    ax.legend(loc='best', ncol=6)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_file = Path(csv_file).stem + '_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plots saved to: {output_file}")
    
    plt.show()

if __name__ == '__main__':
    # If file provided as argument, use it
    if len(sys.argv) >= 2:
        csv_file = sys.argv[1]
    else:
        # Interactive mode - show available files
        logs_dir = Path('logs')
        
        if not logs_dir.exists():
            print("❌ logs/ directory not found!")
            print("Run this script from the project root directory.")
            sys.exit(1)
        
        # Find all CSV files
        all_csv_files = sorted(logs_dir.glob('driver_commands_*.csv'), reverse=True)
        
        if not all_csv_files:
            print("❌ No log files found in logs/ directory")
            sys.exit(1)
        
        # Filter out empty files (only headers, no data)
        csv_files = [f for f in all_csv_files if f.stat().st_size > 500]  # More than just headers
        
        if not csv_files:
            print("❌ No non-empty log files found in logs/ directory")
            print(f"Found {len(all_csv_files)} file(s) but all are empty (header only)")
            sys.exit(1)
        
        # Show only recent files by default
        max_display = 10
        display_files = csv_files[:max_display]
        
        # Show file list
        print("\n" + "="*70)
        print("  Available Driver Log Files")
        if len(csv_files) > max_display:
            print(f"  (Showing {max_display} most recent of {len(csv_files)} total)")
        print("="*70 + "\n")
        
        for idx, file in enumerate(display_files, 1):
            # Get file size
            size = file.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/(1024*1024):.1f}MB"
            
            # Get modification time
            mtime = file.stat().st_mtime
            from datetime import datetime
            dt = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Estimate number of commands
            if size > 500:
                est_commands = (size - 177) // 150  # Rough estimate
                commands_str = f" (~{est_commands} commands)"
            else:
                commands_str = " (empty)"
            
            print(f"  [{idx}] {file.name}")
            print(f"      Size: {size_str:>8s}  |  Modified: {dt}{commands_str}")
            print()
        
        # Get user selection
        print("="*70)
        if len(csv_files) > max_display:
            print(f"\nTip: Type 'all' to see all {len(csv_files)} files")
        
        while True:
            try:
                choice = input(f"\nSelect file [1-{len(display_files)}] or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print("Cancelled.")
                    sys.exit(0)
                
                if choice.lower() == 'all' and len(csv_files) > max_display:
                    # Show all files
                    display_files = csv_files
                    print("\n" + "="*70)
                    print(f"  All {len(csv_files)} Driver Log Files")
                    print("="*70 + "\n")
                    
                    for idx, file in enumerate(display_files, 1):
                        size = file.stat().st_size
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024*1024:
                            size_str = f"{size/1024:.1f}KB"
                        else:
                            size_str = f"{size/(1024*1024):.1f}MB"
                        
                        mtime = file.stat().st_mtime
                        from datetime import datetime
                        dt = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"  [{idx}] {file.name:50s} {size_str:>8s}  {dt}")
                    
                    print("="*70)
                    continue
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(display_files):
                    csv_file = str(display_files[choice_num - 1])
                    break
                else:
                    print(f"Please enter a number between 1 and {len(display_files)}")
            except ValueError:
                print("Invalid input. Enter a number, 'all', or 'q' to quit.")
            except KeyboardInterrupt:
                print("\nCancelled.")
                sys.exit(0)
    
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    analyze_log(csv_file)
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")
