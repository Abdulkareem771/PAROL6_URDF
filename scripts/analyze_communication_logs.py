#!/usr/bin/env python3
"""
Analyze ROS Driver <-> ESP32 Communication Logs

Compares:
- PC log (driver_commands_*.csv from ROS driver)
- ESP32 log (copied from Serial Monitor or SD card)

Generates:
- Latency analysis
- Packet loss detection
- Data integrity verification
- Matplotlib visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
import numpy as np

def load_pc_log(filepath):
    """Load PC-side log from ROS driver"""
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded PC log: {len(df)} commands")
        return df
    except Exception as e:
        print(f"✗ Failed to load PC log: {e}")
        sys.exit(1)

def load_esp32_log(filepath):
    """Load ESP32-side log"""
    # ESP32 logs format: timestamp_us,seq,j1,j2,j3,j4,j5,j6,...
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded ESP32 log: {len(df)} received")
        return df
    except Exception as e:
        print(f"✗ Failed to load ESP32 log: {e}")
        sys.exit(1)

def analyze_packet_loss(pc_df, esp32_df):
    """Detect missing packets"""
    pc_seqs = set(pc_df['seq'].values)
    esp32_seqs = set(esp32_df['Seq'].values) if 'Seq' in esp32_df.columns else set()
    
    sent = len(pc_seqs)
    received = len(esp32_seqs)
    lost = sent - received
    loss_rate = (lost / sent * 100) if sent > 0 else 0
    
    missing_seqs = pc_seqs - esp32_seqs
    
    print("\n" + "="*50)
    print("PACKET LOSS ANALYSIS")
    print("="*50)
    print(f"Commands Sent (PC):     {sent}")
    print(f"Commands Received (ESP): {received}")
    print(f"Packets Lost:            {lost}")
    print(f"Loss Rate:               {loss_rate:.2f}%")
    
    if missing_seqs and len(missing_seqs) < 20:
        print(f"Missing Sequences: {sorted(missing_seqs)}")
    
    return {
        'sent': sent,
        'received': received,
        'lost': lost,
        'loss_rate': loss_rate,
        'missing_seqs': missing_seqs
    }

def analyze_latency(pc_df, esp32_df):
    """Calculate end-to-end latency"""
    # Merge on sequence number
    merged = pd.merge(
        pc_df[['seq', 'timestamp_pc_us']],
        esp32_df[['Seq', 'Timestamp_us']] if 'Timestamp_us' in esp32_df.columns else esp32_df.iloc[:, :2],
        left_on='seq',
        right_on='Seq' if 'Seq' in esp32_df.columns else esp32_df.columns[1],
        how='inner'
    )
    
    if len(merged) == 0:
        print("⚠️  No matching packets found for latency analysis")
        return None
    
    # Calculate latency in milliseconds
    merged['latency_ms'] = (merged.iloc[:, 3] - merged['timestamp_pc_us']) / 1000.0
    
    # Filter out unrealistic values (e.g., clock sync issues)
    merged = merged[merged['latency_ms'].between(-1000, 1000)]
    
    if len(merged) == 0:
        print("⚠️  No valid latency data (check clock synchronization)")
        return None
    
    latencies = merged['latency_ms'].values
    
    print("\n" + "="*50)
    print("LATENCY ANALYSIS")
    print("="*50)
    print(f"Valid Samples:    {len(latencies)}")
    print(f"Mean Latency:     {np.mean(latencies):.2f} ms")
    print(f"Median Latency:   {np.median(latencies):.2f} ms")
    print(f"Min Latency:      {np.min(latencies):.2f} ms")
    print(f"Max Latency:      {np.max(latencies):.2f} ms")
    print(f"Std Deviation:    {np.std(latencies):.2f} ms")
    
    return {
        'latencies': latencies,
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'std': np.std(latencies)
    }

def verify_data_integrity(pc_df, esp32_df):
    """Check if joint values match between PC and ESP32"""
    # This requires matching sequences and comparing joint values
    joint_cols_pc = ['j1_pos', 'j2_pos', 'j3_pos', 'j4_pos', 'j5_pos', 'j6_pos']
    
    # Find corresponding columns in ESP32 log (varies by format)
    # Assuming columns are: Timestamp_us, Seq, J1, J2, J3, J4, J5, J6
    
    merged = pd.merge(
        pc_df[['seq'] + joint_cols_pc],
        esp32_df,
        left_on='seq',
        right_on='Seq' if 'Seq' in esp32_df.columns else esp32_df.columns[1],
        how='inner'
    )
    
    if len(merged) == 0:
        print("⚠️  Cannot verify data integrity (no matching packets)")
        return None
    
    print("\n" + "="*50)
    print("DATA INTEGRITY CHECK")
    print("="*50)
    print(f"Matched Packets: {len(merged)}")
    
    # Compare first few packets
    print("\nSample Comparison (first 3 packets):")
    for i in range(min(3, len(merged))):
        seq = merged.iloc[i]['seq']
        print(f"\n  Seq {seq}:")
        for j in range(6):
            pc_val = merged.iloc[i][joint_cols_pc[j]]
            # ESP32 columns start after Timestamp and Seq
            esp_col_idx = 2 + j if len(esp32_df.columns) > 2 + j else None
            if esp_col_idx:
                esp_val = merged.iloc[i].iloc[len(joint_cols_pc) + 1 + j]
                diff = abs(pc_val - esp_val)
                status = "✓" if diff < 0.01 else "✗"
                print(f"    J{j+1}: PC={pc_val:.4f}, ESP={esp_val:.4f}, Diff={diff:.6f} {status}")
    
    return merged

def generate_report(pc_log, esp32_log, output='comm_analysis_report.png'):
    """Generate comprehensive analysis report"""
    pc_df = load_pc_log(pc_log)
    esp32_df = load_esp32_log(esp32_log)
    
    # Analyses
    loss_stats = analyze_packet_loss(pc_df, esp32_df)
    latency_stats = analyze_latency(pc_df, esp32_df)
    verify_data_integrity(pc_df, esp32_df)
    
    # Generate plots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Packet Loss
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Received', 'Lost']
    sizes = [loss_stats['received'], loss_stats['lost']]
    colors = ['#4CAF50', '#F44336']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Packet Success Rate')
    
    # Plot 2: Latency over time
    ax2 = fig.add_subplot(gs[0, 1])
    if latency_stats and len(latency_stats['latencies']) > 0:
        ax2.plot(latency_stats['latencies'], marker='.', markersize=2, linewidth=0.5)
        ax2.axhline(latency_stats['mean'], color='r', linestyle='--', 
                   label=f"Mean: {latency_stats['mean']:.2f}ms")
        ax2.set_xlabel('Packet Number')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('End-to-End Latency (ROS → ESP32)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Latency histogram
    ax3 = fig.add_subplot(gs[1, 0])
    if latency_stats and len(latency_stats['latencies']) > 0:
        ax3.hist(latency_stats['latencies'], bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(latency_stats['mean'], color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Latency (ms)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Latency Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Command rate over time
    ax4 = fig.add_subplot(gs[1, 1])
    if len(pc_df) > 1:
        time_diffs = np.diff(pc_df['timestamp_pc_us'].values) / 1000.0  # ms
        command_rate = 1000.0 / time_diffs  # Hz
        ax4.plot(command_rate, linewidth=1)
        ax4.set_xlabel('Command Number')
        ax4.set_ylabel('Command Rate (Hz)')
        ax4.set_title('ROS Driver Command Rate')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Statistics summary
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    stats_text = f"""
    ROS DRIVER ↔ ESP32 COMMUNICATION ANALYSIS
    {'='*60}
    
    PACKET STATISTICS:
      Commands Sent (PC):        {loss_stats['sent']}
      Commands Received (ESP32): {loss_stats['received']}
      Packets Lost:              {loss_stats['lost']}
      Loss Rate:                 {loss_stats['loss_rate']:.2f}%
    
    LATENCY METRICS:"""
    
    if latency_stats:
        stats_text += f"""
      Mean Latency:              {latency_stats['mean']:.2f} ms
      Median Latency:            {latency_stats['median']:.2f} ms
      Min Latency:               {latency_stats['min']:.2f} ms
      Max Latency:               {latency_stats['max']:.2f} ms
      Std Deviation:             {latency_stats['std']:.2f} ms
    """
    else:
        stats_text += "\n      No latency data available"
    
    stats_text += f"""
    
    TEST CONFIGURATION:
      PC Log:     {Path(pc_log).name}
      ESP32 Log:  {Path(esp32_log).name}
    """
    
    ax5.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax5.transAxes)
    
    plt.suptitle('ROS-ESP32 Communication Verification Report', 
                fontsize=14, fontweight='bold')
    
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Report saved: {output}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Analyze ROS Driver and ESP32 communication logs'
    )
    parser.add_argument('--pc-log', required=True, 
                       help='Path to PC-side CSV log from ROS driver')
    parser.add_argument('--esp-log', required=True,
                       help='Path to ESP32-side CSV log')
    parser.add_argument('--output', default='comm_analysis_report.png',
                       help='Output report filename')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  ROS-ESP32 Communication Log Analysis")
    print("="*60)
    
    generate_report(args.pc_log, args.esp_log, args.output)
    
    print("\n✓ Analysis complete!\n")

if __name__ == '__main__':
    main()
