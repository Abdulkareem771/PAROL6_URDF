#!/usr/bin/env python3
"""
ROS-ESP32 Communication Integrity Test

This script:
1. Sends synthetic trajectory data to ESP32
2. Measures round-trip latency
3. Detects packet loss
4. Generates performance report with graphs
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import sys

class CommunicationTester:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        """Initialize serial connection"""
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1.0)
            time.sleep(2)  # Wait for ESP32 reset
            print(f"✓ Connected to {port} at {baudrate} baud")
            
            # Wait for READY signal
            start = time.time()
            while time.time() - start < 5:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode().strip()
                    print(f"ESP32: {line}")
                    if "READY" in line:
                        break
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            sys.exit(1)
        
        self.results = {
            'latencies': [],
            'timestamps': [],
            'lost_packets': 0,
            'sent_packets': 0,
            'ack_received': 0
        }
    
    def generate_trajectory_point(self, seq_num):
        """Generate synthetic trajectory waypoint"""
        # Simulate realistic joint values (radians, rad/s, rad/s^2)
        positions = np.random.uniform(-3.14, 3.14, 6)
        velocities = np.random.uniform(-1.0, 1.0, 6)
        accelerations = np.random.uniform(-0.5, 0.5, 6)
        
        # Format: <SEQ,J1,J2,J3,J4,J5,J6,V1,V2,V3,V4,V5,V6,A1,A2,A3,A4,A5,A6>
        values = [seq_num] + list(positions) + list(velocities) + list(accelerations)
        packet = '<' + ','.join([f'{v:.4f}' for v in values]) + '>\n'
        return packet
    
    def send_and_wait(self, seq_num, timeout=0.5):
        """Send packet and wait for ACK"""
        packet = self.generate_trajectory_point(seq_num)
        send_time = time.time()
        
        # Send
        self.ser.write(packet.encode())
        self.results['sent_packets'] += 1
        
        # Wait for ACK: <ACK,SEQ,TIMESTAMP_US>
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            if self.ser.in_waiting:
                response = self.ser.readline().decode().strip()
                
                if response.startswith('<ACK'):
                    recv_time = time.time()
                    latency_ms = (recv_time - send_time) * 1000
                    
                    self.results['latencies'].append(latency_ms)
                    self.results['timestamps'].append(recv_time)
                    self.results['ack_received'] += 1
                    
                    return True, latency_ms
                elif response.startswith('LOG'):
                    # ESP32 is logging, ignore
                    pass
        
        # Timeout - packet lost
        self.results['lost_packets'] += 1
        return False, None
    
    def run_test(self, num_packets=100, delay_ms=50):
        """Run communication test"""
        print(f"\n{'='*50}")
        print(f"Starting Communication Test")
        print(f"  Packets: {num_packets}")
        print(f"  Delay: {delay_ms}ms between packets")
        print(f"{'='*50}\n")
        
        for i in range(num_packets):
            success, latency = self.send_and_wait(i)
            
            if success:
                print(f"[{i+1:3d}/{num_packets}] ✓ ACK received | Latency: {latency:.2f}ms")
            else:
                print(f"[{i+1:3d}/{num_packets}] ✗ TIMEOUT (packet lost)")
            
            time.sleep(delay_ms / 1000.0)
        
        print("\n" + "="*50)
        print("Test Complete!")
        print("="*50)
    
    def generate_report(self, output_file='comm_test_report.png'):
        """Generate performance report with graphs"""
        sent = self.results['sent_packets']
        acked = self.results['ack_received']
        lost = self.results['lost_packets']
        latencies = np.array(self.results['latencies'])
        
        # Calculate stats
        if len(latencies) > 0:
            avg_latency = np.mean(latencies)
            max_latency = np.max(latencies)
            min_latency = np.min(latencies)
            std_latency = np.std(latencies)
        else:
            avg_latency = max_latency = min_latency = std_latency = 0
        
        loss_rate = (lost / sent * 100) if sent > 0 else 0
        
        # Print text report
        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        print(f"Packets Sent:     {sent}")
        print(f"ACKs Received:    {acked}")
        print(f"Packets Lost:     {lost}")
        print(f"Loss Rate:        {loss_rate:.2f}%")
        print(f"Avg Latency:      {avg_latency:.2f} ms")
        print(f"Min Latency:      {min_latency:.2f} ms")
        print(f"Max Latency:      {max_latency:.2f} ms")
        print(f"Std Deviation:    {std_latency:.2f} ms")
        print("="*50)
        
        # Generate graphs
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('ROS-ESP32 Communication Test Report', fontsize=16, fontweight='bold')
        
        # Plot 1: Latency over time
        ax1 = axes[0, 0]
        if len(latencies) > 0:
            ax1.plot(latencies, marker='o', markersize=3, linewidth=1)
            ax1.axhline(avg_latency, color='r', linestyle='--', label=f'Avg: {avg_latency:.2f}ms')
            ax1.set_xlabel('Packet Number')
            ax1.set_ylabel('Latency (ms)')
            ax1.set_title('Round-Trip Latency Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Latency histogram
        ax2 = axes[0, 1]
        if len(latencies) > 0:
            ax2.hist(latencies, bins=20, edgecolor='black', alpha=0.7)
            ax2.axvline(avg_latency, color='r', linestyle='--', linewidth=2, label=f'Mean: {avg_latency:.2f}ms')
            ax2.set_xlabel('Latency (ms)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Latency Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Packet success/loss
        ax3 = axes[1, 0]
        labels = ['ACK Received', 'Lost']
        sizes = [acked, lost]
        colors = ['#4CAF50', '#F44336']
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Packet Success Rate')
        
        # Plot 4: Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""
        COMMUNICATION STATISTICS
        ━━━━━━━━━━━━━━━━━━━━━━━━
        
        Total Packets Sent:    {sent}
        Packets Acknowledged:  {acked}
        Packets Lost:          {lost}
        
        Loss Rate:             {loss_rate:.2f}%
        Success Rate:          {100-loss_rate:.2f}%
        
        Latency (ms):
          Minimum:             {min_latency:.2f}
          Average:             {avg_latency:.2f}
          Maximum:             {max_latency:.2f}
          Std Dev:             {std_latency:.2f}
        
        Test Timestamp:
          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Report saved to: {output_file}")
        
        # Also save raw data
        csv_file = output_file.replace('.png', '.csv')
        with open(csv_file, 'w') as f:
            f.write("seq,latency_ms\n")
            for i, lat in enumerate(latencies):
                f.write(f"{i},{lat:.4f}\n")
        print(f"✓ Raw data saved to: {csv_file}")
        
        plt.show()
    
    def close(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("\n✓ Serial connection closed")

def main():
    parser = argparse.ArgumentParser(description='Test ROS-ESP32 communication integrity')
    parser.add_argument('--port', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--packets', type=int, default=100, help='Number of test packets')
    parser.add_argument('--delay', type=int, default=50, help='Delay between packets (ms)')
    parser.add_argument('--output', default='comm_test_report.png', help='Output report file')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  ROS-ESP32 Communication Integrity Test")
    print("="*60)
    
    try:
        tester = CommunicationTester(port=args.port, baudrate=args.baud)
        tester.run_test(num_packets=args.packets, delay_ms=args.delay)
        tester.generate_report(output_file=args.output)
        tester.close()
        
        print("\n✓ Test completed successfully!\n")
        
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
