"""
docs_tab.py — In-app documentation and help browser for the PAROL6 Firmware Configurator.

Provides a two-panel layout:
  Left  — table of contents / section list
  Right — rich-text content panel
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QScrollArea, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont

# ---------------------------------------------------------------------------
# All documentation sections as (title, rich_html_body) pairs.
# ---------------------------------------------------------------------------
SECTIONS = [
    # ── Overview ──
    ("🏠  Quick Start", """
<h2 style='color:#cba6f7;'>PAROL6 Firmware Configurator — Quick Start</h2>
<p>This tool lets you configure, compile, flash, and operate the PAROL6 6-DOF robot
without touching any C++ source files or bash scripts.</p>

<h3 style='color:#89b4fa;'>First run checklist</h3>
<ol>
<li>Plug in the Teensy 4.1 via USB.</li>
<li>Click <b>🔍 Scan</b> in the toolbar — select the detected port (<code>/dev/ttyACM0</code>).</li>
<li>Click <b>⚡ Connect</b>.</li>
<li>Open the <b>🔬 Protocol</b> tab → select <i>phase0_hardware_check</i> → <b>Load Preset</b>.</li>
<li>Go to <b>⚡ Flash</b> → click <b>Generate config.h</b> then <b>Compile &amp; Flash</b>.</li>
<li>Open <b>💬 Serial</b> — you should see <code>&lt;ACK,0,0.000,…&gt;</code> feedback lines scrolling.</li>
</ol>

<p style='color:#a6e3a1;'>✅ If you see ACK frames, the Teensy is alive and the firmware is running.</p>

<h3 style='color:#89b4fa;'>To launch ROS 2 + MoveIt</h3>
<ol>
<li>Go to <b>🚀 ROS2 Launch</b> tab.</li>
<li>Choose <b>Fake Hardware</b> for simulation, or <b>Real Hardware</b> to control the physical robot.</li>
<li>Click <b>Launch</b>. Logs stream into the terminal panel below the buttons.</li>
</ol>
"""),

    # ── Joints Tab ──
    ("🔩  Joints Tab Guide", """
<h2 style='color:#cba6f7;'>Joints Tab</h2>
<p>Each row is one joint (J1–J6). Every column maps directly to a constant in
<code>generated/config.h</code> that the firmware reads at boot.</p>

<table style='width:100%; border-collapse:collapse; font-size:12px;'>
<tr style='background:#313244; color:#cba6f7;'>
  <th align='left' style='padding:4px;'>Column</th>
  <th align='left' style='padding:4px;'>Effect</th>
</tr>
<tr><td style='padding:4px;'>Enabled</td><td>Disables STEP/DIR output for this axis.</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>STEP/DIR/Enc Pin</td><td>Teensy GPIO numbers — match your wiring schematic.</td></tr>
<tr><td style='padding:4px;'>Gear Ratio</td><td>Used to convert encoder angle → joint angle. From PAROL6 STM32 firmware.</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Dir Inv</td><td>Inverts the STEP pulse direction at Teensy level. J1, J3, J6 are inverted by default.</td></tr>
<tr><td style='padding:4px;'>ROS Inv</td><td>Inverts kinematic sign in <code>parol6_system.cpp</code>. Must stay consistent with Dir Inv.</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Max Vel (rad/s)</td><td>Safety supervisor triggers FAULT if this is exceeded.</td></tr>
<tr><td style='padding:4px;'>Kp / Ki</td><td>PID gains for the closed-loop controller. Start conservative: Kp=2, Ki=0.</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Limit Type</td><td>NONE / MECHANICAL / INDUCTIVE_NPN / INDUCTIVE_PNP</td></tr>
<tr><td style='padding:4px;'>Limit Pin</td><td>Teensy GPIO connected to the optocoupler output.</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Limit Polarity</td><td>FALLING = pin goes LOW when triggered (NPN). RISING = pin goes HIGH (PNP).</td></tr>
<tr><td style='padding:4px;'>Limit Pull</td><td>INPUT_PULLUP for NPN, INPUT_PULLDOWN for PNP. <b>Auto-suggested</b> when you change the sensor type.</td></tr>
</table>

<p style='color:#fab387; margin-top:8px;'>⚠️ After any change here, go to Flash tab → Generate &amp; Flash to apply it.</p>

<h3 style='color:#89b4fa;'>Limit Switch Wiring Quick Reference</h3>
<table style='font-size:12px; border-collapse:collapse;'>
<tr style='color:#cba6f7;'><th align='left'>Sensor</th><th align='left'>Pull</th><th align='left'>Polarity</th><th align='left'>Optocoupler wiring</th></tr>
<tr><td>NPN (open-collector)</td><td>INPUT_PULLUP</td><td>FALLING</td><td>Signal→LED+, LED−→GND, Collector→Pin, Emitter→GND</td></tr>
<tr style='background:#252535;'><td>PNP (sourcing)</td><td>INPUT_PULLDOWN</td><td>RISING</td><td>Signal→LED+, LED−→GND, 3.3V→Collector, Emitter→Pin</td></tr>
<tr><td>Mechanical NC</td><td>INPUT_PULLUP</td><td>FALLING</td><td>COM→Pin, NC→GND (no optocoupler needed)</td></tr>
</table>
"""),

    # ── Features Tab ──
    ("⚙️  Features Tab Guide", """
<h2 style='color:#cba6f7;'>Features Tab</h2>
<p>Toggles compile-time <code>#define</code> flags. Disable features to isolate problems.</p>

<table style='width:100%; border-collapse:collapse; font-size:12px;'>
<tr style='background:#313244; color:#cba6f7;'>
  <th align='left' style='padding:4px;'>Flag</th>
  <th align='left' style='padding:4px;'>Off means…</th>
  <th align='left' style='padding:4px;'>When to disable</th>
</tr>
<tr><td style='padding:4px;'>AlphaBeta Filter</td><td>Raw encoder readings used as velocity</td><td>Debugging encoder noise levels</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Velocity Feedforward</td><td>Pure P-control only</td><td>Isolating tracking lag vs overshoot</td></tr>
<tr><td style='padding:4px;'>Safety Supervisor</td><td>No velocity or ESTOP checks</td><td>Open-loop signal validation only</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Anti-Glitch Filter</td><td>All encoder readings accepted</td><td>Checking raw encoder signal quality</td></tr>
<tr><td style='padding:4px;'>Hardware PWM (FlexPWM)</td><td>Software bit-bang STEP pulses</td><td>CPU load measurement comparison</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Hardware Encoder (QuadTimer)</td><td>Software ISR PWM decode</td><td>Validating QuadTimer vs ISR accuracy</td></tr>
<tr><td style='padding:4px;'>Encoder Test Mode</td><td>Control loop disabled entirely</td><td>Phase 1 &amp; 2: just reading encoders</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Sine Sweep Mode</td><td>ROS commands are used</td><td>Testing kinematics before ROS wiring</td></tr>
</table>

<p style='color:#fab387; margin-top:8px;'>⚠️ <b>Safety Supervisor should ALWAYS be on</b> when the robot is connected to the real hardware.  
Disabling it removes all velocity limiting and ESTOP logic.</p>
"""),

    # ── Comms Tab ──
    ("📡  Comms Tab Guide", """
<h2 style='color:#cba6f7;'>Comms Tab</h2>
<p>Sets how the Teensy communicates with the host computer.</p>

<h3 style='color:#89b4fa;'>Transport Mode</h3>
<table style='font-size:12px; border-collapse:collapse;'>
<tr style='color:#cba6f7;'><th align='left'>Mode</th><th align='left'>Port</th><th align='left'>Notes</th></tr>
<tr><td>USB_CDC_HS</td><td>/dev/ttyACM0</td><td>Default. Plug-and-play, 12 Mbit/s USB.</td></tr>
<tr style='background:#252535;'><td>UART_115200</td><td>/dev/ttyUSB0</td><td>Use with FTDI adapter or isolated RS-485.</td></tr>
<tr><td>ETHERNET</td><td>UDP 192.168.1.177:8888</td><td>Low-latency wired option — set IP/port below.</td></tr>
</table>

<h3 style='color:#89b4fa;'>Rate settings</h3>
<ul>
<li><b>ROS Command Rate</b> — how many position packets per second ROS sends. Must match <code>COMMAND_TIMEOUT_MS</code> (default 200 ms = ≥5 Hz required).</li>
<li><b>Feedback Rate</b> — how often the Teensy sends back encoder telemetry. 10 Hz is fine for logging; 100 Hz for oscilloscope.</li>
<li><b>Control Loop Rate</b> — 1000 Hz is mandatory for the stepper drive. Do not reduce below 500 Hz.</li>
</ul>

<p style='color:#fab387; margin-top:8px;'>⚠️ If you see frequent <i>Command Timeout</i> ESTOPs, <b>increase COMMAND_TIMEOUT_MS</b> to 500 ms while debugging.</p>
"""),

    # ── Flash Tab ──
    ("⚡  Flash Tab Guide", """
<h2 style='color:#cba6f7;'>Flash Tab</h2>
<p>Two-step pipeline: <b>Generate</b> the C config header, then <b>Compile &amp; Flash</b> to the Teensy.</p>

<h3 style='color:#89b4fa;'>Step 1 — Generate config.h</h3>
<ol>
<li>Click <b>Generate config.h</b>.</li>
<li>The file is written to <code>parol6_firmware/generated/config.h</code>.</li>
<li>A fingerprint hash is embedded — this lets you verify which config is on the board.</li>
</ol>

<h3 style='color:#89b4fa;'>Step 2 — Compile &amp; Flash</h3>
<ol>
<li>Make sure the Teensy is plugged in.</li>
<li>Click <b>Compile &amp; Flash</b>. PlatformIO runs inside the Docker container.</li>
<li>The board reboots and starts running the new firmware immediately.</li>
</ol>

<p style='color:#a6e3a1;'>✅ You do NOT need to press the Teensy reset button — the loader handles it automatically.</p>

<h3 style='color:#89b4fa;'>Troubleshooting</h3>
<ul>
<li><b>No device found</b> → Check <code>/dev/ttyACM0</code> exists and container has <code>--privileged</code>.</li>
<li><b>Build errors</b> → Read the output panel for the failing file and line number.</li>
<li><b>Flash OK but no serial</b> → Check baud rate matches (115200 by default).</li>
</ul>
"""),

    # ── Serial Tab ──
    ("💬  Serial Tab Guide", """
<h2 style='color:#cba6f7;'>Serial Tab</h2>
<p>Low-level character terminal to the Teensy. Use this to validate raw telemetry before starting ROS.</p>

<h3 style='color:#89b4fa;'>Expected output (firmware running)</h3>
<pre style='background:#11111b; padding:8px; border-radius:4px; color:#a6e3a1; font-size:11px;'>&lt;ACK,42,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0&gt;</pre>
<p>Format: <code>&lt;ACK, seq, p1..p6, v1..v6, lim_state&gt;</code></p>
<ul>
<li><b>seq</b> — monotonically increasing. Gaps = dropped packets.</li>
<li><b>p1..p6</b> — joint positions in radians.</li>
<li><b>v1..v6</b> — joint velocities in rad/s.</li>
<li><b>lim_state</b> — bitmask: bit 0=J1 triggered, bit 1=J2, … bit 5=J6.</li>
</ul>

<h3 style='color:#89b4fa;'>Sending commands manually</h3>
<pre style='background:#11111b; padding:8px; border-radius:4px; color:#89dceb; font-size:11px;'>&lt;HOME&gt;
&lt;ENABLE&gt;
&lt;0,0.000,0.000,0.000,0.000,0.000,0.000,0.0,0.0,0.0,0.0,0.0,0.0&gt;</pre>
<p><code>&lt;HOME&gt;</code> starts the homing sequence.<br>
<code>&lt;ENABLE&gt;</code> clears a SOFT_ESTOP state.<br>
The last format is a full position+velocity command.</p>
"""),

    # ── Limit Switches ──
    ("🔌  Limit Switches", """
<h2 style='color:#cba6f7;'>Limit Switch Integration</h2>

<h3 style='color:#89b4fa;'>Hardware overview</h3>
<p>PAROL6 uses <b>inductive proximity sensors</b> (NPN open-collector) wired through
<b>optical couplers</b> to Teensy digital inputs. The optocoupler provides:</p>
<ul>
<li>Electrical isolation from the 12–24 V sensor supply</li>
<li>Signal inversion (sensor active → coupler pulls pin LOW)</li>
<li>3.3 V level-shifting safe for Teensy GPIO</li>
</ul>

<h3 style='color:#89b4fa;'>Enabling a limit switch (step by step)</h3>
<ol>
<li>Go to <b>🔩 Joints</b> tab → click a joint row (e.g., J1).</li>
<li>The <b>Wiring Guide</b> panel at the bottom shows sensor-specific advice.</li>
<li>Set <b>Limit Type</b> → <code>INDUCTIVE_NPN</code>.</li>
<li>Pull and Polarity are <i>auto-filled</i>: INPUT_PULLUP + FALLING.</li>
<li>Set the GPIO <b>Pin</b> number (e.g., 20 for J1).</li>
<li>Check the <b>Enabled</b> checkbox in the Limit Pin column.</li>
<li>Go to <b>⚡ Flash</b> → Generate &amp; Flash.</li>
<li>In <b>💬 Serial</b> tab: manually bridge the sensor pin to GND. The <code>lim_state</code> field in ACK frames should show <code>1</code> (bit 0 set).</li>
<li>Send a motion command. The robot should NOT move (firmware FAULT triggered).</li>
</ol>

<h3 style='color:#89b4fa;'>Safety behaviour</h3>
<ul>
<li>Limit switch triggered <b>during normal operation</b> → immediate <code>FAULT</code> ESTOP.</li>
<li>Limit switch triggered <b>during homing</b> → expected; FSM transitions to backoff.</li>
<li>After an ESTOP: send <code>&lt;ENABLE&gt;</code> in the Serial tab OR re-launch the homing sequence.</li>
</ul>
"""),

    # ── Homing ──
    ("🏠  Homing Sequence", """
<h2 style='color:#cba6f7;'>Homing Sequence</h2>
<p>Homing moves each joint slowly toward its limit switch, backs off by
<code>HOMED_OFFSET</code> steps, then zeroes the encoder. The sequence follows
<code>HOMING_ORDER</code> (default: J4→J5→J6→J1→J2→J3 — wrist first).</p>

<h3 style='color:#89b4fa;'>Starting homing</h3>
<p>Click <b>🏠 Home All</b> in the <b>🕹 Jog</b> tab, or send <code>&lt;HOME&gt;</code> from the Serial tab. The firmware responds with:</p>
<pre style='background:#11111b; padding:8px; border-radius:4px; color:#a6e3a1; font-size:11px;'>HOMING_DONE</pre>
<p>or <code>HOMING_FAULT</code> if any axis times out before finding its switch.</p>

<h3 style='color:#89b4fa;'>Configuring homing parameters (Joints tab)</h3>
<table style='font-size:12px; border-collapse:collapse;'>
<tr style='color:#cba6f7;'><th align='left' style='padding:4px;'>Parameter</th><th align='left' style='padding:4px;'>Meaning</th></tr>
<tr><td style='padding:4px;'>Homing Speed (steps/s)</td><td>How fast the motor seeks the limit. Keep ≤500 steps/s initially.</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Homed Offset (steps)</td><td>Steps to back off after the limit triggers. Moves joint to the zero position.</td></tr>
<tr><td style='padding:4px;'>Home Offset Rad</td><td>Optional additional angular offset applied after zeroing.</td></tr>
</table>

<p style='color:#fab387; margin-top:8px;'>⚠️ Always test homing with <b>one joint at a time</b> before enabling the full sequence.
Watch for unexpected direction — if a joint moves <b>away</b> from the switch, invert <b>Dir Inv</b>.</p>
"""),

    # ── MoveIt/RViz ──
    ("🤖  MoveIt & RViz Guide", """
<h2 style='color:#cba6f7;'>MoveIt &amp; RViz — Real Robot Operation</h2>

<h3 style='color:#89b4fa;'>Architecture</h3>
<pre style='background:#11111b; padding:8px; border-radius:4px; color:#cdd6f4; font-size:11px;'>RViz (planning UI)
  └─► MoveIt (motion planner, collision checker)
       └─► ros2_control (JointTrajectoryController)
            └─► parol6_hardware (parol6_system.cpp)
                 └─► USB serial (LibSerial)
                      └─► Teensy 4.1 (parol6_firmware)
                           └─► Stepper motors + Encoders + Limit switches</pre>

<h3 style='color:#89b4fa;'>Launching (GUI Launch tab)</h3>
<table style='font-size:12px; border-collapse:collapse;'>
<tr style='color:#cba6f7;'><th align='left'>Mode</th><th align='left'>Script</th><th align='left'>Use for</th></tr>
<tr><td>Fake Hardware</td><td>launch_moveit_fake.sh</td><td>RViz planning without any robot</td></tr>
<tr style='background:#252535;'><td>Simulation</td><td>launch_moveit_with_gazebo.sh</td><td>Gazebo physics simulation</td></tr>
<tr><td>Real Hardware</td><td>launch_moveit_real_hw.sh</td><td>Physical robot via Teensy USB</td></tr>
</table>

<h3 style='color:#89b4fa;'>Before running Real Hardware mode</h3>
<ol>
<li>Flash firmware with limit switches enabled and tested.</li>
<li>Verify serial port in <code>PAROL6.ros2_control.xacro</code> matches your device.</li>
<li>Run homing sequence successfully at least once.</li>
<li>In RViz: drag the robot to a target pose → <b>Plan &amp; Execute</b>.</li>
</ol>

<h3 style='color:#89b4fa;'>If the robot stops mid-trajectory</h3>
<ul>
<li>Check <b>⚠️ Faults</b> tab for the ESTOP reason.</li>
<li>Most common: <i>Limit Switch J1</i> or <i>Command Timeout</i>.</li>
<li>Send <code>&lt;ENABLE&gt;</code> in Serial tab to clear and re-home.</li>
</ul>
"""),

    # ── Testing Protocol ──
    ("🔬  Testing Protocol", """
<h2 style='color:#cba6f7;'>Testing Protocol — First Power-On</h2>
<p>Follow these phases <b>in order</b>. Do not skip. Each phase verifies the next is safe.</p>

<table style='width:100%; border-collapse:collapse; font-size:12px;'>
<tr style='background:#313244; color:#cba6f7;'>
  <th align='left' style='padding:4px;'>Phase</th>
  <th align='left' style='padding:4px;'>What to do</th>
  <th align='left' style='padding:4px;'>Pass criterion</th>
</tr>
<tr><td style='padding:4px;'>0 — Pre-power</td><td>E-stop test, cables seated, robot at mid-range</td><td>Hardware confirmed safe before power-on</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>1 — Serial link</td><td>Flash → Connect serial → send &lt;ENABLE&gt;</td><td>ACK packets appear at 25 Hz in Serial tab</td></tr>
<tr><td style='padding:4px;'>2 — Direction check</td><td>Jog each joint 0.1 rad via Jog tab</td><td>Joint moves correct direction, encoder sign matches</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>3 — ROS interface</td><td>Launch real_hw, confirm controllers active, check joint_states</td><td>25 Hz joint_states, pose matches physical robot</td></tr>
<tr><td style='padding:4px;'>4 — First trajectory</td><td>Plan &lt;5 cm move in RViz → Execute</td><td>Goal reached, no FAULT, no STALE_CMD</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>5 — Homing</td><td>Enable limit switches in config → verify lim_state bit → Home All</td><td>HOMING_DONE, post-home pose matches HOME_OFFSETS_RAD</td></tr>
</table>

<h3 style='color:#89b4fa;'>Serial strings to recognise</h3>
<table style='font-size:12px; border-collapse:collapse;'>
<tr style='color:#cba6f7;'><th align='left'>String</th><th align='left'>Meaning</th></tr>
<tr><td><code>HOMING_DONE</code></td><td>All axes found their switch and zeroed.</td></tr>
<tr style='background:#252535;'><td><code>HOMING_FAULT</code></td><td>An axis timed out — check switch wiring and Dir Inv.</td></tr>
<tr><td><code>STALE_CMD</code></td><td>Command rejected (seq not newer). Normal after reconnect; send &lt;ENABLE&gt;.</td></tr>
</table>

<p style='color:#fab387; margin-top:8px;'>📄 Full step-by-step guide: <b>docs/FIRST_POWER_ON_CHECKLIST.md</b></p>
"""),

    # ── Troubleshooting ──
    ("🛠️  Troubleshooting", """
<h2 style='color:#cba6f7;'>Troubleshooting</h2>

<table style='width:100%; border-collapse:collapse; font-size:12px;'>
<tr style='background:#313244; color:#cba6f7;'>
  <th align='left' style='padding:4px;'>Symptom</th>
  <th align='left' style='padding:4px;'>Likely cause</th>
  <th align='left' style='padding:4px;'>Fix</th>
</tr>
<tr><td style='padding:4px;'>No ACK frames in Serial tab</td><td>Wrong port or baud rate</td><td>Check <code>dmesg | grep tty</code>, set to 115200; send <code>&lt;ENABLE&gt;</code></td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Robot moves tiny bit then stops</td><td>JTC constraint violation or FAULT/ESTOP</td><td>Check Faults tab; set <code>trajectory: 0.05</code> in ros2_controllers.yaml</td></tr>
<tr><td style='padding:4px;'><code>STALE_CMD</code> appears in serial</td><td>ROS seq counter reset (normal after reconnect)</td><td>Send <code>&lt;ENABLE&gt;</code> to resync, then re-launch controller</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Joint position jumps on startup</td><td>Interpolator seeded in motor space</td><td>Flash latest firmware — already fixed</td></tr>
<tr><td style='padding:4px;'>Limit switch always triggered</td><td>Pull resistor wrong or optocoupler wired backwards</td><td>Check sensor type → auto-suggest pull resistor; verify lim_state bits</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'><code>HOMING_FAULT</code></td><td>Limit switch not connected or axis direction wrong</td><td>Enable one joint, bridge pin to GND, check lim_state bit is 1; check Dir Inv</td></tr>
<tr><td style='padding:4px;'>MoveIt: goal aborted or timed out</td><td>Hardware interface not publishing states, or tolerance too tight</td><td>Check serial connection; run Fake HW first; set trajectory: 0.05 in controllers yaml</td></tr>
<tr style='background:#252535;'><td style='padding:4px;'>Encoder position drifts</td><td>Anti-glitch filter disabled or wrong STEPS_PER_RAD</td><td>Reload correct gear ratio in Joints tab → reflash</td></tr>
<tr><td style='padding:4px;'>Gazebo white screen / robot not spawning</td><td>URDF includes stale gz plugin</td><td>Use <code>libign_ros2_control-system.so</code> in PAROL6.urdf gazebo plugin</td></tr>
</table>
"""),
]


class DocsTab(QWidget):
    """In-app documentation browser with a sidebar TOC and rich-text content panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ── Left: Table of Contents ────────────────────────────────
        left = QWidget()
        left.setFixedWidth(230)
        left.setStyleSheet("background:#181825; border-right:1px solid #45475a;")
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        toc_title = QLabel("  📖  Contents")
        toc_title.setStyleSheet(
            "font-size:13px; font-weight:bold; color:#cba6f7; "
            "padding:12px 8px; border-bottom:1px solid #45475a;"
        )
        left_lay.addWidget(toc_title)

        self._toc = QListWidget()
        self._toc.setStyleSheet("""
            QListWidget {
                background: #181825;
                border: none;
                color: #cdd6f4;
                font-size: 12px;
            }
            QListWidget::item { padding: 8px 12px; border-bottom: 1px solid #313244; }
            QListWidget::item:selected { background: #313244; color: #cba6f7; font-weight: bold; }
            QListWidget::item:hover { background: #252535; }
        """)
        for title, _ in SECTIONS:
            self._toc.addItem(QListWidgetItem(title))
        left_lay.addWidget(self._toc)

        # ── Right: Content panel ───────────────────────────────────
        self._content = QLabel()
        self._content.setTextFormat(Qt.TextFormat.RichText)
        self._content.setWordWrap(True)
        self._content.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._content.setStyleSheet(
            "background:#1e1e2e; color:#cdd6f4; font-size:13px; padding:24px 28px; line-height:1.6;"
        )
        self._content.setOpenExternalLinks(True)

        scroll = QScrollArea()
        scroll.setWidget(self._content)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #1e1e2e; }")

        splitter.addWidget(left)
        splitter.addWidget(scroll)
        splitter.setSizes([230, 900])

        root.addWidget(splitter)

        # Wire selection
        self._toc.currentRowChanged.connect(self._on_section_changed)
        self._toc.setCurrentRow(0)

    def _on_section_changed(self, row: int) -> None:
        if 0 <= row < len(SECTIONS):
            self._content.setText(SECTIONS[row][1])
