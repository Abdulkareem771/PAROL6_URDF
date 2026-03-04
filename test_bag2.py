import sys, os, subprocess
cmd = ['ros2', 'bag', 'play', '/workspace/rosbag2_2026_01_26-23_26_59', '--loop']
full_cmd = "source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && exec " + " ".join(cmd)
env = os.environ.copy()
env.pop('QT_PLUGIN_PATH', None)
env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
proc = subprocess.Popen(
    ['bash', '-c', full_cmd], 
    env=env,
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT,
    text=True, bufsize=1
)
try:
    stdout, _ = proc.communicate(timeout=3)
    print("OUTPUT:", stdout)
except subprocess.TimeoutExpired:
    proc.kill()
    stdout, _ = proc.communicate()
    print("TIMEOUT OUTPUT:", stdout)
