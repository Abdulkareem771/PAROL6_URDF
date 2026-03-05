import subprocess
import os

pkg_dir = os.path.expanduser("~/.platformio/packages")
if os.path.exists(pkg_dir):
    try:
        output = subprocess.check_output("find . -name '*.h' -exec grep -H 'SerialUSB' {} +", shell=True, cwd=pkg_dir)
        print(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print("Grep failed or found nothing:", e)
