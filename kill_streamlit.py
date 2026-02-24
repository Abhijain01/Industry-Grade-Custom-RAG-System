import psutil
import os
import signal

# Loop over all processes
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmd = p.info.get('cmdline')
        if cmd and 'streamlit' in ' '.join(cmd).lower():
            print(f"Killing PID {p.info['pid']}")
            # Force kill on Windows
            p.kill()
    except Exception as e:
        pass
