"""
Persistent Auto-Reconnecting Public Tunnel Manager
Monitors and maintains continuous SSH public reverse tunneling for Streamlit (Port 8501).
Saves live public URL to data/public_url.txt and automatically reconnects on dropouts.
"""

import os
import re
import sys
import time
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URL_FILE = os.path.join(PROJECT_ROOT, "data", "public_url.txt")
os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

def run_tunnel():
    print("[*] Starting persistent tunnel daemon on port 8501...")
    
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=20",
        "-o", "ServerAliveCountMax=6",
        "-R", "80:localhost:8501",
        "nokey@localhost.run"
    ]

    while True:
        try:
            print("[*] Spawning SSH reverse tunnel...")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in iter(proc.stdout.readline, ''):
                sys.stdout.write(line)
                sys.stdout.flush()
                
                # Match https://*.lhr.life or similar URLs
                match = re.search(r'https://[a-zA-Z0-9.-]+\.lhr\.life', line)
                if match:
                    url = match.group(0)
                    print(f"\n==================================================")
                    print(f"[+] ACTIVE LIVE PUBLIC URL: {url}")
                    print(f"==================================================\n")
                    with open(URL_FILE, "w", encoding="utf-8") as f:
                        f.write(url)

            proc.wait()
            print("[!] Tunnel disconnected. Reconnecting in 3 seconds...")
            time.sleep(3)

        except Exception as e:
            print(f"[-] Tunnel error: {e}. Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    run_tunnel()
