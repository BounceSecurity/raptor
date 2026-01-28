#!/usr/bin/env python3
"""
Kill dangling OpenCode instances created by RAPTOR.

This script identifies and terminates OpenCode server processes that were
started by RAPTOR, but only those. It uses process command-line arguments
to identify RAPTOR-managed instances.

RAPTOR starts OpenCode with: opencode serve --port <port> --hostname 127.0.0.1
"""

import sys
import platform
import subprocess
import signal
import os
import time
from pathlib import Path

# Add parent to path for core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import RaptorConfig
from core.logging import get_logger

logger = get_logger()


def find_raptor_opencode_processes():
    """
    Find OpenCode processes started by RAPTOR.
    
    RAPTOR starts opencode with: opencode serve --port <port> --hostname 127.0.0.1
    
    Returns:
        List of (pid, port, cmdline) tuples
    """
    processes = []
    system = platform.system().lower()
    
    # Binary name based on OS (same logic as server_manager.py)
    binary_name = "opencode.exe" if system == "windows" else "opencode"
    
    if system == "windows":
        try:
            # Use tasklist to find processes, then wmic for command line
            # First, get PIDs of opencode.exe processes
            result = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {binary_name}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            pids = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    # CSV format: "opencode.exe","1234","Session Name","Session#","Mem Usage"
                    parts = line.split('","')
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1].strip('"'))
                            pids.append(pid)
                        except (ValueError, IndexError):
                            continue
            
            # Get command line for each PID
            for pid in pids:
                try:
                    result = subprocess.run(
                        ["wmic", "process", "where", f"ProcessId={pid}", "get", "CommandLine", "/format:list"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    cmdline = ""
                    for line in result.stdout.split('\n'):
                        if line.startswith("CommandLine="):
                            cmdline = line.split("=", 1)[1].strip()
                            break
                    
                    # Check if it matches RAPTOR pattern AND has RAPTOR_MANAGED env var
                    # Get environment variables for this process
                    env_result = subprocess.run(
                        ["wmic", "process", "where", f"ProcessId={pid}", "get", "Environment", "/format:list"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    has_raptor_marker = False
                    for env_line in env_result.stdout.split('\n'):
                        if 'RAPTOR_MANAGED=1' in env_line:
                            has_raptor_marker = True
                            break
                    
                    # Must match RAPTOR pattern AND have the marker
                    if (cmdline and "serve" in cmdline and "--hostname" in cmdline and 
                        "127.0.0.1" in cmdline and has_raptor_marker):
                        # Extract port
                        port = None
                        if "--port" in cmdline:
                            try:
                                port_idx = cmdline.index("--port")
                                port_str = cmdline[port_idx:].split()[1]
                                port = int(port_str)
                            except (ValueError, IndexError):
                                pass
                        
                        processes.append((pid, port, cmdline))
                except Exception as e:
                    logger.debug(f"Failed to get command line for PID {pid}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to query processes on Windows: {e}")
            return []
    else:
        # Linux/macOS - use ps
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            for line in result.stdout.split('\n'):
                if binary_name in line and 'serve' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            cmdline = ' '.join(parts[10:])  # Command starts at index 10
                            
                            # Check if it matches RAPTOR pattern
                            # On Unix, we can check environment via /proc/<pid>/environ
                            has_raptor_marker = False
                            try:
                                environ_path = Path(f"/proc/{pid}/environ")
                                if environ_path.exists():
                                    env_data = environ_path.read_bytes()
                                    if b'RAPTOR_MANAGED=1' in env_data:
                                        has_raptor_marker = True
                            except Exception:
                                # If we can't read /proc, fall back to command line pattern only
                                # This is less safe but better than nothing
                                pass
                            
                            # Must match RAPTOR pattern AND have the marker (or pattern match if /proc unavailable)
                            if "--hostname" in cmdline and "127.0.0.1" in cmdline and has_raptor_marker:
                                # Extract port
                                port = None
                                if "--port" in cmdline:
                                    try:
                                        port_idx = cmdline.index("--port")
                                        port_str = cmdline[port_idx:].split()[1]
                                        port = int(port_str)
                                    except (ValueError, IndexError):
                                        pass
                                
                                processes.append((pid, port, cmdline))
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            logger.error(f"Failed to query processes on Unix: {e}")
            return []
    
    return processes


def kill_process(pid):
    """Kill a process by PID."""
    system = platform.system().lower()
    
    if system == "windows":
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to kill process {pid}: {e}")
            return False
    else:
        # Linux/macOS
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait a bit, then force kill if still running
            time.sleep(1)
            try:
                os.kill(pid, 0)  # Check if still exists
                os.kill(pid, signal.SIGKILL)  # Force kill
            except ProcessLookupError:
                pass  # Already dead
            return True
        except ProcessLookupError:
            return True  # Already dead
        except Exception as e:
            logger.error(f"Failed to kill process {pid}: {e}")
            return False


def main():
    """Main entry point."""
    print("Searching for RAPTOR-managed OpenCode processes...")
    
    processes = find_raptor_opencode_processes()
    
    if not processes:
        print("No RAPTOR-managed OpenCode processes found.")
        return 0
    
    print(f"\nFound {len(processes)} RAPTOR-managed OpenCode process(es):")
    for pid, port, cmdline in processes:
        port_str = f"port {port}" if port else "unknown port"
        print(f"  PID {pid} ({port_str}): {cmdline[:80]}...")
    
    print("\nKilling processes...")
    killed = 0
    for pid, port, cmdline in processes:
        if kill_process(pid):
            print(f"  [OK] Killed PID {pid}")
            killed += 1
        else:
            print(f"  [FAIL] Failed to kill PID {pid}")
    
    print(f"\nKilled {killed}/{len(processes)} process(es).")
    return 0 if killed == len(processes) else 1


if __name__ == "__main__":
    sys.exit(main())
