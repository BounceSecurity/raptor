#!/usr/bin/env python3
"""
OpenCode Server Manager

Manages the lifecycle of the OpenCode server process.
Automatically starts server if not running, handles cleanup on exit.
"""

import atexit
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TYPE_CHECKING
from urllib.parse import urlparse

import requests

from core.config import RaptorConfig
from core.logging import get_logger
from .exceptions import OpenCodeConnectionError

logger = get_logger()

# Global registry to track managed servers and avoid duplicate atexit handlers
_managed_servers: set = set()
_atexit_registered = False
_singleton_manager = None  # Will be OpenCodeServerManager instance


def _global_shutdown():
    """Global shutdown handler for all managed servers."""
    for server_manager in list(_managed_servers):
        try:
            server_manager.shutdown()
        except Exception as e:
            logger.warning(f"Error shutting down server manager: {e}")


class OpenCodeServerManager:
    """Manages OpenCode server lifecycle."""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        """
        Initialize server manager.
        
        Args:
            server_url: OpenCode server URL (default: http://localhost:8080)
        """
        self.server_url = server_url
        self.process: Optional[subprocess.Popen] = None
        self.managed = False  # True if we started it
        self._port = None  # Cached port number
        self.repo_path: Optional[Path] = None  # Repository path for server cwd
        
        # Register global cleanup handler once
        global _atexit_registered
        if not _atexit_registered:
            atexit.register(_global_shutdown)
            _atexit_registered = True
        
        # Register this instance for cleanup
        _managed_servers.add(self)
    
    def ensure_running(self, repo_path: Optional[Path] = None) -> bool:
        """
        Ensure OpenCode server is running.
        
        Args:
            repo_path: Optional repository path to set as working directory
        
        Returns True if server is available (either already running or we started it).
        Raises OpenCodeConnectionError if server cannot be started or connected to.
        """
        # Store repo_path if provided (for server startup)
        if repo_path:
            self.repo_path = Path(repo_path).resolve()
        
        # Check if server is already running
        if self._check_server_health():
            logger.debug("OpenCode server already running")
            return True
        
        # Try to start server
        if self._start_server():
            self.managed = True
            logger.info("Started OpenCode server (RAPTOR-managed)")
            return True
        
        # If we get here, we couldn't start the server
        raise OpenCodeConnectionError(
            f"Cannot start or connect to OpenCode server at {self.server_url}. "
            "Please ensure opencode binary is installed and in PATH, "
            "or set OPENCODE_SERVER_BINARY environment variable."
        )
    
    def _check_server_health(self) -> bool:
        """Check if OpenCode HTTP server is responding."""
        try:
            # Check if HTTP server is responding
            import requests
            health_url = f"{self.server_url.rstrip('/')}/health"
            response = requests.get(health_url, timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Server health check failed: {e}")
            return False
    
    def _start_server(self) -> bool:
        """Start OpenCode server as subprocess."""
        # Find opencode binary
        server_binary = self._find_server_binary()
        if not server_binary:
            logger.error("opencode binary not found")
            return False
        
        # Get port from URL
        port = self._get_port()
        
        # Start OpenCode HTTP server (opencode serve)
        # The SDK uses HTTP API, so we need the server running
        try:
            logger.info(f"Starting OpenCode HTTP server on port {port}...")
            # Set environment variable to mark this as RAPTOR-managed
            env = os.environ.copy()
            env['RAPTOR_MANAGED'] = '1'
            env['RAPTOR_PID'] = str(os.getpid())  # Store parent PID for additional verification
            
            self.process = subprocess.Popen(
                [str(server_binary), "serve", "--port", str(port), "--hostname", "127.0.0.1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent
                cwd=str(self.repo_path) if hasattr(self, 'repo_path') else None,
                env=env
            )
            
            # Wait for server to be ready (with timeout)
            timeout = RaptorConfig.OPENCODE_STARTUP_TIMEOUT
            for _ in range(timeout):
                time.sleep(1)
                if self._check_server_health():
                    logger.info(f"OpenCode server started successfully on port {port}")
                    return True
            
            # Server didn't start in time
            logger.error(f"OpenCode server did not start within {timeout} seconds")
            if self.process:
                self.process.kill()
                self.process = None
            return False
        except Exception as e:
            logger.error(f"Failed to start OpenCode server: {e}")
            if self.process:
                self.process.kill()
                self.process = None
            return False
    
    def _find_server_binary(self) -> Optional[Path]:
        """Find opencode binary, auto-installing if needed."""
        # Check custom path first
        custom_path = RaptorConfig.OPENCODE_SERVER_BINARY
        if custom_path:
            path = Path(custom_path)
            if path.exists() and path.is_file():
                return path
        
        # Binary name is "opencode" on Linux/macOS, "opencode.exe" on Windows
        binary_name = "opencode.exe" if platform.system().lower() == "windows" else "opencode"
        
        # Check common locations
        possible_paths = [
            shutil.which("opencode"),
            shutil.which("opencode.exe"),
            Path.home() / ".local/bin" / binary_name,
            Path("/usr/local/bin") / binary_name,
            Path("/usr/bin") / binary_name,
        ]
        
        for path in possible_paths:
            if path and Path(path).exists():
                return Path(path)
        
        # Check RAPTOR-managed binary directory
        raptor_binary = RaptorConfig.OPENCODE_BIN_DIR / binary_name
        if raptor_binary.exists() and raptor_binary.is_file():
            return raptor_binary
        
        # Auto-install if enabled
        if RaptorConfig.OPENCODE_AUTO_INSTALL:
            logger.info("OpenCode server binary not found, attempting auto-install...")
            installed = self._install_server_binary()
            if installed and raptor_binary.exists():
                return raptor_binary
        
        return None
    
    def _install_server_binary(self) -> bool:
        """
        Automatically download and install OpenCode server binary.
        
        Returns:
            True if installation successful, False otherwise
        """
        try:
            # Determine platform
            system = platform.system().lower()
            machine = platform.machine().lower()
            
            # Debug logging
            logger.debug(f"Platform detection: system={system}, machine={machine}")
            
            # Map to OpenCode release asset names
            # Format: opencode-{platform}-{arch}.{ext} (binary inside is named "opencode" or "opencode.exe")
            if system == "linux":
                if machine in ["x86_64", "amd64"]:
                    arch = "x86_64"
                elif machine in ["aarch64", "arm64"]:
                    arch = "arm64"
                else:
                    logger.error(f"Unsupported architecture: {machine}")
                    return False
                ext = "tar.gz"
            elif system == "darwin":  # macOS
                if machine in ["x86_64", "amd64"]:
                    arch = "x86_64"
                elif machine in ["aarch64", "arm64"]:
                    arch = "arm64"
                else:
                    logger.error(f"Unsupported architecture: {machine}")
                    return False
                ext = "tar.gz"
            elif system == "windows":
                arch = "x86_64"  # Assume 64-bit
                ext = "zip"
            else:
                logger.error(f"Unsupported platform: {system}")
                return False
            
            # Get latest release info from GitHub API
            logger.info("Fetching latest OpenCode release information...")
            try:
                api_url = "https://api.github.com/repos/anomalyco/opencode/releases/latest"
                api_response = requests.get(api_url, timeout=10)
                api_response.raise_for_status()
                release_data = api_response.json()
                
                # Log all available assets for debugging
                all_assets = release_data.get("assets", [])
                logger.debug(f"Found {len(all_assets)} assets in release")
                server_assets = [a["name"] for a in all_assets if "server" in a["name"].lower()]
                logger.debug(f"Server assets: {server_assets}")
                
                # Find the asset matching our platform
                download_url = None
                matched_asset = None
                
                # First, try to find exact matches from release assets
                # OpenCode naming pattern: opencode-{platform}-{arch}.tar.gz (or .zip for Windows)
                # Examples: opencode-linux-x64.tar.gz, opencode-linux-arm64.tar.gz
                # Note: Server binaries do NOT have "server" in the name, only desktop apps have "desktop"
                for asset in all_assets:
                    asset_name = asset["name"]
                    asset_name_lower = asset_name.lower()
                    
                    # CRITICAL: Skip desktop apps, .dmg, .rpm, .deb, .exe, .sig files
                    if ("desktop" in asset_name_lower or 
                        ".app" in asset_name_lower or 
                        asset_name_lower.endswith(".dmg") or
                        asset_name_lower.endswith(".rpm") or
                        asset_name_lower.endswith(".deb") or
                        asset_name_lower.endswith(".exe") or
                        asset_name_lower.endswith(".sig") or
                        asset_name_lower == "latest.json"):
                        logger.debug(f"Skipping non-server asset: {asset_name}")
                        continue
                    
                    # Must start with "opencode-" and be an archive (.tar.gz or .zip)
                    if not asset_name_lower.startswith("opencode-") or not (asset_name_lower.endswith(".tar.gz") or asset_name_lower.endswith(".zip")):
                        logger.debug(f"Skipping asset (wrong format): {asset_name}")
                        continue
                    
                    # Check if it matches our platform (explicit platform check)
                    platform_match = False
                    if system == "linux":
                        # Linux: must contain "linux" and NOT contain "darwin" or "windows"
                        platform_match = ("linux" in asset_name_lower) and ("darwin" not in asset_name_lower) and ("windows" not in asset_name_lower) and ("macos" not in asset_name_lower)
                    elif system == "darwin":
                        # macOS: must contain "darwin" or "macos" and NOT contain "linux" or "windows"
                        platform_match = (("darwin" in asset_name_lower or "macos" in asset_name_lower) and 
                                         ("linux" not in asset_name_lower) and ("windows" not in asset_name_lower))
                    elif system == "windows":
                        # Windows: must contain "windows" or "win" and NOT contain "linux" or "darwin"
                        platform_match = (("windows" in asset_name_lower or "win" in asset_name_lower) and 
                                         ("linux" not in asset_name_lower) and ("darwin" not in asset_name_lower))
                    
                    if not platform_match:
                        logger.debug(f"Platform mismatch for {asset_name}: system={system}")
                        continue
                    
                    # Check architecture match
                    # OpenCode uses: x64 (for x86_64), arm64 (for ARM64/aarch64)
                    arch_match = False
                    if arch == "x86_64":
                        # Accept x64, x86_64, or amd64
                        arch_match = ("x64" in asset_name_lower or "x86_64" in asset_name_lower or "amd64" in asset_name_lower)
                    elif arch == "arm64":
                        # Accept arm64 or aarch64
                        arch_match = ("arm64" in asset_name_lower or "aarch64" in asset_name_lower)
                    
                    if not arch_match:
                        logger.debug(f"Architecture mismatch for {asset_name}: arch={arch}")
                        continue
                    
                    # Prefer non-musl, non-baseline versions, but accept them if that's all that's available
                    # We'll prioritize exact matches first
                    is_preferred = "musl" not in asset_name_lower and "baseline" not in asset_name_lower
                    
                    # Found a match!
                    if not download_url or is_preferred:
                        download_url = asset["browser_download_url"]
                        matched_asset = asset_name
                        if is_preferred:
                            logger.info(f"Found matching server binary: {matched_asset}")
                            break  # Found preferred version, stop searching
                        else:
                            logger.debug(f"Found fallback server binary: {matched_asset} (continuing to search for preferred)")
                
                if download_url:
                    logger.info(f"Selected server binary: {matched_asset}")
                
                if not download_url:
                    # Fallback: try to construct URL from release tag
                    tag = release_data.get("tag_name", "latest")
                    # OpenCode naming: opencode-{platform}-{arch}.{ext}
                    # Map our arch to OpenCode's naming (x86_64 -> x64, arm64 -> arm64)
                    opencode_arch = "x64" if arch == "x86_64" else "arm64"
                    possible_patterns = [
                        f"opencode-{system}-{opencode_arch}.{ext}",  # e.g., opencode-linux-x64.tar.gz
                        f"opencode-{system}-{opencode_arch}-musl.{ext}",  # musl variant
                        f"opencode-{system}-{opencode_arch}-baseline.{ext}",  # baseline variant
                    ]
                    
                    for pattern in possible_patterns:
                        fallback_url = f"https://github.com/anomalyco/opencode/releases/download/{tag}/{pattern}"
                        # Test if URL exists (HEAD request)
                        try:
                            head_response = requests.head(fallback_url, timeout=5, allow_redirects=True)
                            if head_response.status_code == 200:
                                download_url = fallback_url
                                logger.info(f"Using fallback URL pattern: {pattern}")
                                break
                        except:
                            continue
                
                if not download_url:
                    # List available assets for debugging
                    available_assets = [a["name"] for a in release_data.get("assets", [])]
                    # Server binaries are named opencode-{platform}-{arch}.tar.gz (no "server" in name)
                    server_assets = [a for a in available_assets 
                                   if a.lower().startswith("opencode-") 
                                   and (a.lower().endswith(".tar.gz") or a.lower().endswith(".zip"))
                                   and "desktop" not in a.lower()
                                   and not a.lower().endswith(".sig")
                                   and not a.lower().endswith(".dmg")
                                   and not a.lower().endswith(".rpm")
                                   and not a.lower().endswith(".deb")
                                   and not a.lower().endswith(".exe")]
                    platform_assets = [a for a in server_assets if system in a.lower()]
                    
                    logger.error(f"Could not find OpenCode server binary for {system}-{arch}")
                    logger.error(f"Total assets: {len(available_assets)}")
                    logger.error(f"Server assets (excluding desktop/packages): {len(server_assets)}")
                    if server_assets:
                        logger.error(f"Server assets: {', '.join(server_assets[:10])}")
                    if platform_assets:
                        logger.error(f"{system.capitalize()} server assets: {', '.join(platform_assets[:10])}")
                    
                    raise ValueError(
                        f"Could not find OpenCode server binary for {system}-{arch}. "
                        f"Found {len(server_assets)} server assets (excluding desktop/packages) out of {len(available_assets)} total. "
                        f"{system.capitalize()} server assets: {len(platform_assets)}. "
                        f"Please check https://github.com/anomalyco/opencode/releases/latest "
                        f"and ensure a {system} server binary is available."
                    )
                
                # Validate that we're not downloading a desktop app
                if download_url and ("desktop" in download_url.lower() or ".app" in download_url.lower()):
                    logger.error(f"ERROR: Download URL points to desktop app, not server: {download_url}")
                    raise ValueError("Download URL is a desktop app, not a server binary. This should not happen.")
                
                release_url = download_url
            except Exception as e:
                logger.warning(f"Failed to query GitHub API: {e}, using fallback URL")
                # Fallback to direct download URL (but this might not exist)
                # OpenCode naming: opencode-{platform}-{arch}.{ext}
                opencode_arch = "x64" if arch == "x86_64" else "arm64"
                release_url = f"https://github.com/anomalyco/opencode/releases/latest/download/opencode-{system}-{opencode_arch}.{ext}"
            
            # Create bin directory
            RaptorConfig.OPENCODE_BIN_DIR.mkdir(parents=True, exist_ok=True)
            
            # Validate URL before downloading
            if "desktop" in release_url.lower() or ".app" in release_url.lower():
                raise ValueError(f"Download URL appears to be a desktop app: {release_url}")
            
            # Determine if this is a standalone binary or archive
            # Check Content-Type header or file extension
            logger.info(f"Downloading OpenCode server from {release_url}...")
            response = requests.get(release_url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Double-check the downloaded file name from Content-Disposition or URL
            content_disposition = response.headers.get("Content-Disposition", "")
            if "desktop" in content_disposition.lower() or ".app" in content_disposition.lower():
                raise ValueError(f"Downloaded file appears to be a desktop app: {content_disposition}")
            
            content_type = response.headers.get("Content-Type", "").lower()
            is_archive = ext in ["tar.gz", "zip"] or "archive" in content_type or "compressed" in content_type
            
            if is_archive:
                # Handle archive (extract first)
                download_path = RaptorConfig.OPENCODE_BIN_DIR / f"opencode.{ext}"
                # Binary name is "opencode" on Linux/macOS, "opencode.exe" on Windows
                binary_name = "opencode.exe" if system == "windows" else "opencode"
                binary_path = RaptorConfig.OPENCODE_BIN_DIR / binary_name
                
                # Save archive to file
                with open(download_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded archive to {download_path}")
                
                # Extract archive
                if ext == "tar.gz":
                    logger.info("Extracting tar.gz archive...")
                    with tarfile.open(download_path, "r:gz") as tar:
                        # Find the binary in the archive
                        members = tar.getmembers()
                        binary_member = None
                        
                        # Look for the binary (named "opencode" on Linux/macOS)
                        for member in members:
                            name_lower = member.name.lower()
                            # Skip desktop apps, directories, and symlinks
                            if ".app" in name_lower or member.isdir() or member.issym():
                                continue
                            # Look for binary named "opencode" (not "opencode-server")
                            if (member.name == "opencode" or 
                                member.name.endswith("/opencode")):
                                binary_member = member
                                logger.debug(f"Found binary in archive: {member.name}")
                                break
                        
                        if binary_member:
                            # Extract with proper name
                            original_name = binary_member.name
                            binary_member.name = binary_name
                            tar.extract(binary_member, RaptorConfig.OPENCODE_BIN_DIR)
                            extracted_path = RaptorConfig.OPENCODE_BIN_DIR / binary_name
                            if extracted_path.exists() and extracted_path.is_file():
                                binary_path = extracted_path
                                logger.debug(f"Extracted binary from {original_name} -> {binary_name}")
                            else:
                                logger.warning(f"Extracted file not found at expected path, trying full extract...")
                                binary_member = None  # Fall through to full extract
                        
                        if not binary_member or not binary_path.exists():
                            # Try extracting all and finding the binary
                            logger.debug("Extracting entire archive to find binary...")
                            tar.extractall(RaptorConfig.OPENCODE_BIN_DIR)
                            # Look for binary in extracted files (skip .app directories)
                            for extracted_file in RaptorConfig.OPENCODE_BIN_DIR.rglob("*"):
                                if extracted_file.is_file() and not extracted_file.is_symlink():
                                    name_lower = str(extracted_file).lower()
                                    # Skip desktop apps
                                    if ".app" in name_lower or "desktop" in name_lower:
                                        continue
                                    # Check if it's the binary (opencode or opencode.exe)
                                    if extracted_file.name == binary_name:
                                        if extracted_file != binary_path:
                                            if binary_path.exists():
                                                binary_path.unlink()  # Remove old file
                                            extracted_file.rename(binary_path)
                                            logger.info(f"Found and renamed binary: {extracted_file} -> {binary_path}")
                                        break
                elif ext == "zip":
                    logger.info("Extracting zip archive...")
                    with zipfile.ZipFile(download_path, "r") as zip_ref:
                        # Find binary in zip (opencode.exe on Windows, opencode on macOS)
                        found_binary_name = None
                        for name in zip_ref.namelist():
                            if name == binary_name or name.endswith(f"/{binary_name}"):
                                found_binary_name = name
                                break
                        
                        if found_binary_name:
                            zip_ref.extract(found_binary_name, RaptorConfig.OPENCODE_BIN_DIR)
                            extracted = RaptorConfig.OPENCODE_BIN_DIR / found_binary_name
                            if extracted.exists():
                                if extracted != binary_path:
                                    if binary_path.exists():
                                        binary_path.unlink()
                                    extracted.rename(binary_path)
                                logger.info(f"Extracted binary: {found_binary_name} -> {binary_name}")
                        else:
                            # Extract all and find binary
                            logger.debug("Extracting entire archive to find binary...")
                            zip_ref.extractall(RaptorConfig.OPENCODE_BIN_DIR)
                            for extracted_file in RaptorConfig.OPENCODE_BIN_DIR.rglob(binary_name):
                                if extracted_file.is_file():
                                    if extracted_file != binary_path:
                                        if binary_path.exists():
                                            binary_path.unlink()
                                        extracted_file.rename(binary_path)
                                    logger.info(f"Found and renamed binary: {extracted_file} -> {binary_path}")
                                    break
                
                    # Clean up archive
                    download_path.unlink()
            else:
                # Standalone binary - save directly and run in place
                # Binary name is already set above: "opencode.exe" on Windows, "opencode" on Linux/macOS
                binary_path = RaptorConfig.OPENCODE_BIN_DIR / binary_name
                
                # Save binary directly
                with open(binary_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Downloaded standalone binary to {binary_path}")
            
            # Make executable (Unix-like systems)
            if system != "windows":
                binary_path.chmod(binary_path.stat().st_mode | stat.S_IEXEC)
            
            # Verify binary exists
            if not binary_path.exists():
                logger.error("Binary not found after download/extraction")
                return False
            
            # Test binary (basic version check)
            try:
                result = subprocess.run(
                    [str(binary_path), "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"OpenCode server ready: {binary_path}")
                    logger.info(f"Version: {result.stdout.decode().strip()}")
                    return True
            except Exception as e:
                logger.warning(f"Version check failed, but binary exists: {e}")
                # Binary exists, assume it's valid
                logger.info(f"OpenCode server ready: {binary_path}")
                return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download OpenCode server: {e}")
            logger.info("You can manually install OpenCode server from: https://github.com/anomalyco/opencode/releases")
            logger.info("Or set OPENCODE_AUTO_INSTALL=false to disable auto-installation")
            return False
        except Exception as e:
            logger.error(f"Failed to install OpenCode server: {e}")
            return False
        
        return False
    
    def _get_port(self) -> int:
        """Get port number from server URL."""
        if self._port is not None:
            return self._port
        
        try:
            parsed = urlparse(self.server_url)
            port = parsed.port
            if port is None:
                # Default port based on scheme
                port = 8080 if parsed.scheme == "http" else 443
            self._port = port
            return port
        except Exception:
            # Fallback to default
            return 8080
    
    def shutdown(self):
        """Shutdown managed server."""
        # Remove from registry to avoid duplicate shutdown attempts
        _managed_servers.discard(self)
        
        if self.process and self.managed:
            logger.info("Shutting down OpenCode server (RAPTOR-managed)...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not terminate gracefully, killing...")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.warning(f"Error during server shutdown: {e}")
            finally:
                self.process = None
                logger.info("OpenCode server shut down")


def get_opencode_server_manager(server_url: str = "http://localhost:8080") -> OpenCodeServerManager:
    """
    Get or create a singleton OpenCodeServerManager instance.
    
    This ensures only one server manager exists per URL and registers
    the atexit handler only once.
    
    Args:
        server_url: Server URL (default: from config)
    
    Returns:
        OpenCodeServerManager instance
    """
    global _atexit_registered, _singleton_manager
    
    # Use singleton if URL matches
    if _singleton_manager and _singleton_manager.server_url == server_url:
        return _singleton_manager
    
    # Create new manager (will be added to _managed_servers in __init__)
    _singleton_manager = OpenCodeServerManager(server_url)
    
    # Register global shutdown handler once
    if not _atexit_registered:
        atexit.register(_global_shutdown)
        _atexit_registered = True
    
    return _singleton_manager
