#!/usr/bin/env python3
"""
OpenCode Client using custom HTTP client

Uses direct HTTP requests to interact with OpenCode HTTP server.
Based on OpenAPI 3.1 spec available at: https://opencode.ai/docs/server/

API Documentation: https://opencode.ai/docs/server/
OpenAPI Spec: http://<hostname>:<port>/doc (e.g., http://localhost:8080/doc)
"""

import json
import os
import signal
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

import requests

from core.config import RaptorConfig
from core.logging import get_logger
from .server_manager import OpenCodeServerManager, get_opencode_server_manager
from .exceptions import OpenCodeUnavailableError, OpenCodeConnectionError

logger = get_logger()


class OpenCodeClient:
    """
    Client for OpenCode operations using direct HTTP requests.
    
    Uses the OpenCode HTTP server API based on OpenAPI 3.1 spec.
    The server must be running (managed by OpenCodeServerManager).
    
    API Reference: https://opencode.ai/docs/server/
    OpenAPI Spec: http://<hostname>:<port>/doc
    """
    
    def __init__(self, repo_path: Path, server_url: Optional[str] = None):
        """
        Initialize OpenCode client.
        
        Args:
            repo_path: Repository path to analyze
            server_url: Server URL (default: from config)
        
        Raises:
            OpenCodeUnavailableError: If OpenCode cannot be started or connected to
        """
        self.repo_path = Path(repo_path).resolve()
        self.server_url = (server_url or RaptorConfig.OPENCODE_SERVER_URL).rstrip('/')
        
        # Initialize server manager (starts `opencode serve`)
        self.server_manager = get_opencode_server_manager(self.server_url)
        
        # Automatically ensure server is running (pass repo_path for correct cwd)
        try:
            self.server_manager.ensure_running(repo_path=self.repo_path)
        except OpenCodeConnectionError as e:
            raise OpenCodeUnavailableError(
                f"Cannot connect to OpenCode server: {e}. "
                "Please ensure opencode binary is installed and accessible. "
                "See docs/OPENCODE_INTEGRATION.md for installation instructions."
            ) from e
        
               # Create or get session for this repo
               self.session_id: Optional[str] = None
               self._create_session()
               
               # Track seen message IDs to detect new messages
               self._seen_message_ids: Set[str] = set()
               
               # Register signal handler for cleanup on Ctrl-C
               self._register_signal_handlers()

               logger.info(f"Initialized OpenCodeClient for {self.repo_path}")
    
    def _create_session(self):
        """Create a new session for this repository."""
        try:
            # POST /session
            # Body: { parentID?, title? }
            # Returns: Session object with id
            response = requests.post(
                f"{self.server_url}/session",
                json={},  # Empty body for new session
                timeout=10
            )
            response.raise_for_status()
            
            session_data = response.json()
            self.session_id = session_data.get('id')
            
            if not self.session_id:
                raise OpenCodeConnectionError("Session created but no ID returned")
            
            logger.debug(f"Created session: {self.session_id} (repo: {self.repo_path})")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create session: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.text
                    logger.debug(f"Error response: {error_detail}")
                except:
                    pass
            raise OpenCodeConnectionError(f"Failed to create session: {e}") from e
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("Received interrupt signal, shutting down OpenCode server...")
            try:
                self.server_manager.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down OpenCode server: {e}")
            raise KeyboardInterrupt
        
        # Register handler for SIGINT (Ctrl-C)
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
    
    def verify_connection(self) -> bool:
        """
        Verify connection to OpenCode server.
        
        Returns:
            True if server is responding
        """
        return self.server_manager._check_server_health()
    
    def send_message(self, message: str, show_progress: bool = False) -> str:
        """
        Send a message/prompt to OpenCode and wait for the complete response.
        
        This uses the async endpoint (prompt_async) and then polls for the complete
        response. This allows for status updates while waiting. In debug mode 
        (RAPTOR_DEBUG=1), status updates and response content are logged.
        
        Args:
            message: Message/prompt to send
            show_progress: If True, show progress updates (auto-enabled in debug mode)
        
        Returns:
            Complete response text from OpenCode
        """
        if not self.session_id:
            raise OpenCodeConnectionError("No session. Call _create_session() first.")
        
        # Check if debug mode is enabled
        import os
        debug_mode = os.getenv("RAPTOR_DEBUG", "").lower() in ("1", "true", "yes", "debug")
        show_progress = show_progress or debug_mode
        
        try:
            payload = {
                "parts": [
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            }
            
            # Step 1: Send message asynchronously (returns immediately)
            # POST /session/:id/prompt_async
            logger.debug("Sending message asynchronously...")
            response = requests.post(
                f"{self.server_url}/session/{self.session_id}/prompt_async",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            # Returns 204 No Content - message is queued
            
            # Step 2: Poll for the complete response
            logger.debug("Polling for complete response...")
            if show_progress:
                logger.info(f"Processing: {message[:60]}...")
            
            # Poll GET /session/:id/message until we get a complete response
            max_wait = 600  # 10 minutes max
            start_time = time.time()
            last_message_id = None
            last_message_length = 0
            printed_lengths = {}  # Track printed lengths by message ID: {msg_id: length}
            
            while time.time() - start_time < max_wait:
                try:
                    # Poll GET /session/:id/message to check for response
                    messages_response = requests.get(
                        f"{self.server_url}/session/{self.session_id}/message",
                        timeout=10
                    )
                    messages_response.raise_for_status()
                    messages = messages_response.json()
                    
                    if show_progress:
                        elapsed = int(time.time() - start_time)
                        logger.debug(f"Polling status... ({elapsed}s, {len(messages) if messages else 0} messages)")
                    
                    # Process all messages to detect new ones and show content
                    if messages:
                        for msg in messages:
                            msg_info = msg.get('info', {})
                            role = msg_info.get('role', 'unknown')
                            msg_id = msg_info.get('id', 'unknown')
                            parts = msg.get('parts', [])
                            
                            # Check if this is a new message we haven't seen
                            is_new_message = msg_id not in self._seen_message_ids
                            
                            if is_new_message:
                                self._seen_message_ids.add(msg_id)
                                
                                # Extract text content
                                text_parts = []
                                for part in parts:
                                    if part.get('type') == 'text' and 'text' in part:
                                        text_parts.append(part['text'])
                                
                                # Show user messages (full content)
                                if role == 'user' and text_parts:
                                    user_text = "\n".join(text_parts)
                                    if show_progress:
                                        logger.debug(f"User message ({len(user_text)} chars):\n{user_text}")
                                
                                # Show assistant messages
                                elif role == 'assistant':
                                    if text_parts:
                                        response_text = "\n".join(text_parts)
                                        if show_progress:
                                            logger.info(f"OpenCode response ({len(response_text)} chars):\n{response_text}")
                                    else:
                                        # Check for tool calls or other non-text parts
                                        tool_calls = [p for p in parts if p.get('type') in ('tool_call', 'tool', 'function_call', 'file_read', 'code_search')]
                                        if tool_calls and show_progress:
                                            logger.debug(f"Assistant message {msg_id} has {len(tool_calls)} tool call(s) - OpenCode is working")
                                        elif show_progress:
                                            logger.debug(f"Assistant message {msg_id} has {len(parts)} part(s) (no text yet)")
                            
                            # Check for updates to existing assistant messages
                            elif role == 'assistant' and msg_id in self._seen_message_ids:
                                text_parts = []
                                for part in parts:
                                    if part.get('type') == 'text' and 'text' in part:
                                        text_parts.append(part['text'])
                                
                                if text_parts:
                                    response_text = "\n".join(text_parts)
                                    current_length = len(response_text)
                                    
                                    # Check for non-text parts that might indicate work in progress
                                    has_tool_calls = any(
                                        part.get('type') in ('tool_call', 'tool', 'function_call', 'file_read', 'code_search')
                                        for part in parts
                                    )
                                    
                                    # Show update if message has grown
                                    if current_length > printed_lengths.get(msg_id, 0):
                                        printed_lengths[msg_id] = current_length
                                        last_message_id = msg_id
                                        last_message_length = current_length
                                        if show_progress:
                                            logger.info(f"OpenCode response updated ({current_length} chars):\n{response_text}")
                                    
                                    # Track the latest assistant message for stability checking
                                    if msg_id not in printed_lengths:
                                        printed_lengths[msg_id] = current_length
                                        last_message_id = msg_id
                                        last_message_length = current_length
                                    
                                    # Check for tool calls (only log once when tool calls appear)
                                    if has_tool_calls:
                                        # Track tool call status to avoid repeated logging
                                        if not hasattr(self, '_tool_call_status'):
                                            self._tool_call_status = {}
                                        if not self._tool_call_status.get(msg_id, False):
                                            self._tool_call_status[msg_id] = True
                                            if show_progress:
                                                logger.debug(f"Message has tool calls - OpenCode is still working")
                        
                        # Find the latest assistant message for stability checking
                        for msg in reversed(messages):
                            msg_info = msg.get('info', {})
                            if msg_info.get('role') == 'assistant':
                                msg_id = msg_info.get('id')
                                parts = msg.get('parts', [])
                                
                                text_parts = []
                                for part in parts:
                                    if part.get('type') == 'text' and 'text' in part:
                                        text_parts.append(part['text'])
                                
                                if text_parts:
                                    response_text = "\n".join(text_parts)
                                    current_length = len(response_text)
                                    
                                    # Track for stability checking (only if not already tracked)
                                    if msg_id not in printed_lengths:
                                        printed_lengths[msg_id] = current_length
                                    last_message_id = msg_id
                                    last_message_length = current_length
                                    break  # Found latest assistant message
                                    
                except requests.exceptions.RequestException as e:
                    if show_progress:
                        logger.debug(f"Poll error: {e}")
                    pass  # Continue polling
                
                # After each poll cycle, check if message is stable (hasn't grown)
                # Only check if we've found a message and it's been stable for a while
                if last_message_id and last_message_id in printed_lengths:
                    # Track how long the message has been stable
                    if not hasattr(self, '_stability_start_time'):
                        self._stability_start_time = {}
                    
                    # Check if message has grown
                    try:
                        stability_check = requests.get(
                            f"{self.server_url}/session/{self.session_id}/message",
                            timeout=10
                        )
                        if stability_check.status_code == 200:
                            stability_messages = stability_check.json()
                            for stability_msg in reversed(stability_messages):
                                stability_info = stability_msg.get('info', {})
                                if stability_info.get('role') == 'assistant' and stability_info.get('id') == last_message_id:
                                    stability_parts = stability_msg.get('parts', [])
                                    
                                    # Check for non-text parts that indicate work in progress
                                    has_tool_calls = any(
                                        part.get('type') in ('tool_call', 'tool', 'function_call', 'file_read', 'code_search')
                                        for part in stability_parts
                                    )
                                    
                                    stability_text_parts = []
                                    for part in stability_parts:
                                        if part.get('type') == 'text' and 'text' in part:
                                            stability_text_parts.append(part['text'])
                                    if stability_text_parts:
                                        stability_response_text = "\n".join(stability_text_parts)
                                        stability_length = len(stability_response_text)
                                        
                                        # If message has tool calls, it's definitely still processing
                                        if has_tool_calls:
                                            if show_progress:
                                                logger.debug(f"Message has tool calls - OpenCode is still working")
                                            # Reset stability timer
                                            self._stability_start_time.pop(last_message_id, None)
                                            break
                                        
                                        # If message hasn't grown since we last printed it
                                        if stability_length == printed_lengths.get(last_message_id, 0):
                                            # Track when stability started
                                            if last_message_id not in self._stability_start_time:
                                                self._stability_start_time[last_message_id] = time.time()
                                            
                                            # Message must be stable for at least 15 seconds before we consider it complete
                                            stability_duration = time.time() - self._stability_start_time[last_message_id]
                                            if stability_duration >= 15:
                                                # Message has been stable for 15+ seconds - it's likely complete
                                                if show_progress:
                                                    logger.info(f"Response complete ({stability_length} chars, stable for {int(stability_duration)}s)")
                                                    logger.debug(f"OpenCode final response:\n{stability_response_text}")
                                                return stability_response_text
                                            elif show_progress:
                                                logger.debug(f"Message stable for {int(stability_duration)}s, waiting for 15s total...")
                                        else:
                                            # Message grew - update and reset stability timer
                                            if stability_length > printed_lengths.get(last_message_id, 0):
                                                printed_lengths[last_message_id] = stability_length
                                                self._stability_start_time.pop(last_message_id, None)  # Reset timer
                                                if show_progress:
                                                    logger.info(f"OpenCode response updated ({stability_length} chars):\n{stability_response_text}")
                                    break
                    except requests.exceptions.RequestException:
                        pass  # Continue polling if check fails
                
                time.sleep(5)  # Poll every 5 seconds
            
            # Timeout - try to get any available response
            logger.warning("Timeout waiting for response, attempting to get latest message...")
            messages_response = requests.get(
                f"{self.server_url}/session/{self.session_id}/message?limit=1",
                timeout=10
            )
            if messages_response.status_code == 200:
                messages = messages_response.json()
                if messages:
                    msg = messages[-1]
                    parts = msg.get('parts', [])
                    text_parts = []
                    for part in parts:
                        if part.get('type') == 'text' and 'text' in part:
                            text_parts.append(part['text'])
                    if text_parts:
                        return "\n".join(text_parts)
            
            raise OpenCodeConnectionError("Timeout waiting for OpenCode response")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send message: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.text
                    logger.debug(f"Error response: {error_detail}")
                except:
                    pass
            raise OpenCodeConnectionError(f"Failed to send message: {e}") from e
    
    def read_file(self, file_path: str, start_line: Optional[int] = None,
                  end_line: Optional[int] = None) -> str:
        """
        Read file contents with optional line range.
        
        Args:
            file_path: Path to file (relative to repo root)
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (1-indexed)
        
        Returns:
            File contents (or line range if specified)
        """
        try:
            # GET /file/content?path=<path>
            # Query params: path (required)
            params = {"path": str(self.repo_path / file_path)}
            
            response = requests.get(
                f"{self.server_url}/file/content",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            file_data = response.json()
            content = file_data.get('content', '')
            
            # Handle line range if specified
            if start_line or end_line:
                lines = content.split('\n')
                start = (start_line or 1) - 1  # Convert to 0-indexed
                end = end_line if end_line else len(lines)
                content = '\n'.join(lines[start:end])
            
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            # Fallback to direct file read
            full_path = self.repo_path / file_path
            if full_path.exists():
                return full_path.read_text(encoding='utf-8', errors='replace')
            raise OpenCodeConnectionError(f"Failed to read file: {e}") from e
    
    def list_files(self, pattern: Optional[str] = None) -> List[str]:
        """
        List files in repository.
        
        Args:
            pattern: Optional file pattern filter (e.g., "*.py")
        
        Returns:
            List of file paths (relative to repo root)
        """
        try:
            # GET /file?path=<path>
            # Returns: FileNode[]
            params = {"path": str(self.repo_path)}
            
            # Increase timeout for large repositories
            response = requests.get(
                f"{self.server_url}/file",
                params=params,
                timeout=120  # 2 minutes for large repos
            )
            response.raise_for_status()
            
            file_nodes = response.json()
            
            files = []
            def collect_files(nodes, base_path=""):
                for node in nodes:
                    node_path = node.get('path', '')
                    node_type = node.get('type', '')
                    
                    if node_type == 'file':
                        rel_path = Path(node_path).relative_to(self.repo_path)
                        if pattern is None or rel_path.match(pattern):
                            files.append(str(rel_path))
                    elif node_type == 'directory' and 'children' in node:
                        collect_files(node['children'], node_path)
            
            collect_files(file_nodes)
            return sorted(files)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to list files via API, using direct access: {e}")
            # Fallback to direct filesystem access
            import glob
            if pattern:
                matches = list(self.repo_path.rglob(pattern))
            else:
                matches = list(self.repo_path.rglob("*"))
            
            files = []
            for match in matches:
                if match.is_file():
                    rel_path = match.relative_to(self.repo_path)
                    files.append(str(rel_path))
            
            return sorted(files)
    
    def detect_languages(self) -> List[str]:
        """
        Detect programming languages in repository.
        
        Returns:
            List of detected language names
        """
        # Fallback: scan filesystem if no file list available
        # This is slower but works if list_files() wasn't called first
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
        }
        
        seen_extensions = set()
        file_count = 0
        max_files_to_check = 1000  # Limit to avoid long delays
        
        try:
            for file_path in self.repo_path.rglob("*"):
                if file_count >= max_files_to_check:
                    break
                if file_path.is_file():
                    file_count += 1
                    ext = file_path.suffix.lower()
                    if ext in language_map and ext not in seen_extensions:
                        seen_extensions.add(ext)
        except Exception as e:
            logger.debug(f"Error during language detection: {e}")
        
        return [language_map[ext] for ext in seen_extensions if ext in language_map]
    
    def detect_languages_from_files(self, files: List[str]) -> List[str]:
        """
        Detect programming languages from a list of file paths.
        
        This is much faster than scanning the filesystem.
        
        Args:
            files: List of file paths (relative to repo root)
        
        Returns:
            List of detected language names
        """
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
        }
        
        seen_languages = set()
        
        # Check extensions from file list (much faster than filesystem scan)
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            if ext in language_map:
                lang = language_map[ext]
                if lang not in seen_languages:
                    seen_languages.add(lang)
        
        return sorted(list(seen_languages))
    
    def get_event_stream(self, show_events: bool = False):
        """
        Get Server-Sent Events (SSE) stream for status updates.
        
        Args:
            show_events: If True, print events to console as they arrive (for debug mode)
        
        Returns:
            Generator that yields event objects from the /event endpoint
        
        Note:
            GET /event - Server-sent events stream
            First event is "server.connected", then bus events
            Use this to monitor progress of async operations or get real-time status updates
        
        Example:
            for event in client.get_event_stream(show_events=True):
                if event.get('type') == 'message.complete':
                    break
        """
        try:
            # GET /event - Server-sent events stream
            response = requests.get(
                f"{self.server_url}/event",
                stream=True,
                timeout=None  # SSE streams don't timeout
            )
            response.raise_for_status()
            
            # Parse SSE format (data: {...}\n\n)
            buffer = ""
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                if chunk:
                    buffer += chunk
                    # Process complete events (separated by \n\n)
                    while "\n\n" in buffer:
                        event_block, buffer = buffer.split("\n\n", 1)
                        for line in event_block.split("\n"):
                            if line.startswith("data: "):
                                data = line[6:]  # Remove "data: " prefix
                                try:
                                    event = json.loads(data)
                                    if show_events:
                                        event_type = event.get('type', 'unknown')
                                        event_msg = event.get('message', event.get('data', ''))
                                        if event_msg:
                                            logger.debug(f"Event: {event_type}: {event_msg}")
                                    yield event
                                except json.JSONDecodeError:
                                    event = {"raw": data}
                                    if show_events:
                                        logger.debug(f"Event: {data}")
                                    yield event
        except requests.exceptions.ChunkedEncodingError as e:
            # ChunkedEncodingError is expected when the stream closes normally
            # This happens when the server closes the connection after sending all events
            logger.debug(f"Event stream closed normally: {e}")
            return  # End generator gracefully (don't raise StopIteration in Python 3.7+)
        except requests.exceptions.RequestException as e:
            # Only log as error if it's not a normal stream close
            error_msg = str(e)
            if "prematurely" not in error_msg.lower() and "connection" not in error_msg.lower():
                logger.error(f"Failed to get event stream: {e}")
                raise OpenCodeConnectionError(f"Failed to get event stream: {e}") from e
            else:
                logger.debug(f"Event stream closed: {e}")
                return  # End generator gracefully
    
    def close(self):
        """Close connection (no-op for HTTP, but kept for compatibility)."""
        # HTTP client doesn't need explicit closing
        # Server shutdown is handled by server_manager
        pass
