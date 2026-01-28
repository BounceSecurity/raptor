#!/usr/bin/env python3
"""
OpenCode Client using official Python SDK

Uses the official OpenCode Python SDK (generated from OpenAPI spec) to provide
a clean interface for RAPTOR packages. Maintains the same external interface as client.py.

SDK: https://github.com/stainless-sdks/opencode-sdk-fork-python
Install: pip install git+ssh://git@github.com/stainless-sdks/opencode-sdk-fork-python.git

OpenAPI Spec: https://raw.githubusercontent.com/anomalyco/opencode/refs/heads/dev/packages/sdk/openapi.json
API Documentation: https://opencode.ai/docs/server/
"""

import os
import signal
import time
from pathlib import Path
from typing import List, Optional, Set

from core.config import RaptorConfig
from core.logging import get_logger
from .server_manager import OpenCodeServerManager, get_opencode_server_manager
from .exceptions import OpenCodeUnavailableError, OpenCodeConnectionError

logger = get_logger()

# Import the official SDK
try:
    from opencode_sdk_fork import OpencodeSDKFork
except ImportError as e:
    logger.error(f"Failed to import OpenCode SDK: {e}")
    raise ImportError(
        "OpenCode SDK not installed. Install with: "
        "pip install git+ssh://git@github.com/stainless-sdks/opencode-sdk-fork-python.git"
    ) from e


class OpenCodeClient:
    """
    Client for OpenCode operations using official Python SDK.
    
    Uses the official OpenCode Python SDK (generated from OpenAPI spec) to interact
    with the OpenCode server. The server must be running (managed by OpenCodeServerManager).
    
    This provides the same external interface as client.py (custom HTTP implementation),
    but uses the official SDK internally, which handles all HTTP communication.
    
    OpenAPI Spec: https://raw.githubusercontent.com/anomalyco/opencode/refs/heads/dev/packages/sdk/openapi.json
    API Reference: https://opencode.ai/docs/server/
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
        
        # Initialize SDK client
        try:
            # SDK: OpencodeSDKFork from opencode_sdk_fork
            # For local server, we can pass None for api_key (no auth required)
            # base_url is required to point to our local server
            self.sdk_client = OpencodeSDKFork(
                base_url=self.server_url,
                api_key=None  # None for local server (no auth required)
            )
            logger.debug(f"Using official OpenCode SDK (OpencodeSDKFork)")
        except Exception as e:
            raise OpenCodeUnavailableError(
                f"Failed to initialize OpenCode SDK: {e}. "
                "Please ensure the SDK is properly installed."
            ) from e
        
        # Create or get session for this repo
        self.session_id: Optional[str] = None
        self._create_session()
        
        # Track seen message IDs to detect new messages
        self._seen_message_ids: Set[str] = set()
        
        # Register signal handler for cleanup on Ctrl-C
        self._register_signal_handlers()

        logger.info(f"Initialized OpenCodeClient (SDK) for {self.repo_path}")
    
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
    
    def _create_session(self):
        """
        Create a new session for this repository.
        
        OpenAPI: POST /session
        SDK: client.session.create() -> Session
        """
        try:
            # Use SDK to create session
            # SDK: session.create() returns Session object with .id attribute
            session = self.sdk_client.session.create()
            self.session_id = session.id
            
            if not self.session_id:
                raise OpenCodeConnectionError("Session created but no ID returned")
            
            logger.debug(f"Created session via SDK: {self.session_id} (repo: {self.repo_path})")
                
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise OpenCodeConnectionError(f"Failed to create session: {e}") from e
    
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
        
        Uses SDK methods to send message and poll for response.
        In debug mode (RAPTOR_DEBUG=1), status updates and response content are logged.
        
        OpenAPI: POST /session/{sessionID}/prompt_async + GET /session/{sessionID}/message
        SDK: client.session.prompt_async() + client.session.message.list()
        
        Args:
            message: Message/prompt to send
            show_progress: If True, show progress updates (auto-enabled in debug mode)
        
        Returns:
            Complete response text from OpenCode
        """
        if not self.session_id:
            raise OpenCodeConnectionError("No session. Call _create_session() first.")
        
        # Check if debug mode is enabled
        debug_mode = os.getenv("RAPTOR_DEBUG", "").lower() in ("1", "true", "yes", "debug")
        show_progress = show_progress or debug_mode
        
        try:
            # Use SDK to send message asynchronously
            # SDK: session.send_async_message(session_id, parts=[...])
            logger.debug("Sending message asynchronously via SDK...")
            self.sdk_client.session.send_async_message(
                session_id=self.session_id,
                parts=[{"type": "text", "text": message}]
            )
            
            # Poll for the complete response
            logger.debug("Polling for complete response...")
            if show_progress:
                logger.info(f"Processing: {message[:60]}...")
            
            # Poll using SDK until we get a complete response
            # OpenAPI: GET /session/{sessionID}/message returns Message[]
            max_wait = 600  # 10 minutes max
            start_time = time.time()
            last_message_id = None
            printed_lengths = {}  # Track printed lengths by message ID
            
            while time.time() - start_time < max_wait:
                try:
                    # Get messages using SDK
                    # SDK: session.message.list(session_id) returns list of Message objects
                    messages = self.sdk_client.session.message.list(session_id=self.session_id)
                    
                    # Convert to list if it's an iterator
                    if not isinstance(messages, list):
                        messages = list(messages) if hasattr(messages, '__iter__') else [messages]
                    
                    if show_progress:
                        elapsed = int(time.time() - start_time)
                        msg_count = len(messages) if hasattr(messages, '__len__') else 0
                        logger.debug(f"Polling status... ({elapsed}s, {msg_count} messages)")
                    
                    # Process messages to detect new ones and show content
                    if messages:
                        for msg in messages:
                            # Extract message info (SDK structure from OpenAPI spec)
                            # Message schema: { info: { role, id }, parts: [{ type, text? }] }
                            msg_info = msg.info if hasattr(msg, 'info') else getattr(msg, 'info', {})
                            role = msg_info.role if hasattr(msg_info, 'role') else msg_info.get('role', 'unknown')
                            msg_id = msg_info.id if hasattr(msg_info, 'id') else msg_info.get('id', 'unknown')
                            parts = msg.parts if hasattr(msg, 'parts') else getattr(msg, 'parts', [])
                            
                            # Check if this is a new message we haven't seen
                            is_new_message = msg_id not in self._seen_message_ids
                            
                            if is_new_message:
                                self._seen_message_ids.add(msg_id)
                                
                                # Extract text content from message parts
                                text_parts = []
                                for part in parts:
                                    part_text = part.text if hasattr(part, 'text') else (part.get('text', '') if isinstance(part, dict) else '')
                                    part_type = part.type if hasattr(part, 'type') else (part.get('type', '') if isinstance(part, dict) else '')
                                    if part_type == 'text' and part_text:
                                        text_parts.append(part_text)
                                
                                # Show user messages (full content)
                                if role == 'user' and text_parts:
                                    user_text = "\n".join(text_parts)
                                    if show_progress:
                                        logger.debug(f"User message ({len(user_text)} chars):\n{user_text}")
                                
                                # Show assistant messages (only once when first seen)
                                elif role == 'assistant':
                                    if text_parts:
                                        response_text = "\n".join(text_parts)
                                        if show_progress:
                                            logger.info(f"OpenCode response ({len(response_text)} chars):\n{response_text}")
                                    else:
                                        # Check for tool calls or other non-text parts
                                        tool_calls = [
                                            p for p in parts
                                            if (p.type if hasattr(p, 'type') else (p.get('type', '') if isinstance(p, dict) else ''))
                                            in ('tool_call', 'tool', 'function_call', 'file_read', 'code_search')
                                        ]
                                        if tool_calls and show_progress:
                                            logger.debug(f"Assistant message {msg_id} has {len(tool_calls)} tool call(s) - OpenCode is working")
                            
                            # Check for updates to existing assistant messages
                            elif role == 'assistant' and msg_id in self._seen_message_ids:
                                text_parts = []
                                for part in parts:
                                    part_text = part.text if hasattr(part, 'text') else (part.get('text', '') if isinstance(part, dict) else '')
                                    part_type = part.type if hasattr(part, 'type') else (part.get('type', '') if isinstance(part, dict) else '')
                                    if part_type == 'text' and part_text:
                                        text_parts.append(part_text)
                                
                                if text_parts:
                                    response_text = "\n".join(text_parts)
                                    current_length = len(response_text)
                                    
                                    # Check for non-text parts that might indicate work in progress
                                    has_tool_calls = any(
                                        (p.type if hasattr(p, 'type') else (p.get('type', '') if isinstance(p, dict) else ''))
                                        in ('tool_call', 'tool', 'function_call', 'file_read', 'code_search')
                                        for p in parts
                                    )
                                    
                                    # Show update if message has grown
                                    if current_length > printed_lengths.get(msg_id, 0):
                                        printed_lengths[msg_id] = current_length
                                        last_message_id = msg_id
                                        if show_progress:
                                            logger.info(f"OpenCode response updated ({current_length} chars):\n{response_text}")
                                    
                                    # Track the latest assistant message for stability checking
                                    if msg_id not in printed_lengths:
                                        printed_lengths[msg_id] = current_length
                                        last_message_id = msg_id
                                    
                                    # Check for tool calls (only log once when tool calls appear)
                                    if has_tool_calls:
                                        if not hasattr(self, '_tool_call_status'):
                                            self._tool_call_status = {}
                                        if not self._tool_call_status.get(msg_id, False):
                                            self._tool_call_status[msg_id] = True
                                            if show_progress:
                                                logger.debug(f"Message has tool calls - OpenCode is still working")
                        
                        # Find the latest assistant message for stability checking
                        for msg in reversed(messages):
                            msg_info = msg.info if hasattr(msg, 'info') else getattr(msg, 'info', {})
                            role = msg_info.role if hasattr(msg_info, 'role') else msg_info.get('role', 'unknown')
                            
                            if role == 'assistant':
                                msg_id = msg_info.id if hasattr(msg_info, 'id') else msg_info.get('id', None)
                                parts = msg.parts if hasattr(msg, 'parts') else getattr(msg, 'parts', [])
                                
                                text_parts = []
                                for part in parts:
                                    part_text = part.text if hasattr(part, 'text') else (part.get('text', '') if isinstance(part, dict) else '')
                                    part_type = part.type if hasattr(part, 'type') else (part.get('type', '') if isinstance(part, dict) else '')
                                    if part_type == 'text' and part_text:
                                        text_parts.append(part_text)
                                
                                if text_parts:
                                    response_text = "\n".join(text_parts)
                                    current_length = len(response_text)
                                    
                                    # Track for stability checking (only if not already tracked)
                                    if msg_id not in printed_lengths:
                                        printed_lengths[msg_id] = current_length
                                    last_message_id = msg_id
                                    last_message_length = current_length
                                    break  # Found latest assistant message
                                    
                except Exception as e:
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
                        stability_messages = self.sdk_client.session.message.list(session_id=self.session_id)
                        for stability_msg in reversed(stability_messages):
                            msg_info = stability_msg.info if hasattr(stability_msg, 'info') else getattr(stability_msg, 'info', {})
                            role = msg_info.role if hasattr(msg_info, 'role') else msg_info.get('role', 'unknown')
                            msg_id = msg_info.id if hasattr(msg_info, 'id') else msg_info.get('id', 'unknown')
                            
                            if role == 'assistant' and msg_id == last_message_id:
                                parts = stability_msg.parts if hasattr(stability_msg, 'parts') else getattr(stability_msg, 'parts', [])
                                
                                # Check for non-text parts that indicate work in progress
                                has_tool_calls = any(
                                    (p.type if hasattr(p, 'type') else (p.get('type', '') if isinstance(p, dict) else ''))
                                    in ('tool_call', 'tool', 'function_call', 'file_read', 'code_search')
                                    for p in parts
                                )
                                
                                stability_text_parts = []
                                for part in parts:
                                    part_text = part.text if hasattr(part, 'text') else (part.get('text', '') if isinstance(part, dict) else '')
                                    part_type = part.type if hasattr(part, 'type') else (part.get('type', '') if isinstance(part, dict) else '')
                                    if part_type == 'text' and part_text:
                                        stability_text_parts.append(part_text)
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
                    except Exception:
                        pass  # Continue polling if check fails
                
                time.sleep(5)  # Poll every 5 seconds
            
            # Timeout - try to get any available response
            logger.warning("Timeout waiting for response, attempting to get latest message...")
            messages = self.sdk_client.session.message.list(session_id=self.session_id)
            if messages:
                # Get last message
                last_msg = list(messages)[-1] if hasattr(messages, '__iter__') else messages[-1]
                parts = last_msg.parts if hasattr(last_msg, 'parts') else getattr(last_msg, 'parts', [])
                text_parts = []
                for part in parts:
                    part_text = part.text if hasattr(part, 'text') else (part.get('text', '') if isinstance(part, dict) else '')
                    part_type = part.type if hasattr(part, 'type') else (part.get('type', '') if isinstance(part, dict) else '')
                    if part_type == 'text' and part_text:
                        text_parts.append(part_text)
                if text_parts:
                    return "\n".join(text_parts)
            
            raise OpenCodeConnectionError("Timeout waiting for OpenCode response")
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise OpenCodeConnectionError(f"Failed to send message: {e}") from e
    
    def read_file(self, file_path: str, start_line: Optional[int] = None,
                  end_line: Optional[int] = None) -> str:
        """
        Read file contents with optional line range.
        
        OpenAPI: GET /file/content?path=<path>
        SDK: client.file.content.retrieve(path=...)
        
        Args:
            file_path: Path to file (relative to repo root)
            start_line: Optional start line (1-indexed)
            end_line: Optional end line (1-indexed)
        
        Returns:
            File contents (or line range if specified)
        """
        try:
            # Use SDK to read file
            # SDK: file.read(path=...) returns file content
            full_path = str(self.repo_path / file_path)
            file_content = self.sdk_client.file.read(path=full_path)
            
            # Extract content (SDK returns content directly or as object with .content)
            if hasattr(file_content, 'content'):
                content = file_content.content
            elif isinstance(file_content, str):
                content = file_content
            else:
                # Try to get content attribute or convert to string
                content = getattr(file_content, 'content', str(file_content))
            
            # Handle line range
            if start_line or end_line:
                lines = content.split('\n')
                start = (start_line or 1) - 1  # Convert to 0-indexed
                end = end_line if end_line else len(lines)
                content = '\n'.join(lines[start:end])
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            # Fallback to direct file read
            full_path = self.repo_path / file_path
            if full_path.exists():
                return full_path.read_text(encoding='utf-8', errors='replace')
            raise OpenCodeConnectionError(f"Failed to read file: {e}") from e
    
    def list_files(self, pattern: Optional[str] = None) -> List[str]:
        """
        List files in repository.
        
        OpenAPI: GET /file?path=<path>
        SDK: client.file.list(path=...)
        
        Args:
            pattern: Optional file pattern filter (e.g., "*.py")
        
        Returns:
            List of file paths (relative to repo root)
        """
        try:
            # Use SDK to list files
            # OpenAPI: GET /file?path=<path> returns FileNode[]
            file_nodes = self.sdk_client.file.list(path=str(self.repo_path))
            
            # Convert to list if needed
            if not isinstance(file_nodes, list):
                file_nodes = list(file_nodes) if hasattr(file_nodes, '__iter__') else [file_nodes]
            
            files = []
            def collect_files(nodes, base_path=""):
                for node in nodes:
                    # Extract node info (SDK structure from OpenAPI spec)
                    # FileNode schema: { path: string, type: "file" | "directory", children?: FileNode[] }
                    node_path = node.path if hasattr(node, 'path') else (node.get('path', '') if isinstance(node, dict) else '')
                    node_type = node.type if hasattr(node, 'type') else (node.get('type', '') if isinstance(node, dict) else '')
                    children = node.children if hasattr(node, 'children') else (node.get('children', []) if isinstance(node, dict) else [])
                    
                    if node_type == 'file':
                        rel_path = Path(node_path).relative_to(self.repo_path)
                        if pattern is None or rel_path.match(pattern):
                            files.append(str(rel_path))
                    elif node_type == 'directory' and children:
                        collect_files(children, node_path)
            
            collect_files(file_nodes)
            return sorted(files)
            
        except Exception as e:
            logger.warning(f"Failed to list files via SDK, using direct access: {e}")
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
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.cxx': 'cpp',
            '.cc': 'cpp',
            '.c': 'c',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
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
        
        OpenAPI: GET /global/event (returns text/event-stream)
        SDK: client.global.event() or similar
        
        Args:
            show_events: If True, print events to console as they arrive (for debug mode)
        
        Returns:
            Generator that yields event objects from the /event endpoint
        """
        try:
            # Use SDK to get event stream
            # SDK: global_.subscribe_to_events() returns event stream
            events = self.sdk_client.global_.subscribe_to_events()
            
            for event in events:
                if show_events:
                    event_type = getattr(event, 'type', None) or (event.get('type', 'unknown') if isinstance(event, dict) else 'unknown')
                    event_msg = getattr(event, 'message', None) or getattr(event, 'data', None) or (event.get('message', event.get('data', '')) if isinstance(event, dict) else '')
                    if event_msg:
                        logger.debug(f"Event: {event_type}: {event_msg}")
                yield event
                
        except AttributeError:
            # SDK doesn't have event streaming, fall back to HTTP
            import requests
            import json
            try:
                # GET /global/event - Server-sent events stream
                response = requests.get(
                    f"{self.server_url}/global/event",
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
                logger.debug(f"Event stream closed normally: {e}")
                return
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                if "prematurely" not in error_msg.lower() and "connection" not in error_msg.lower():
                    logger.error(f"Failed to get event stream: {e}")
                    raise OpenCodeConnectionError(f"Failed to get event stream: {e}") from e
                else:
                    logger.debug(f"Event stream closed: {e}")
                    return
        except Exception as e:
            logger.error(f"Failed to get event stream: {e}")
            raise OpenCodeConnectionError(f"Failed to get event stream: {e}") from e
    
    def close(self):
        """Close connection (no-op for SDK, but kept for compatibility)."""
        # SDK client doesn't need explicit closing
        # Server shutdown is handled by server_manager
        pass
