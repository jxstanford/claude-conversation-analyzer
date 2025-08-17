"""Scanner for discovering and loading Claude Code conversation files."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass, field
from datetime import datetime
import jsonlines
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str
    content: Any
    timestamp: Optional[datetime] = None
    uuid: Optional[str] = None
    type: Optional[str] = None
    tool_use: Optional[Dict[str, Any]] = None
    parent_uuid: Optional[str] = None


@dataclass
class ConversationFile:
    """Represents a conversation file with metadata."""
    file_path: Path
    conversation_id: str
    project_path: str
    message_count: int = 0
    has_user_messages: bool = False
    has_assistant_messages: bool = False
    has_interventions: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    git_branch: Optional[str] = None
    cwd: Optional[str] = None


@dataclass
class ValidationReport:
    """Report of conversation validation results."""
    total_files: int = 0
    valid_files: int = 0
    corrupted_files: List[Path] = field(default_factory=list)
    empty_files: List[Path] = field(default_factory=list)
    total_messages: int = 0
    total_conversations: int = 0
    files_by_project: Dict[str, int] = field(default_factory=dict)


class ConversationScanner:
    """Discovers and loads Claude Code conversation files."""
    
    def __init__(self):
        self.intervention_signals = [
            "[Request interrupted by user]",
            "stop", "wait", "hold on", "actually",
            "no don't", "not that", "instead"
        ]
    
    def scan_directory(self, path: str) -> List[ConversationFile]:
        """Scan directory for conversation files."""
        base_path = Path(path)
        if not base_path.exists():
            raise ValueError(f"Directory not found: {path}")
        
        logger.info(f"Starting scan of directory: {path}")
        
        conversation_files = []
        jsonl_files = list(base_path.rglob("*.jsonl"))
        
        logger.info(f"Found {len(jsonl_files)} JSONL files to process")
        
        # Track statistics
        files_with_interventions = 0
        total_messages = 0
        
        for file_path in tqdm(jsonl_files, desc="Scanning conversations"):
            try:
                conv_file = self._process_conversation_file(file_path, base_path)
                if conv_file and conv_file.message_count > 0:
                    conversation_files.append(conv_file)
                    total_messages += conv_file.message_count
                    if conv_file.has_interventions:
                        files_with_interventions += 1
                    logger.debug(f"Processed {conv_file.conversation_id}: {conv_file.message_count} messages")
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
        
        logger.info(f"Scan complete: {len(conversation_files)} valid conversations")
        logger.info(f"Total messages: {total_messages:,}")
        logger.info(f"Conversations with interventions: {files_with_interventions}")
        
        return conversation_files
    
    def _process_conversation_file(self, file_path: Path, base_path: Path) -> Optional[ConversationFile]:
        """Process a single conversation file."""
        try:
            # Extract project path from file structure
            relative_path = file_path.relative_to(base_path)
            project_path = str(relative_path.parent)
            conversation_id = file_path.stem
            
            conv_file = ConversationFile(
                file_path=file_path,
                conversation_id=conversation_id,
                project_path=project_path
            )
            
            # Quick scan for metadata
            with jsonlines.open(file_path) as reader:
                messages = list(reader)
                if not messages:
                    return None
                
                conv_file.message_count = len(messages)
                
                # Extract metadata from first message
                first_msg = messages[0]
                if 'timestamp' in first_msg:
                    conv_file.start_time = self._parse_timestamp(first_msg['timestamp'])
                if 'gitBranch' in first_msg:
                    conv_file.git_branch = first_msg['gitBranch']
                if 'cwd' in first_msg:
                    conv_file.cwd = first_msg['cwd']
                
                # Check message types and interventions
                for msg in messages:
                    if msg.get('type') == 'user':
                        conv_file.has_user_messages = True
                        # Check for intervention signals
                        content = self._extract_content(msg)
                        if content and any(signal in str(content).lower() for signal in self.intervention_signals):
                            conv_file.has_interventions = True
                    
                    elif msg.get('type') == 'assistant':
                        conv_file.has_assistant_messages = True
                
                # Get end time from last message
                if messages and 'timestamp' in messages[-1]:
                    conv_file.end_time = self._parse_timestamp(messages[-1]['timestamp'])
            
            return conv_file
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return None
    
    def _extract_content(self, message: Dict[str, Any]) -> Optional[str]:
        """Extract text content from various message formats."""
        if 'message' in message and isinstance(message['message'], dict):
            content = message['message'].get('content')
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle structured content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                return ' '.join(text_parts)
        return None
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime."""
        try:
            # Handle ISO format with Z suffix
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'
            return datetime.fromisoformat(timestamp_str)
        except:
            return None
    
    def load_conversation(self, conv_file: ConversationFile) -> List[Message]:
        """Load full conversation from file."""
        messages = []
        
        with jsonlines.open(conv_file.file_path) as reader:
            for entry in reader:
                msg = self._parse_message(entry)
                if msg:
                    messages.append(msg)
        
        return messages
    
    def _parse_message(self, entry: Dict[str, Any]) -> Optional[Message]:
        """Parse a message entry."""
        try:
            msg_data = entry.get('message', {})
            
            # Handle different message formats
            if isinstance(msg_data, dict):
                role = msg_data.get('role', entry.get('type', 'unknown'))
                content = msg_data.get('content', '')
            else:
                role = entry.get('type', 'unknown')
                content = msg_data
            
            return Message(
                role=role,
                content=content,
                timestamp=self._parse_timestamp(entry['timestamp']) if 'timestamp' in entry else None,
                uuid=entry.get('uuid'),
                type=entry.get('type'),
                parent_uuid=entry.get('parentUuid')
            )
        except Exception as e:
            logger.warning(f"Failed to parse message: {e}")
            return None
    
    def validate_conversations(self, conversation_files: List[ConversationFile]) -> ValidationReport:
        """Validate conversation files and generate report."""
        logger.info(f"Starting validation of {len(conversation_files)} conversations")
        
        report = ValidationReport()
        report.total_files = len(conversation_files)
        
        for conv_file in tqdm(conversation_files, desc="Validating conversations"):
            try:
                # Check if file exists and is readable
                if not conv_file.file_path.exists():
                    report.corrupted_files.append(conv_file.file_path)
                    logger.debug(f"File not found: {conv_file.file_path}")
                    continue
                
                if conv_file.message_count == 0:
                    report.empty_files.append(conv_file.file_path)
                    logger.debug(f"Empty file: {conv_file.file_path}")
                else:
                    report.valid_files += 1
                    report.total_messages += conv_file.message_count
                    report.total_conversations += 1
                    
                    # Track by project
                    project = conv_file.project_path
                    report.files_by_project[project] = report.files_by_project.get(project, 0) + 1
                    
            except Exception as e:
                logger.error(f"Validation error for {conv_file.file_path}: {e}")
                report.corrupted_files.append(conv_file.file_path)
        
        logger.info(f"Validation complete: {report.valid_files} valid, {len(report.corrupted_files)} corrupted, {len(report.empty_files)} empty")
        
        return report
    
    def filter_conversations(self, 
                           conversation_files: List[ConversationFile],
                           has_interventions: Optional[bool] = None,
                           min_messages: Optional[int] = None,
                           project_filter: Optional[str] = None) -> List[ConversationFile]:
        """Filter conversations based on criteria."""
        filtered = conversation_files
        
        if has_interventions is not None:
            filtered = [cf for cf in filtered if cf.has_interventions == has_interventions]
        
        if min_messages is not None:
            filtered = [cf for cf in filtered if cf.message_count >= min_messages]
        
        if project_filter:
            filtered = [cf for cf in filtered if project_filter in cf.project_path]
        
        return filtered