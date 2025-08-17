"""Track which conversations have been processed to avoid reprocessing."""

import json
import logging
from pathlib import Path
from typing import Dict, Set, Optional, Any
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ProcessingTracker:
    """Tracks processed conversations to enable incremental analysis."""
    
    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state: Dict[str, Dict] = self._load_state()
        self._migrate_state_format()
    
    def _load_state(self) -> Dict[str, Dict]:
        """Load existing processing state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load processing state: {e}")
                return {}
        return {}
    
    def _migrate_state_format(self):
        """Migrate old state format to new per-phase format."""
        migrated = False
        for conv_id, conv_state in self.state.items():
            if 'phases' not in conv_state:
                # Old format - convert to new format
                conv_state['phases'] = {
                    'classification': {
                        'completed': True,  # Assume old entries completed all phases
                        'completed_at': conv_state.get('processed_at'),
                        'result': None
                    },
                    'intervention_analysis': {
                        'completed': True,
                        'completed_at': conv_state.get('processed_at'),
                        'result': None
                    },
                    'deep_analysis': {
                        'completed': True,
                        'completed_at': conv_state.get('processed_at'),
                        'result': None
                    }
                }
                migrated = True
        
        if migrated:
            self._save_state()
            logger.info("Migrated processing state to new per-phase format")
    
    def _save_state(self):
        """Save processing state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save processing state: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file modification time and size."""
        stat = file_path.stat()
        # Use mtime and size for quick change detection
        content = f"{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_processed(self, conversation_id: str, file_path: Path) -> bool:
        """Check if conversation has already been processed."""
        if conversation_id not in self.state:
            return False
        
        # Check if file has been modified since last processing
        current_hash = self._get_file_hash(file_path)
        stored_hash = self.state[conversation_id].get('file_hash')
        
        if current_hash != stored_hash:
            logger.debug(f"Conversation {conversation_id} has been modified")
            return False
        
        # Check if all phases are completed
        phases = self.state[conversation_id].get('phases', {})
        all_completed = all(
            phase_data.get('completed', False) 
            for phase_data in phases.values()
        )
        
        return all_completed
    
    def mark_processed(self, conversation_id: str, file_path: Path):
        """Mark a conversation as fully processed (all phases complete)."""
        self.state[conversation_id] = {
            'file_path': str(file_path),
            'file_hash': self._get_file_hash(file_path),
            'processed_at': datetime.now().isoformat(),
            'file_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'phases': {
                'classification': {
                    'completed': True,
                    'completed_at': datetime.now().isoformat(),
                    'result': None
                },
                'intervention_analysis': {
                    'completed': True,
                    'completed_at': datetime.now().isoformat(),
                    'result': None
                },
                'deep_analysis': {
                    'completed': True,
                    'completed_at': datetime.now().isoformat(),
                    'result': None
                }
            }
        }
        self._save_state()
    
    def get_unprocessed_conversations(self, conversation_files: list, force_all: bool = False) -> list:
        """Filter conversations to only return unprocessed ones."""
        if force_all:
            logger.info("Processing all conversations (--all flag used)")
            return conversation_files
        
        unprocessed = []
        for conv_file in conversation_files:
            if not self.is_processed(conv_file.conversation_id, conv_file.file_path):
                unprocessed.append(conv_file)
        
        logger.info(f"Found {len(unprocessed)} new/modified conversations out of {len(conversation_files)} total")
        return unprocessed
    
    def mark_all_processed(self, conversation_files: list):
        """Mark multiple conversations as processed."""
        for conv_file in conversation_files:
            self.mark_processed(conv_file.conversation_id, conv_file.file_path)
        logger.debug(f"Marked {len(conversation_files)} conversations as processed")
    
    def save_phase_result(self, conversation_id: str, file_path: Path, phase: str, result: Any):
        """Save the result of a specific phase for a conversation."""
        if conversation_id not in self.state:
            # Initialize conversation state if it doesn't exist
            self.state[conversation_id] = {
                'file_path': str(file_path),
                'file_hash': self._get_file_hash(file_path),
                'file_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'phases': {
                    'classification': {'completed': False, 'result': None},
                    'intervention_analysis': {'completed': False, 'result': None},
                    'deep_analysis': {'completed': False, 'result': None}
                }
            }
        
        # Update the specific phase
        if phase in self.state[conversation_id]['phases']:
            self.state[conversation_id]['phases'][phase] = {
                'completed': True,
                'completed_at': datetime.now().isoformat(),
                'result': result
            }
            self._save_state()
            logger.debug(f"Saved {phase} result for {conversation_id}")
    
    def get_phase_result(self, conversation_id: str, phase: str) -> Optional[Any]:
        """Get the cached result of a specific phase for a conversation."""
        if conversation_id not in self.state:
            return None
        
        phases = self.state[conversation_id].get('phases', {})
        phase_data = phases.get(phase, {})
        
        if phase_data.get('completed', False):
            return phase_data.get('result')
        
        return None
    
    def is_phase_completed(self, conversation_id: str, file_path: Path, phase: str) -> bool:
        """Check if a specific phase is completed for a conversation."""
        if conversation_id not in self.state:
            return False
        
        # Check if file has been modified
        current_hash = self._get_file_hash(file_path)
        stored_hash = self.state[conversation_id].get('file_hash')
        
        if current_hash != stored_hash:
            logger.debug(f"Conversation {conversation_id} has been modified, phases invalid")
            return False
        
        phases = self.state[conversation_id].get('phases', {})
        phase_data = phases.get(phase, {})
        
        return phase_data.get('completed', False)
    
    def get_conversations_needing_phase(self, conversation_files: list, phase: str, force_all: bool = False) -> list:
        """Get conversations that need a specific phase to be processed."""
        if force_all:
            logger.info(f"Processing all conversations for {phase} (--all flag used)")
            return conversation_files
        
        need_processing = []
        for conv_file in conversation_files:
            if not self.is_phase_completed(conv_file.conversation_id, conv_file.file_path, phase):
                need_processing.append(conv_file)
        
        logger.info(f"Found {len(need_processing)} conversations needing {phase} out of {len(conversation_files)} total")
        return need_processing
    
    def get_statistics(self) -> Dict:
        """Get statistics about processed conversations."""
        if not self.state:
            return {
                'total_conversations': 0,
                'fully_processed': 0,
                'phase_statistics': {},
                'last_processed': None,
                'oldest_processed': None
            }
        
        # Count completed phases
        phase_stats = {
            'classification': 0,
            'intervention_analysis': 0,
            'deep_analysis': 0
        }
        
        fully_processed = 0
        processed_times = []
        
        for conv_id, conv_state in self.state.items():
            phases = conv_state.get('phases', {})
            all_phases_complete = True
            
            for phase_name, phase_data in phases.items():
                if phase_data.get('completed', False):
                    phase_stats[phase_name] = phase_stats.get(phase_name, 0) + 1
                else:
                    all_phases_complete = False
            
            if all_phases_complete:
                fully_processed += 1
            
            if 'processed_at' in conv_state:
                processed_times.append(datetime.fromisoformat(conv_state['processed_at']))
        
        return {
            'total_conversations': len(self.state),
            'fully_processed': fully_processed,
            'phase_statistics': phase_stats,
            'last_processed': max(processed_times).isoformat() if processed_times else None,
            'oldest_processed': min(processed_times).isoformat() if processed_times else None
        }