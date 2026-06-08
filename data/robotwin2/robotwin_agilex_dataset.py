# RoboTwin2 dataset loader for EfficientWAM.
# Supports multi-task RoboTwin video and action sampling.

import os
import random
import h5py
import numpy as np
import cv2
import json
import torch
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import warnings

# Import image processing utilities
from data.utils.image_utils import (
    apply_image_augmentation,
    load_video_frames, get_video_frame_count
)
from data.robotwin2.action_normalization import RoboTwinQposNormalizer, normalize_action_normalization_config

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)

class RoboTwinTaskDataset(data.Dataset):
    """
    Dataset for RoboTwin data with task-level organization and flexible sampling.
    
    Data structure:
    /path/to/converted_robotwin/
    ├── clean/
    │   ├── adjust_bottle/
    │   │   ├── qpos/           # Robot position files (.pt)
    │   │   ├── videos/         # MP4 video files  
    │   │   └── umt5_wan/       # Pre-encoded language embeddings (.pt)
    │   ├── beat_block_hammer/
    │   └── ...
    └── randomized/
        ├── adjust_bottle/
        └── ...
    """
    
    def __init__(
        self,
        dataset_dir: str = "/path/to/converted_robotwin",
        data_mode: str = "clean",  # "clean", "randomized", or "both"
        task_mode: str = "multi",  # "single" or "multi" 
        task_name: Optional[str] = None,  # Required for single task mode
        randomized_limit_per_task: Optional[int] = None,  # Limit randomized episodes per task (take first N)
        
        # Sampling parameters
        global_downsample_rate: int = 3,  # Global downsampling (e.g., 30Hz -> 10Hz)
        video_action_freq_ratio: int = 5,  # Video:Action frequency ratio  
        num_video_frames: int = 3,  # Number of video frames to predict
        
        # Standard parameters
        video_size: Tuple[int, int] = (320, 384),
        max_episodes: Optional[int] = None,
        upsample_rate: int = 1,  # For compatibility with H_RDT
        val: bool = False,
        image_aug: bool = False,
        action_normalization: Optional[Dict[str, Any]] = None,
        
    ):
        """
        Initialize RoboTwin dataset with flexible sampling.
        
        Args:
            dataset_dir: Root directory containing clean/ and randomized/ folders
            data_mode: Which data split to use ("clean", "randomized", or "both")
            task_mode: Single task or multi-task ("single" or "multi")
            task_name: Task name for single task mode (e.g., "adjust_bottle")
            
            global_downsample_rate: Global downsampling rate (e.g., 3 for 30Hz->10Hz)
            video_action_freq_ratio: Frequency ratio between video and action
            num_video_frames: Number of video frames to predict
            
            video_size: Target video resolution (H, W)
            max_episodes: Maximum number of episodes to load (for debugging)
            upsample_rate: Temporal data upsampling rate (for H_RDT compatibility)
            val: Whether this is validation set
            image_aug: Whether to apply image augmentation
            action_normalization: Optional qpos mean/std normalization config
        """
        self.dataset_dir = Path(dataset_dir)
        self.data_mode = data_mode
        self.task_mode = task_mode
        self.task_name = task_name
        self.randomized_limit_per_task = randomized_limit_per_task
        
        # Sampling parameters
        self.global_downsample_rate = global_downsample_rate
        self.video_action_freq_ratio = video_action_freq_ratio
        self.num_video_frames = num_video_frames
        
        # Calculate action sequence length
        self.action_chunk_size = num_video_frames * video_action_freq_ratio
        
        # Standard parameters
        self.video_size = video_size
        self.max_episodes = max_episodes
        self.upsample_rate = upsample_rate
        self.val = val
        self.image_aug = image_aug
        self._frame_count_cache: Dict[str, int] = {}
        self._qpos_frame_count_cache: Dict[str, int] = {}
        self.action_normalization_config = normalize_action_normalization_config(action_normalization)
        self.action_normalizer = RoboTwinQposNormalizer.from_config(self.action_normalization_config)
        self.sample_index: List[Tuple[int, int]] = []
        
        # Validate parameters
        if task_mode == "single" and not task_name:
            raise ValueError("Single task mode requires task_name parameter")
        
        assert data_mode in ["clean", "randomized", "both"], \
            f"data_mode must be 'clean', 'randomized', or 'both', got {data_mode}"
        
        # Initialize data structures
        if task_mode == "single":
            self.episode_files = []  # List of episode files for single task
        else:
            self.task_to_episodes = {}  # Task name -> episode files mapping
            self.task_weights = {}      # Task sampling weights
        self.all_episodes = []
        
        self.total_episodes = 0
        
        logger.info("RoboTwin dataset initialized:")
        logger.info(f"  Data mode: {data_mode}")
        logger.info(f"  Task mode: {task_mode}")
        if task_name:
            logger.info(f"  Task name: {task_name}")
        logger.info(f"  Global downsample rate: {global_downsample_rate}")
        logger.info(f"  Video:Action frequency ratio: {video_action_freq_ratio}")
        logger.info(f"  Action chunk size: {self.action_chunk_size}")
        logger.info(f"  Video frames to predict: {num_video_frames}")
        logger.info(f"  Action normalization: {self.action_normalization_config}")
        logger.info("  Sample indexing: exhaustive")
        
        # Load dataset episodes
        self._load_episodes()
        logger.info(f"  Total episodes: {self.total_episodes}")
        logger.info(f"  Total samples: {len(self)}")

    def _get_video_frame_count_cached(self, video_path: str) -> int:
        cached = self._frame_count_cache.get(video_path)
        if cached is not None:
            return cached
        frame_count = get_video_frame_count(video_path)
        self._frame_count_cache[video_path] = frame_count
        return frame_count

    def _get_qpos_frame_count_cached(self, qpos_path: str) -> int:
        cached = self._qpos_frame_count_cache.get(qpos_path)
        if cached is not None:
            return cached
        qpos_data = torch.load(qpos_path, map_location='cpu')
        if not isinstance(qpos_data, torch.Tensor):
            qpos_data = torch.as_tensor(qpos_data)
        frame_count = int(qpos_data.shape[0]) if qpos_data.dim() > 0 else 0
        self._qpos_frame_count_cache[qpos_path] = frame_count
        return frame_count
    
    def _limit_episodes_first_n(self, episodes: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        """
        Limit to the first N episodes based on sorted episode_name.
        Numeric names are sorted numerically; otherwise lexicographically.
        """
        if n is None or n <= 0 or not episodes:
            return episodes
        def _sort_key(ep: Dict[str, Any]):
            name = ep.get('episode_name', '')
            try:
                return (0, int(name))
            except Exception:
                return (1, str(name))
        episodes_sorted = sorted(episodes, key=_sort_key)
        return episodes_sorted[:n]
    
    def _scan_task_folder(self, task_path: Path, split_name: str) -> List[Dict[str, Any]]:
        """
        Scan a single task folder.
        
        Args:
            task_path: Path to task folder (e.g., .../clean/adjust_bottle)
            
        Returns:
            List of valid episode identifiers
        """
        qpos_dir = task_path / "qpos"
        videos_dir = task_path / "videos"
        umt5_dir = task_path / "umt5_wan"
        
        # Check if all required directories exist
        if not all([qpos_dir.exists(), videos_dir.exists(), umt5_dir.exists()]):
            logger.warning(f"Missing data directories in {task_path}")
            return []
        
        # Find valid episodes (those that have all three data types)
        valid_episodes = []
        
        # Get all qpos files as base (.pt format)
        qpos_files = list(qpos_dir.glob("*.pt"))
        
        for qpos_file in qpos_files:
            episode_name = qpos_file.stem
            
            # Check if corresponding video and language files exist
            video_file = videos_dir / f"{episode_name}.mp4"
            lang_file = umt5_dir / f"{episode_name}.pt"
            
            if video_file.exists() and lang_file.exists():
                # Store full paths
                episode_data = {
                    'episode_name': episode_name,
                    'task_name': task_path.name,
                    'split': split_name,
                    'qpos_path': str(qpos_file),
                    'video_path': str(video_file),
                    'lang_path': str(lang_file),
                }
                valid_episodes.append(episode_data)
        
        logger.info(f"Task {task_path.name} ({task_path.parent.name}): Found {len(valid_episodes)} valid episodes")
        return valid_episodes
    
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Initialize dataset by scanning folders."""
        logger.info("Initializing dataset...")
        
        # Determine which data splits to scan
        if self.data_mode == "both":
            data_splits = ["clean", "randomized"]
        else:
            data_splits = [self.data_mode]
        
        all_episodes = []
        
        for split in data_splits:
            split_dir = self.dataset_dir / split
            
            if not split_dir.exists():
                logger.warning(f"Split directory not found: {split_dir}")
                continue
            
            if self.task_mode == "single":
                # Single task mode: scan specific task
                task_dir = split_dir / self.task_name
                if task_dir.exists():
                    episodes = self._scan_task_folder(task_dir, split)
                    # If scanning randomized split, optionally limit to first N episodes per task
                    if split == "randomized" and self.randomized_limit_per_task is not None:
                        before = len(episodes)
                        episodes = self._limit_episodes_first_n(episodes, self.randomized_limit_per_task)
                        logger.info(f"Applied randomized per-task limit ({self.randomized_limit_per_task}) for {task_dir.name}: {before} -> {len(episodes)}")
                    all_episodes.extend(episodes)
                else:
                    logger.warning(f"Task directory not found: {task_dir}")
            
            else:
                # Multi task mode: scan all tasks
                task_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                
                for task_dir in task_dirs:
                    episodes = self._scan_task_folder(task_dir, split)
                    
                    # If scanning randomized split, optionally limit to first N episodes per task
                    if split == "randomized" and self.randomized_limit_per_task is not None:
                        before = len(episodes)
                        episodes = self._limit_episodes_first_n(episodes, self.randomized_limit_per_task)
                        logger.info(f"Applied randomized per-task limit ({self.randomized_limit_per_task}) for {task_dir.name}: {before} -> {len(episodes)}")
                    
                    # Group by task for multi-task sampling
                    task_name = task_dir.name
                    if task_name not in self.task_to_episodes:
                        self.task_to_episodes[task_name] = []
                    self.task_to_episodes[task_name].extend(episodes)
                    all_episodes.extend(episodes)
        
        if self.task_mode == "single":
            self.episode_files = all_episodes
            
            # Limit episodes if requested
            if self.max_episodes is not None:
                self.episode_files = self.episode_files[:self.max_episodes]
            
            self.total_episodes = len(self.episode_files)
            
            if self.total_episodes == 0:
                raise ValueError(f"No valid episodes found for task {self.task_name}")
        
        else:
            # Multi-task mode: calculate sampling weights
            if not self.task_to_episodes:
                raise ValueError("No valid episodes found for any task")
            
            # Equal weight for all tasks
            num_tasks = len(self.task_to_episodes)
            for task_name in self.task_to_episodes.keys():
                self.task_weights[task_name] = 1.0 / num_tasks
            
            self.total_episodes = sum(len(episodes) for episodes in self.task_to_episodes.values())
            
            logger.info(f"Multi-task dataset with {num_tasks} tasks:")
            for task_name, episodes in self.task_to_episodes.items():
                logger.info(f"  {task_name}: {len(episodes)} episodes")
        
        self.all_episodes = list(all_episodes)
        self._build_sample_index()

        # Return all episodes for consistency with other datasets
        return all_episodes

    def _effective_episode_frame_count(self, episode_data: Dict[str, Any]) -> int:
        video_frames = self._get_video_frame_count_cached(episode_data['video_path'])
        qpos_frames = self._get_qpos_frame_count_cached(episode_data['qpos_path'])
        return min(video_frames, qpos_frames)

    def _condition_indices_for_episode(self, total_frames: int) -> List[int]:
        physical_chunk_size = self.action_chunk_size * self.global_downsample_rate
        valid_count = total_frames - physical_chunk_size
        if valid_count <= 0:
            return []

        return list(range(0, valid_count))

    def _build_sample_index(self) -> None:
        self.sample_index = []
        dropped_short = 0
        dropped_errors = 0
        for episode_index, episode_data in enumerate(self.all_episodes):
            try:
                total_frames = self._effective_episode_frame_count(episode_data)
                condition_indices = self._condition_indices_for_episode(total_frames)
            except Exception as exc:
                dropped_errors += 1
                logger.warning(
                    "Skipping episode %s/%s while building sample index: %s",
                    episode_data.get("task_name", "?"),
                    episode_data.get("episode_name", "?"),
                    exc,
                )
                continue

            if not condition_indices:
                dropped_short += 1
                continue

            self.sample_index.extend((episode_index, condition_idx) for condition_idx in condition_indices)

        if not self.sample_index:
            raise ValueError("No valid exhaustive RoboTwin samples found")
        logger.info(
            "Built exhaustive RoboTwin sample index: samples=%d episodes=%d dropped_short=%d dropped_errors=%d",
            len(self.sample_index),
            len(self.all_episodes),
            dropped_short,
            dropped_errors,
        )

    def _load_robot_data(self, qpos_path: str, action_indices: List[int], initial_state_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load robot position data.
        
        Args:
            qpos_path: Path to qpos .pt file
            action_indices: List of action frame indices
            initial_state_idx: Index for initial state (should match condition frame)
            
        Returns:
            - initial_state: Robot state at condition frame [state_dim]
            - action_sequence: Actions at specified indices [len(action_indices), action_dim]
        """
        qpos_data = torch.load(qpos_path, map_location='cpu')  # [T, feature_dim]
        
        # Get initial state at the condition frame index
        if initial_state_idx >= len(qpos_data):
            initial_state_idx = len(qpos_data) - 1
        initial_state = qpos_data[initial_state_idx].float()
        
        # Get action sequence at specified indices
        actions = []
        for idx in action_indices:
            if idx >= len(qpos_data):
                raise IndexError(f"Action index {idx} out of bounds for qpos data length {len(qpos_data)}")
            else:
                action = qpos_data[idx]
            actions.append(action)
        
        action_sequence = torch.stack(actions).float()

        if self.action_normalizer.enabled:
            initial_state = self.action_normalizer.normalize(initial_state)
            action_sequence = self.action_normalizer.normalize(action_sequence)
        
        return initial_state, action_sequence
    
    def _load_language_embedding(self, lang_path: str, instruction_idx: Optional[int] = None) -> tuple[torch.Tensor, int]:
        """Load pre-encoded language embedding and return the selected index."""
        try:
            embedding_data = torch.load(lang_path, map_location='cpu')
            
            # RoboTwin embedding is always a list of tensors
            if instruction_idx is None:
                selected_idx = random.randint(0, len(embedding_data) - 1)
            else:
                selected_idx = int(instruction_idx) % len(embedding_data)
            embeddings = embedding_data[selected_idx]  # [seq_len, 4096]
            
            # Remove batch dimension if present
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(0)
            
            return embeddings, selected_idx
            
        except Exception as e:
            logger.error(f"Error loading language embedding from {lang_path}: {e}")
            raise
    
    def _load_text_instruction(self, task_name: str, episode_name: str, instruction_idx: int = None, split: Optional[str] = None) -> str:
        """Load text instruction from RoboTwin meta files.

        Args:
            task_name: Task name (e.g., "adjust_bottle")
            episode_name: Episode identifier (e.g., "429")
            instruction_idx: Optional index to select a deterministic instruction
            split: Dataset split to read from ("clean" or "randomized"). If None,
                   falls back to self.data_mode when it is not "both".
        """
        try:
            # Try to read from meta file first
            # Path structure: dataset_dir/split/task_name/metas/episode_name.txt
            if split is None:
                if self.data_mode in ["clean", "randomized"]:
                    split_to_use = self.data_mode
                else:
                    # Fallback to clean when split cannot be inferred
                    split_to_use = "clean"
            else:
                split_to_use = split

            meta_file = self.dataset_dir / split_to_use / task_name / "metas" / f"{episode_name}.txt"
            
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                    # Filter out empty lines
                    instructions = [line.strip() for line in lines if line.strip()]
                
                if instructions:
                    if instruction_idx is not None and 0 <= instruction_idx < len(instructions):
                        # Use specific index to match language_embedding
                        return instructions[instruction_idx]
                    else:
                        # Random selection (fallback for old behavior)
                        import random
                        return random.choice(instructions)
                else:
                    # No fallback - raise error if no instructions found
                    raise ValueError(f"No instructions found in meta file for {task_name}/{episode_name}")
            else:
                raise FileNotFoundError(f"Meta file not found: {meta_file}")
                
        except Exception as e:
            logger.error(f"Failed to load text instruction for {task_name}/{episode_name}: {e}")
            raise
    
    def _resolve_condition_frame_idx(
        self,
        total_frames: int,
        condition_frame_idx: Optional[int] = None,
        subclip_id: Optional[int] = None,
        state_id: Optional[int] = None,
        subclips_per_episode: int = 1,
        states_per_subclip: int = 1,
    ) -> int:
        """Resolve a condition frame either randomly or from deterministic subclip/state indices."""
        physical_chunk_size = self.action_chunk_size * self.global_downsample_rate
        max_condition_idx = total_frames - physical_chunk_size - 1

        if max_condition_idx < 0:
            return 0

        if condition_frame_idx is not None:
            if condition_frame_idx < 0 or condition_frame_idx > max_condition_idx:
                raise ValueError(
                    f"condition_frame_idx {condition_frame_idx} out of valid range [0, {max_condition_idx}]"
                )
            return int(condition_frame_idx)

        if subclip_id is None or state_id is None:
            return random.randint(0, max_condition_idx)

        if subclips_per_episode <= 0:
            raise ValueError(f"subclips_per_episode must be positive, got {subclips_per_episode}")
        if states_per_subclip <= 0:
            raise ValueError(f"states_per_subclip must be positive, got {states_per_subclip}")
        if not 0 <= subclip_id < subclips_per_episode:
            raise ValueError(f"subclip_id {subclip_id} out of range for {subclips_per_episode} subclips")
        if not 0 <= state_id < states_per_subclip:
            raise ValueError(f"state_id {state_id} out of range for {states_per_subclip} states")

        total_positions = max_condition_idx + 1
        subclip_start = (subclip_id * total_positions) // subclips_per_episode
        subclip_end = ((subclip_id + 1) * total_positions) // subclips_per_episode - 1
        if subclip_end < subclip_start:
            raise ValueError(
                f"Subclip {subclip_id} has no valid condition frames for total_frames={total_frames}"
            )

        span = subclip_end - subclip_start + 1
        offset = min((state_id * span) // states_per_subclip, span - 1)
        return subclip_start + offset

    def _calculate_sampling_indices(
        self,
        total_frames: int,
        condition_frame_idx: Optional[int] = None,
        subclip_id: Optional[int] = None,
        state_id: Optional[int] = None,
        subclips_per_episode: int = 1,
        states_per_subclip: int = 1,
    ) -> Tuple[int, List[int], List[int]]:
        """
        Calculate sampling indices.
        
        Args:
            total_frames: Total number of frames in the episode
            
        Returns:
            - condition_frame_idx: Index of condition frame (corresponds to initial state)
            - video_indices: List of video frame indices to predict
            - action_indices: List of action frame indices to predict
        """
        condition_frame_idx = self._resolve_condition_frame_idx(
            total_frames=total_frames,
            condition_frame_idx=condition_frame_idx,
            subclip_id=subclip_id,
            state_id=state_id,
            subclips_per_episode=subclips_per_episode,
            states_per_subclip=states_per_subclip,
        )
        
        # Action indices: from condition_frame_idx+1 onwards, with downsampling
        action_indices = []
        for i in range(self.action_chunk_size):
            # Each action is separated by global_downsample_rate frames
            action_idx = condition_frame_idx + (i + 1) * self.global_downsample_rate
            action_indices.append(min(action_idx, total_frames - 1))
        
        # Video indices: sample at frequency ratio intervals from action indices
        # Example: ratio=5, frames=[5, 10, 15, 20] for 4 video frames
        video_indices = []
        for i in range(self.num_video_frames):
            action_step = (i + 1) * self.video_action_freq_ratio - 1
            if action_step < len(action_indices):
                video_indices.append(action_indices[action_step])
            else:
                video_indices.append(action_indices[-1])
        
        # Verify spacing
        if len(action_indices) > 1:
            intervals = [action_indices[i+1] - action_indices[i] for i in range(len(action_indices)-1)]
            # print(f"  Action interval verification: {set(intervals)} (should all be {self.global_downsample_rate})")
        
        return condition_frame_idx, video_indices, action_indices

    def get_episode(self, episode_index: int) -> Dict[str, Any]:
        if not self.all_episodes:
            raise ValueError("Dataset contains no episodes")
        if episode_index < 0 or episode_index >= len(self.all_episodes):
            raise IndexError(f"episode_index {episode_index} out of range for {len(self.all_episodes)} episodes")
        return self.all_episodes[episode_index]

    def build_sample(
        self,
        episode_data: Dict[str, Any],
        *,
        condition_frame_idx: Optional[int] = None,
        subclip_id: Optional[int] = None,
        state_id: Optional[int] = None,
        subclips_per_episode: int = 1,
        states_per_subclip: int = 1,
        instruction_idx: Optional[int] = None,
        load_video: bool = True,
    ) -> Dict[str, Any]:
        """Build a single sample for a specific episode, optionally with deterministic slicing."""
        total_frames = self._get_video_frame_count_cached(episode_data['video_path'])
        if total_frames < 2:
            raise ValueError(f"Episode {episode_data['episode_name']} has fewer than 2 frames")

        resolved_condition_idx, video_indices, action_indices = self._calculate_sampling_indices(
            total_frames=total_frames,
            condition_frame_idx=condition_frame_idx,
            subclip_id=subclip_id,
            state_id=state_id,
            subclips_per_episode=subclips_per_episode,
            states_per_subclip=states_per_subclip,
        )

        initial_state, action_sequence = self._load_robot_data(
            episode_data['qpos_path'],
            action_indices,
            resolved_condition_idx,
        )
        language_embedding, resolved_instruction_idx = self._load_language_embedding(
            episode_data['lang_path'],
            instruction_idx=instruction_idx,
        )

        sample = {
            'episode_name': episode_data['episode_name'],
            'task_name': episode_data['task_name'],
            'split': episode_data.get('split', self.data_mode),
            'qpos_path': episode_data['qpos_path'],
            'video_path': episode_data['video_path'],
            'lang_path': episode_data['lang_path'],
            'instruction_idx': resolved_instruction_idx,
            'condition_frame_idx': resolved_condition_idx,
            'video_indices': video_indices,
            'action_indices': action_indices,
            'initial_state': initial_state,
            'action_sequence': action_sequence,
            'language_embedding': language_embedding,
        }
        if load_video:
            sampled_frames = load_video_frames(
                episode_data['video_path'],
                [resolved_condition_idx] + video_indices,
                self.video_size,
            )
            sample['first_frame'] = sampled_frames[0]
            sample['video_frames'] = sampled_frames[1:]
        return sample

    def build_indexed_sample(
        self,
        episode_index: int,
        *,
        condition_frame_idx: Optional[int] = None,
        subclip_id: Optional[int] = None,
        state_id: Optional[int] = None,
        subclips_per_episode: int = 1,
        states_per_subclip: int = 1,
        instruction_idx: Optional[int] = None,
        load_video: bool = True,
    ) -> Dict[str, Any]:
        episode_data = self.get_episode(episode_index)
        return self.build_sample(
            episode_data,
            condition_frame_idx=condition_frame_idx,
            subclip_id=subclip_id,
            state_id=state_id,
            subclips_per_episode=subclips_per_episode,
            states_per_subclip=states_per_subclip,
            instruction_idx=instruction_idx,
            load_video=load_video,
        )
    
    def __len__(self) -> int:
        """Return the real exhaustive sample count."""
        return len(self.sample_index)
    
    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Get a training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing training data
        """
        if not self.sample_index:
            raise ValueError("Dataset contains no indexed samples")
        episode_index, condition_frame_idx = self.sample_index[idx % len(self.sample_index)]
        episode_data = self.get_episode(episode_index)
        return self.build_sample(
            episode_data,
            condition_frame_idx=condition_frame_idx,
        )
