"""
Onset检测器 (Onset Detector)
使用多种算法检测音频中的击打点，专为编铙等打击乐器优化
"""

import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Optional, Union
from pathlib import Path
import logging

from .data_structures import (
    AudioSegment, 
    OnsetDetectionError, 
    AudioFormatError,
    PreprocessingError
)

logger = logging.getLogger(__name__)


class OnsetDetector:
    """
    音频击打点检测和分割器
    
    使用spectral flux、energy和complex domain多种方法融合检测onset，
    特别适用于编铙等快速衰减的打击乐器。
    """
    
    def __init__(
        self, 
        hop_length: int = 512,
        sr: int = 44100,
        threshold: float = 0.5,
        min_interval: float = 0.1,
        pre_max: int = 3,
        post_max: int = 3,
        pre_avg: int = 3,
        post_avg: int = 5,
        delta: float = 0.07,
        wait: int = 10
    ):
        """
        初始化Onset检测器
        
        Args:
            hop_length: STFT hop长度
            sr: 采样率
            threshold: 检测阈值 (0-1)
            min_interval: 最小间隔时间 (秒)，防止重复检测
            pre_max: peak_pick前向最大值窗口
            post_max: peak_pick后向最大值窗口
            pre_avg: peak_pick前向平均值窗口
            post_avg: peak_pick后向平均值窗口
            delta: peak_pick相对阈值
            wait: peak_pick等待时间
        """
        self.hop_length = hop_length
        self.sr = sr
        self.threshold = threshold
        self.min_interval = min_interval
        self.peak_pick_params = {
            'pre_max': pre_max,
            'post_max': post_max,
            'pre_avg': pre_avg,
            'post_avg': post_avg,
            'delta': delta,
            'wait': wait
        }
        
        # 内部状态
        self._last_onsets = []
        
        logger.info(f"OnsetDetector初始化完成 - threshold: {threshold}, hop_length: {hop_length}")
    
    def detect_onsets(self, audio_file: Union[str, Path]) -> List[float]:
        """
        检测音频文件中的击打点时间
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            List[float]: 击打点时间列表 (秒)
            
        Raises:
            FileNotFoundError: 音频文件不存在
            AudioFormatError: 不支持的音频格式
            OnsetDetectionError: 检测过程中的错误
        """
        try:
            # 加载音频文件
            audio_path = Path(audio_file)
            if not audio_path.exists():
                raise FileNotFoundError(f"音频文件不存在: {audio_file}")
            
            logger.info(f"加载音频文件: {audio_file}")
            y, sr = librosa.load(str(audio_path), sr=self.sr)
            
            if len(y) == 0:
                raise AudioFormatError(f"无法读取音频数据: {audio_file}")
            
            # 检测onset
            onsets = self._detect_onsets_from_array(y, sr)
            
            logger.info(f"检测到 {len(onsets)} 个onset")
            return onsets
            
        except Exception as e:
            if isinstance(e, (FileNotFoundError, AudioFormatError, OnsetDetectionError)):
                raise
            else:
                raise OnsetDetectionError(f"Onset检测失败: {e}")
    
    def _detect_onsets_from_array(self, y: np.ndarray, sr: int) -> List[float]:
        """
        从音频数组检测onset
        
        Args:
            y: 音频数据
            sr: 采样率
            
        Returns:
            List[float]: onset时间列表
        """
        try:
            # 方法1: Spectral flux
            onset_spectral = self._detect_spectral_flux(y, sr)
            
            # 方法2: Energy-based
            onset_energy = self._detect_energy_based(y, sr)
            
            # 方法3: Complex domain
            onset_complex = self._detect_complex_domain(y, sr)
            
            # 融合多种检测结果
            onsets_fused = self._fuse_onset_detections([
                onset_spectral, onset_energy, onset_complex
            ])
            
            # 后处理：过滤过近的onset
            onsets_filtered = self._filter_close_onsets(onsets_fused)
            
            return onsets_filtered
            
        except Exception as e:
            raise OnsetDetectionError(f"Onset检测计算失败: {e}")
    
    def _detect_spectral_flux(self, y: np.ndarray, sr: int) -> List[float]:
        """使用spectral flux检测onset"""
        try:
            onset_envelope = librosa.onset.onset_strength(
                y=y, 
                sr=sr, 
                hop_length=self.hop_length,
                aggregate=np.median,  # 使用median聚合，对噪声更鲁棒
                fmax=8000,  # 限制频率范围，编铙主要能量在8kHz以下
                n_mels=128
            )
            
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_envelope,
                sr=sr,
                hop_length=self.hop_length,
                **self.peak_pick_params,
                units='time'
            )
            
            return onsets.tolist()
            
        except Exception as e:
            logger.warning(f"Spectral flux检测失败: {e}")
            return []
    
    def _detect_energy_based(self, y: np.ndarray, sr: int) -> List[float]:
        """使用能量变化检测onset"""
        try:
            # 计算短时能量
            frame_length = 2048
            energy = []
            
            for i in range(0, len(y) - frame_length, self.hop_length):
                frame = y[i:i + frame_length]
                frame_energy = np.sum(frame ** 2)
                energy.append(frame_energy)
            
            energy = np.array(energy)
            if len(energy) == 0:
                return []
            
            # 计算能量差分
            energy_diff = np.diff(energy)
            energy_diff = np.maximum(energy_diff, 0)  # 只保留正向变化
            
            # 标准化
            if np.max(energy_diff) > 0:
                energy_diff = energy_diff / np.max(energy_diff)
            
            # 检测峰值
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(
                energy_diff, 
                height=self.threshold * 0.8,  # 稍微降低阈值
                distance=int(self.min_interval * sr / self.hop_length)
            )
            
            # 转换为时间
            onset_times = peaks * self.hop_length / sr
            
            return onset_times.tolist()
            
        except Exception as e:
            logger.warning(f"Energy-based检测失败: {e}")
            return []
    
    def _detect_complex_domain(self, y: np.ndarray, sr: int) -> List[float]:
        """使用complex domain检测onset"""
        try:
            # 使用complex domain onset detection
            onset_envelope = librosa.onset.onset_strength(
                y=y,
                sr=sr,
                hop_length=self.hop_length,
                feature=librosa.feature.chroma_stft,  # 使用chroma特征
                aggregate=np.mean
            )
            
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_envelope,
                sr=sr,
                hop_length=self.hop_length,
                **self.peak_pick_params,
                units='time'
            )
            
            return onsets.tolist()
            
        except Exception as e:
            logger.warning(f"Complex domain检测失败: {e}")
            return []
    
    def _fuse_onset_detections(self, onset_lists: List[List[float]]) -> List[float]:
        """
        融合多种onset检测结果
        
        Args:
            onset_lists: 多个onset时间列表
            
        Returns:
            List[float]: 融合后的onset时间列表
        """
        if not onset_lists:
            return []
        
        # 过滤空列表
        valid_lists = [onsets for onsets in onset_lists if onsets]
        if not valid_lists:
            return []
        
        # 如果只有一个有效列表，直接返回
        if len(valid_lists) == 1:
            return valid_lists[0]
        
        # 合并所有onset
        all_onsets = []
        for onsets in valid_lists:
            all_onsets.extend(onsets)
        
        if not all_onsets:
            return []
        
        # 排序
        all_onsets.sort()
        
        # 聚类：将相近的onset合并
        clustered_onsets = []
        tolerance = 0.05  # 50ms容差
        
        current_cluster = [all_onsets[0]]
        
        for onset in all_onsets[1:]:
            if onset - current_cluster[-1] <= tolerance:
                current_cluster.append(onset)
            else:
                # 计算当前聚类的代表值（取中位数）
                cluster_representative = np.median(current_cluster)
                clustered_onsets.append(cluster_representative)
                current_cluster = [onset]
        
        # 处理最后一个聚类
        if current_cluster:
            cluster_representative = np.median(current_cluster)
            clustered_onsets.append(cluster_representative)
        
        return clustered_onsets
    
    def _filter_close_onsets(self, onsets: List[float]) -> List[float]:
        """过滤过于接近的onset"""
        if not onsets:
            return []
        
        filtered = [onsets[0]]
        
        for onset in onsets[1:]:
            if onset - filtered[-1] >= self.min_interval:
                filtered.append(onset)
        
        return filtered
    
    def slice_audio(
        self, 
        audio_file: Union[str, Path], 
        padding: float = 0.1,
        min_segment_length: float = 0.2,
        max_segment_length: float = 5.0
    ) -> List[AudioSegment]:
        """
        根据onset检测结果分割音频
        
        Args:
            audio_file: 音频文件路径
            padding: 分割时的padding时间 (秒)
            min_segment_length: 最小片段长度 (秒)
            max_segment_length: 最大片段长度 (秒)
            
        Returns:
            List[AudioSegment]: 分割后的音频片段列表
        """
        try:
            # 检测onset
            onset_times = self.detect_onsets(audio_file)
            
            if not onset_times:
                logger.warning(f"未检测到onset，返回整个音频: {audio_file}")
                # 如果没有检测到onset，返回整个音频
                y, sr = librosa.load(str(audio_file), sr=self.sr)
                segment = AudioSegment(
                    audio_data=y,
                    sample_rate=sr,
                    onset_time=0.0,
                    duration=len(y) / sr,
                    source_file=str(audio_file),
                    segment_id="full_audio"
                )
                return [segment]
            
            # 加载音频
            y, sr = librosa.load(str(audio_file), sr=self.sr)
            audio_duration = len(y) / sr
            
            segments = []
            
            for i, onset_time in enumerate(onset_times):
                # 计算片段起始和结束时间
                start_time = max(0, onset_time - padding)
                
                if i < len(onset_times) - 1:
                    # 不是最后一个onset，结束时间是下一个onset
                    end_time = min(audio_duration, onset_times[i + 1] - padding/2)
                else:
                    # 最后一个onset，使用固定长度或到文件结尾
                    end_time = min(audio_duration, onset_time + max_segment_length)
                
                segment_duration = end_time - start_time
                
                # 检查片段长度
                if segment_duration < min_segment_length:
                    logger.debug(f"跳过过短片段: {segment_duration:.3f}s < {min_segment_length}s")
                    continue
                
                # 提取音频片段
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = y[start_sample:end_sample]
                
                if len(segment_audio) == 0:
                    continue
                
                segment = AudioSegment(
                    audio_data=segment_audio,
                    sample_rate=sr,
                    onset_time=onset_time,
                    duration=segment_duration,
                    source_file=str(audio_file),
                    segment_id=f"segment_{i:03d}"
                )
                
                segments.append(segment)
                logger.debug(f"创建片段 {i}: {onset_time:.3f}s - {end_time:.3f}s ({segment_duration:.3f}s)")
            
            logger.info(f"成功分割 {len(segments)} 个音频片段")
            return segments
            
        except Exception as e:
            raise OnsetDetectionError(f"音频分割失败: {e}")
    
    def set_parameters(self, **kwargs) -> None:
        """
        动态设置检测参数
        
        Args:
            **kwargs: 参数字典 (threshold, hop_length, min_interval等)
        """
        if 'threshold' in kwargs:
            self.threshold = float(kwargs['threshold'])
            logger.info(f"更新threshold: {self.threshold}")
        
        if 'hop_length' in kwargs:
            self.hop_length = int(kwargs['hop_length'])
            logger.info(f"更新hop_length: {self.hop_length}")
        
        if 'min_interval' in kwargs:
            self.min_interval = float(kwargs['min_interval'])
            logger.info(f"更新min_interval: {self.min_interval}")
        
        # 更新peak_pick参数
        for param in ['pre_max', 'post_max', 'pre_avg', 'post_avg', 'delta', 'wait']:
            if param in kwargs:
                self.peak_pick_params[param] = kwargs[param]
                logger.info(f"更新{param}: {kwargs[param]}")
    
    def get_onset_strength(self, audio_file: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取onset强度函数，用于可视化和调试
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (时间轴, onset强度)
        """
        try:
            y, sr = librosa.load(str(audio_file), sr=self.sr)
            
            onset_envelope = librosa.onset.onset_strength(
                y=y,
                sr=sr,
                hop_length=self.hop_length
            )
            
            # 生成时间轴
            times = librosa.times_like(onset_envelope, sr=sr, hop_length=self.hop_length)
            
            return times, onset_envelope
            
        except Exception as e:
            raise OnsetDetectionError(f"获取onset强度失败: {e}")
