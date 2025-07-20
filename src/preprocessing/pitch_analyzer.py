"""
音高分析器 (Pitch Analyzer)
专为编铙等打击乐器的音高检测优化
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional, Union
from pathlib import Path
import logging

from .data_structures import (
    AudioSegment,
    PitchInfo,
    PitchDetectionError,
    PreprocessingError
)

logger = logging.getLogger(__name__)


class PitchAnalyzer:
    """
    音高分析器
    
    使用多种算法分析音频片段的音高信息，
    特别适用于编铙等非谐波打击乐器。
    """
    
    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        method: str = 'yin',
        fmin: float = 50.0,
        fmax: float = 2000.0,
        threshold: float = 0.1
    ):
        """
        初始化音高分析器
        
        Args:
            frame_length: 分析帧长度
            hop_length: 帧移
            method: 分析方法 ('yin', 'pyin', 'spectral')
            fmin: 最小频率 (Hz)
            fmax: 最大频率 (Hz)
            threshold: YIN算法阈值
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.method = method
        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold
        
        logger.info(f"PitchAnalyzer初始化完成 - method: {method}, fmin: {fmin}, fmax: {fmax}")
    
    def analyze_pitch(self, audio_segment: AudioSegment) -> PitchInfo:
        """
        分析音频片段的音高
        
        Args:
            audio_segment: 音频片段
            
        Returns:
            PitchInfo: 音高分析结果
            
        Raises:
            PitchDetectionError: 音高检测失败
        """
        try:
            y = audio_segment.audio_data
            sr = audio_segment.sample_rate
            
            if len(y) == 0:
                raise PitchDetectionError("音频数据为空")
            
            # 使用指定方法检测音高
            if self.method == 'yin':
                fundamental_freq, confidence, trajectory = self._analyze_with_yin(y, sr)
            elif self.method == 'pyin':
                fundamental_freq, confidence, trajectory = self._analyze_with_pyin(y, sr)
            elif self.method == 'spectral':
                fundamental_freq, confidence, trajectory = self._analyze_with_spectral(y, sr)
            else:
                # 默认使用多算法融合
                fundamental_freq, confidence, trajectory = self._analyze_multi_method(y, sr)
            
            # 转换为音符名称
            note_name, octave, cents_deviation = self._freq_to_note(fundamental_freq)
            
            # 计算音高稳定性
            pitch_stability = self._calculate_pitch_stability(trajectory)
            
            # 检测泛音
            harmonics = self._detect_harmonics(y, sr, fundamental_freq)
            
            pitch_info = PitchInfo(
                fundamental_freq=fundamental_freq,
                note_name=note_name,
                octave=octave,
                cents_deviation=cents_deviation,
                confidence=confidence,
                harmonics=harmonics,
                pitch_stability=pitch_stability,
                frequency_trajectory=trajectory,
                detection_method=self.method
            )
            
            logger.debug(f"音高分析完成: {fundamental_freq:.2f}Hz ({note_name}), 置信度: {confidence:.3f}")
            return pitch_info
            
        except Exception as e:
            if isinstance(e, PitchDetectionError):
                raise
            else:
                raise PitchDetectionError(f"音高分析失败: {e}")
    
    def _analyze_with_yin(self, y: np.ndarray, sr: int) -> Tuple[float, float, List[float]]:
        """使用YIN算法分析音高"""
        try:
            f0 = librosa.yin(
                y,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                hop_length=self.hop_length,
                frame_length=self.frame_length
            )
            
            # 确保f0是numpy数组
            f0 = np.asarray(f0)
            
            # 过滤无效值：NaN、无穷大、负值和零值
            valid_mask = np.isfinite(f0) & (f0 > 0)
            valid_f0 = f0[valid_mask]
            
            if len(valid_f0) == 0:
                return 0.0, 0.0, [0.0]
            
            # 计算基频（中位数，对异常值更鲁棒）
            fundamental_freq = float(np.median(valid_f0))
            
            # 计算置信度（基于稳定性）
            confidence = self._calculate_confidence(valid_f0.tolist(), fundamental_freq)
            
            # 频率轨迹，将NaN和无效值替换为0
            trajectory = []
            for freq in f0:
                if np.isfinite(freq) and freq > 0:
                    trajectory.append(float(freq))
                else:
                    trajectory.append(0.0)
            
            return fundamental_freq, confidence, trajectory
            
        except Exception as e:
            logger.warning(f"YIN算法失败: {e}")
            return 0.0, 0.0, [0.0]
    
    def _analyze_with_pyin(self, y: np.ndarray, sr: int) -> Tuple[float, float, List[float]]:
        """使用PYIN算法分析音高（处理噪声信号）"""
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                hop_length=self.hop_length,
                frame_length=self.frame_length
            )
            
            # 只考虑有声帧
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) == 0:
                return 0.0, 0.0, [0.0]
            
            fundamental_freq = float(np.median(voiced_f0))
            
            # 置信度基于有声概率的平均值
            confidence = float(np.mean(voiced_probs[voiced_flag]))
            
            trajectory = f0.tolist()
            
            return fundamental_freq, confidence, trajectory
            
        except Exception as e:
            logger.warning(f"PYIN算法失败: {e}")
            return 0.0, 0.0, [0.0]
    
    def _analyze_with_spectral(self, y: np.ndarray, sr: int) -> Tuple[float, float, List[float]]:
        """使用频谱峰值检测音高"""
        try:
            # 计算STFT
            stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)
            
            # 对每帧找到峰值频率
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
            f0_trajectory = []
            
            for frame in range(magnitude.shape[1]):
                frame_mag = magnitude[:, frame]
                
                # 在指定频率范围内找峰值
                freq_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
                if not np.any(freq_mask):
                    f0_trajectory.append(0.0)
                    continue
                
                masked_freqs = freqs[freq_mask]
                masked_mag = frame_mag[freq_mask]
                
                if len(masked_mag) == 0:
                    f0_trajectory.append(0.0)
                    continue
                
                # 找到最大幅度对应的频率
                peak_idx = np.argmax(masked_mag)
                peak_freq = masked_freqs[peak_idx]
                f0_trajectory.append(float(peak_freq))
            
            # 过滤有效值
            valid_f0 = [f for f in f0_trajectory if f > 0]
            if not valid_f0:
                return 0.0, 0.0, [0.0]
            
            fundamental_freq = float(np.median(valid_f0))
            confidence = self._calculate_confidence(valid_f0, fundamental_freq)
            
            return fundamental_freq, confidence, f0_trajectory
            
        except Exception as e:
            logger.warning(f"频谱分析失败: {e}")
            return 0.0, 0.0, [0.0]
    
    def _analyze_multi_method(self, y: np.ndarray, sr: int) -> Tuple[float, float, List[float]]:
        """多算法融合分析"""
        try:
            # 尝试多种方法
            results = []
            
            # YIN
            f0_yin, conf_yin, traj_yin = self._analyze_with_yin(y, sr)
            if f0_yin > 0:
                results.append((f0_yin, conf_yin, traj_yin, 'yin'))
            
            # PYIN
            f0_pyin, conf_pyin, traj_pyin = self._analyze_with_pyin(y, sr)
            if f0_pyin > 0:
                results.append((f0_pyin, conf_pyin, traj_pyin, 'pyin'))
            
            # 频谱方法
            f0_spec, conf_spec, traj_spec = self._analyze_with_spectral(y, sr)
            if f0_spec > 0:
                results.append((f0_spec, conf_spec, traj_spec, 'spectral'))
            
            if not results:
                return 0.0, 0.0, [0.0]
            
            # 选择置信度最高的结果
            best_result = max(results, key=lambda x: x[1])
            fundamental_freq, confidence, trajectory, method = best_result
            
            self.method = method  # 更新使用的方法
            return fundamental_freq, confidence, trajectory
            
        except Exception as e:
            logger.warning(f"多算法融合失败: {e}")
            return 0.0, 0.0, [0.0]
    
    def _freq_to_note(self, freq: float) -> Tuple[str, int, float]:
        """将频率转换为音符名称"""
        if freq <= 0:
            return "N/A", 0, 0.0
        
        # A4 = 440Hz
        A4_freq = 440.0
        
        # 计算相对于A4的半音数
        semitones_from_A4 = 12 * np.log2(freq / A4_freq)
        
        # 计算音符和八度
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # A4是第9个音符（从C开始数）
        semitone_in_octave = (9 + semitones_from_A4) % 12
        note_index = int(round(semitone_in_octave)) % 12
        note_name = note_names[note_index]
        
        # 计算八度
        octave = 4 + int((9 + semitones_from_A4) // 12)
        
        # 计算音分偏差
        exact_semitone = 9 + semitones_from_A4
        rounded_semitone = round(exact_semitone)
        cents_deviation = (exact_semitone - rounded_semitone) * 100
        
        return note_name, octave, cents_deviation
    
    def _calculate_confidence(self, f0_values: List[float], target_freq: float) -> float:
        """计算音高检测置信度"""
        if not f0_values or target_freq <= 0:
            return 0.0
        
        # 计算相对标准差
        f0_array = np.array(f0_values)
        relative_std = np.std(f0_array) / target_freq
        
        # 置信度与稳定性成反比
        confidence = np.exp(-relative_std * 10)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _calculate_pitch_stability(self, trajectory: List[float]) -> float:
        """计算音高稳定性"""
        if not trajectory:
            return 0.0
        
        valid_values = [f for f in trajectory if f > 0]
        if len(valid_values) < 2:
            return 0.0
        
        # 计算变异系数
        mean_freq = np.mean(valid_values)
        std_freq = np.std(valid_values)
        
        if mean_freq == 0:
            return 0.0
        
        cv = std_freq / mean_freq
        stability = np.exp(-cv * 5)  # 变异系数越小，稳定性越高
        
        return float(np.clip(stability, 0.0, 1.0))
    
    def _detect_harmonics(self, y: np.ndarray, sr: int, fundamental_freq: float) -> List[dict]:
        """检测泛音"""
        if fundamental_freq <= 0:
            return []
        
        try:
            # 计算频谱
            stft = librosa.stft(y, n_fft=4096)
            magnitude = np.mean(np.abs(stft), axis=1)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
            
            harmonics = []
            
            # 检测前5个泛音
            for harmonic_num in range(2, 7):  # 2倍频到6倍频
                harmonic_freq = fundamental_freq * harmonic_num
                
                # 在泛音频率附近寻找峰值
                tolerance = fundamental_freq * 0.1  # 10%容差
                freq_mask = (freqs >= harmonic_freq - tolerance) & (freqs <= harmonic_freq + tolerance)
                
                if np.any(freq_mask):
                    harmonic_magnitudes = magnitude[freq_mask]
                    harmonic_freqs = freqs[freq_mask]
                    
                    if len(harmonic_magnitudes) > 0:
                        peak_idx = np.argmax(harmonic_magnitudes)
                        peak_freq = harmonic_freqs[peak_idx]
                        peak_amplitude = harmonic_magnitudes[peak_idx]
                        
                        harmonics.append({
                            "frequency": float(peak_freq),
                            "amplitude": float(peak_amplitude)
                        })
            
            return harmonics
            
        except Exception as e:
            logger.warning(f"泛音检测失败: {e}")
            return []
    
    def estimate_note_name(self, frequency: float) -> str:
        """将频率转换为音符名称"""
        note_name, octave, _ = self._freq_to_note(frequency)
        return f"{note_name}{octave}"
    
    def calculate_confidence(self, audio_segment: AudioSegment) -> float:
        """计算音高检测的置信度"""
        try:
            pitch_info = self.analyze_pitch(audio_segment)
            return pitch_info.confidence
        except Exception as e:
            logger.warning(f"置信度计算失败: {e}")
            return 0.0
