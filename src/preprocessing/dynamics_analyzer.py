"""
动态分析器 (Dynamics Analyzer)
分析音频片段的力度和动态特征，专为编铙等打击乐器优化
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional, Union
import logging

from .data_structures import (
    AudioSegment,
    DynamicsInfo,
    DynamicsAnalysisError,
    PreprocessingError
)

logger = logging.getLogger(__name__)


class DynamicsAnalyzer:
    """
    动态分析器
    
    分析音频片段的力度、响度、包络等动态特征，
    特别适用于编铙等打击乐器的表达性分析。
    """
    
    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
        attack_threshold: float = 0.1,
        decay_threshold: float = 0.1
    ):
        """
        初始化动态分析器
        
        Args:
            frame_length: 分析帧长度
            hop_length: 帧移
            attack_threshold: 攻击检测阈值
            decay_threshold: 衰减检测阈值
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.attack_threshold = attack_threshold
        self.decay_threshold = decay_threshold
        
        logger.info(f"DynamicsAnalyzer初始化完成")
    
    def analyze_dynamics(self, audio_segment: AudioSegment) -> DynamicsInfo:
        """
        分析音频片段的动态特征
        
        Args:
            audio_segment: 音频片段
            
        Returns:
            DynamicsInfo: 动态分析结果
            
        Raises:
            DynamicsAnalysisError: 动态分析失败
        """
        try:
            y = audio_segment.audio_data
            sr = audio_segment.sample_rate
            
            if len(y) == 0:
                raise DynamicsAnalysisError("音频数据为空")
            
            # 计算基本能量指标
            rms_energy = self._calculate_rms_energy(y)
            peak_amplitude = self._calculate_peak_amplitude(y)
            
            # 计算响度 (简化的LUFS计算)
            loudness_lufs = self._calculate_loudness(y, sr)
            
            # 计算动态范围
            dynamic_range_db = self._calculate_dynamic_range(y)
            
            # 分类动态等级
            dynamic_level = self._classify_dynamic_level(rms_energy)
            
            # 分析包络特征 (ADSR)
            envelope = self._calculate_envelope(y)
            adsr_params = self._analyze_adsr_envelope(envelope, sr)
            
            dynamics_info = DynamicsInfo(
                rms_energy=rms_energy,
                peak_amplitude=peak_amplitude,
                loudness_lufs=loudness_lufs,
                dynamic_range_db=dynamic_range_db,
                dynamic_level=dynamic_level,
                attack_time_ms=adsr_params['attack_time_ms'],
                decay_time_ms=adsr_params['decay_time_ms'],
                sustain_level=adsr_params['sustain_level'],
                release_time_ms=adsr_params['release_time_ms'],
                envelope_shape=adsr_params['envelope_shape']
            )
            
            logger.debug(f"动态分析完成: RMS={rms_energy:.4f}, 等级={dynamic_level}")
            return dynamics_info
            
        except Exception as e:
            if isinstance(e, DynamicsAnalysisError):
                raise
            else:
                raise DynamicsAnalysisError(f"动态分析失败: {e}")
    
    def _calculate_rms_energy(self, y: np.ndarray) -> float:
        """计算RMS能量"""
        if len(y) == 0:
            return 0.0
        
        rms = np.sqrt(np.mean(y ** 2))
        return float(rms)
    
    def _calculate_peak_amplitude(self, y: np.ndarray) -> float:
        """计算峰值幅度"""
        if len(y) == 0:
            return 0.0
        
        peak = np.max(np.abs(y))
        return float(peak)
    
    def _calculate_loudness(self, y: np.ndarray, sr: int) -> float:
        """
        计算响度 (简化的LUFS)
        这里使用A加权滤波器近似
        """
        try:
            # 使用librosa计算频谱
            stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.frame_length)
            magnitude = np.abs(stft)
            
            # 频率轴
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
            
            # 简化的A加权滤波 (只在关键频率点)
            # 实际A加权比这复杂，这里只是近似
            a_weights = np.ones_like(freqs)
            
            # 1kHz附近增强，低频和高频衰减
            for i, freq in enumerate(freqs):
                if freq < 100:
                    a_weights[i] = 0.1
                elif freq < 1000:
                    a_weights[i] = freq / 1000
                elif freq < 3000:
                    a_weights[i] = 1.0
                else:
                    a_weights[i] = max(0.1, 3000 / freq)
            
            # 应用A加权
            weighted_magnitude = magnitude * a_weights[:, np.newaxis]
            
            # 计算平均响度
            mean_power = np.mean(weighted_magnitude ** 2)
            loudness_lufs = 10 * np.log10(mean_power + 1e-10) - 23  # 相对于-23 LUFS
            
            return float(loudness_lufs)
            
        except Exception as e:
            logger.warning(f"响度计算失败: {e}")
            # 回退到简单的dB计算
            rms = self._calculate_rms_energy(y)
            db = 20 * np.log10(rms + 1e-10)
            return float(db)
    
    def _calculate_dynamic_range(self, y: np.ndarray) -> float:
        """计算动态范围"""
        if len(y) == 0:
            return 0.0
        
        # 计算95%和5%分位数的差值
        y_abs = np.abs(y)
        percentile_95 = np.percentile(y_abs, 95)
        percentile_5 = np.percentile(y_abs, 5)
        
        if percentile_5 > 0:
            dynamic_range = 20 * np.log10(percentile_95 / percentile_5)
        else:
            dynamic_range = 20 * np.log10(percentile_95 / (np.mean(y_abs) + 1e-10))
        
        return float(dynamic_range)
    
    def _classify_dynamic_level(self, rms_energy: float) -> str:
        """
        根据RMS能量分类动态等级
        
        参考古典音乐动态标记：
        pp (pianissimo) - 很弱
        p (piano) - 弱
        mp (mezzo-piano) - 中弱
        mf (mezzo-forte) - 中强
        f (forte) - 强
        ff (fortissimo) - 很强
        """
        # 这些阈值需要根据实际编铙录音数据调整
        if rms_energy < 0.01:
            return "pp"
        elif rms_energy < 0.05:
            return "p"
        elif rms_energy < 0.15:
            return "mp"
        elif rms_energy < 0.35:
            return "mf"
        elif rms_energy < 0.7:
            return "f"
        else:
            return "ff"
    
    def _calculate_envelope(self, y: np.ndarray) -> np.ndarray:
        """计算音频包络"""
        try:
            # 使用scipy的Hilbert变换计算瞬时幅度
            from scipy.signal import hilbert
            analytic_signal = hilbert(y)
            envelope = np.abs(analytic_signal)
            
            # 平滑处理
            window_size = len(y) // 200  # 平滑窗口大小
            if window_size > 1:
                envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
            
            return envelope
            
        except Exception as e:
            logger.warning(f"包络计算失败: {e}")
            # 回退到简单的滑动RMS
            frame_size = len(y) // 100
            if frame_size < 1:
                return np.abs(y)
            
            envelope = []
            for i in range(0, len(y), frame_size):
                frame = y[i:i+frame_size]
                if len(frame) > 0:
                    rms = np.sqrt(np.mean(frame ** 2))
                    envelope.extend([rms] * len(frame))
            
            return np.array(envelope[:len(y)])
    
    def _analyze_adsr_envelope(self, envelope: np.ndarray, sr: int) -> dict:
        """
        分析ADSR包络参数
        Attack-Decay-Sustain-Release
        """
        try:
            if len(envelope) == 0:
                return {
                    'attack_time_ms': 0.0,
                    'decay_time_ms': 0.0,
                    'sustain_level': 0.0,
                    'release_time_ms': 0.0,
                    'envelope_shape': {'type': 'unknown'}
                }
            
            # 标准化包络
            max_amp = np.max(envelope)
            if max_amp == 0:
                return {
                    'attack_time_ms': 0.0,
                    'decay_time_ms': 0.0,
                    'sustain_level': 0.0,
                    'release_time_ms': 0.0,
                    'envelope_shape': {'type': 'silence'}
                }
            
            norm_envelope = envelope / max_amp
            
            # 找到峰值位置
            peak_idx = np.argmax(norm_envelope)
            
            # Attack时间：从开始到峰值
            attack_samples = peak_idx
            attack_time_ms = (attack_samples / sr) * 1000
            
            # 找到sustain level (峰值后稳定的电平)
            if peak_idx < len(norm_envelope) - 1:
                post_peak = norm_envelope[peak_idx:]
                # 取后半段的中位数作为sustain level
                mid_point = len(post_peak) // 2
                if mid_point > 0:
                    sustain_level = float(np.median(post_peak[mid_point:]))
                else:
                    sustain_level = float(norm_envelope[-1])
            else:
                sustain_level = float(norm_envelope[-1])
            
            # Decay时间：从峰值到sustain level
            decay_idx = peak_idx
            sustain_threshold = max_amp * 0.8  # 80%作为decay结束点
            
            for i in range(peak_idx, len(norm_envelope)):
                if norm_envelope[i] <= sustain_threshold:
                    decay_idx = i
                    break
            
            decay_samples = decay_idx - peak_idx
            decay_time_ms = (decay_samples / sr) * 1000
            
            # Release时间：从sustain到结束
            # 找到开始release的点 (sustain结束)
            release_start_idx = decay_idx
            
            # 从后往前找，找到明显开始衰减的点
            for i in range(len(norm_envelope) - 1, decay_idx, -1):
                if norm_envelope[i] > sustain_level * 1.2:  # 高于sustain 20%
                    release_start_idx = i
                    break
            
            release_samples = len(norm_envelope) - release_start_idx
            release_time_ms = (release_samples / sr) * 1000
            
            # 分析包络形状
            envelope_shape = self._classify_envelope_shape(norm_envelope, peak_idx)
            
            return {
                'attack_time_ms': float(attack_time_ms),
                'decay_time_ms': float(decay_time_ms),
                'sustain_level': sustain_level,
                'release_time_ms': float(release_time_ms),
                'envelope_shape': envelope_shape
            }
            
        except Exception as e:
            logger.warning(f"ADSR分析失败: {e}")
            return {
                'attack_time_ms': 0.0,
                'decay_time_ms': 0.0,
                'sustain_level': 0.0,
                'release_time_ms': 0.0,
                'envelope_shape': {'type': 'error'}
            }
    
    def _classify_envelope_shape(self, envelope: np.ndarray, peak_idx: int) -> dict:
        """分类包络形状"""
        try:
            if len(envelope) < 3:
                return {'type': 'too_short'}
            
            # 计算攻击斜率
            if peak_idx > 0:
                attack_slope = envelope[peak_idx] / peak_idx
            else:
                attack_slope = 0
            
            # 计算衰减斜率
            if peak_idx < len(envelope) - 1:
                decay_section = envelope[peak_idx:]
                if len(decay_section) > 1:
                    decay_slope = (decay_section[-1] - decay_section[0]) / len(decay_section)
                else:
                    decay_slope = 0
            else:
                decay_slope = 0
            
            # 分类
            if attack_slope > 0.1 and abs(decay_slope) > 0.01:
                envelope_type = "percussive"  # 快攻击，明显衰减
            elif attack_slope > 0.05:
                envelope_type = "plucked"     # 中等攻击
            elif attack_slope < 0.02:
                envelope_type = "bowed"       # 慢攻击
            else:
                envelope_type = "generic"
            
            return {
                'type': envelope_type,
                'attack_slope': float(attack_slope),
                'decay_slope': float(decay_slope)
            }
            
        except Exception as e:
            logger.warning(f"包络形状分类失败: {e}")
            return {'type': 'unknown'}
    
    def estimate_volume_level(self, audio_segment: AudioSegment) -> str:
        """估计音量等级"""
        try:
            dynamics_info = self.analyze_dynamics(audio_segment)
            return dynamics_info.dynamic_level
        except Exception as e:
            logger.warning(f"音量等级估计失败: {e}")
            return "mp"  # 默认中等音量
    
    def calculate_loudness_similarity(self, segment1: AudioSegment, segment2: AudioSegment) -> float:
        """计算两个音频片段的响度相似性"""
        try:
            dynamics1 = self.analyze_dynamics(segment1)
            dynamics2 = self.analyze_dynamics(segment2)
            
            # 基于RMS能量计算相似性
            rms_diff = abs(dynamics1.rms_energy - dynamics2.rms_energy)
            max_rms = max(dynamics1.rms_energy, dynamics2.rms_energy)
            
            if max_rms == 0:
                return 1.0
            
            similarity = 1.0 - (rms_diff / max_rms)
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"响度相似性计算失败: {e}")
            return 0.0
