"""
音频预处理模块的数据结构定义
包含音频片段、音高信息、力度信息等核心数据类型
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class AudioSegment:
    """音频片段数据结构"""
    audio_data: np.ndarray    # 音频数据
    sample_rate: int          # 采样率
    onset_time: float         # 起始时间 (秒)
    duration: float           # 持续时间 (秒)
    source_file: str          # 源文件路径
    segment_id: Optional[str] = None  # 片段ID


@dataclass
class PitchInfo:
    """音高分析结果"""
    fundamental_freq: float           # 基频 (Hz)
    note_name: str                   # 音符名称 (如 "A4", "C#3")
    octave: int                      # 八度
    cents_deviation: float           # 音分偏差
    confidence: float                # 检测置信度 (0-1)
    harmonics: List[Dict[str, float]] # 泛音频率和幅度
    pitch_stability: float           # 音高稳定性 (0-1)
    frequency_trajectory: List[float] # 频率轨迹
    detection_method: str = "yin"    # 检测方法


@dataclass
class DynamicsInfo:
    """动态分析结果"""
    rms_energy: float                # RMS能量
    peak_amplitude: float            # 峰值幅度
    loudness_lufs: float            # LUFS响度
    dynamic_range_db: float         # 动态范围 (dB)
    dynamic_level: str              # 分类等级 ("pp", "p", "mp", "mf", "f", "ff")
    attack_time_ms: float           # 攻击时间 (毫秒)
    decay_time_ms: float            # 衰减时间 (毫秒)
    sustain_level: float            # 延音电平
    release_time_ms: float          # 释放时间 (毫秒)
    envelope_shape: Dict[str, str] = None  # 包络形状描述


@dataclass
class SpectralFeatures:
    """频谱特征"""
    spectral_centroid: float        # 频谱质心
    spectral_rolloff: float         # 频谱滚降
    spectral_bandwidth: float       # 频谱带宽
    spectral_flatness: float        # 频谱平坦度
    spectral_contrast: List[float]  # 频谱对比度
    zero_crossing_rate: float       # 过零率
    mfcc: np.ndarray               # MFCC系数
    chroma: np.ndarray             # 色度特征
    tonnetz: np.ndarray            # Tonnetz特征


@dataclass
class TemporalFeatures:
    """时间域特征"""
    duration: float                 # 持续时间
    onset_strength: float           # onset强度
    offset_strength: float          # offset强度
    transient_ratio: float          # 瞬态比例
    steady_state_ratio: float       # 稳态比例
    tempo_stability: float = 0.0    # 节拍稳定性
    rhythmic_pattern: Optional[List[float]] = None  # 节奏模式


@dataclass
class AudioFeatures:
    """完整的音频特征集合"""
    pitch: PitchInfo
    dynamics: DynamicsInfo
    spectral: SpectralFeatures
    temporal: TemporalFeatures
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        return {
            "pitch": {
                "fundamental_freq": self.pitch.fundamental_freq,
                "note_name": self.pitch.note_name,
                "octave": self.pitch.octave,
                "cents_deviation": self.pitch.cents_deviation,
                "confidence": self.pitch.confidence,
                "harmonics": self.pitch.harmonics,
                "pitch_stability": self.pitch.pitch_stability,
                "frequency_trajectory": self.pitch.frequency_trajectory,
                "detection_method": self.pitch.detection_method,
            },
            "dynamics": {
                "rms_energy": self.dynamics.rms_energy,
                "peak_amplitude": self.dynamics.peak_amplitude,
                "loudness_lufs": self.dynamics.loudness_lufs,
                "dynamic_range_db": self.dynamics.dynamic_range_db,
                "dynamic_level": self.dynamics.dynamic_level,
                "attack_time_ms": self.dynamics.attack_time_ms,
                "decay_time_ms": self.dynamics.decay_time_ms,
                "sustain_level": self.dynamics.sustain_level,
                "release_time_ms": self.dynamics.release_time_ms,
                "envelope_shape": self.dynamics.envelope_shape,
            },
            "spectral": {
                "spectral_centroid": self.spectral.spectral_centroid,
                "spectral_rolloff": self.spectral.spectral_rolloff,
                "spectral_bandwidth": self.spectral.spectral_bandwidth,
                "spectral_flatness": self.spectral.spectral_flatness,
                "spectral_contrast": self.spectral.spectral_contrast.tolist() if hasattr(self.spectral.spectral_contrast, 'tolist') else self.spectral.spectral_contrast,
                "zero_crossing_rate": self.spectral.zero_crossing_rate,
                "mfcc": self.spectral.mfcc.tolist(),
                "chroma": self.spectral.chroma.tolist(),
                "tonnetz": self.spectral.tonnetz.tolist(),
            },
            "temporal": {
                "duration": self.temporal.duration,
                "onset_strength": self.temporal.onset_strength,
                "offset_strength": self.temporal.offset_strength,
                "transient_ratio": self.temporal.transient_ratio,
                "steady_state_ratio": self.temporal.steady_state_ratio,
                "tempo_stability": self.temporal.tempo_stability,
                "rhythmic_pattern": self.temporal.rhythmic_pattern,
            }
        }


# 异常类定义
class PreprocessingError(Exception):
    """预处理模块基础异常"""
    pass


class OnsetDetectionError(PreprocessingError):
    """Onset检测异常"""
    pass


class PitchDetectionError(PreprocessingError):
    """音高检测异常"""
    def __init__(self, message: str, confidence: float = 0.0):
        super().__init__(message)
        self.confidence = confidence


class DynamicsAnalysisError(PreprocessingError):
    """动态分析异常"""
    pass


class PitchShiftingError(PreprocessingError):
    """音高变换异常"""
    pass


class AudioFormatError(PreprocessingError):
    """音频格式异常"""
    pass
