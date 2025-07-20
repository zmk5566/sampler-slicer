"""
音频预处理模块 (Preprocessing Module)
提供音频分割、特征分析和音高变换功能
"""

from .onset_detector import OnsetDetector
from .pitch_analyzer import PitchAnalyzer
from .dynamics_analyzer import DynamicsAnalyzer
from .pitch_shifter import PitchShifter
from .data_structures import AudioSegment, PitchInfo, DynamicsInfo

__all__ = [
    "OnsetDetector",
    "PitchAnalyzer", 
    "DynamicsAnalyzer",
    "PitchShifter",
    "AudioSegment",
    "PitchInfo",
    "DynamicsInfo",
]
