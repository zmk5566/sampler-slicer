"""
音高变换器 (Pitch Shifter)
提供不变速音高变换功能，专为编铙音色保持优化
"""

import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path
import logging

from .data_structures import (
    AudioSegment,
    PitchShiftingError,
    PreprocessingError
)

logger = logging.getLogger(__name__)


class PitchShifter:
    """
    音高变换器
    
    提供高质量的音高变换功能，保持音频的时长不变，
    特别适用于编铙等打击乐器的音色保持。
    """
    
    def __init__(
        self,
        method: str = 'psola',
        frame_length: int = 2048,
        hop_length: int = 512,
        n_fft: int = 2048
    ):
        """
        初始化音高变换器
        
        Args:
            method: 变换方法 ('psola', 'phase_vocoder', 'harmonic')
            frame_length: 分析帧长度
            hop_length: 帧移
            n_fft: FFT长度
        """
        self.method = method
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        
        logger.info(f"PitchShifter初始化完成 - method: {method}")
    
    def shift_pitch(
        self, 
        audio_segment: AudioSegment, 
        semitones: float
    ) -> AudioSegment:
        """
        对音频片段进行音高变换
        
        Args:
            audio_segment: 输入音频片段
            semitones: 变换的半音数 (正数为升高，负数为降低)
            
        Returns:
            AudioSegment: 变换后的音频片段
            
        Raises:
            PitchShiftingError: 音高变换失败
        """
        try:
            y = audio_segment.audio_data
            sr = audio_segment.sample_rate
            
            if len(y) == 0:
                raise PitchShiftingError("音频数据为空")
            
            if semitones == 0:
                # 不需要变换
                return audio_segment
            
            # 根据方法选择变换算法
            if self.method == 'psola':
                y_shifted = self._shift_with_psola(y, sr, semitones)
            elif self.method == 'phase_vocoder':
                y_shifted = self._shift_with_phase_vocoder(y, sr, semitones)
            elif self.method == 'harmonic':
                y_shifted = self._shift_with_harmonic_method(y, sr, semitones)
            else:
                # 默认使用librosa的pitch_shift
                y_shifted = self._shift_with_librosa(y, sr, semitones)
            
            # 创建新的音频片段
            shifted_segment = AudioSegment(
                audio_data=y_shifted,
                sample_rate=sr,
                onset_time=audio_segment.onset_time,
                duration=len(y_shifted) / sr,
                source_file=audio_segment.source_file,
                segment_id=f"{audio_segment.segment_id}_shift_{semitones:+.1f}st"
            )
            
            logger.debug(f"音高变换完成: {semitones:+.1f} 半音")
            return shifted_segment
            
        except Exception as e:
            if isinstance(e, PitchShiftingError):
                raise
            else:
                raise PitchShiftingError(f"音高变换失败: {e}")
    
    def _shift_with_librosa(self, y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """使用librosa的pitch_shift"""
        try:
            y_shifted = librosa.effects.pitch_shift(
                y, 
                sr=sr, 
                n_steps=semitones,
                bins_per_octave=12
            )
            return y_shifted
            
        except Exception as e:
            raise PitchShiftingError(f"Librosa pitch shift失败: {e}")
    
    def _shift_with_psola(self, y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """
        使用PSOLA (Pitch Synchronous Overlap and Add) 方法
        这种方法对打击乐器效果较好
        """
        try:
            # PSOLA实现比较复杂，这里使用简化版本
            # 实际项目中可能需要使用专门的PSOLA库
            
            # 计算变换比例
            shift_ratio = 2 ** (semitones / 12.0)
            
            # 使用相位声码器作为PSOLA的近似
            stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # 频率轴变换
            shifted_stft = np.zeros_like(stft)
            n_freqs = stft.shape[0]
            
            for i in range(n_freqs):
                # 计算目标频率bin
                target_freq_idx = int(i * shift_ratio)
                if 0 <= target_freq_idx < n_freqs:
                    shifted_stft[target_freq_idx] += stft[i]
            
            # 重构音频
            y_shifted = librosa.istft(shifted_stft, hop_length=self.hop_length)
            
            # 长度匹配
            if len(y_shifted) > len(y):
                y_shifted = y_shifted[:len(y)]
            elif len(y_shifted) < len(y):
                y_shifted = np.pad(y_shifted, (0, len(y) - len(y_shifted)), 'constant')
            
            return y_shifted
            
        except Exception as e:
            logger.warning(f"PSOLA方法失败，回退到librosa: {e}")
            return self._shift_with_librosa(y, sr, semitones)
    
    def _shift_with_phase_vocoder(self, y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """使用相位声码器方法"""
        try:
            # 计算变换比例
            shift_ratio = 2 ** (semitones / 12.0)
            
            # STFT分析
            stft = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # 频率轴重采样
            n_freqs, n_frames = stft.shape
            shifted_magnitude = np.zeros_like(magnitude)
            shifted_phase = np.zeros_like(phase)
            
            for frame in range(n_frames):
                for freq_bin in range(n_freqs):
                    # 计算目标频率bin
                    target_bin = freq_bin * shift_ratio
                    
                    if 0 <= target_bin < n_freqs:
                        target_bin_int = int(target_bin)
                        
                        # 插值
                        if target_bin_int < n_freqs - 1:
                            weight = target_bin - target_bin_int
                            shifted_magnitude[target_bin_int, frame] += magnitude[freq_bin, frame] * (1 - weight)
                            shifted_magnitude[target_bin_int + 1, frame] += magnitude[freq_bin, frame] * weight
                            
                            # 相位处理
                            shifted_phase[target_bin_int, frame] = phase[freq_bin, frame]
                        else:
                            shifted_magnitude[target_bin_int, frame] += magnitude[freq_bin, frame]
                            shifted_phase[target_bin_int, frame] = phase[freq_bin, frame]
            
            # 重构STFT
            shifted_stft = shifted_magnitude * np.exp(1j * shifted_phase)
            
            # 逆变换
            y_shifted = librosa.istft(shifted_stft, hop_length=self.hop_length)
            
            # 长度匹配
            if len(y_shifted) > len(y):
                y_shifted = y_shifted[:len(y)]
            elif len(y_shifted) < len(y):
                y_shifted = np.pad(y_shifted, (0, len(y) - len(y_shifted)), 'constant')
            
            return y_shifted
            
        except Exception as e:
            logger.warning(f"相位声码器方法失败，回退到librosa: {e}")
            return self._shift_with_librosa(y, sr, semitones)
    
    def _shift_with_harmonic_method(self, y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
        """
        使用谐波分析方法
        特别适用于有明确谐波结构的打击乐器
        """
        try:
            # 分离谐波和冲击成分
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # 对谐波部分进行音高变换
            y_harmonic_shifted = librosa.effects.pitch_shift(
                y_harmonic, 
                sr=sr, 
                n_steps=semitones
            )
            
            # 对冲击部分应用轻微的频谱变换
            if semitones != 0:
                stft_perc = librosa.stft(y_percussive, n_fft=self.n_fft, hop_length=self.hop_length)
                
                # 对冲击成分进行轻微的频谱调整
                shift_factor = 2 ** (semitones / 24.0)  # 只变换一半，保持冲击特性
                
                # 简单的频谱移位
                magnitude = np.abs(stft_perc)
                phase = np.angle(stft_perc)
                
                shifted_stft_perc = np.zeros_like(stft_perc)
                n_freqs = stft_perc.shape[0]
                
                for i in range(n_freqs):
                    target_idx = int(i * shift_factor)
                    if 0 <= target_idx < n_freqs:
                        shifted_stft_perc[target_idx] = stft_perc[i]
                
                y_percussive_shifted = librosa.istft(shifted_stft_perc, hop_length=self.hop_length)
                
                # 长度匹配
                min_length = min(len(y_harmonic_shifted), len(y_percussive_shifted))
                y_harmonic_shifted = y_harmonic_shifted[:min_length]
                y_percussive_shifted = y_percussive_shifted[:min_length]
            else:
                y_percussive_shifted = y_percussive
            
            # 重新合成
            y_shifted = y_harmonic_shifted + y_percussive_shifted
            
            # 长度匹配
            if len(y_shifted) > len(y):
                y_shifted = y_shifted[:len(y)]
            elif len(y_shifted) < len(y):
                y_shifted = np.pad(y_shifted, (0, len(y) - len(y_shifted)), 'constant')
            
            return y_shifted
            
        except Exception as e:
            logger.warning(f"谐波方法失败，回退到librosa: {e}")
            return self._shift_with_librosa(y, sr, semitones)
    
    def generate_pitch_variants(
        self, 
        audio_segment: AudioSegment, 
        semitone_range: Tuple[float, float] = (-12.0, 12.0),
        step_size: float = 1.0
    ) -> Dict[float, AudioSegment]:
        """
        生成音频片段的音高变体
        
        Args:
            audio_segment: 输入音频片段
            semitone_range: 音高变换范围 (最小值, 最大值) 半音
            step_size: 步长（半音）
            
        Returns:
            Dict[float, AudioSegment]: 半音数 -> 变换后音频片段的字典
        """
        try:
            variants = {}
            
            min_semitones, max_semitones = semitone_range
            
            # 生成半音步长序列
            semitone_steps = np.arange(min_semitones, max_semitones + step_size, step_size)
            
            for semitones in semitone_steps:
                try:
                    if semitones == 0:
                        # 原始版本
                        variants[0.0] = audio_segment
                    else:
                        # 变换版本
                        shifted_segment = self.shift_pitch(audio_segment, semitones)
                        variants[float(semitones)] = shifted_segment
                        
                    logger.debug(f"生成音高变体: {semitones:+.1f} 半音")
                    
                except Exception as e:
                    logger.warning(f"生成 {semitones:+.1f} 半音变体失败: {e}")
                    continue
            
            logger.info(f"成功生成 {len(variants)} 个音高变体")
            return variants
            
        except Exception as e:
            raise PitchShiftingError(f"生成音高变体失败: {e}")
    
    def batch_pitch_shift(
        self, 
        segments: List[AudioSegment], 
        semitones: float
    ) -> List[AudioSegment]:
        """
        批量音高变换
        
        Args:
            segments: 音频片段列表
            semitones: 变换的半音数
            
        Returns:
            List[AudioSegment]: 变换后的音频片段列表
        """
        try:
            shifted_segments = []
            
            for i, segment in enumerate(segments):
                try:
                    shifted_segment = self.shift_pitch(segment, semitones)
                    shifted_segments.append(shifted_segment)
                    logger.debug(f"批量处理进度: {i+1}/{len(segments)}")
                    
                except Exception as e:
                    logger.warning(f"处理片段 {i} 失败: {e}")
                    # 跳过失败的片段
                    continue
            
            logger.info(f"批量音高变换完成: {len(shifted_segments)}/{len(segments)} 个片段成功")
            return shifted_segments
            
        except Exception as e:
            raise PitchShiftingError(f"批量音高变换失败: {e}")
    
    def save_shifted_audio(
        self, 
        shifted_segment: AudioSegment, 
        output_path: Union[str, Path]
    ) -> None:
        """
        保存变换后的音频
        
        Args:
            shifted_segment: 变换后的音频片段
            output_path: 输出文件路径
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(
                str(output_path),
                shifted_segment.audio_data,
                shifted_segment.sample_rate
            )
            
            logger.info(f"音频保存成功: {output_path}")
            
        except Exception as e:
            raise PitchShiftingError(f"保存音频失败: {e}")
    
    def estimate_quality(
        self, 
        original: AudioSegment, 
        shifted: AudioSegment
    ) -> float:
        """
        估计音高变换的质量
        
        Args:
            original: 原始音频片段
            shifted: 变换后音频片段
            
        Returns:
            float: 质量评分 (0-1, 1为最好)
        """
        try:
            y_orig = original.audio_data
            y_shift = shifted.audio_data
            
            # 长度匹配
            min_length = min(len(y_orig), len(y_shift))
            y_orig = y_orig[:min_length]
            y_shift = y_shift[:min_length]
            
            if min_length == 0:
                return 0.0
            
            # 计算频谱相似性
            stft_orig = librosa.stft(y_orig, n_fft=self.n_fft, hop_length=self.hop_length)
            stft_shift = librosa.stft(y_shift, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # 幅度谱相关性
            mag_orig = np.abs(stft_orig)
            mag_shift = np.abs(stft_shift)
            
            # 计算每帧的相关性
            correlations = []
            for frame in range(min(mag_orig.shape[1], mag_shift.shape[1])):
                corr = np.corrcoef(mag_orig[:, frame], mag_shift[:, frame])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            if not correlations:
                return 0.0
            
            # 平均相关性
            avg_correlation = np.mean(correlations)
            
            # 转换为0-1的质量评分
            quality = (avg_correlation + 1) / 2  # 从[-1,1]映射到[0,1]
            
            return float(np.clip(quality, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"质量评估失败: {e}")
            return 0.5  # 默认中等质量
    
    def set_method(self, method: str) -> None:
        """设置音高变换方法"""
        if method in ['psola', 'phase_vocoder', 'harmonic', 'librosa']:
            self.method = method
            logger.info(f"音高变换方法设置为: {method}")
        else:
            logger.warning(f"未知的变换方法: {method}，保持当前方法: {self.method}")
