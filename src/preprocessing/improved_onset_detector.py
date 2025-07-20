"""
改进的Onset检测器
实现多层过滤机制和完整的onset/offset检测
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

def setup_chinese_font():
    """设置中文字体"""
    try:
        # 抑制字体警告
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
        # 尝试设置中文字体
        chinese_fonts = [
            'Arial Unicode MS',  # macOS
            'SimHei',           # Windows
            'DejaVu Sans',      # 回退选项
        ]
        
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                # 测试字体是否可用
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '测试', fontsize=12)
                plt.close(fig)
                logger.info(f"成功设置中文字体: {font}")
                return True
            except:
                continue
        
        # 如果都失败，使用英文标题
        logger.warning("无法设置中文字体，将使用英文标题")
        return False
        
    except Exception as e:
        logger.warning(f"字体设置失败: {e}")
        return False

# 全局字体设置
CHINESE_FONT_AVAILABLE = setup_chinese_font()

@dataclass
class OnsetConfig:
    """Onset检测配置"""
    # 基础参数
    hop_length: int = 512
    sr: int = 44100
    
    # 多层过滤参数
    energy_percentile: float = 75.0  # 能量门限百分位数
    min_segment_duration: float = 0.2  # 最小片段长度(秒)
    max_segment_duration: float = 3.0   # 最大片段长度(秒)
    
    # Onset检测参数
    onset_threshold: float = 0.6  # 提高阈值
    min_onset_interval: float = 0.1  # 最小onset间隔(秒)
    
    # Offset检测参数
    offset_method: str = "energy_decay"  # "energy_decay" 或 "silence"
    energy_decay_ratio: float = 0.1  # 衰减到峰值的比例
    silence_threshold_db: float = -40  # 静音阈值(dB)
    
    # 后处理参数
    rms_threshold_percentile: float = 30.0  # RMS阈值百分位数
    snr_threshold_db: float = 10.0  # 信噪比阈值

@dataclass
class SegmentInfo:
    """音频片段信息"""
    onset_time: float
    offset_time: float
    duration: float
    peak_time: float
    rms_energy: float
    peak_amplitude: float
    snr_db: float
    confidence: float

class ImprovedOnsetDetector:
    """改进的Onset检测器"""
    
    def __init__(self, config: OnsetConfig = None):
        self.config = config or OnsetConfig()
        self.y = None
        self.sr = None
        self.onset_frames = None
        self.onset_times = None
        
    def load_audio(self, file_path: str) -> bool:
        """加载音频文件"""
        try:
            self.y, self.sr = librosa.load(file_path, sr=self.config.sr)
            logger.info(f"加载音频: {file_path}")
            logger.info(f"  时长: {len(self.y)/self.sr:.2f}秒")
            logger.info(f"  采样率: {self.sr}Hz")
            return True
        except Exception as e:
            logger.error(f"加载音频失败: {e}")
            return False
    
    def _calculate_energy_threshold(self) -> float:
        """计算能量门限"""
        # 计算短时能量
        frame_length = self.config.hop_length * 2
        energy = librosa.feature.rms(
            y=self.y, 
            frame_length=frame_length,
            hop_length=self.config.hop_length
        )[0]
        
        # 使用百分位数作为门限
        threshold = np.percentile(energy, self.config.energy_percentile)
        logger.info(f"能量门限: {threshold:.6f} (第{self.config.energy_percentile}百分位数)")
        return threshold
    
    def _detect_onsets(self) -> np.ndarray:
        """检测onset"""
        # 方法1: 基于spectral flux的检测
        onset_env_spectral = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.config.hop_length
        )
        onset_frames_spectral = librosa.util.peak_pick(
            onset_env_spectral,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=self.config.onset_threshold,
            wait=10
        )
        
        # 方法2: 基于能量变化的检测
        # 直接计算RMS能量
        rms_energy = librosa.feature.rms(
            y=self.y, hop_length=self.config.hop_length
        )[0]
        
        # 计算能量差分
        rms_diff = np.diff(rms_energy)
        rms_diff = np.pad(rms_diff, (1, 0), mode='constant', constant_values=0)
        
        # 找到能量突增的点
        onset_frames_energy = librosa.util.peak_pick(
            rms_diff,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=self.config.onset_threshold * 0.1,  # 降低阈值
            wait=10
        )
        
        # 合并结果
        if len(onset_frames_spectral) > 0 and len(onset_frames_energy) > 0:
            onset_frames = np.unique(np.concatenate([onset_frames_spectral, onset_frames_energy]))
        elif len(onset_frames_spectral) > 0:
            onset_frames = onset_frames_spectral
        elif len(onset_frames_energy) > 0:
            onset_frames = onset_frames_energy
        else:
            onset_frames = np.array([])
        
        # 转换为时间
        if len(onset_frames) > 0:
            onset_times = librosa.frames_to_time(
                onset_frames, 
                sr=self.sr, 
                hop_length=self.config.hop_length
            )
        else:
            onset_times = np.array([])
        
        # 最小间隔过滤
        filtered_times = []
        last_time = -np.inf
        
        for time in onset_times:
            if time - last_time >= self.config.min_onset_interval:
                filtered_times.append(time)
                last_time = time
        
        logger.info(f"检测到 {len(onset_times)} 个onset，过滤后 {len(filtered_times)} 个")
        return np.array(filtered_times)
    
    def _detect_offset(self, onset_time: float, next_onset_time: Optional[float] = None) -> float:
        """检测单个onset的offset"""
        onset_sample = int(onset_time * self.sr)
        
        # 确定搜索范围
        if next_onset_time is not None:
            search_end = int(next_onset_time * self.sr)
        else:
            search_end = len(self.y)
        
        # 限制最大搜索长度
        max_search_samples = int(self.config.max_segment_duration * self.sr)
        search_end = min(search_end, onset_sample + max_search_samples)
        
        if search_end <= onset_sample:
            return onset_time + self.config.min_segment_duration
        
        search_audio = self.y[onset_sample:search_end]
        
        if self.config.offset_method == "energy_decay":
            return self._detect_offset_energy_decay(onset_time, search_audio)
        else:
            return self._detect_offset_silence(onset_time, search_audio)
    
    def _detect_offset_energy_decay(self, onset_time: float, search_audio: np.ndarray) -> float:
        """基于能量衰减检测offset"""
        # 计算滑动窗口RMS
        window_size = int(0.05 * self.sr)  # 50ms窗口
        hop_size = int(0.01 * self.sr)     # 10ms跳跃
        
        rms_values = []
        for i in range(0, len(search_audio) - window_size, hop_size):
            window = search_audio[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        if len(rms_values) == 0:
            return onset_time + self.config.min_segment_duration
        
        rms_values = np.array(rms_values)
        
        # 找到峰值
        peak_idx = np.argmax(rms_values)
        peak_rms = rms_values[peak_idx]
        
        # 找到衰减点
        decay_threshold = peak_rms * self.config.energy_decay_ratio
        
        for i in range(peak_idx, len(rms_values)):
            if rms_values[i] <= decay_threshold:
                offset_time = onset_time + (i * hop_size) / self.sr
                # 确保最小长度
                min_offset = onset_time + self.config.min_segment_duration
                return max(offset_time, min_offset)
        
        # 如果没找到衰减点，使用默认长度
        return onset_time + self.config.min_segment_duration
    
    def _detect_offset_silence(self, onset_time: float, search_audio: np.ndarray) -> float:
        """基于静音检测offset"""
        # 转换为dB
        audio_db = librosa.amplitude_to_db(np.abs(search_audio))
        
        # 滑动窗口检测静音
        window_size = int(0.1 * self.sr)  # 100ms窗口
        
        for i in range(window_size, len(audio_db)):
            window = audio_db[i-window_size:i]
            if np.mean(window) < self.config.silence_threshold_db:
                offset_time = onset_time + i / self.sr
                min_offset = onset_time + self.config.min_segment_duration
                return max(offset_time, min_offset)
        
        return onset_time + self.config.min_segment_duration
    
    def _calculate_segment_features(self, onset_time: float, offset_time: float) -> Dict:
        """计算片段特征"""
        onset_sample = int(onset_time * self.sr)
        offset_sample = int(offset_time * self.sr)
        
        segment_audio = self.y[onset_sample:offset_sample]
        
        if len(segment_audio) == 0:
            return {
                'rms_energy': 0.0,
                'peak_amplitude': 0.0,
                'snr_db': -np.inf,
                'peak_time': onset_time
            }
        
        # 基础特征
        rms_energy = np.sqrt(np.mean(segment_audio ** 2))
        peak_amplitude = np.max(np.abs(segment_audio))
        
        # 峰值时间
        peak_idx = np.argmax(np.abs(segment_audio))
        peak_time = onset_time + peak_idx / self.sr
        
        # 计算信噪比 (简化版本)
        # 假设前10%和后10%是噪音
        noise_start = segment_audio[:len(segment_audio)//10]
        noise_end = segment_audio[-len(segment_audio)//10:]
        noise_samples = np.concatenate([noise_start, noise_end])
        
        if len(noise_samples) > 0:
            noise_rms = np.sqrt(np.mean(noise_samples ** 2))
            if noise_rms > 0:
                snr_db = 20 * np.log10(rms_energy / noise_rms)
            else:
                snr_db = 60.0  # 很高的SNR
        else:
            snr_db = 60.0
        
        return {
            'rms_energy': rms_energy,
            'peak_amplitude': peak_amplitude,
            'snr_db': snr_db,
            'peak_time': peak_time
        }
    
    def _filter_segments(self, segments: List[SegmentInfo]) -> List[SegmentInfo]:
        """多层过滤"""
        if not segments:
            return segments
        
        logger.info(f"开始过滤 {len(segments)} 个片段...")
        
        # 1. 时长过滤
        duration_filtered = [
            seg for seg in segments 
            if self.config.min_segment_duration <= seg.duration <= self.config.max_segment_duration
        ]
        logger.info(f"时长过滤: {len(duration_filtered)}/{len(segments)}")
        
        # 2. RMS能量过滤
        if duration_filtered:
            rms_values = [seg.rms_energy for seg in duration_filtered]
            rms_threshold = np.percentile(rms_values, self.config.rms_threshold_percentile)
            
            energy_filtered = [
                seg for seg in duration_filtered 
                if seg.rms_energy >= rms_threshold
            ]
            logger.info(f"能量过滤: {len(energy_filtered)}/{len(duration_filtered)} (阈值: {rms_threshold:.6f})")
        else:
            energy_filtered = duration_filtered
        
        # 3. 信噪比过滤
        snr_filtered = [
            seg for seg in energy_filtered 
            if seg.snr_db >= self.config.snr_threshold_db
        ]
        logger.info(f"信噪比过滤: {len(snr_filtered)}/{len(energy_filtered)}")
        
        # 4. 计算置信度
        for seg in snr_filtered:
            # 基于多个因素计算置信度
            duration_score = min(seg.duration / 1.0, 1.0)  # 理想时长1秒
            energy_score = min(seg.rms_energy * 100, 1.0)  # 归一化能量
            snr_score = min(seg.snr_db / 30.0, 1.0)       # 归一化SNR
            
            seg.confidence = (duration_score + energy_score + snr_score) / 3.0
        
        # 5. 按置信度排序
        snr_filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"最终保留 {len(snr_filtered)} 个高质量片段")
        return snr_filtered
    
    def detect_segments(self) -> List[SegmentInfo]:
        """检测音频片段"""
        if self.y is None:
            logger.error("请先加载音频文件")
            return []
        
        # 1. 检测onset
        onset_times = self._detect_onsets()
        if len(onset_times) == 0:
            logger.warning("未检测到任何onset")
            return []
        
        # 2. 检测offset和计算特征
        segments = []
        for i, onset_time in enumerate(onset_times):
            next_onset = onset_times[i + 1] if i + 1 < len(onset_times) else None
            offset_time = self._detect_offset(onset_time, next_onset)
            
            duration = offset_time - onset_time
            features = self._calculate_segment_features(onset_time, offset_time)
            
            segment = SegmentInfo(
                onset_time=onset_time,
                offset_time=offset_time,
                duration=duration,
                peak_time=features['peak_time'],
                rms_energy=features['rms_energy'],
                peak_amplitude=features['peak_amplitude'],
                snr_db=features['snr_db'],
                confidence=0.0  # 待计算
            )
            segments.append(segment)
        
        # 3. 多层过滤
        filtered_segments = self._filter_segments(segments)
        
        return filtered_segments
    
    def visualize_detection(self, segments: List[SegmentInfo], output_path: str = None):
        """可视化检测结果"""
        if self.y is None:
            return
        
        plt.figure(figsize=(15, 8))
        
        # 子图1: 波形和检测结果
        plt.subplot(3, 1, 1)
        time_axis = np.linspace(0, len(self.y) / self.sr, len(self.y))
        plt.plot(time_axis, self.y, alpha=0.7, color='blue', linewidth=0.5)
        
        for seg in segments:
            plt.axvline(seg.onset_time, color='green', alpha=0.8, linestyle='--', label='Onset' if seg == segments[0] else '')
            plt.axvline(seg.offset_time, color='red', alpha=0.8, linestyle='--', label='Offset' if seg == segments[0] else '')
            plt.axvspan(seg.onset_time, seg.offset_time, alpha=0.2, color='yellow')
        
        plt.title('音频波形和检测结果')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: RMS能量
        plt.subplot(3, 1, 2)
        hop_length = self.config.hop_length
        rms = librosa.feature.rms(y=self.y, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr, hop_length=hop_length)
        plt.plot(times, rms, color='orange', linewidth=1)
        
        for seg in segments:
            plt.axvspan(seg.onset_time, seg.offset_time, alpha=0.3, color='yellow')
        
        plt.title('RMS能量')
        plt.xlabel('时间 (秒)')
        plt.ylabel('RMS')
        plt.grid(True, alpha=0.3)
        
        # 子图3: 片段质量
        plt.subplot(3, 1, 3)
        if segments:
            indices = range(len(segments))
            confidences = [seg.confidence for seg in segments]
            snrs = [seg.snr_db for seg in segments]
            
            plt.bar(indices, confidences, alpha=0.7, color='green', label='置信度')
            plt.plot(indices, np.array(snrs) / 50.0, 'ro-', alpha=0.7, label='SNR/50')
            
        plt.title('片段质量评估')
        plt.xlabel('片段索引')
        plt.ylabel('得分')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"可视化结果保存到: {output_path}")
        else:
            plt.show()
    
    def export_segments(self, segments: List[SegmentInfo], output_dir: str) -> List[str]:
        """导出音频片段"""
        if self.y is None:
            return []
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        for i, seg in enumerate(segments):
            onset_sample = int(seg.onset_time * self.sr)
            offset_sample = int(seg.offset_time * self.sr)
            
            segment_audio = self.y[onset_sample:offset_sample]
            
            if len(segment_audio) == 0:
                continue
            
            # 文件名包含更多信息
            filename = f"segment_{i:03d}_t{seg.onset_time:.2f}_d{seg.duration:.2f}_c{seg.confidence:.2f}.wav"
            file_path = output_path / filename
            
            sf.write(file_path, segment_audio, self.sr)
            exported_files.append(str(file_path))
            
            logger.info(f"导出片段 {i}: {filename}")
            logger.info(f"  时间: {seg.onset_time:.2f}s - {seg.offset_time:.2f}s")
            logger.info(f"  置信度: {seg.confidence:.3f}")
            logger.info(f"  SNR: {seg.snr_db:.1f}dB")
        
        logger.info(f"成功导出 {len(exported_files)} 个片段到 {output_dir}")
        return exported_files

def test_improved_detector():
    """测试改进的检测器"""
    config = OnsetConfig(
        onset_threshold=0.5,  # 降低onset阈值
        energy_percentile=70.0,  # 降低能量门限
        min_segment_duration=0.2,
        max_segment_duration=3.0,
        rms_threshold_percentile=20.0,  # 降低RMS阈值
        snr_threshold_db=5.0  # 大幅降低SNR阈值
    )
    
    detector = ImprovedOnsetDetector(config)
    
    # 测试文件
    test_file = "samples/编铙/biannao-正鼓音.wav"
    if not Path(test_file).exists():
        logger.error(f"测试文件不存在: {test_file}")
        return
    
    # 加载音频
    if not detector.load_audio(test_file):
        return
    
    # 检测片段
    segments = detector.detect_segments()
    
    if segments:
        # 可视化
        detector.visualize_detection(segments, "onset_detection_analysis.png")
        
        # 导出测试片段
        detector.export_segments(segments, "test_segments_improved")
        
        # 统计信息
        logger.info(f"\n=== 检测统计 ===")
        logger.info(f"总片段数: {len(segments)}")
        logger.info(f"平均时长: {np.mean([s.duration for s in segments]):.2f}秒")
        logger.info(f"平均置信度: {np.mean([s.confidence for s in segments]):.3f}")
        logger.info(f"平均SNR: {np.mean([s.snr_db for s in segments]):.1f}dB")
    else:
        logger.warning("未检测到任何有效片段")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_improved_detector()
