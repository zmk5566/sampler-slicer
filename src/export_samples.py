"""
编铙采样器导出脚本
导出处理好的音频片段、音高变体和元数据
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import soundfile as sf
import logging
from typing import List, Dict, Optional

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.onset_detector import OnsetDetector
from preprocessing.pitch_analyzer import PitchAnalyzer
from preprocessing.dynamics_analyzer import DynamicsAnalyzer
from preprocessing.pitch_shifter import PitchShifter
from preprocessing.data_structures import AudioSegment

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleExporter:
    """编铙采样器导出器"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化导出器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.segments_dir = self.output_dir / "segments"
        self.variants_dir = self.output_dir / "pitch_variants"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.segments_dir, self.variants_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"SampleExporter初始化完成 - 输出目录: {self.output_dir}")
    
    def export_all(
        self, 
        audio_file: str,
        onset_threshold: float = 0.3,
        min_segment_length: float = 0.3,
        semitone_range: tuple = (-6, 6),
        step_size: float = 1.0
    ) -> Dict:
        """
        完整导出流程
        
        Args:
            audio_file: 输入音频文件
            onset_threshold: Onset检测阈值
            min_segment_length: 最小片段长度
            semitone_range: 音高变换范围（半音）
            step_size: 音高变换步长
            
        Returns:
            Dict: 导出结果统计
        """
        logger.info("=" * 60)
        logger.info("开始完整导出流程...")
        logger.info("=" * 60)
        
        # 1. 初始化分析器
        logger.info("初始化分析器...")
        onset_detector = OnsetDetector(threshold=onset_threshold)
        pitch_analyzer = PitchAnalyzer(method='yin', fmin=100.0, fmax=800.0)
        dynamics_analyzer = DynamicsAnalyzer()
        pitch_shifter = PitchShifter(method='harmonic')
        
        # 2. 音频分割
        logger.info(f"分割音频文件: {audio_file}")
        segments = onset_detector.slice_audio(audio_file, min_segment_length=min_segment_length)
        logger.info(f"检测到 {len(segments)} 个音频片段")
        
        if len(segments) == 0:
            logger.warning("没有检测到有效的音频片段")
            return {"segments": 0, "variants": 0, "metadata_files": 0}
        
        # 3. 导出原始片段
        logger.info("导出原始音频片段...")
        segment_metadata = self._export_segments(segments, pitch_analyzer, dynamics_analyzer)
        
        # 4. 生成并导出音高变体
        logger.info("生成并导出音高变体...")
        variant_stats = self._export_pitch_variants(
            segments, pitch_shifter, semitone_range, step_size
        )
        
        # 5. 导出汇总元数据
        logger.info("导出汇总元数据...")
        summary_metadata = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "source_file": str(audio_file),
                "onset_threshold": onset_threshold,
                "min_segment_length": min_segment_length,
                "semitone_range": semitone_range,
                "step_size": step_size
            },
            "statistics": {
                "total_segments": len(segments),
                "total_variants": variant_stats["total_variants"],
                "pitch_range": variant_stats["pitch_range"],
                "dynamics_distribution": self._analyze_dynamics_distribution(segment_metadata)
            },
            "segments": segment_metadata
        }
        
        summary_file = self.metadata_dir / "export_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_metadata, f, indent=2, ensure_ascii=False)
        
        # 6. 创建文件索引
        self._create_file_index(summary_metadata)
        
        logger.info("=" * 60)
        logger.info("导出完成！")
        logger.info(f"原始片段: {len(segments)} 个")
        logger.info(f"音高变体: {variant_stats['total_variants']} 个")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)
        
        return {
            "segments": len(segments),
            "variants": variant_stats["total_variants"],
            "metadata_files": 3,  # segments.json, variants.json, summary.json
            "output_dir": str(self.output_dir)
        }
    
    def _export_segments(
        self, 
        segments: List[AudioSegment], 
        pitch_analyzer: PitchAnalyzer, 
        dynamics_analyzer: DynamicsAnalyzer
    ) -> List[Dict]:
        """导出原始音频片段"""
        segment_metadata = []
        
        for i, segment in enumerate(segments):
            # 分析音高和动态特征
            try:
                pitch_info = pitch_analyzer.analyze_pitch(segment)
                dynamics_info = dynamics_analyzer.analyze_dynamics(segment)
                
                # 生成文件名
                note_name = pitch_info.note_name if pitch_info.note_name != "N/A" else "Unknown"
                dynamic_level = dynamics_info.dynamic_level
                filename = f"segment_{i:03d}_{note_name}_{dynamic_level}_{pitch_info.fundamental_freq:.1f}Hz.wav"
                
                # 保存音频文件
                output_path = self.segments_dir / filename
                sf.write(str(output_path), segment.audio_data, segment.sample_rate)
                
                # 创建元数据
                metadata = {
                    "segment_id": segment.segment_id,
                    "index": i,
                    "filename": filename,
                    "onset_time": segment.onset_time,
                    "duration": segment.duration,
                    "source_file": str(segment.source_file),
                    "pitch_info": {
                        "fundamental_freq": pitch_info.fundamental_freq,
                        "note_name": pitch_info.note_name,
                        "octave": pitch_info.octave,
                        "cents_deviation": pitch_info.cents_deviation,
                        "confidence": pitch_info.confidence,
                        "pitch_stability": pitch_info.pitch_stability,
                        "harmonics": pitch_info.harmonics
                    },
                    "dynamics_info": {
                        "rms_energy": dynamics_info.rms_energy,
                        "peak_amplitude": dynamics_info.peak_amplitude,
                        "loudness_lufs": dynamics_info.loudness_lufs,
                        "dynamic_range_db": dynamics_info.dynamic_range_db,
                        "dynamic_level": dynamics_info.dynamic_level,
                        "attack_time_ms": dynamics_info.attack_time_ms,
                        "decay_time_ms": dynamics_info.decay_time_ms,
                        "sustain_level": dynamics_info.sustain_level,
                        "release_time_ms": dynamics_info.release_time_ms,
                        "envelope_shape": dynamics_info.envelope_shape
                    }
                }
                
                segment_metadata.append(metadata)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(segments)} 个片段")
                    
            except Exception as e:
                logger.error(f"处理片段 {i} 失败: {e}")
                continue
        
        # 保存片段元数据
        segments_metadata_file = self.metadata_dir / "segments_metadata.json"
        with open(segments_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(segment_metadata, f, indent=2, ensure_ascii=False)
        
        return segment_metadata
    
    def _export_pitch_variants(
        self, 
        segments: List[AudioSegment], 
        pitch_shifter: PitchShifter, 
        semitone_range: tuple, 
        step_size: float
    ) -> Dict:
        """导出音高变体"""
        min_semitones, max_semitones = semitone_range
        semitone_steps = np.arange(min_semitones, max_semitones + step_size, step_size)
        
        variant_metadata = []
        total_variants = 0
        pitch_range = {"min_freq": float('inf'), "max_freq": 0}
        
        for i, segment in enumerate(segments[:10]):  # 只处理前10个片段作为示例
            segment_variants = {}
            
            for semitones in semitone_steps:
                try:
                    if semitones == 0:
                        # 原始版本
                        variant_segment = segment
                        suffix = "original"
                    else:
                        # 变换版本
                        variant_segment = pitch_shifter.shift_pitch(segment, semitones)
                        suffix = f"{semitones:+.0f}st"
                    
                    # 生成文件名
                    filename = f"segment_{i:03d}_variant_{suffix}.wav"
                    
                    # 保存音频文件
                    output_path = self.variants_dir / filename
                    sf.write(str(output_path), variant_segment.audio_data, variant_segment.sample_rate)
                    
                    # 记录变体信息
                    segment_variants[f"{semitones:+.0f}st"] = {
                        "filename": filename,
                        "semitones": semitones,
                        "original_segment_id": segment.segment_id
                    }
                    
                    total_variants += 1
                    
                    # 更新音高范围（估算）
                    if hasattr(segment, 'estimated_freq'):
                        transformed_freq = segment.estimated_freq * (2 ** (semitones / 12))
                        pitch_range["min_freq"] = min(pitch_range["min_freq"], transformed_freq)
                        pitch_range["max_freq"] = max(pitch_range["max_freq"], transformed_freq)
                    
                except Exception as e:
                    logger.warning(f"生成片段 {i} 的 {semitones:+.0f} 半音变体失败: {e}")
                    continue
            
            if segment_variants:
                variant_metadata.append({
                    "original_segment_index": i,
                    "variants": segment_variants
                })
        
        # 保存变体元数据
        variants_metadata_file = self.metadata_dir / "variants_metadata.json"
        with open(variants_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(variant_metadata, f, indent=2, ensure_ascii=False)
        
        return {
            "total_variants": total_variants,
            "pitch_range": pitch_range if pitch_range["min_freq"] != float('inf') else {"min_freq": 0, "max_freq": 0}
        }
    
    def _analyze_dynamics_distribution(self, segment_metadata: List[Dict]) -> Dict:
        """分析动态分布"""
        dynamics_count = {}
        rms_values = []
        
        for segment in segment_metadata:
            dynamic_level = segment["dynamics_info"]["dynamic_level"]
            dynamics_count[dynamic_level] = dynamics_count.get(dynamic_level, 0) + 1
            rms_values.append(segment["dynamics_info"]["rms_energy"])
        
        return {
            "level_distribution": dynamics_count,
            "rms_statistics": {
                "mean": float(np.mean(rms_values)) if rms_values else 0,
                "std": float(np.std(rms_values)) if rms_values else 0,
                "min": float(np.min(rms_values)) if rms_values else 0,
                "max": float(np.max(rms_values)) if rms_values else 0
            }
        }
    
    def _create_file_index(self, summary_metadata: Dict) -> None:
        """创建文件索引"""
        index = {
            "export_timestamp": summary_metadata["export_info"]["timestamp"],
            "total_files": 0,
            "directories": {
                "segments": str(self.segments_dir),
                "variants": str(self.variants_dir),
                "metadata": str(self.metadata_dir)
            },
            "files": {
                "segments": [],
                "variants": [],
                "metadata": [
                    "segments_metadata.json",
                    "variants_metadata.json",
                    "export_summary.json",
                    "file_index.json"
                ]
            }
        }
        
        # 统计实际文件
        if self.segments_dir.exists():
            index["files"]["segments"] = [f.name for f in self.segments_dir.glob("*.wav")]
        
        if self.variants_dir.exists():
            index["files"]["variants"] = [f.name for f in self.variants_dir.glob("*.wav")]
        
        index["total_files"] = (
            len(index["files"]["segments"]) + 
            len(index["files"]["variants"]) + 
            len(index["files"]["metadata"])
        )
        
        # 保存索引文件
        index_file = self.metadata_dir / "file_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)


def main():
    """主函数"""
    # 配置参数
    audio_file = "samples/编铙/biannao-正鼓音.wav"
    output_dir = "exported_samples"
    
    if not Path(audio_file).exists():
        logger.error(f"音频文件不存在: {audio_file}")
        return
    
    # 创建导出器
    exporter = SampleExporter(output_dir)
    
    # 执行导出
    try:
        results = exporter.export_all(
            audio_file=audio_file,
            onset_threshold=0.3,
            min_segment_length=0.3,
            semitone_range=(-6, 6),
            step_size=2.0  # 2半音步长，减少文件数量
        )
        
        print("\n🎉 导出完成！")
        print(f"📁 输出目录: {results['output_dir']}")
        print(f"🎵 原始片段: {results['segments']} 个")
        print(f"🎼 音高变体: {results['variants']} 个")
        print(f"📄 元数据文件: {results['metadata_files']} 个")
        
    except Exception as e:
        logger.error(f"导出失败: {e}")
        raise


if __name__ == "__main__":
    main()
