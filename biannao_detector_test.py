"""
编铙专用检测器测试
针对编铙特征优化的onset检测配置
"""

import logging
from pathlib import Path
from src.preprocessing.improved_onset_detector import ImprovedOnsetDetector, OnsetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_biannao_configurations():
    """测试多种编铙检测配置"""
    
    # 配置1: 高灵敏度配置
    config_sensitive = OnsetConfig(
        onset_threshold=0.2,  # 很低的阈值捕获弱击
        energy_percentile=50.0,  # 较低的能量门限
        min_segment_duration=0.3,  # 编铙最短持续时间
        max_segment_duration=5.0,  # 允许很长的衰减
        min_onset_interval=0.15,  # 较小的最小间隔
        rms_threshold_percentile=5.0,  # 非常宽松的RMS过滤
        snr_threshold_db=2.0,  # 极低的SNR阈值
        energy_decay_ratio=0.03,  # 检测更长的衰减
        offset_method="energy_decay"
    )
    
    # 配置2: 中等灵敏度配置
    config_medium = OnsetConfig(
        onset_threshold=0.4,
        energy_percentile=65.0,
        min_segment_duration=0.4,
        max_segment_duration=4.0,
        min_onset_interval=0.2,
        rms_threshold_percentile=15.0,
        snr_threshold_db=4.0,
        energy_decay_ratio=0.05,
        offset_method="energy_decay"
    )
    
    # 配置3: 保守配置（高质量但可能遗漏）
    config_conservative = OnsetConfig(
        onset_threshold=0.6,
        energy_percentile=75.0,
        min_segment_duration=0.5,
        max_segment_duration=3.0,
        min_onset_interval=0.3,
        rms_threshold_percentile=25.0,
        snr_threshold_db=8.0,
        energy_decay_ratio=0.08,
        offset_method="energy_decay"
    )
    
    configs = [
        ("高灵敏度", config_sensitive),
        ("中等灵敏度", config_medium),
        ("保守配置", config_conservative)
    ]
    
    test_file = "samples/编铙/biannao-正鼓音.wav"
    if not Path(test_file).exists():
        logger.error(f"测试文件不存在: {test_file}")
        return
    
    results = {}
    
    for name, config in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试配置: {name}")
        logger.info(f"{'='*50}")
        
        detector = ImprovedOnsetDetector(config)
        
        if not detector.load_audio(test_file):
            continue
        
        segments = detector.detect_segments()
        
        if segments:
            # 导出测试片段
            output_dir = f"test_biannao_{name.replace(' ', '_')}"
            detector.export_segments(segments, output_dir)
            
            # 可视化
            viz_path = f"biannao_detection_{name.replace(' ', '_')}.png"
            detector.visualize_detection(segments, viz_path)
            
            # 统计信息
            durations = [s.duration for s in segments]
            confidences = [s.confidence for s in segments]
            snrs = [s.snr_db for s in segments]
            times = [s.onset_time for s in segments]
            
            results[name] = {
                'count': len(segments),
                'avg_duration': sum(durations) / len(durations),
                'avg_confidence': sum(confidences) / len(confidences),
                'avg_snr': sum(snrs) / len(snrs),
                'time_range': (min(times), max(times)),
                'segments': segments
            }
            
            logger.info(f"检测到 {len(segments)} 个片段")
            logger.info(f"平均时长: {results[name]['avg_duration']:.2f}秒")
            logger.info(f"平均置信度: {results[name]['avg_confidence']:.3f}")
            logger.info(f"平均SNR: {results[name]['avg_snr']:.1f}dB")
            logger.info(f"时间范围: {results[name]['time_range'][0]:.1f}s - {results[name]['time_range'][1]:.1f}s")
            
            # 显示时间分布
            time_buckets = [0] * 11  # 0-10秒为一个区间，共11个区间覆盖110秒
            for t in times:
                bucket = min(int(t // 10), 10)
                time_buckets[bucket] += 1
            
            logger.info("时间分布 (每10秒一个区间):")
            for i, count in enumerate(time_buckets):
                if count > 0:
                    start_time = i * 10
                    end_time = (i + 1) * 10 if i < 10 else "end"
                    logger.info(f"  {start_time}-{end_time}秒: {count} 个片段")
        else:
            logger.warning(f"配置 {name} 未检测到任何有效片段")
            results[name] = {'count': 0}
    
    # 比较结果
    logger.info(f"\n{'='*50}")
    logger.info("配置比较总结")
    logger.info(f"{'='*50}")
    
    for name, result in results.items():
        if result['count'] > 0:
            logger.info(f"{name}: {result['count']} 个片段, "
                       f"平均时长 {result['avg_duration']:.2f}s, "
                       f"置信度 {result['avg_confidence']:.3f}")
        else:
            logger.info(f"{name}: 0 个片段")
    
    # 推荐最佳配置
    if results:
        # 选择片段数量适中且质量较高的配置
        best_config = None
        best_score = 0
        
        for name, result in results.items():
            if result['count'] > 0:
                # 评分：片段数量 * 平均置信度，但太多片段会降低评分
                count_score = min(result['count'] / 20.0, 1.0)  # 20个片段为满分
                quality_score = result['avg_confidence']
                total_score = count_score * quality_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_config = name
        
        if best_config:
            logger.info(f"\n推荐配置: {best_config}")
            logger.info(f"评分: {best_score:.3f}")

def analyze_time_distribution():
    """分析原始音频的时间分布特征"""
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    test_file = "samples/编铙/biannao-正鼓音.wav"
    if not Path(test_file).exists():
        logger.error(f"测试文件不存在: {test_file}")
        return
    
    logger.info("分析原始音频的能量分布...")
    
    y, sr = librosa.load(test_file, sr=44100)
    
    # 计算每秒的RMS能量
    hop_length = sr  # 1秒的hop
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = np.arange(len(rms))
    
    # 绘制能量分布图
    plt.figure(figsize=(15, 6))
    plt.plot(times, rms, 'b-', linewidth=1, alpha=0.7)
    plt.title('编铙录音每秒RMS能量分布')
    plt.xlabel('时间 (秒)')
    plt.ylabel('RMS能量')
    plt.grid(True, alpha=0.3)
    
    # 标记高能量区域
    energy_threshold = np.percentile(rms, 75)
    high_energy_indices = np.where(rms > energy_threshold)[0]
    
    for idx in high_energy_indices:
        plt.axvline(idx, color='red', alpha=0.5, linestyle='--')
    
    plt.savefig('biannao_energy_distribution.png', dpi=150, bbox_inches='tight')
    logger.info("能量分布图保存为: biannao_energy_distribution.png")
    
    # 统计信息
    logger.info(f"总时长: {len(y)/sr:.1f}秒")
    logger.info(f"平均RMS: {np.mean(rms):.6f}")
    logger.info(f"最大RMS: {np.max(rms):.6f}")
    logger.info(f"75%百分位RMS: {energy_threshold:.6f}")
    logger.info(f"高能量区域数量: {len(high_energy_indices)}")
    
    if len(high_energy_indices) > 0:
        logger.info(f"高能量区域分布: {high_energy_indices.tolist()}")

if __name__ == "__main__":
    # 先分析原始音频特征
    analyze_time_distribution()
    
    # 然后测试不同配置
    test_biannao_configurations()
