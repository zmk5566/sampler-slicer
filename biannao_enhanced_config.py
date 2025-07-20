"""
编铙增强配置
解决offset结束过早和字体问题的优化配置
"""

import logging
from pathlib import Path
from src.preprocessing.improved_onset_detector import ImprovedOnsetDetector, OnsetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_biannao_configs():
    """创建针对编铙优化的配置"""
    
    # 配置1: 长衰减配置 - 专门针对编铙的长衰减特性
    config_long_decay = OnsetConfig(
        onset_threshold=0.15,
        energy_percentile=65.0,
        min_segment_duration=0.25,  # 最小0.5秒
        max_segment_duration=8.0,  # 允许8秒长衰减
        min_onset_interval=0.3,    # 编铙敲击间隔通常较长
        rms_threshold_percentile=25,
        snr_threshold_db=0.5,
        energy_decay_ratio=0.02,   # 衰减到峰值的2%（更长衰减）
        offset_method="energy_decay"
    )
    
    # 配置2: 混合检测配置 - 能量衰减+静音检测
    config_hybrid = OnsetConfig(
        onset_threshold=0.4,
        energy_percentile=65.0,
        min_segment_duration=0.6,
        max_segment_duration=6.0,
        min_onset_interval=0.25,
        rms_threshold_percentile=15.0,
        snr_threshold_db=4.0,
        energy_decay_ratio=0.03,
        silence_threshold_db=-50,  # 更低的静音阈值
        offset_method="energy_decay"  # 将修改为混合模式
    )
    
    # 配置3: 保守长衰减配置 - 高质量但完整衰减
    config_conservative_long = OnsetConfig(
        onset_threshold=0.5,
        energy_percentile=70.0,
        min_segment_duration=0.8,
        max_segment_duration=5.0,
        min_onset_interval=0.4,
        rms_threshold_percentile=25.0,
        snr_threshold_db=6.0,
        energy_decay_ratio=0.025,
        offset_method="energy_decay"
    )
    
    return {
        "长衰减配置": config_long_decay,
        "混合检测配置": config_hybrid,
        "保守长衰减配置": config_conservative_long
    }

def test_enhanced_biannao_detection():
    """测试增强的编铙检测配置"""
    
    configs = create_biannao_configs()
    test_file = "samples/编铙/biannao-正鼓音.wav"
    
    if not Path(test_file).exists():
        logger.error(f"测试文件不存在: {test_file}")
        return
    
    results = {}
    
    for name, config in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"测试配置: {name}")
        logger.info(f"{'='*60}")
        logger.info(f"配置参数:")
        logger.info(f"  最大片段长度: {config.max_segment_duration}秒")
        logger.info(f"  能量衰减比例: {config.energy_decay_ratio}")
        logger.info(f"  最小片段长度: {config.min_segment_duration}秒")
        
        detector = ImprovedOnsetDetector(config)
        
        if not detector.load_audio(test_file):
            continue
        
        segments = detector.detect_segments()
        
        if segments:
            # 导出测试片段
            output_dir = f"test_enhanced_{name.replace(' ', '_')}"
            detector.export_segments(segments, output_dir)
            
            # 可视化
            viz_path = f"enhanced_detection_{name.replace(' ', '_')}.png"
            detector.visualize_detection(segments, viz_path)
            
            # 详细统计
            durations = [s.duration for s in segments]
            confidences = [s.confidence for s in segments]
            snrs = [s.snr_db for s in segments]
            times = [s.onset_time for s in segments]
            
            results[name] = {
                'count': len(segments),
                'durations': durations,
                'avg_duration': sum(durations) / len(durations),
                'max_duration': max(durations),
                'min_duration': min(durations),
                'avg_confidence': sum(confidences) / len(confidences),
                'avg_snr': sum(snrs) / len(snrs),
                'time_range': (min(times), max(times)) if times else (0, 0),
                'segments': segments
            }
            
            logger.info(f"检测结果:")
            logger.info(f"  片段数量: {len(segments)}")
            logger.info(f"  平均时长: {results[name]['avg_duration']:.2f}秒")
            logger.info(f"  最长片段: {results[name]['max_duration']:.2f}秒")
            logger.info(f"  最短片段: {results[name]['min_duration']:.2f}秒")
            logger.info(f"  平均置信度: {results[name]['avg_confidence']:.3f}")
            logger.info(f"  平均SNR: {results[name]['avg_snr']:.1f}dB")
            
            # 时长分布分析
            long_segments = [d for d in durations if d > 1.0]
            medium_segments = [d for d in durations if 0.5 <= d <= 1.0]
            short_segments = [d for d in durations if d < 0.5]
            
            logger.info(f"  时长分布:")
            logger.info(f"    长片段(>1.0s): {len(long_segments)} 个")
            logger.info(f"    中等片段(0.5-1.0s): {len(medium_segments)} 个")
            logger.info(f"    短片段(<0.5s): {len(short_segments)} 个")
            
            # 检查是否有足够的长衰减片段
            if len(long_segments) > 0:
                logger.info(f"  ✅ 成功捕获了 {len(long_segments)} 个长衰减片段")
                logger.info(f"  最长衰减: {max(long_segments):.2f}秒")
            else:
                logger.warning(f"  ⚠️ 没有捕获到长衰减片段，可能需要调整参数")
                
        else:
            logger.warning(f"配置 {name} 未检测到任何有效片段")
            results[name] = {'count': 0}
    
    # 比较和推荐
    logger.info(f"\n{'='*60}")
    logger.info("配置比较和推荐")
    logger.info(f"{'='*60}")
    
    best_for_decay = None
    best_decay_score = 0
    
    for name, result in results.items():
        if result['count'] > 0:
            # 评估衰减捕获能力
            long_count = len([d for d in result['durations'] if d > 1.0])
            avg_duration = result['avg_duration']
            max_duration = result['max_duration']
            
            # 衰减评分：考虑长片段数量、平均时长和最大时长
            decay_score = (long_count / result['count']) * 0.4 + \
                         min(avg_duration / 2.0, 1.0) * 0.3 + \
                         min(max_duration / 4.0, 1.0) * 0.3
            
            logger.info(f"{name}:")
            logger.info(f"  片段数: {result['count']}, 平均时长: {avg_duration:.2f}s")
            logger.info(f"  长衰减片段: {long_count}, 最大时长: {max_duration:.2f}s")
            logger.info(f"  衰减捕获评分: {decay_score:.3f}")
            
            if decay_score > best_decay_score:
                best_decay_score = decay_score
                best_for_decay = name
        else:
            logger.info(f"{name}: 0 个片段")
    
    if best_for_decay:
        logger.info(f"\n🎯 推荐配置（最佳衰减捕获）: {best_for_decay}")
        logger.info(f"评分: {best_decay_score:.3f}")
        logger.info(f"该配置最好地平衡了片段数量和衰减完整性")
        
        # 输出推荐配置的详细参数
        recommended_config = configs[best_for_decay]
        logger.info(f"\n推荐配置参数:")
        logger.info(f"  onset_threshold: {recommended_config.onset_threshold}")
        logger.info(f"  max_segment_duration: {recommended_config.max_segment_duration}")
        logger.info(f"  energy_decay_ratio: {recommended_config.energy_decay_ratio}")
        logger.info(f"  min_segment_duration: {recommended_config.min_segment_duration}")

def analyze_decay_characteristics():
    """分析编铙衰减特征"""
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    
    test_file = "samples/编铙/biannao-正鼓音.wav"
    if not Path(test_file).exists():
        logger.error(f"测试文件不存在: {test_file}")
        return
    
    logger.info("分析编铙衰减特征...")
    
    y, sr = librosa.load(test_file, sr=44100)
    
    # 找到一些高能量区域进行衰减分析
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # 找到RMS峰值
    peak_threshold = np.percentile(rms, 90)
    peak_indices = np.where(rms > peak_threshold)[0]
    
    if len(peak_indices) == 0:
        logger.warning("未找到明显的能量峰值")
        return
    
    logger.info(f"找到 {len(peak_indices)} 个高能量区域")
    
    # 分析几个典型的衰减曲线
    decay_analysis = []
    
    for i, peak_idx in enumerate(peak_indices[:5]):  # 分析前5个峰值
        peak_time = times[peak_idx]
        peak_rms = rms[peak_idx]
        
        # 分析后续的衰减
        search_length = min(400, len(rms) - peak_idx)  # 约9秒的搜索范围
        decay_curve = rms[peak_idx:peak_idx + search_length]
        decay_times = times[peak_idx:peak_idx + search_length] - peak_time
        
        # 计算衰减到不同比例需要的时间
        decay_ratios = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        decay_durations = {}
        
        for ratio in decay_ratios:
            threshold = peak_rms * ratio
            decay_idx = np.where(decay_curve <= threshold)[0]
            if len(decay_idx) > 0:
                decay_durations[ratio] = decay_times[decay_idx[0]]
            else:
                decay_durations[ratio] = None
        
        decay_analysis.append({
            'peak_time': peak_time,
            'peak_rms': peak_rms,
            'decay_durations': decay_durations
        })
        
        logger.info(f"峰值 {i+1} (时间: {peak_time:.1f}s):")
        for ratio, duration in decay_durations.items():
            if duration is not None:
                logger.info(f"  衰减到 {ratio*100:.0f}%: {duration:.2f}秒")
            else:
                logger.info(f"  衰减到 {ratio*100:.0f}%: >9秒")
    
    # 统计典型衰减时间
    all_decay_50 = [d['decay_durations'][0.5] for d in decay_analysis if d['decay_durations'][0.5] is not None]
    all_decay_10 = [d['decay_durations'][0.1] for d in decay_analysis if d['decay_durations'][0.1] is not None]
    all_decay_02 = [d['decay_durations'][0.02] for d in decay_analysis if d['decay_durations'][0.02] is not None]
    
    logger.info(f"\n编铙衰减特征统计:")
    if all_decay_50:
        logger.info(f"  衰减到50%平均时间: {np.mean(all_decay_50):.2f}秒")
    if all_decay_10:
        logger.info(f"  衰减到10%平均时间: {np.mean(all_decay_10):.2f}秒")
    if all_decay_02:
        logger.info(f"  衰减到2%平均时间: {np.mean(all_decay_02):.2f}秒")
    
    # 建议的配置参数
    if all_decay_02:
        suggested_max_duration = max(all_decay_02) + 1.0  # 留1秒余量
        suggested_decay_ratio = 0.02
    elif all_decay_10:
        suggested_max_duration = max(all_decay_10) + 1.0
        suggested_decay_ratio = 0.05
    else:
        suggested_max_duration = 6.0
        suggested_decay_ratio = 0.1
    
    logger.info(f"\n基于衰减分析的建议配置:")
    logger.info(f"  max_segment_duration: {suggested_max_duration:.1f}秒")
    logger.info(f"  energy_decay_ratio: {suggested_decay_ratio}")

if __name__ == "__main__":
    # 先分析衰减特征
    analyze_decay_characteristics()
    
    # 然后测试增强配置
    test_enhanced_biannao_detection()
