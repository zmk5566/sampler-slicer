"""
音频电平修复脚本
修复输出音频音量过小的问题
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_audio_levels(audio_path):
    """分析音频电平"""
    try:
        y, sr = sf.read(audio_path)
        
        if len(y) == 0:
            return None
        
        rms = np.sqrt(np.mean(y ** 2))
        peak = np.max(np.abs(y))
        
        # 计算响度（简化）
        if rms > 0:
            loudness_db = 20 * np.log10(rms)
        else:
            loudness_db = -100
        
        return {
            'rms': rms,
            'peak': peak,
            'loudness_db': loudness_db,
            'length': len(y),
            'sample_rate': sr
        }
    except Exception as e:
        logger.error(f"分析音频失败 {audio_path}: {e}")
        return None

def enhance_audio_levels(input_path, output_path, target_rms=0.1, target_peak=0.8):
    """增强音频电平"""
    try:
        y, sr = sf.read(input_path)
        
        if len(y) == 0:
            logger.warning(f"音频为空: {input_path}")
            return False
        
        # 移除DC偏移
        y = y - np.mean(y)
        
        # 计算当前RMS和峰值
        current_rms = np.sqrt(np.mean(y ** 2))
        current_peak = np.max(np.abs(y))
        
        if current_rms == 0 or current_peak == 0:
            logger.warning(f"音频没有信号: {input_path}")
            return False
        
        # 计算增益
        # 基于RMS计算增益，但限制峰值不超标
        rms_gain = target_rms / current_rms
        peak_gain = target_peak / current_peak
        
        # 使用较小的增益以避免削波
        gain = min(rms_gain, peak_gain)
        
        # 应用增益
        y_enhanced = y * gain
        
        # 最终检查和限制
        final_peak = np.max(np.abs(y_enhanced))
        if final_peak > 0.95:  # 留一些余量
            y_enhanced = y_enhanced * (0.95 / final_peak)
        
        # 保存增强后的音频
        sf.write(output_path, y_enhanced, sr)
        
        # 计算最终统计
        final_rms = np.sqrt(np.mean(y_enhanced ** 2))
        final_peak = np.max(np.abs(y_enhanced))
        
        logger.info(f"增强完成: {Path(input_path).name}")
        logger.info(f"  RMS: {current_rms:.6f} → {final_rms:.6f} (增益: {gain:.1f}x)")
        logger.info(f"  峰值: {current_peak:.6f} → {final_peak:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"增强音频失败 {input_path}: {e}")
        return False

def fix_exported_samples():
    """修复导出的音频样本"""
    
    # 输入和输出目录
    input_dir = Path("exported_samples")
    output_dir = Path("exported_samples_fixed")
    
    if not input_dir.exists():
        logger.error("找不到exported_samples目录")
        return
    
    # 创建输出目录结构
    segments_in = input_dir / "segments"
    variants_in = input_dir / "pitch_variants"
    
    segments_out = output_dir / "segments"
    variants_out = output_dir / "pitch_variants"
    metadata_out = output_dir / "metadata"
    
    for dir_path in [segments_out, variants_out, metadata_out]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("开始修复音频电平...")
    
    # 修复原始片段
    if segments_in.exists():
        wav_files = list(segments_in.glob("*.wav"))
        logger.info(f"修复 {len(wav_files)} 个原始片段...")
        
        success_count = 0
        for wav_file in wav_files:
            output_file = segments_out / wav_file.name
            if enhance_audio_levels(wav_file, output_file):
                success_count += 1
        
        logger.info(f"原始片段: {success_count}/{len(wav_files)} 个修复成功")
    
    # 修复音高变体
    if variants_in.exists():
        wav_files = list(variants_in.glob("*.wav"))
        logger.info(f"修复 {len(wav_files)} 个音高变体...")
        
        success_count = 0
        for wav_file in wav_files:
            output_file = variants_out / wav_file.name
            if enhance_audio_levels(wav_file, output_file):
                success_count += 1
        
        logger.info(f"音高变体: {success_count}/{len(wav_files)} 个修复成功")
    
    # 复制元数据文件
    metadata_in = input_dir / "metadata"
    if metadata_in.exists():
        for metadata_file in metadata_in.glob("*.json"):
            import shutil
            shutil.copy2(metadata_file, metadata_out / metadata_file.name)
        logger.info("元数据文件已复制")
    
    logger.info("音频电平修复完成！")
    logger.info(f"修复后的文件保存在: {output_dir}")

def test_specific_file():
    """测试特定文件"""
    test_file = "exported_samples/segments/segment_000_B_pp_246.2Hz.wav"
    
    if not Path(test_file).exists():
        logger.error(f"测试文件不存在: {test_file}")
        return
    
    logger.info("分析原始文件...")
    original = analyze_audio_levels(test_file)
    if original:
        logger.info(f"原始音频:")
        logger.info(f"  RMS: {original['rms']:.6f}")
        logger.info(f"  峰值: {original['peak']:.6f}")
        logger.info(f"  响度: {original['loudness_db']:.1f} dB")
        logger.info(f"  长度: {original['length']} 样本 ({original['length']/original['sample_rate']:.3f}秒)")
    
    # 创建增强版本
    enhanced_file = "test_enhanced.wav"
    if enhance_audio_levels(test_file, enhanced_file):
        logger.info("分析增强文件...")
        enhanced = analyze_audio_levels(enhanced_file)
        if enhanced:
            logger.info(f"增强音频:")
            logger.info(f"  RMS: {enhanced['rms']:.6f}")
            logger.info(f"  峰值: {enhanced['peak']:.6f}")
            logger.info(f"  响度: {enhanced['loudness_db']:.1f} dB")
            
            # 计算改善
            rms_improvement = enhanced['rms'] / original['rms']
            peak_improvement = enhanced['peak'] / original['peak']
            db_improvement = enhanced['loudness_db'] - original['loudness_db']
            
            logger.info(f"改善情况:")
            logger.info(f"  RMS提升: {rms_improvement:.1f}倍")
            logger.info(f"  峰值提升: {peak_improvement:.1f}倍") 
            logger.info(f"  响度提升: {db_improvement:+.1f} dB")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_specific_file()
    else:
        fix_exported_samples()
