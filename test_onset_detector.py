"""
测试OnsetDetector的基本功能
"""

import sys
import numpy as np
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, 'src')

from preprocessing.onset_detector import OnsetDetector
from preprocessing.data_structures import OnsetDetectionError

def test_onset_detector():
    """测试OnsetDetector的基本功能"""
    print("开始测试OnsetDetector...")
    
    # 创建检测器
    detector = OnsetDetector(threshold=0.3, min_interval=0.05)
    print(f"✓ OnsetDetector创建成功")
    
    # 检查样本文件是否存在
    sample_files = [
        "samples/编铙/biannao-正鼓音.wav",
        "samples/编铙/biannao-侧鼓音.wav"
    ]
    
    for sample_file in sample_files:
        if Path(sample_file).exists():
            print(f"找到样本文件: {sample_file}")
            
            try:
                # 检测onset
                onsets = detector.detect_onsets(sample_file)
                print(f"✓ 检测到 {len(onsets)} 个onset: {onsets}")
                
                # 分割音频
                segments = detector.slice_audio(sample_file)
                print(f"✓ 分割出 {len(segments)} 个音频片段")
                
                for i, segment in enumerate(segments):
                    print(f"  片段 {i}: {segment.onset_time:.3f}s, 长度: {segment.duration:.3f}s")
                
                # 获取onset强度
                times, strength = detector.get_onset_strength(sample_file)
                print(f"✓ 获取onset强度: {len(times)} 个时间点")
                
                return True
                
            except Exception as e:
                print(f"✗ 处理 {sample_file} 时出错: {e}")
                return False
        else:
            print(f"样本文件不存在: {sample_file}")
    
    print("未找到样本文件，将创建测试音频...")
    # 如果没有真实的样本文件，创建一个测试音频
    create_test_audio()
    return test_with_synthetic_audio(detector)

def create_test_audio():
    """创建一个测试音频文件"""
    import soundfile as sf
    
    # 创建一个包含几个音调的测试音频
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 创建几个短音调，模拟打击声音
    audio = np.zeros_like(t)
    
    # 在不同时间点添加音调
    onset_times = [0.5, 1.0, 1.5, 2.0]
    for onset_time in onset_times:
        start_idx = int(onset_time * sr)
        end_idx = int((onset_time + 0.1) * sr)  # 100ms的音调
        
        if end_idx < len(audio):
            # 创建衰减的正弦波，模拟编铙声音
            note_duration = end_idx - start_idx
            note_t = np.linspace(0, 0.1, note_duration)
            freq = 440 + np.random.randn() * 50  # 随机频率
            
            # 指数衰减包络
            envelope = np.exp(-note_t * 20)
            note = envelope * np.sin(2 * np.pi * freq * note_t)
            
            audio[start_idx:end_idx] = note
    
    # 保存测试音频
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test_biannao.wav"
    
    sf.write(str(test_file), audio, sr)
    print(f"创建测试音频: {test_file}")
    
    return str(test_file)

def test_with_synthetic_audio(detector):
    """使用合成音频测试"""
    test_file = create_test_audio()
    
    try:
        # 检测onset
        onsets = detector.detect_onsets(test_file)
        print(f"✓ 测试音频检测到 {len(onsets)} 个onset: {onsets}")
        
        # 分割音频
        segments = detector.slice_audio(test_file)
        print(f"✓ 分割出 {len(segments)} 个音频片段")
        
        for i, segment in enumerate(segments):
            print(f"  片段 {i}: {segment.onset_time:.3f}s, 长度: {segment.duration:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_onset_detector()
    if success:
        print("\n🎉 OnsetDetector测试通过！")
    else:
        print("\n❌ OnsetDetector测试失败")
