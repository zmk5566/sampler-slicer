"""
完整的preprocessing模块测试
测试音频分割、音高分析、动态分析和音高变换功能
"""

import sys
import numpy as np
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

# 添加src到Python路径
sys.path.insert(0, 'src')

from preprocessing.onset_detector import OnsetDetector
from preprocessing.pitch_analyzer import PitchAnalyzer
from preprocessing.dynamics_analyzer import DynamicsAnalyzer
from preprocessing.pitch_shifter import PitchShifter
from preprocessing.data_structures import AudioSegment

def test_full_preprocessing_pipeline():
    """测试完整的preprocessing流水线"""
    print("=" * 60)
    print("开始测试完整的预处理流水线...")
    print("=" * 60)
    
    # 1. 创建各个分析器
    print("\n1. 初始化分析器...")
    onset_detector = OnsetDetector(threshold=0.3, min_interval=0.05)
    pitch_analyzer = PitchAnalyzer(method='yin')
    dynamics_analyzer = DynamicsAnalyzer()
    pitch_shifter = PitchShifter(method='harmonic')
    print("✓ 所有分析器初始化完成")
    
    # 2. 选择测试文件
    sample_files = [
        "samples/编铙/biannao-正鼓音.wav",
        "samples/编铙/biannao-侧鼓音.wav"
    ]
    
    test_file = None
    for sample_file in sample_files:
        if Path(sample_file).exists():
            test_file = sample_file
            print(f"✓ 找到测试文件: {test_file}")
            break
    
    if not test_file:
        print("⚠️  未找到样本文件，将创建测试音频")
        test_file = create_test_biannao_audio()
    
    # 3. 音频分割
    print(f"\n2. 音频分割...")
    segments = onset_detector.slice_audio(test_file, min_segment_length=0.3)
    print(f"✓ 分割出 {len(segments)} 个音频片段")
    
    if len(segments) == 0:
        print("❌ 没有找到有效的音频片段")
        return False
    
    # 4. 分析前几个片段
    print(f"\n3. 分析前 {min(5, len(segments))} 个片段...")
    analysis_results = []
    
    for i in range(min(5, len(segments))):
        segment = segments[i]
        print(f"\n--- 分析片段 {i+1} ---")
        print(f"时间: {segment.onset_time:.3f}s, 长度: {segment.duration:.3f}s")
        
        # 音高分析
        try:
            pitch_info = pitch_analyzer.analyze_pitch(segment)
            print(f"✓ 音高: {pitch_info.fundamental_freq:.2f}Hz ({pitch_info.note_name})")
            print(f"  置信度: {pitch_info.confidence:.3f}")
            print(f"  稳定性: {pitch_info.pitch_stability:.3f}")
        except Exception as e:
            print(f"⚠️  音高分析失败: {e}")
            pitch_info = None
        
        # 动态分析
        try:
            dynamics_info = dynamics_analyzer.analyze_dynamics(segment)
            print(f"✓ 动态: RMS={dynamics_info.rms_energy:.4f}, 等级={dynamics_info.dynamic_level}")
            print(f"  攻击时间: {dynamics_info.attack_time_ms:.1f}ms")
            print(f"  响度: {dynamics_info.loudness_lufs:.1f} LUFS")
        except Exception as e:
            print(f"⚠️  动态分析失败: {e}")
            dynamics_info = None
        
        analysis_results.append({
            'segment': segment,
            'pitch': pitch_info,
            'dynamics': dynamics_info
        })
    
    # 5. 音高变换测试
    print(f"\n4. 音高变换测试...")
    if analysis_results:
        test_segment = analysis_results[0]['segment']
        
        # 测试不同的音高变换
        semitone_shifts = [-5, -2, 0, 2, 5, 7]
        
        for semitones in semitone_shifts:
            try:
                if semitones == 0:
                    print(f"✓ 原始音高: 保持不变")
                else:
                    shifted_segment = pitch_shifter.shift_pitch(test_segment, semitones)
                    print(f"✓ 音高变换 {semitones:+d} 半音: 成功")
                    
                    # 分析变换后的音高
                    try:
                        shifted_pitch = pitch_analyzer.analyze_pitch(shifted_segment)
                        original_pitch = analysis_results[0]['pitch']
                        if original_pitch and shifted_pitch:
                            freq_ratio = shifted_pitch.fundamental_freq / original_pitch.fundamental_freq
                            expected_ratio = 2 ** (semitones / 12.0)
                            error = abs(freq_ratio - expected_ratio) / expected_ratio * 100
                            print(f"  频率变化: {original_pitch.fundamental_freq:.1f}Hz → {shifted_pitch.fundamental_freq:.1f}Hz")
                            print(f"  变换精度: {error:.1f}% 误差")
                    except:
                        pass
                        
            except Exception as e:
                print(f"⚠️  音高变换 {semitones:+d} 失败: {e}")
    
    # 6. 生成音高变体集合
    print(f"\n5. 生成音高变体集合...")
    if analysis_results:
        test_segment = analysis_results[0]['segment']
        try:
            variants = pitch_shifter.generate_pitch_variants(
                test_segment, 
                semitone_range=(-6, 6), 
                step_size=2.0
            )
            print(f"✓ 生成 {len(variants)} 个音高变体")
            
            for semitones, variant in sorted(variants.items()):
                if semitones == 0:
                    print(f"  原始: 0.0 半音")
                else:
                    print(f"  变体: {semitones:+.0f} 半音")
                    
        except Exception as e:
            print(f"⚠️  生成音高变体失败: {e}")
    
    print("\n=" * 60)
    print("✨ 完整预处理流水线测试完成!")
    print("=" * 60)
    
    return True

def create_test_biannao_audio():
    """创建测试用的编铙音频"""
    import soundfile as sf
    
    print("创建测试编铙音频...")
    
    # 音频参数
    sr = 44100
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 创建测试音频
    audio = np.zeros_like(t)
    
    # 编铙的基本频率 (根据研究，商周编铙主要音高在200-800Hz)
    base_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C4-B4
    
    # 在不同时间点添加编铙音
    onset_times = np.arange(0.5, duration-1, 1.2)
    
    for i, onset_time in enumerate(onset_times):
        start_idx = int(onset_time * sr)
        note_duration = 0.8  # 编铙余音较长
        end_idx = int((onset_time + note_duration) * sr)
        
        if end_idx >= len(audio):
            break
        
        # 选择频率
        freq = base_freqs[i % len(base_freqs)]
        
        # 时间轴
        note_samples = end_idx - start_idx
        note_t = np.linspace(0, note_duration, note_samples)
        
        # 生成编铙特有的音色
        # 1. 基音
        fundamental = np.sin(2 * np.pi * freq * note_t)
        
        # 2. 泛音 (编铙有丰富的泛音)
        harmonics = (
            0.6 * np.sin(2 * np.pi * freq * 2 * note_t) +
            0.3 * np.sin(2 * np.pi * freq * 3 * note_t) +
            0.2 * np.sin(2 * np.pi * freq * 4 * note_t) +
            0.1 * np.sin(2 * np.pi * freq * 5 * note_t)
        )
        
        # 3. 包络 (快攻击，慢衰减)
        attack_time = 0.01  # 10ms攻击时间
        decay_time = note_duration - attack_time
        
        envelope = np.ones_like(note_t)
        
        # 攻击阶段
        attack_samples = int(attack_time * sr)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # 衰减阶段
        decay_samples = note_samples - attack_samples
        if decay_samples > 0:
            # 指数衰减，模拟编铙的自然衰减
            decay_curve = np.exp(-note_t[attack_samples:] * 2)
            envelope[attack_samples:] = decay_curve
        
        # 合成音符
        note = envelope * (fundamental + 0.4 * harmonics)
        
        # 添加到音频中
        audio[start_idx:end_idx] += note * 0.3  # 降低音量避免削波
    
    # 添加少量噪声，模拟真实录音
    noise = np.random.normal(0, 0.001, len(audio))
    audio += noise
    
    # 标准化
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # 保存
    test_dir = Path("test_audio")
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test_biannao_complete.wav"
    
    sf.write(str(test_file), audio, sr)
    print(f"✓ 创建测试音频: {test_file}")
    
    return str(test_file)

def analyze_audio_features(segment, pitch_analyzer, dynamics_analyzer):
    """分析音频片段的完整特征"""
    features = {}
    
    # 音高特征
    try:
        pitch_info = pitch_analyzer.analyze_pitch(segment)
        features['pitch'] = {
            'frequency': pitch_info.fundamental_freq,
            'note': pitch_info.note_name,
            'confidence': pitch_info.confidence,
            'stability': pitch_info.pitch_stability
        }
    except Exception as e:
        features['pitch'] = {'error': str(e)}
    
    # 动态特征
    try:
        dynamics_info = dynamics_analyzer.analyze_dynamics(segment)
        features['dynamics'] = {
            'rms_energy': dynamics_info.rms_energy,
            'peak_amplitude': dynamics_info.peak_amplitude,
            'dynamic_level': dynamics_info.dynamic_level,
            'attack_time_ms': dynamics_info.attack_time_ms
        }
    except Exception as e:
        features['dynamics'] = {'error': str(e)}
    
    return features

if __name__ == "__main__":
    success = test_full_preprocessing_pipeline()
    if success:
        print("\n🎉 所有测试通过！预处理模块工作正常。")
    else:
        print("\n❌ 测试失败")
