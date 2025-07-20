# API 文档 (API Documentation)

## 概述

本文档详细描述了商周编铙采样器系统各模块的API接口。所有API遵循一致的设计原则，提供清晰的输入输出规范和错误处理机制。

## 模块结构

```
src/
├── preprocessing/
│   ├── onset_detector.py
│   ├── pitch_analyzer.py
│   ├── dynamics_analyzer.py
│   └── pitch_shifter.py
├── database/
│   ├── sample_database.py
│   └── metadata_manager.py
└── sampler/
    ├── similarity_matcher.py
    ├── sequencer.py
    └── playback_engine.py
```

## 1. 音频预处理模块 (Preprocessing Module)

### 1.1 OnsetDetector

检测音频中的击打点并进行音频分割。

```python
class OnsetDetector:
    """音频击打点检测和分割器"""
    
    def __init__(self, 
                 hop_length: int = 512,
                 sr: int = 44100,
                 threshold: float = 0.5):
        """
        初始化检测器
        
        Args:
            hop_length: STFT hop长度
            sr: 采样率
            threshold: 检测阈值
        """
```

#### 方法

```python
def detect_onsets(self, audio_file: str) -> List[float]:
    """
    检测音频文件中的击打点时间
    
    Args:
        audio_file: 音频文件路径
        
    Returns:
        List[float]: 击打点时间列表 (秒)
        
    Raises:
        FileNotFoundError: 音频文件不存在
        AudioFormatError: 不支持的音频格式
    """

def slice_audio(self, audio_file: str, 
                padding: float = 0.1) -> List[AudioSegment]:
    """
    根据击打点分割音频
    
    Args:
        audio_file: 音频文件路径
        padding: 分割时的padding时间 (秒)
        
    Returns:
        List[AudioSegment]: 分割后的音频片段列表
    """

def set_parameters(self, **kwargs) -> None:
    """
    动态设置检测参数
    
    Args:
        **kwargs: 参数字典 (threshold, hop_length等)
    """
```

#### 数据类型

```python
@dataclass
class AudioSegment:
    """音频片段数据结构"""
    audio_data: np.ndarray    # 音频数据
    sample_rate: int          # 采样率
    onset_time: float         # 起始时间
    duration: float           # 持续时间
    source_file: str          # 源文件路径
```

### 1.2 PitchAnalyzer

分析音频片段的音高信息。

```python
class PitchAnalyzer:
    """音高分析器"""
    
    def __init__(self, 
                 frame_length: int = 2048,
                 method: str = 'yin'):
        """
        初始化音高分析器
        
        Args:
            frame_length: 分析帧长度
            method: 分析方法 ('yin', 'pyin', 'spectral')
        """
```

#### 方法

```python
def analyze_pitch(self, audio_segment: AudioSegment) -> PitchInfo:
    """
    分析音频片段的音高
    
    Args:
        audio_segment: 音频片段
        
    Returns:
        PitchInfo: 音高分析结果
    """

def estimate_note_name(self, frequency: float) -> str:
    """
    将频率转换为音符名称
    
    Args:
        frequency: 基频 (Hz)
        
    Returns:
        str: 音符名称 (如 "A4", "C#3")
    """

def calculate_confidence(self, audio_segment: AudioSegment) -> float:
    """
    计算音高检测的置信度
    
    Args:
        audio_segment: 音频片段
        
    Returns:
        float: 置信度 (0-1)
    """
```

#### 数据类型

```python
@dataclass
class PitchInfo:
    """音高分析结果"""
    fundamental_freq: float   # 基频 (Hz)
    note_name: str           # 音符名称
    octave: int              # 八度
    cents_deviation: float   # 音分偏差
    confidence: float        # 检测置信度
    harmonics: List[float]   # 泛音频率列表
```

### 1.3 DynamicsAnalyzer

分析音频的力度和动态特征。

```python
class DynamicsAnalyzer:
    """动态分析器"""
    
    def __init__(self, frame_length: int = 2048):
        """初始化动态分析器"""
```

#### 方法

```python
def analyze_dynamics(self, audio_segment: AudioSegment) -> DynamicsInfo:
    """
    分析音频的动态特征
    
    Args:
        audio_segment: 音频片段
        
    Returns:
        DynamicsInfo: 动态分析结果
    """

def classify_dynamic_level(self, loudness_lufs: float) -> str:
    """
    将响度值分类为动态等级
    
    Args:
        loudness_lufs: LUFS响度值
        
    Returns:
        str: 动态等级 ("pp", "p", "mp", "mf", "f", "ff")
    """
```

#### 数据类型

```python
@dataclass
class DynamicsInfo:
    """动态分析结果"""
    rms_energy: float         # RMS能量
    peak_amplitude: float     # 峰值幅度
    loudness_lufs: float      # LUFS响度
    dynamic_range: float      # 动态范围
    attack_time: float        # 攻击时间
    decay_time: float         # 衰减时间
    dynamic_level: str        # 分类等级
```

### 1.4 PitchShifter

生成不同音高的样本变体。

```python
class PitchShifter:
    """音高变换器"""
    
    def __init__(self, method: str = 'psola'):
        """
        初始化音高变换器
        
        Args:
            method: 变换方法 ('psola', 'phase_vocoder')
        """
```

#### 方法

```python
def generate_pitch_variants(self, 
                          audio_segment: AudioSegment,
                          semitone_range: Tuple[int, int] = (-12, 12)) -> Dict[int, AudioSegment]:
    """
    生成音高变体
    
    Args:
        audio_segment: 原始音频片段
        semitone_range: 半音范围 (最小值, 最大值)
        
    Returns:
        Dict[int, AudioSegment]: 半音偏移 -> 变换后音频
    """

def shift_pitch(self, 
                audio_segment: AudioSegment, 
                semitones: float) -> AudioSegment:
    """
    变换指定半音数
    
    Args:
        audio_segment: 音频片段
        semitones: 半音偏移量
        
    Returns:
        AudioSegment: 变换后的音频
    """
```

## 2. 数据库模块 (Database Module)

### 2.1 SampleDatabase

管理样本数据的存储和检索。

```python
class SampleDatabase:
    """样本数据库"""
    
    def __init__(self, db_path: str):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径
        """
```

#### 方法

```python
def add_sample(self, sample: Sample) -> str:
    """
    添加样本到数据库
    
    Args:
        sample: 样本对象
        
    Returns:
        str: 生成的样本ID
    """

def get_sample(self, sample_id: str) -> Optional[Sample]:
    """
    根据ID获取样本
    
    Args:
        sample_id: 样本ID
        
    Returns:
        Optional[Sample]: 样本对象或None
    """

def search_samples(self, criteria: SearchCriteria) -> List[Sample]:
    """
    根据条件搜索样本
    
    Args:
        criteria: 搜索条件
        
    Returns:
        List[Sample]: 匹配的样本列表
    """

def update_sample(self, sample_id: str, updates: Dict) -> bool:
    """
    更新样本信息
    
    Args:
        sample_id: 样本ID
        updates: 更新字段字典
        
    Returns:
        bool: 更新是否成功
    """

def delete_sample(self, sample_id: str) -> bool:
    """
    删除样本
    
    Args:
        sample_id: 样本ID
        
    Returns:
        bool: 删除是否成功
    """
```

#### 数据类型

```python
@dataclass
class Sample:
    """样本数据结构"""
    sample_id: str
    source_file: str
    instrument_type: str
    sound_type: str
    audio_features: AudioFeatures
    pitch_variants: Dict[int, str]
    processing_info: ProcessingInfo
    metadata: Dict[str, Any]

@dataclass
class SearchCriteria:
    """搜索条件"""
    instrument_type: Optional[str] = None
    sound_type: Optional[str] = None
    pitch_range: Optional[Tuple[float, float]] = None
    dynamics_range: Optional[Tuple[float, float]] = None
    quality_threshold: Optional[float] = None
```

### 2.2 MetadataManager

管理样本元数据和索引。

```python
class MetadataManager:
    """元数据管理器"""
    
    def __init__(self, metadata_path: str):
        """初始化元数据管理器"""
```

#### 方法

```python
def create_index(self, field: str, index_type: str = 'btree') -> None:
    """
    创建字段索引
    
    Args:
        field: 字段名
        index_type: 索引类型
    """

def query_by_features(self, features: AudioFeatures, 
                     tolerance: float = 0.1) -> List[str]:
    """
    根据音频特征查询样本
    
    Args:
        features: 目标特征
        tolerance: 容差范围
        
    Returns:
        List[str]: 匹配的样本ID列表
    """

def get_statistics(self) -> Dict[str, Any]:
    """
    获取数据库统计信息
    
    Returns:
        Dict: 统计信息
    """
```

## 3. 采样器引擎 (Sampler Engine)

### 3.1 SimilarityMatcher

实现基于特征的相似度匹配。

```python
class SimilarityMatcher:
    """相似度匹配器"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        初始化匹配器
        
        Args:
            weights: 特征权重字典
        """
```

#### 方法

```python
def find_best_match(self, 
                   target_features: AudioFeatures,
                   candidates: List[str] = None) -> str:
    """
    找到最佳匹配样本
    
    Args:
        target_features: 目标特征
        candidates: 候选样本ID列表
        
    Returns:
        str: 最佳匹配的样本ID
    """

def calculate_similarity(self, 
                        features1: AudioFeatures,
                        features2: AudioFeatures) -> float:
    """
    计算特征相似度
    
    Args:
        features1: 特征1
        features2: 特征2
        
    Returns:
        float: 相似度分数 (0-1)
    """

def set_weights(self, weights: Dict[str, float]) -> None:
    """
    设置特征权重
    
    Args:
        weights: 权重字典
    """
```

### 3.2 Sequencer

音序器控制和MIDI处理。

```python
class Sequencer:
    """音序器"""
    
    def __init__(self, sampler_engine):
        """初始化音序器"""
```

#### 方法

```python
def create_sequence(self, events: List[MidiEvent]) -> Sequence:
    """
    创建音序
    
    Args:
        events: MIDI事件列表
        
    Returns:
        Sequence: 音序对象
    """

def play_sequence(self, sequence: Sequence, 
                 mode: PlaybackMode = PlaybackMode.SIMILARITY) -> None:
    """
    播放音序
    
    Args:
        sequence: 音序对象
        mode: 播放模式
    """

def stop_sequence(self) -> None:
    """停止当前播放"""

def set_tempo(self, bpm: float) -> None:
    """
    设置播放速度
    
    Args:
        bpm: 每分钟节拍数
    """
```

#### 数据类型

```python
@dataclass
class MidiEvent:
    """MIDI事件"""
    timestamp: float
    note: int
    velocity: int
    duration: float

class PlaybackMode(Enum):
    """播放模式"""
    SIMILARITY = "similarity"
    RANDOM = "random"
```

### 3.3 PlaybackEngine

音频播放引擎。

```python
class PlaybackEngine:
    """播放引擎"""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 buffer_size: int = 1024):
        """初始化播放引擎"""
```

#### 方法

```python
def play_sample(self, 
                sample_id: str,
                velocity: int = 80,
                pitch_offset: int = 0) -> None:
    """
    播放样本
    
    Args:
        sample_id: 样本ID
        velocity: 演奏力度 (0-127)
        pitch_offset: 音高偏移 (半音)
    """

def stop_sample(self, sample_id: str) -> None:
    """
    停止指定样本
    
    Args:
        sample_id: 样本ID
    """

def stop_all_samples(self) -> None:
    """停止所有播放中的样本"""

def set_volume(self, sample_id: str, volume: float) -> None:
    """
    设置样本音量
    
    Args:
        sample_id: 样本ID
        volume: 音量 (0.0-1.0)
    """
```

## 4. 公共数据类型

### 4.1 AudioFeatures

```python
@dataclass
class AudioFeatures:
    """音频特征集合"""
    pitch: PitchInfo
    dynamics: DynamicsInfo
    spectral: SpectralFeatures
    temporal: TemporalFeatures

@dataclass
class SpectralFeatures:
    """频谱特征"""
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    mfcc: np.ndarray
    chroma: np.ndarray
    zero_crossing_rate: float

@dataclass
class TemporalFeatures:
    """时间域特征"""
    duration: float
    attack_time: float
    decay_time: float
    sustain_level: float
    release_time: float
```

## 5. 错误处理

### 5.1 异常类型

```python
class SamplerError(Exception):
    """采样器基础异常"""
    pass

class AudioFormatError(SamplerError):
    """音频格式错误"""
    pass

class PitchDetectionError(SamplerError):
    """音高检测错误"""
    pass

class DatabaseError(SamplerError):
    """数据库操作错误"""
    pass

class PlaybackError(SamplerError):
    """播放错误"""
    pass
```

### 5.2 错误码

```python
class ErrorCode(Enum):
    """错误码定义"""
    SUCCESS = 0
    FILE_NOT_FOUND = 1001
    INVALID_FORMAT = 1002
    PROCESSING_FAILED = 1003
    DATABASE_ERROR = 2001
    PLAYBACK_ERROR = 3001
```

## 6. 配置和常量

### 6.1 系统配置

```python
class Config:
    """系统配置"""
    SAMPLE_RATE = 44100
    BUFFER_SIZE = 1024
    MAX_PITCH_SHIFT = 12
    MIN_PITCH_SHIFT = -12
    DEFAULT_THRESHOLD = 0.5
    CACHE_SIZE = 100
```

## 7. 使用示例

### 7.1 完整处理流程

```python
from src.preprocessing import OnsetDetector, PitchAnalyzer, DynamicsAnalyzer, PitchShifter
from src.database import SampleDatabase, MetadataManager
from src.sampler import SamplerEngine

# 1. 音频预处理
detector = OnsetDetector()
pitch_analyzer = PitchAnalyzer()
dynamics_analyzer = DynamicsAnalyzer()
pitch_shifter = PitchShifter()

# 分割音频
segments = detector.slice_audio("samples/编铙/biannao-正鼓音.wav")

# 分析特征
for segment in segments:
    pitch_info = pitch_analyzer.analyze_pitch(segment)
    dynamics_info = dynamics_analyzer.analyze_dynamics(segment)
    
    # 生成音高变体
    variants = pitch_shifter.generate_pitch_variants(segment)
    
    # 创建样本对象
    sample = Sample(
        sample_id=generate_id(),
        source_file=segment.source_file,
        audio_features=AudioFeatures(pitch=pitch_info, dynamics=dynamics_info),
        pitch_variants=variants
    )
    
    # 存储到数据库
    db = SampleDatabase("samples.db")
    db.add_sample(sample)

# 2. 创建采样器
sampler = SamplerEngine()
sampler.load_database("samples.db")

# 3. 播放样本
sampler.play_note(pitch=220, velocity=80, mode="similarity")
```

这个API设计提供了完整的功能覆盖，支持从音频预处理到最终播放的完整工作流程，同时保持了良好的模块化和扩展性。
