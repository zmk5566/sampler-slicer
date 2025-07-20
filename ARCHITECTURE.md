# 系统架构设计 (System Architecture)

## 总体架构

商周编铙采样器采用模块化设计，分为三个主要层次：

```
┌─────────────────────────────────────────┐
│           用户界面层 (UI Layer)            │
├─────────────────────────────────────────┤
│          业务逻辑层 (Logic Layer)          │
│  ┌─────────────┬─────────────┬─────────────┐ │
│  │ 音频预处理模块 │   数据库模块  │  采样器引擎   │ │
│  │Preprocessing│  Database   │   Sampler   │ │
│  └─────────────┴─────────────┴─────────────┘ │
├─────────────────────────────────────────┤
│           数据层 (Data Layer)             │
│  ┌─────────────┬─────────────┬─────────────┐ │
│  │  原始音频    │  处理音频    │   元数据     │ │
│  │Raw Samples  │Processed    │  Metadata   │ │
│  │             │Samples      │             │ │
│  └─────────────┴─────────────┴─────────────┘ │
└─────────────────────────────────────────┘
```

## 核心模块设计

### 1. 音频预处理模块 (Preprocessing Module)

负责原始音频的分析和处理，包含以下子模块：

#### 1.1 Onset Detection (击打点检测)
```python
class OnsetDetector:
    """
    使用librosa的onset detection算法检测音频中的击打点
    """
    def detect_onsets(self, audio_file: str) -> List[float]
    def slice_audio(self, audio_file: str) -> List[AudioSegment]
```

**算法选择**:
- 主算法: `librosa.onset.onset_detect()` with spectral flux
- 备选算法: Complex domain onset detection
- 后处理: 最小间隔过滤 (防止重复检测)

#### 1.2 Pitch Analysis (音高分析)
```python
class PitchAnalyzer:
    """
    分析每个音频片段的基频和音高信息
    """
    def analyze_pitch(self, audio_segment: AudioSegment) -> PitchInfo
    def estimate_note_name(self, frequency: float) -> str
```

**算法选择**:
- 主算法: YIN algorithm (`librosa.yin()`)
- 备选算法: PYIN for noisy signals
- 置信度评估: 基于autocorrelation峰值强度

#### 1.3 Dynamics Analysis (力度分析)
```python
class DynamicsAnalyzer:
    """
    分析音频的响度和动态特征
    """
    def analyze_dynamics(self, audio_segment: AudioSegment) -> DynamicsInfo
    def classify_dynamic_level(self, loudness: float) -> str
```

**特征提取**:
- RMS Energy: 整体能量水平
- Peak Amplitude: 峰值响度
- LUFS: 感知响度标准
- Dynamic Range: 动态范围分析

#### 1.4 Pitch Shifting (音高变换)
```python
class PitchShifter:
    """
    生成不同音高的样本变体 (-12 to +12 semitones)
    """
    def generate_pitch_variants(self, audio_segment: AudioSegment) -> Dict[int, AudioSegment]
```

**技术实现**:
- 算法: Phase Vocoder (PSOLA)
- 范围: -12 到 +12 半音
- 质量控制: 保持音频长度不变

### 2. 数据库模块 (Database Module)

管理样本数据和元数据的存储与检索：

#### 2.1 Sample Database
```python
class SampleDatabase:
    """
    样本数据库管理
    """
    def add_sample(self, sample: Sample) -> str
    def get_sample(self, sample_id: str) -> Sample
    def search_samples(self, criteria: SearchCriteria) -> List[Sample]
```

#### 2.2 Metadata Manager
```python
class MetadataManager:
    """
    元数据管理和索引
    """
    def create_index(self, field: str) -> None
    def query_by_features(self, features: AudioFeatures) -> List[str]
```

**数据结构**:
- 存储格式: JSON + SQLite索引
- 索引字段: 音高、力度、音色特征
- 缓存策略: LRU cache for frequent access

### 3. 采样器引擎 (Sampler Engine)

核心播放和控制逻辑：

#### 3.1 Similarity Matcher (相似度匹配)
```python
class SimilarityMatcher:
    """
    基于特征相似度的样本匹配算法
    """
    def find_best_match(self, target_features: AudioFeatures) -> str
    def calculate_similarity(self, features1: AudioFeatures, features2: AudioFeatures) -> float
```

**相似度算法**:
```
similarity_score = w1 * pitch_similarity + 
                   w2 * dynamics_similarity + 
                   w3 * timbre_similarity

where:
- w1 = 0.6 (音高权重)
- w2 = 0.3 (力度权重) 
- w3 = 0.1 (音色权重)
```

#### 3.2 Sequencer (音序器)
```python
class Sequencer:
    """
    音序播放控制
    """
    def create_sequence(self, events: List[MidiEvent]) -> Sequence
    def play_sequence(self, sequence: Sequence, mode: PlaybackMode) -> None
```

**播放模式**:
- **Similarity Mode**: 根据输入特征找最相似样本
- **Random Mode**: 在特征范围内随机选择样本

#### 3.3 Playback Engine (播放引擎)
```python
class PlaybackEngine:
    """
    音频播放和实时控制
    """
    def play_sample(self, sample_id: str, velocity: int) -> None
    def stop_all_samples(self) -> None
```

## 数据流设计

### 预处理流程
```
原始音频 → Onset Detection → 音频切割 → 特征分析 → Pitch Shifting → 元数据存储
     ↓                                                                    ↓
  samples/编铙/               processed_samples/              metadata/samples.json
```

### 播放流程
```
MIDI输入 → 特征提取 → 相似度匹配 → 样本选择 → 音频播放
    ↓           ↓           ↓          ↓         ↓
  Note On    Target     Best Match   Sample    Audio Out
           Features    Algorithm     Loading
```

## 性能优化策略

### 1. 内存管理
- **样本缓存**: LRU缓存常用样本
- **延迟加载**: 按需加载音频数据
- **内存池**: 复用AudioSegment对象

### 2. 计算优化
- **特征预计算**: 预计算所有样本特征
- **索引加速**: 多维索引加速相似度搜索
- **并行处理**: 多线程处理音频切割

### 3. 实时性能
- **预加载**: 预加载可能使用的样本
- **缓冲区**: 音频输出缓冲区优化
- **中断优先级**: 实时音频任务优先级

## 扩展性设计

### 1. 模块插件化
- 每个算法模块可独立替换
- 支持自定义特征提取器
- 支持第三方音频效果器

### 2. 乐器扩展
- 配置驱动的乐器定义
- 可扩展的音色分析算法
- 通用的样本处理流程

### 3. 格式兼容
- 支持多种音频格式输入
- 标准化的元数据格式
- 兼容主流采样器格式

## 错误处理和质量控制

### 1. 输入验证
- 音频格式检查
- 采样率统一
- 文件完整性验证

### 2. 质量评估
- 样本质量评分
- 特征提取置信度
- 异常样本过滤

### 3. 容错机制
- 算法降级策略
- 缺失数据处理
- 错误恢复机制

## 未来演进路径

### Phase 1: Python原型 (当前)
- 核心算法实现
- 基础功能验证
- 性能基准测试

### Phase 2: 优化版本
- 算法优化
- 用户界面开发
- 完整功能集成

### Phase 3: 插件版本
- C++重写核心算法
- VST/AU插件开发
- DAW集成测试

这个架构设计确保了系统的可扩展性、可维护性和高性能，为商周编铙采样器的成功实现奠定了坚实基础。
