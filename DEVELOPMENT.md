# 开发指南 (Development Guide)

## 概述

本文档为商周编铙采样器项目的开发者提供详细的开发环境配置、编码规范、测试策略和贡献指南。

## 环境配置

### 系统要求

- **操作系统**: Windows 10+, macOS 10.15+, 或 Linux (Ubuntu 18.04+)
- **Python**: 3.8 - 3.11
- **内存**: 最少 8GB RAM (推荐 16GB)
- **存储**: 至少 5GB 可用空间
- **音频设备**: 支持44.1kHz或更高采样率的音频接口

### 开发环境安装

#### 1. Python环境设置

```bash
# 使用pyenv管理Python版本 (推荐)
curl https://pyenv.run | bash
pyenv install 3.10.12
pyenv local 3.10.12

# 或直接使用系统Python
python3 --version  # 确保版本 >= 3.8
```

#### 2. 虚拟环境创建

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux  
source venv/bin/activate
```

#### 3. 依赖安装

```bash
# 更新pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt

# 安装开发依赖
pip install -r requirements-dev.txt
```

#### 4. 音频库安装

```bash
# Linux额外依赖
sudo apt-get install libsndfile1 libasound2-dev

# macOS额外依赖
brew install libsndfile portaudio

# Windows额外依赖 (使用conda)
conda install -c conda-forge libsndfile
```

### 依赖文件

#### requirements.txt
```
numpy>=1.21.0,<2.0.0
scipy>=1.7.0,<2.0.0
librosa>=0.9.0,<1.0.0
soundfile>=0.10.0,<1.0.0
scikit-learn>=1.0.0,<2.0.0
matplotlib>=3.5.0,<4.0.0
pandas>=1.3.0,<3.0.0
tqdm>=4.62.0,<5.0.0
PyYAML>=6.0,<7.0
click>=8.0.0,<9.0.0
```

#### requirements-dev.txt
```
pytest>=7.0.0,<8.0.0
pytest-cov>=4.0.0,<5.0.0
black>=22.0.0,<24.0.0
flake8>=5.0.0,<7.0.0
mypy>=1.0.0,<2.0.0
isort>=5.10.0,<6.0.0
pre-commit>=2.20.0,<4.0.0
sphinx>=5.0.0,<7.0.0
jupyter>=1.0.0,<2.0.0
ipykernel>=6.15.0,<7.0.0
```

### IDE配置

#### VS Code推荐插件

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter", 
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    "ms-python.isort",
    "redhat.vscode-yaml",
    "ms-vscode.vscode-json",
    "formulahendry.auto-rename-tag"
  ]
}
```

#### VS Code设置 (.vscode/settings.json)

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/venv": true
  }
}
```

## 项目结构

```
sampler-slicer/
├── src/                          # 源代码
│   ├── __init__.py
│   ├── preprocessing/            # 音频预处理模块
│   │   ├── __init__.py
│   │   ├── onset_detector.py
│   │   ├── pitch_analyzer.py
│   │   ├── dynamics_analyzer.py
│   │   └── pitch_shifter.py
│   ├── database/                 # 数据库模块
│   │   ├── __init__.py
│   │   ├── sample_database.py
│   │   └── metadata_manager.py
│   ├── sampler/                  # 采样器引擎
│   │   ├── __init__.py
│   │   ├── similarity_matcher.py
│   │   ├── sequencer.py
│   │   └── playback_engine.py
│   ├── utils/                    # 工具模块
│   │   ├── __init__.py
│   │   ├── audio_utils.py
│   │   ├── file_utils.py
│   │   └── validation.py
│   └── cli/                      # 命令行界面
│       ├── __init__.py
│       └── commands.py
├── tests/                        # 测试文件
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_preprocessing/
│   ├── test_database/
│   ├── test_sampler/
│   └── test_utils/
├── samples/                      # 音频样本
├── processed_samples/            # 处理后样本
├── metadata/                     # 元数据存储
├── docs/                         # 文档
├── scripts/                      # 脚本工具
├── config/                       # 配置文件
└── notebooks/                    # Jupyter notebooks
```

## 编码规范

### Python编码标准

我们遵循 [PEP 8](https://pep8.org/) 和 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)。

#### 代码格式化

```bash
# 使用Black进行代码格式化
black src/ tests/

# 使用isort整理import语句
isort src/ tests/

# 使用flake8检查代码风格
flake8 src/ tests/
```

#### 类型注解

所有函数和方法必须包含类型注解：

```python
from typing import List, Optional, Dict, Tuple
import numpy as np

def analyze_pitch(
    audio_data: np.ndarray, 
    sample_rate: int,
    frame_length: int = 2048
) -> Tuple[float, float]:
    """
    分析音频的音高信息
    
    Args:
        audio_data: 音频数据数组
        sample_rate: 采样率
        frame_length: 分析帧长度
        
    Returns:
        Tuple[float, float]: (基频, 置信度)
    """
    # 实现代码
    pass
```

#### 文档字符串

使用Google风格的docstring：

```python
def detect_onsets(
    audio_file: str, 
    threshold: float = 0.5
) -> List[float]:
    """检测音频文件中的击打点
    
    使用spectral flux方法检测音频中的onset时间点，
    适用于打击乐器的快速攻击特征。
    
    Args:
        audio_file: 音频文件路径
        threshold: 检测阈值，范围0-1，值越小越敏感
        
    Returns:
        击打点时间列表，单位为秒
        
    Raises:
        FileNotFoundError: 当音频文件不存在时
        AudioFormatError: 当音频格式不支持时
        
    Example:
        >>> detector = OnsetDetector()
        >>> onsets = detector.detect_onsets("sample.wav", threshold=0.3)
        >>> print(f"检测到 {len(onsets)} 个击打点")
    """
    pass
```

### 命名规范

```python
# 类名: PascalCase
class OnsetDetector:
    pass

# 函数和变量名: snake_case
def analyze_pitch():
    pass

sample_rate = 44100

# 常量: UPPER_SNAKE_CASE
DEFAULT_THRESHOLD = 0.5
MAX_PITCH_SHIFT = 12

# 私有成员: 前缀下划线
class AudioProcessor:
    def __init__(self):
        self._buffer_size = 1024
        self.__secret_key = "hidden"
```

### 错误处理

```python
# 自定义异常类
class SamplerError(Exception):
    """采样器基础异常"""
    pass

class PitchDetectionError(SamplerError):
    """音高检测异常"""
    def __init__(self, message: str, confidence: float = 0.0):
        super().__init__(message)
        self.confidence = confidence

# 错误处理模式
def process_audio(file_path: str) -> AudioSegment:
    try:
        audio_data = load_audio(file_path)
        return analyze_audio(audio_data)
    except FileNotFoundError:
        logger.error(f"音频文件未找到: {file_path}")
        raise
    except Exception as e:
        logger.error(f"处理音频失败: {e}")
        raise ProcessingError(f"无法处理音频文件 {file_path}: {e}")
```

## 测试策略

### 测试框架配置

#### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
python_classes = Test*
addopts = 
    --verbose
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: 单元测试
    integration: 集成测试
    slow: 慢速测试
    audio: 需要音频文件的测试
```

### 测试分类

#### 单元测试

```python
# tests/test_preprocessing/test_onset_detector.py
import pytest
import numpy as np
from src.preprocessing.onset_detector import OnsetDetector

class TestOnsetDetector:
    
    @pytest.fixture
    def detector(self):
        return OnsetDetector(threshold=0.5)
    
    @pytest.fixture
    def sample_audio(self):
        """生成测试用音频信号"""
        duration = 2.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 生成包含突然变化的信号
        signal = np.zeros_like(t)
        signal[int(0.5 * sample_rate):int(0.6 * sample_rate)] = np.sin(2 * np.pi * 440 * t[int(0.5 * sample_rate):int(0.6 * sample_rate)])
        signal[int(1.0 * sample_rate):int(1.1 * sample_rate)] = np.sin(2 * np.pi * 880 * t[int(1.0 * sample_rate):int(1.1 * sample_rate)])
        
        return signal, sample_rate
    
    @pytest.mark.unit
    def test_detect_onsets_basic(self, detector, sample_audio):
        """测试基本onset检测功能"""
        audio, sr = sample_audio
        onsets = detector._detect_onsets_from_array(audio, sr)
        
        assert len(onsets) >= 2, "应该检测到至少2个onset"
        assert all(0 <= onset <= 2.0 for onset in onsets), "onset时间应该在有效范围内"
    
    @pytest.mark.unit
    def test_threshold_sensitivity(self, detector, sample_audio):
        """测试阈值对检测敏感度的影响"""
        audio, sr = sample_audio
        
        # 高阈值应该检测到更少的onset
        detector.set_threshold(0.8)
        onsets_high = detector._detect_onsets_from_array(audio, sr)
        
        # 低阈值应该检测到更多的onset
        detector.set_threshold(0.2)
        onsets_low = detector._detect_onsets_from_array(audio, sr)
        
        assert len(onsets_low) >= len(onsets_high), "低阈值应该检测到更多onset"
```

#### 集成测试

```python
# tests/test_integration/test_full_pipeline.py
import pytest
import tempfile
import soundfile as sf
from src.preprocessing import OnsetDetector, PitchAnalyzer
from src.database import SampleDatabase

@pytest.mark.integration
class TestFullPipeline:
    
    @pytest.fixture
    def temp_audio_file(self):
        """创建临时音频文件"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # 生成测试音频
            audio = generate_test_audio()
            sf.write(f.name, audio, 44100)
            yield f.name
        # 清理
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_database(self):
        """创建临时数据库"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db = SampleDatabase(f.name)
            yield db
        # 清理
        os.unlink(f.name)
    
    def test_end_to_end_processing(self, temp_audio_file, temp_database):
        """测试从音频文件到数据库存储的完整流程"""
        # 1. 检测onset
        detector = OnsetDetector()
        segments = detector.slice_audio(temp_audio_file)
        assert len(segments) > 0, "应该检测到音频片段"
        
        # 2. 分析音高
        analyzer = PitchAnalyzer()
        for segment in segments:
            pitch_info = analyzer.analyze_pitch(segment)
            assert pitch_info.confidence > 0.5, "音高检测置信度应该足够高"
        
        # 3. 存储到数据库
        for i, segment in enumerate(segments):
            sample_id = temp_database.add_sample({
                'segment': segment,
                'pitch_info': pitch_info,
                'metadata': {'test': True}
            })
            assert sample_id is not None, "样本应该成功存储"
```

#### 性能测试

```python
# tests/test_performance/test_benchmarks.py
import pytest
import time
import numpy as np
from src.preprocessing.onset_detector import OnsetDetector

@pytest.mark.slow
class TestPerformance:
    
    def test_onset_detection_speed(self):
        """测试onset检测的性能"""
        detector = OnsetDetector()
        
        # 生成较长的音频文件 (10秒)
        duration = 10.0
        sample_rate = 44100
        audio = np.random.randn(int(duration * sample_rate))
        
        start_time = time.time()
        onsets = detector._detect_onsets_from_array(audio, sample_rate)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # 应该在合理时间内完成处理
        assert processing_time < 2.0, f"处理时间过长: {processing_time:.2f}s"
        
        # 计算实时因子
        real_time_factor = duration / processing_time
        assert real_time_factor > 5.0, f"实时因子过低: {real_time_factor:.2f}x"
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定标记的测试
pytest -m unit           # 只运行单元测试
pytest -m integration    # 只运行集成测试
pytest -m "not slow"     # 跳过慢速测试

# 运行特定文件的测试
pytest tests/test_preprocessing/

# 生成覆盖率报告
pytest --cov=src --cov-report=html

# 运行性能测试
pytest -m slow --benchmark-only
```

## 代码质量工具

### Pre-commit配置

创建 `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        additional_dependencies: [flake8-docstrings]
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

安装和启用：

```bash
# 安装pre-commit hooks
pre-commit install

# 手动运行所有hooks
pre-commit run --all-files
```

### 持续集成配置

#### GitHub Actions (.github/workflows/ci.yml)

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
        
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y libsndfile1 libasound2-dev
        
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        
    - name: Lint with flake8
      run: |
        flake8 src/ tests/
        
    - name: Format check with black
      run: |
        black --check src/ tests/
        
    - name: Import sort check
      run: |
        isort --check-only src/ tests/
        
    - name: Type check with mypy
      run: |
        mypy src/
        
    - name: Test with pytest
      run: |
        pytest --cov=src --cov-report=xml
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 调试和性能分析

### 日志配置

```python
# src/utils/logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(level: str = "INFO", log_file: str = None):
    """配置项目日志"""
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件handler (可选)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

# 使用示例
import logging
logger = logging.getLogger(__name__)

def process_audio(file_path: str):
    logger.info(f"开始处理音频文件: {file_path}")
    try:
        # 处理逻辑
        logger.debug("完成onset检测")
        logger.debug("完成音高分析") 
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        raise
```

### 性能分析

```python
# scripts/profiling.py
import cProfile
import pstats
from src.preprocessing.onset_detector import OnsetDetector

def profile_onset_detection():
    """分析onset检测的性能瓶颈"""
    detector = OnsetDetector()
    
    # 生成测试数据
    audio_file = "samples/test.wav"
    
    # 运行profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 执行要分析的代码
    segments = detector.slice_audio(audio_file)
    
    profiler.disable()
    
    # 分析结果
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 显示前20个最耗时的函数

if __name__ == "__main__":
    profile_onset_detection()
```

## 贡献指南

### 分支策略

我们使用 Git Flow 分支模型：

- **main**: 稳定的生产版本
- **develop**: 开发分支，集成最新功能
- **feature/xxx**: 功能分支
- **hotfix/xxx**: 紧急修复分支
- **release/xxx**: 发布准备分支

### 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型说明：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档变更
- `style`: 代码格式变更
- `refactor`: 重构代码
- `test`: 添加或修改测试
- `chore`: 构建过程或辅助工具的变动

示例：
```
feat(preprocessing): 添加多算法融合的音高检测

实现了YIN和频谱峰值检测的融合算法，提高了编铙等
打击乐器的音高检测准确率。

- 支持置信度加权融合
- 增加了鲁棒性检测
- 更新了相关测试用例

Closes #123
```

### Pull Request流程

1. **创建feature分支**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/new-pitch-algorithm
   ```

2. **开发和测试**
   ```bash
   # 编写代码
   # 运行测试
   pytest
   # 检查代码质量
   pre-commit run --all-files
   ```

3. **提交代码**
   ```bash
   git add .
   git commit -m "feat(pitch): 实现新的音高检测算法"
   git push origin feature/new-pitch-algorithm
   ```

4. **创建Pull Request**
   - 使用PR模板
   - 包含详细的变更说明
   - 关联相关的issue
   - 添加适当的标签

5. **代码审查**
   - 至少需要一位审查者批准
   - 所有CI检查必须通过
   - 解决所有审查意见

6. **合并**
   - 使用"Squash and merge"
   - 删除feature分支

### 发布流程

1. **创建release分支**
   ```bash
   git checkout develop
   git checkout -b release/1.1.0
   ```

2. **更新版本信息**
   - 更新 `__version__`
   - 更新 CHANGELOG.md
   - 更新文档

3. **最终测试**
   ```bash
   pytest --cov=src
   ```

4. **合并到main**
   ```bash
   git checkout main
   git merge --no-ff release/1.1.0
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin main --tags
   ```

5. **回合并到develop**
   ```bash
   git checkout develop  
   git merge --no-ff release/1.1.0
   git push origin develop
   ```

### Issue模板

#### Bug Report
```markdown
**描述bug**
简要描述发生了什么问题。

**重现步骤**
1. 执行 '...'
2. 点击 '....'
3. 看到错误

**期望行为**
描述您期望发生什么。

**实际行为**
描述实际发生了什么。

**环境信息**
- OS: [如 Windows 10]
- Python版本: [如 3.10.12]
- 项目版本: [如 1.0.0]

**附加信息**
添加任何其他有助于解释问题的信息。
```

## 常见问题解决

### 常见开发问题

1. **音频库安装失败**
   ```bash
   # macOS
   brew install libsndfile portaudio
   export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"
   
   # Linux
   sudo apt-get install libsndfile1-dev libasound2-dev
   
   # Windows
   conda install -c conda-forge libsndfile
   ```

2. **import错误**
   ```bash
   # 确保项目根目录在Python路径中
   export PYTHONPATH="${PYTHONPATH}:${PWD}"
   # 或在代码中添加
   import sys
   sys.path.insert(0, '.')
   ```

3. **测试数据缺失**
   ```bash
   # 下载测试数据
   python scripts/download_test_data.py
   ```

### 性能优化建议

1. **使用numpy向量化操作**
2. **避免Python循环处理大数组**
3. **合理使用缓存机制**
4. **考虑使用numba加速计算密集型函数**

这个开发指南为项目贡献者提供了完整的开发环境配置和最佳实践指导，确保代码质量和开发效率。
