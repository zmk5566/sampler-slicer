# 商周编铙采样器 (Sampler-Slicer)

一个专为中国古代打击乐器设计的智能采样器，专注于商周时期编铙音色的采集、分析和重现。

## 项目概述

本项目旨在构建一个完整的采样器系统，能够：

- 🎵 **自动音频切割**: 使用onset detection技术自动识别和分割音频中的单个打击声音
- 🏷️ **智能标注**: 自动分析音高、力度和音色特征
- 🎹 **智能播放**: 支持相似度匹配和随机模式的采样播放
- 🔄 **音高变换**: 为每个样本生成-12到+12半音的所有变体
- 🎚️ **实时控制**: 提供音序器功能，支持MIDI控制

## 技术特色

- **MIR算法**: 集成音乐信息检索技术进行音频分析
- **Python生态**: 基于librosa、numpy等成熟音频处理库
- **元数据驱动**: 完整的样本特征数据库
- **插件兼容**: 未来支持VST/AU插件格式

## 快速开始

### 环境要求

- Python 3.8+
- librosa >= 0.9.0
- numpy >= 1.21.0
- soundfile >= 0.10.0
- scipy >= 1.7.0

### 安装

```bash
# 克隆项目
git clone https://github.com/zmk5566/sampler-slicer.git
cd sampler-slicer

# 安装依赖
pip install -r requirements.txt
```

### 基本使用

```python
from src.preprocessing import OnsetDetector, PitchAnalyzer
from src.sampler import SamplerEngine

# 1. 音频预处理
detector = OnsetDetector()
samples = detector.slice_audio("samples/编铙/biannao-正鼓音.wav")

# 2. 特征分析
analyzer = PitchAnalyzer()
metadata = analyzer.analyze_samples(samples)

# 3. 创建采样器
sampler = SamplerEngine()
sampler.load_samples(metadata)

# 4. 播放样本
sampler.play_note(pitch=220, velocity=80, mode="similarity")
```

## 项目结构

```
sampler-slicer/
├── src/                    # 源代码
│   ├── preprocessing/      # 音频预处理模块
│   ├── database/          # 数据库管理
│   └── sampler/           # 采样器引擎
├── samples/               # 原始音频样本
├── processed_samples/     # 处理后的样本
├── metadata/             # 元数据存储
├── docs/                 # 文档
└── tests/                # 测试用例
```

## 支持的乐器类型

- **编铙** (Biannao): 商周时期青铜编钟类乐器
  - 正鼓音: 主要敲击音色
  - 侧鼓音: 侧面敲击音色

## 文档

- [系统架构](ARCHITECTURE.md) - 详细的系统设计说明
- [API文档](API.md) - 模块接口说明
- [元数据规范](METADATA_SPEC.md) - 数据格式定义
- [开发指南](DEVELOPMENT.md) - 开发环境配置和贡献指南
- [部署指南](DEPLOYMENT.md) - 插件打包和部署说明

## 开发状态

- [x] 项目架构设计
- [x] 文档编写
- [ ] 音频预处理模块
- [ ] 特征分析算法
- [ ] 采样器引擎
- [ ] 用户界面
- [ ] VST/AU插件

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
