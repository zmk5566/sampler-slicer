# 部署和插件转换指南 (Deployment & Plugin Conversion Guide)

## 概述

本文档详细说明了商周编铙采样器从Python原型到生产环境部署，以及最终转换为VST/AU音频插件的完整流程。

## 部署策略概览

### 分阶段部署方案

```
Phase 1: Python原型              Phase 2: 优化版本               Phase 3: 插件版本
┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
│ Python + librosa│              │ Python + C扩展  │              │ C++ + JUCE框架  │
│ 命令行工具      │  ────────→   │ 桌面应用程序    │  ────────→   │ VST/AU插件     │
│ 算法验证        │              │ GUI界面         │              │ DAW集成        │
└─────────────────┘              └─────────────────┘              └─────────────────┘
```

## Phase 1: Python原型部署

### 1.1 环境准备

#### 系统要求
```yaml
minimum_requirements:
  os: ["Windows 10", "macOS 10.15", "Ubuntu 18.04"]
  python: "3.8+"
  memory: "8GB"
  storage: "5GB"

recommended_requirements:
  os: ["Windows 11", "macOS 12+", "Ubuntu 20.04+"]
  python: "3.10+"
  memory: "16GB"
  storage: "20GB"
  audio_interface: "Low-latency ASIO driver"
```

#### 依赖打包
```bash
# 生成requirements.txt (锁定版本)
pip freeze > requirements-prod.txt

# 创建wheels缓存
pip wheel -r requirements-prod.txt -w wheels/

# 离线安装包
pip install --no-index --find-links wheels/ -r requirements-prod.txt
```

### 1.2 应用打包

#### 使用PyInstaller打包

```bash
# 安装PyInstaller
pip install pyinstaller

# 创建spec文件
pyi-makespec --onefile --windowed --add-data "config:config" src/cli/main.py

# 编辑spec文件 (main.spec)
```

```python
# main.spec
a = Analysis(
    ['src/cli/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config', 'config'),
        ('samples', 'samples'), 
        ('metadata', 'metadata')
    ],
    hiddenimports=[
        'librosa',
        'soundfile',
        'scipy.sparse.csgraph._validation',
        'sklearn.utils._weight_vector'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='sampler-slicer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

```bash
# 构建应用
pyinstaller main.spec

# 生成的应用在 dist/ 目录
```

#### Docker部署

```dockerfile
# Dockerfile
FROM python:3.10-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements-prod.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements-prod.txt

# 复制应用代码
COPY src/ ./src/
COPY config/ ./config/
COPY samples/ ./samples/

# 设置环境变量
ENV PYTHONPATH=/app
ENV SAMPLER_CONFIG_PATH=/app/config

# 暴露端口(如果有web接口)
EXPOSE 8000

# 启动命令
CMD ["python", "src/cli/main.py", "--server"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  sampler-slicer:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./samples:/app/samples
      - ./processed_samples:/app/processed_samples
      - ./metadata:/app/metadata
    environment:
      - LOG_LEVEL=INFO
      - CACHE_SIZE=1000
```

### 1.3 命令行界面

```python
# src/cli/main.py
import click
from pathlib import Path
from src.preprocessing import OnsetDetector, PitchAnalyzer, DynamicsAnalyzer, PitchShifter
from src.database import SampleDatabase
from src.sampler import SamplerEngine

@click.group()
@click.option('--config', default='config/default.yaml', help='配置文件路径')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
@click.pass_context
def cli(ctx, config, verbose):
    """商周编铙采样器命令行工具"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output-dir', default='processed_samples', help='输出目录')
@click.option('--threshold', default=0.5, help='onset检测阈值')
@click.pass_context
def process(ctx, input_file, output_dir, threshold):
    """处理音频文件，生成样本数据库"""
    click.echo(f"处理音频文件: {input_file}")
    
    # 初始化处理器
    detector = OnsetDetector(threshold=threshold)
    pitch_analyzer = PitchAnalyzer()
    dynamics_analyzer = DynamicsAnalyzer()
    pitch_shifter = PitchShifter()
    
    # 处理流程
    with click.progressbar(length=5, label='处理进度') as bar:
        # 1. 检测onset
        segments = detector.slice_audio(input_file)
        bar.update(1)
        
        # 2. 分析特征
        samples = []
        for segment in segments:
            pitch_info = pitch_analyzer.analyze_pitch(segment)
            dynamics_info = dynamics_analyzer.analyze_dynamics(segment)
            samples.append({
                'segment': segment,
                'pitch': pitch_info,
                'dynamics': dynamics_info
            })
        bar.update(1)
        
        # 3. 生成音高变体
        for sample in samples:
            variants = pitch_shifter.generate_pitch_variants(sample['segment'])
            sample['variants'] = variants
        bar.update(1)
        
        # 4. 存储到数据库
        db = SampleDatabase('samples.db')
        for sample in samples:
            db.add_sample(sample)
        bar.update(1)
        
        # 5. 生成报告
        click.echo(f"处理完成! 生成 {len(samples)} 个样本")
        bar.update(1)

@cli.command()
@click.option('--database', default='samples.db', help='样本数据库')
@click.option('--port', default=8000, help='服务端口')
def serve(database, port):
    """启动采样器服务"""
    click.echo(f"启动采样器服务，端口: {port}")
    
    sampler = SamplerEngine()
    sampler.load_database(database)
    
    # 启动web服务或MIDI服务
    # 实现细节...

if __name__ == '__main__':
    cli()
```

## Phase 2: 桌面应用程序

### 2.1 GUI框架选择

#### 选项对比

| 框架 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| **Tkinter** | 内置、轻量 | 界面简陋 | 简单工具 |
| **PyQt6/PySide6** | 功能强大、专业 | 体积大、复杂 | 专业应用 |
| **Kivy** | 跨平台、现代 | 学习曲线陡 | 触屏应用 |
| **Dear PyGui** | 高性能、游戏级 | 生态较新 | 实时应用 |

**推荐**: Dear PyGui (适合音频应用的实时特性)

### 2.2 Dear PyGui实现

```python
# src/gui/main_window.py
import dearpygui.dearpygui as dpg
import threading
import numpy as np
from src.sampler import SamplerEngine

class SamplerGUI:
    def __init__(self):
        self.sampler = SamplerEngine()
        self.is_playing = False
        
    def setup_ui(self):
        dpg.create_context()
        
        # 主窗口
        with dpg.window(label="商周编铙采样器", tag="main_window"):
            
            # 文件加载区域
            with dpg.group(horizontal=True):
                dpg.add_button(label="加载音频", callback=self.load_audio_callback)
                dpg.add_text("未加载文件", tag="file_status")
            
            dpg.add_separator()
            
            # 参数控制区域
            with dpg.collapsing_header(label="检测参数"):
                dpg.add_slider_float(
                    label="Onset阈值", 
                    default_value=0.5, 
                    min_value=0.1, 
                    max_value=1.0,
                    tag="onset_threshold"
                )
                dpg.add_slider_int(
                    label="Hop长度", 
                    default_value=512, 
                    min_value=256, 
                    max_value=2048,
                    tag="hop_length"
                )
            
            # 处理控制
            with dpg.group(horizontal=True):
                dpg.add_button(label="开始处理", callback=self.process_callback)
                dpg.add_button(label="停止处理", callback=self.stop_callback, enabled=False)
            
            # 进度条
            dpg.add_progress_bar(tag="progress_bar", default_value=0.0)
            
            dpg.add_separator()
            
            # 样本列表
            with dpg.collapsing_header(label="样本库", default_open=True):
                with dpg.table(header_row=True, tag="sample_table"):
                    dpg.add_table_column(label="ID")
                    dpg.add_table_column(label="音高")
                    dpg.add_table_column(label="力度")
                    dpg.add_table_column(label="时长")
                    dpg.add_table_column(label="操作")
            
            dpg.add_separator()
            
            # 播放控制
            with dpg.collapsing_header(label="播放控制", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="播放", callback=self.play_callback)
                    dpg.add_button(label="停止", callback=self.stop_playback_callback)
                
                # 虚拟键盘
                self.create_virtual_keyboard()
            
            # 状态栏
            dpg.add_text("就绪", tag="status_text")
    
    def create_virtual_keyboard(self):
        """创建虚拟MIDI键盘"""
        with dpg.group():
            dpg.add_text("虚拟键盘:")
            
            # 白键
            with dpg.group(horizontal=True):
                for note in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
                    dpg.add_button(
                        label=note, 
                        width=40, 
                        height=100,
                        callback=lambda s, a, note=note: self.play_note(note, 4)
                    )
            
            # 黑键
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=20)
                for note in ['C#', 'D#', '', 'F#', 'G#', 'A#', '']:
                    if note:
                        dpg.add_button(
                            label=note, 
                            width=30, 
                            height=60,
                            callback=lambda s, a, note=note: self.play_note(note, 4)
                        )
                    else:
                        dpg.add_spacer(width=30)
    
    def load_audio_callback(self):
        """加载音频文件对话框"""
        def file_selected(sender, app_data):
            file_path = app_data['file_path_name']
            dpg.set_value("file_status", f"已加载: {file_path}")
            self.current_file = file_path
        
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=file_selected,
            file_count=1,
            width=700,
            height=400
        ):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".wav", color=(255, 255, 0, 255))
            dpg.add_file_extension(".mp3", color=(255, 0, 255, 255))
    
    def process_callback(self):
        """开始处理音频"""
        if not hasattr(self, 'current_file'):
            dpg.set_value("status_text", "请先加载音频文件")
            return
        
        # 在新线程中处理，避免界面冻结
        threading.Thread(target=self.process_audio_thread, daemon=True).start()
    
    def process_audio_thread(self):
        """音频处理线程"""
        try:
            dpg.set_value("status_text", "正在处理...")
            dpg.configure_item("process_button", enabled=False)
            
            # 模拟处理进度
            for i in range(101):
                dpg.set_value("progress_bar", i / 100.0)
                time.sleep(0.05)  # 模拟处理时间
            
            dpg.set_value("status_text", "处理完成")
            self.update_sample_table()
            
        except Exception as e:
            dpg.set_value("status_text", f"处理失败: {e}")
        finally:
            dpg.configure_item("process_button", enabled=True)
    
    def update_sample_table(self):
        """更新样本表格"""
        # 清空现有行
        dpg.delete_item("sample_table", children_only=True)
        
        # 重新添加表头
        dpg.add_table_column(label="ID", parent="sample_table")
        dpg.add_table_column(label="音高", parent="sample_table")
        dpg.add_table_column(label="力度", parent="sample_table")
        dpg.add_table_column(label="时长", parent="sample_table")
        dpg.add_table_column(label="操作", parent="sample_table")
        
        # 添加样本数据
        samples = self.sampler.get_all_samples()
        for sample in samples:
            with dpg.table_row(parent="sample_table"):
                dpg.add_text(sample.id)
                dpg.add_text(f"{sample.pitch:.1f} Hz")
                dpg.add_text(sample.dynamics)
                dpg.add_text(f"{sample.duration:.2f}s")
                dpg.add_button(
                    label="播放", 
                    callback=lambda s, a, sid=sample.id: self.play_sample(sid)
                )
    
    def play_note(self, note: str, octave: int):
        """播放指定音符"""
        try:
            self.sampler.play_note(note, octave, velocity=80)
        except Exception as e:
            dpg.set_value("status_text", f"播放失败: {e}")
    
    def run(self):
        """运行GUI"""
        self.setup_ui()
        
        dpg.create_viewport(title="商周编铙采样器", width=800, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)
        dpg.start_dearpygui()
        dpg.destroy_context()

def main():
    app = SamplerGUI()
    app.run()

if __name__ == "__main__":
    main()
```

### 2.3 性能优化

#### Cython扩展

```python
# src/extensions/audio_processing.pyx
import numpy as np
cimport numpy as cnp
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_onset_detection(cnp.ndarray[cnp.float64_t, ndim=1] audio, 
                        double threshold, 
                        int hop_length):
    """Cython优化的onset检测"""
    cdef int n_frames = len(audio) // hop_length
    cdef cnp.ndarray[cnp.float64_t, ndim=1] onsets = np.zeros(n_frames)
    cdef int i, j
    cdef double energy_curr, energy_prev = 0.0
    
    for i in range(1, n_frames):
        energy_curr = 0.0
        for j in range(hop_length):
            energy_curr += audio[i * hop_length + j] ** 2
        
        if energy_curr - energy_prev > threshold:
            onsets[i] = 1.0
        
        energy_prev = energy_curr
    
    return onsets
```

```python
# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("src/extensions/*.pyx"),
    include_dirs=[numpy.get_include()]
)
```

## Phase 3: VST/AU插件开发

### 3.1 技术栈选择

#### JUCE框架
- **优势**: 成熟、跨平台、专业级
- **劣势**: C++学习曲线、开发周期长
- **适用**: 商业级插件开发

#### Steinberg VST SDK
- **优势**: 官方SDK、标准兼容
- **劣势**: 仅支持VST、C++only
- **适用**: VST专用开发

#### iPlug2框架
- **优势**: 开源、现代C++、多格式
- **劣势**: 社区相对较小
- **适用**: 独立开发者

**推荐**: JUCE (工业标准)

### 3.2 JUCE项目结构

```
SamplerSlicerPlugin/
├── Source/                           # 源代码
│   ├── PluginProcessor.h/.cpp       # 主处理器
│   ├── PluginEditor.h/.cpp          # GUI编辑器
│   ├── AudioEngine/                 # 音频引擎
│   │   ├── SampleDatabase.h/.cpp
│   │   ├── OnsetDetector.h/.cpp
│   │   └── PitchAnalyzer.h/.cpp
│   └── Utils/                       # 工具类
│       └── AudioUtils.h/.cpp
├── JuceLibraryCode/                 # JUCE库代码
├── Builds/                          # 构建文件
│   ├── VisualStudio2022/           # Windows
│   ├── Xcode/                      # macOS
│   └── LinuxMakefile/              # Linux
└── SamplerSlicer.jucer             # 项目配置
```

### 3.3 核心插件代码

#### 插件处理器

```cpp
// Source/PluginProcessor.h
#pragma once
#include <JuceHeader.h>
#include "AudioEngine/SampleDatabase.h"
#include "AudioEngine/SamplerEngine.h"

class SamplerSlicerAudioProcessor : public juce::AudioProcessor
{
public:
    SamplerSlicerAudioProcessor();
    ~SamplerSlicerAudioProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

private:
    std::unique_ptr<SamplerEngine> samplerEngine;
    std::unique_ptr<SampleDatabase> sampleDatabase;
    
    // 参数管理
    juce::AudioProcessorValueTreeState parameters;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SamplerSlicerAudioProcessor)
};
```

```cpp
// Source/PluginProcessor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"

SamplerSlicerAudioProcessor::SamplerSlicerAudioProcessor()
     : AudioProcessor (BusesProperties()
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)),
       parameters (*this, nullptr, juce::Identifier ("SamplerSlicer"),
                   createParameterLayout())
{
    samplerEngine = std::make_unique<SamplerEngine>();
    sampleDatabase = std::make_unique<SampleDatabase>();
}

void SamplerSlicerAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    samplerEngine->prepareToPlay (sampleRate, samplesPerBlock);
}

void SamplerSlicerAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, 
                                               juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    
    // 清空输出缓冲区
    buffer.clear();
    
    // 处理MIDI消息
    for (const auto metadata : midiMessages)
    {
        auto message = metadata.getMessage();
        
        if (message.isNoteOn())
        {
            int noteNumber = message.getNoteNumber();
            int velocity = message.getVelocity();
            
            // 根据MIDI音符触发样本播放
            samplerEngine->playNote (noteNumber, velocity);
        }
        else if (message.isNoteOff())
        {
            int noteNumber = message.getNoteNumber();
            samplerEngine->stopNote (noteNumber);
        }
    }
    
    // 渲染音频
    samplerEngine->renderNextBlock (buffer, 0, buffer.getNumSamples());
}

juce::AudioProcessorValueTreeState::ParameterLayout 
SamplerSlicerAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    
    // 添加参数
    params.push_back (std::make_unique<juce::AudioParameterFloat> (
        "threshold", "Onset Threshold", 0.0f, 1.0f, 0.5f));
    
    params.push_back (std::make_unique<juce::AudioParameterChoice> (
        "mode", "Playback Mode", 
        juce::StringArray { "Similarity", "Random" }, 0));
    
    return { params.begin(), params.end() };
}
```

#### 插件编辑器

```cpp
// Source/PluginEditor.h
#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

class SamplerSlicerAudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    SamplerSlicerAudioProcessorEditor (SamplerSlicerAudioProcessor&);
    ~SamplerSlicerAudioProcessorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    SamplerSlicerAudioProcessor& audioProcessor;
    
    // UI组件
    juce::Slider thresholdSlider;
    juce::Label thresholdLabel;
    juce::ComboBox modeComboBox;
    juce::Label modeLabel;
    juce::TextButton loadButton;
    juce::TextButton processButton;
    
    // 样本列表
    juce::ListBox sampleListBox;
    
    // 虚拟键盘
    juce::MidiKeyboardComponent midiKeyboard;
    juce::MidiKeyboardState keyboardState;
    
    // 参数附件
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> thresholdAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> modeAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SamplerSlicerAudioProcessorEditor)
};
```

### 3.4 音频引擎移植

#### 算法移植策略

```cpp
// AudioEngine/OnsetDetector.h
#pragma once
#include <JuceHeader.h>
#include <vector>

class OnsetDetector
{
public:
    OnsetDetector();
    ~OnsetDetector();
    
    void setParameters (float threshold, int hopLength);
    std::vector<double> detectOnsets (const juce::AudioBuffer<float>& buffer, 
                                    double sampleRate);

private:
    float threshold;
    int hopLength;
    
    // FFT相关
    std::unique_ptr<juce::dsp::FFT> fft;
    juce::AudioBuffer<float> fftBuffer;
    std::vector<float> magnitudeSpectrum;
    std::vector<float> previousSpectrum;
    
    void computeSpectralFlux (const std::vector<float>& current,
                            const std::vector<float>& previous,
                            std::vector<float>& flux);
};
```

```cpp
// AudioEngine/OnsetDetector.cpp
#include "OnsetDetector.h"

OnsetDetector::OnsetDetector()
    : threshold (0.5f), hopLength (512)
{
    fft = std::make_unique<juce::dsp::FFT>(10); // 1024 point FFT
    fftBuffer.setSize (1, 1024);
}

std::vector<double> OnsetDetector::detectOnsets (const juce::AudioBuffer<float>& buffer, 
                                                double sampleRate)
{
    std::vector<double> onsetTimes;
    
    const int numSamples = buffer.getNumSamples();
    const float* audioData = buffer.getReadPointer (0);
    
    // 按帧处理
    for (int pos = 0; pos < numSamples - 1024; pos += hopLength)
    {
        // 复制音频数据到FFT缓冲区
        for (int i = 0; i < 1024; ++i)
        {
            fftBuffer.setSample (0, i, audioData[pos + i]);
        }
        
        // 执行FFT
        fft->performFrequencyOnlyForwardTransform (fftBuffer.getWritePointer (0));
        
        // 计算幅度谱
        magnitudeSpectrum.resize (512);
        for (int i = 0; i < 512; ++i)
        {
            auto real = fftBuffer.getSample (0, i * 2);
            auto imag = fftBuffer.getSample (0, i * 2 + 1);
            magnitudeSpectrum[i] = std::sqrt (real * real + imag * imag);
        }
        
        // 计算spectral flux
        if (!previousSpectrum.empty())
        {
            float flux = 0.0f;
            for (int i = 0; i < 512; ++i)
            {
                float diff = magnitudeSpectrum[i] - previousSpectrum[i];
                if (diff > 0.0f)
                    flux += diff;
            }
            
            // 检测peak
            if (flux > threshold)
            {
                double onsetTime = pos / sampleRate;
                onsetTimes.push_back (onsetTime);
            }
        }
        
        previousSpectrum = magnitudeSpectrum;
    }
    
    return onsetTimes;
}
```

### 3.5 构建和分发

#### CMake配置

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(SamplerSlicerPlugin VERSION 1.0.0)

# 添加JUCE
add_sub
