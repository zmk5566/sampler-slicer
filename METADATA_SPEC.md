# 元数据规范 (Metadata Specification)

## 概述

本文档定义了商周编铙采样器系统中所有元数据的格式规范、数据结构和存储标准。统一的元数据规范确保了系统各组件间的数据一致性和互操作性。

## 元数据层次结构

```
元数据系统
├── 项目级元数据 (Project Metadata)
├── 乐器级元数据 (Instrument Metadata)
├── 样本级元数据 (Sample Metadata)
└── 处理级元数据 (Processing Metadata)
```

## 1. 项目级元数据 (Project Metadata)

定义整个采样器项目的基本信息和配置。

### 1.1 数据结构

```json
{
  "project_info": {
    "name": "商周编铙采样器",
    "version": "1.0.0",
    "created_date": "2025-07-20T15:30:00Z",
    "last_updated": "2025-07-20T15:30:00Z",
    "description": "专为商周时期编铙设计的智能采样器",
    "author": "研究团队",
    "license": "MIT"
  },
  
  "technical_info": {
    "sample_rate": 44100,
    "bit_depth": 24,
    "audio_format": "wav",
    "processing_version": "1.0.0",
    "schema_version": "1.0"
  },
  
  "statistics": {
    "total_instruments": 1,
    "total_samples": 0,
    "total_variants": 0,
    "database_size_mb": 0.0,
    "last_backup": "2025-07-20T15:30:00Z"
  },
  
  "configuration": {
    "pitch_shift_range": {
      "min_semitones": -12,
      "max_semitones": 12
    },
    "quality_thresholds": {
      "min_snr_db": 20.0,
      "min_confidence": 0.7
    },
    "processing_settings": {
      "onset_threshold": 0.5,
      "hop_length": 512,
      "frame_length": 2048
    }
  }
}
```

### 1.2 字段规范

| 字段名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| `project_info.name` | string | ✓ | 项目名称 |
| `project_info.version` | string | ✓ | 项目版本号 (遵循语义版本) |
| `project_info.created_date` | datetime | ✓ | 项目创建时间 (ISO 8601) |
| `technical_info.sample_rate` | integer | ✓ | 音频采样率 (Hz) |
| `technical_info.bit_depth` | integer | ✓ | 音频位深度 |
| `statistics.total_samples` | integer | ✓ | 样本总数 |

## 2. 乐器级元数据 (Instrument Metadata)

定义乐器类型的特性和分类信息。

### 2.1 数据结构

```json
{
  "instrument_catalog": {
    "编铙": {
      "instrument_info": {
        "chinese_name": "编铙",
        "english_name": "Biannao",
        "category": "idiophone",
        "subcategory": "bell",
        "material": "bronze",
        "period": "商周",
        "cultural_context": "ritual_music"
      },
      
      "sound_types": [
        {
          "type_id": "zhenggu",
          "chinese_name": "正鼓音",
          "english_name": "center_strike",
          "description": "正面中心击打产生的主要音色",
          "primary_frequency_range": [100, 2000],
          "typical_pitch_range": ["C3", "C6"]
        },
        {
          "type_id": "cegu", 
          "chinese_name": "侧鼓音",
          "english_name": "side_strike",
          "description": "侧面击打产生的辅助音色",
          "primary_frequency_range": [200, 3000],
          "typical_pitch_range": ["D3", "D6"]
        }
      ],
      
      "physical_properties": {
        "size_range": {
          "diameter_cm": [8, 25],
          "height_cm": [6, 20]
        },
        "weight_range_g": [100, 2000],
        "material_composition": {
          "copper_percent": 70,
          "tin_percent": 30
        }
      },
      
      "acoustic_properties": {
        "fundamental_frequency_range": [100, 1000],
        "harmonic_structure": "inharmonic",
        "decay_time_range_s": [0.5, 3.0],
        "dynamic_range_db": 40
      },
      
      "sample_statistics": {
        "total_samples": 0,
        "samples_by_type": {
          "zhenggu": 0,
          "cegu": 0
        },
        "pitch_distribution": {},
        "dynamics_distribution": {}
      }
    }
  }
}
```

## 3. 样本级元数据 (Sample Metadata)

定义单个音频样本的详细信息和特征数据。

### 3.1 核心结构

```json
{
  "sample_id": "biannao_zhenggu_001",
  "basic_info": {
    "source_file": "samples/编铙/biannao-正鼓音.wav",
    "instrument_type": "编铙",
    "sound_type": "正鼓音",
    "recording_info": {
      "date": "2025-07-20T10:00:00Z",
      "location": "录音室A",
      "microphone": "Neumann U87",
      "distance_cm": 30,
      "room_acoustics": "干声"
    }
  },
  
  "temporal_info": {
    "onset_time": 1.234,
    "duration": 0.856,
    "end_time": 2.090,
    "original_file_duration": 10.5
  },
  
  "file_info": {
    "processed_file_path": "processed_samples/biannao_zhenggu_001.wav",
    "file_size_bytes": 123456,
    "checksum_md5": "d41d8cd98f00b204e9800998ecf8427e",
    "format": "wav",
    "channels": 1,
    "sample_rate": 44100,
    "bit_depth": 24
  }
}
```

### 3.2 音频特征 (Audio Features)

```json
{
  "audio_features": {
    "pitch": {
      "fundamental_freq": 220.5,
      "confidence": 0.85,
      "note_name": "A3",
      "octave": 3,
      "cents_deviation": -15.2,
      "detection_method": "yin",
      "harmonics": [
        {"frequency": 441.0, "amplitude": 0.8},
        {"frequency": 661.5, "amplitude": 0.6},
        {"frequency": 882.0, "amplitude": 0.4}
      ],
      "pitch_stability": 0.92,
      "frequency_trajectory": [220.1, 220.3, 220.8, 220.5, 220.2]
    },
    
    "dynamics": {
      "rms_energy": 0.234,
      "peak_amplitude": 0.789,
      "loudness_lufs": -18.5,
      "dynamic_range_db": 35.2,
      "dynamic_level": "mf",
      "attack_time_ms": 12.5,
      "decay_time_ms": 245.0,
      "sustain_level": 0.65,
      "release_time_ms": 890.0,
      "envelope": {
        "attack_shape": "exponential",
        "decay_shape": "exponential",
        "sustain_shape": "linear"
      }
    },
    
    "spectral": {
      "spectral_centroid": 1500.2,
      "spectral_rolloff": 3000.8,
      "spectral_bandwidth": 800.5,
      "spectral_flatness": 0.15,
      "spectral_contrast": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
      "zero_crossing_rate": 0.045,
      "mfcc": [1.2, -0.5, 0.8, 1.1, -0.3, 0.6, -0.2, 0.4, -0.1, 0.3, -0.15, 0.25, 0.1],
      "chroma": [0.8, 0.1, 0.05, 0.02, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.25],
      "tonnetz": [0.2, -0.1, 0.3, 0.1, -0.2, 0.4]
    },
    
    "temporal": {
      "tempo_stability": 0.0,
      "rhythmic_pattern": null,
      "onset_strength": 0.85,
      "offset_strength": 0.65,
      "transient_ratio": 0.75,
      "steady_state_ratio": 0.25
    }
  }
}
```

### 3.3 音高变体信息

```json
{
  "pitch_variants": {
    "-12": {
      "file_path": "processed_samples/biannao_zhenggu_001_-12.wav",
      "target_frequency": 110.25,
      "actual_frequency": 110.3,
      "quality_score": 0.88,
      "processing_artifacts": ["slight_formant_shift"]
    },
    "-6": {
      "file_path": "processed_samples/biannao_zhenggu_001_-6.wav",
      "target_frequency": 155.93,
      "actual_frequency": 155.95,
      "quality_score": 0.94,
      "processing_artifacts": []
    },
    "0": {
      "file_path": "processed_samples/biannao_zhenggu_001_0.wav",
      "target_frequency": 220.5,
      "actual_frequency": 220.5,
      "quality_score": 1.0,
      "processing_artifacts": []
    },
    "+6": {
      "file_path": "processed_samples/biannao_zhenggu_001_+6.wav",
      "target_frequency": 311.73,
      "actual_frequency": 311.68,
      "quality_score": 0.91,
      "processing_artifacts": []
    },
    "+12": {
      "file_path": "processed_samples/biannao_zhenggu_001_+12.wav",
      "target_frequency": 441.0,
      "actual_frequency": 440.8,
      "quality_score": 0.87,
      "processing_artifacts": ["pitch_fluctuation"]
    }
  }
}
```

## 4. 处理级元数据 (Processing Metadata)

记录音频处理过程的详细信息。

### 4.1 数据结构

```json
{
  "processing_info": {
    "processing_id": "proc_20250720_001",
    "created_date": "2025-07-20T15:30:00Z",
    "processing_version": "1.0.0",
    "total_processing_time_s": 45.2,
    
    "preprocessing_steps": [
      {
        "step_name": "onset_detection",
        "algorithm": "spectral_flux",
        "parameters": {
          "hop_length": 512,
          "threshold": 0.5,
          "pre_max": 3,
          "post_max": 3,
          "pre_avg": 3,
          "post_avg": 5,
          "delta": 0.07,
          "wait": 10
        },
        "execution_time_s": 2.1,
        "success": true,
        "detected_onsets": 15
      },
      {
        "step_name": "audio_segmentation",
        "algorithm": "onset_based",
        "parameters": {
          "padding_s": 0.1,
          "min_segment_length_s": 0.2,
          "max_segment_length_s": 5.0
        },
        "execution_time_s": 1.5,
        "success": true,
        "segments_created": 15
      },
      {
        "step_name": "pitch_analysis",
        "algorithm": "yin",
        "parameters": {
          "frame_length": 2048,
          "hop_length": 512,
          "fmin": 50.0,
          "fmax": 2000.0,
          "threshold": 0.1
        },
        "execution_time_s": 8.3,
        "success": true,
        "average_confidence": 0.82
      },
      {
        "step_name": "dynamics_analysis", 
        "algorithm": "multi_feature",
        "parameters": {
          "frame_length": 2048,
          "hop_length": 512
        },
        "execution_time_s": 3.8,
        "success": true
      },
      {
        "step_name": "pitch_shifting",
        "algorithm": "psola",
        "parameters": {
          "semitone_range": [-12, 12],
          "quality_target": 0.85
        },
        "execution_time_s": 29.5,
        "success": true,
        "variants_created": 25
      }
    ],
    
    "quality_assessment": {
      "overall_score": 0.92,
      "snr_db": 28.5,
      "thd_percent": 0.8,
      "frequency_response": "flat",
      "artifacts_detected": [],
      "quality_notes": "高质量样本，适合所有应用场景"
    },
    
    "validation": {
      "audio_integrity_check": true,
      "metadata_completeness": true,
      "feature_extraction_success": true,
      "pitch_variant_success": true,
      "database_insertion_success": true
    }
  }
}
```

## 5. 数据验证规范

### 5.1 必需字段验证

```python
REQUIRED_FIELDS = {
    'sample_metadata': [
        'sample_id',
        'basic_info.source_file',
        'basic_info.instrument_type',
        'basic_info.sound_type',
        'temporal_info.onset_time',
        'temporal_info.duration',
        'audio_features.pitch.fundamental_freq',
        'audio_features.dynamics.rms_energy'
    ],
    'project_metadata': [
        'project_info.name',
        'project_info.version',
        'technical_info.sample_rate'
    ]
}
```

### 5.2 数据类型验证

```python
FIELD_TYPES = {
    'sample_id': str,
    'audio_features.pitch.fundamental_freq': float,
    'audio_features.pitch.confidence': float,
    'temporal_info.onset_time': float,
    'temporal_info.duration': float,
    'technical_info.sample_rate': int,
    'pitch_variants': dict
}
```

### 5.3 取值范围验证

```python
VALUE_RANGES = {
    'audio_features.pitch.confidence': (0.0, 1.0),
    'audio_features.dynamics.rms_energy': (0.0, 1.0),
    'technical_info.sample_rate': (8000, 192000),
    'temporal_info.duration': (0.1, 10.0),
    'processing_info.quality_assessment.overall_score': (0.0, 1.0)
}
```

## 6. 存储格式规范

### 6.1 文件组织结构

```
metadata/
├── project.json              # 项目级元数据
├── instruments/              # 乐器级元数据
│   └── biannao.json
├── samples/                  # 样本级元数据
│   ├── biannao_zhenggu_001.json
│   ├── biannao_zhenggu_002.json
│   └── ...
├── processing/               # 处理记录
│   ├── proc_20250720_001.json
│   └── ...
└── indexes/                  # 索引文件
    ├── pitch_index.json
    ├── dynamics_index.json
    └── temporal_index.json
```

### 6.2 JSON 格式规范

- **编码**: UTF-8
- **缩进**: 2空格
- **键名**: snake_case (小写+下划线)
- **时间格式**: ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`)
- **数值精度**: 浮点数保留6位小数

### 6.3 文件命名规范

```python
# 样本元数据文件命名
sample_metadata_filename = f"{instrument_type}_{sound_type}_{sequence:03d}.json"

# 处理记录文件命名  
processing_filename = f"proc_{date}_{sequence:03d}.json"

# 示例
"biannao_zhenggu_001.json"
"proc_20250720_001.json"
```

## 7. 版本控制和兼容性

### 7.1 Schema版本管理

```json
{
  "schema_info": {
    "version": "1.0",
    "compatibility": {
      "backward_compatible_versions": ["1.0"],
      "migration_required_from": ["0.x"]
    },
    "changelog": {
      "1.0": "初始版本",
      "1.1": "增加音色特征字段",
      "2.0": "重构特征结构"
    }
  }
}
```

### 7.2 数据迁移规范

```python
class MetadataMigrator:
    def migrate_v1_to_v2(self, old_metadata: dict) -> dict:
        """从版本1.0迁移到2.0的规则"""
        pass
        
    def validate_schema_version(self, metadata: dict) -> bool:
        """验证schema版本兼容性"""
        pass
```

## 8. 索引和查询优化

### 8.1 索引定义

```json
{
  "indexes": {
    "pitch_index": {
      "field": "audio_features.pitch.fundamental_freq",
      "type": "btree",
      "bins": 100,
      "range": [50, 2000]
    },
    "dynamics_index": {
      "field": "audio_features.dynamics.rms_energy", 
      "type": "btree",
      "bins": 50,
      "range": [0.0, 1.0]
    },
    "instrument_index": {
      "field": "basic_info.instrument_type",
      "type": "hash"
    },
    "sound_type_index": {
      "field": "basic_info.sound_type",
      "type": "hash"
    }
  }
}
```

### 8.2 查询性能优化

```python
# 复合索引示例
COMPOSITE_INDEXES = [
    ["basic_info.instrument_type", "basic_info.sound_type"],
    ["audio_features.pitch.fundamental_freq", "audio_features.dynamics.rms_energy"]
]
```

## 9. 错误处理和日志

### 9.1 错误记录格式

```json
{
  "error_log": {
    "error_id": "err_20250720_001",
    "timestamp": "2025-07-20T15:30:00Z",
    "error_type": "validation_error",
    "severity": "warning",
    "context": {
      "sample_id": "biannao_zhenggu_001",
      "field": "audio_features.pitch.confidence",
      "expected_range": [0.0, 1.0],
      "actual_value": 1.2
    },
    "message": "置信度值超出有效范围",
    "resolution": "值已截断到1.0"
  }
}
```

## 10. 导出格式

### 10.1 标准导出格式

支持多种标准格式的元数据导出：

- **SFZ**: 采样器标准格式
- **EXS24**: Logic Pro采样器格式  
- **Kontakt**: Native Instruments格式
- **CSV**: 表格数据分析格式

### 10.2 导出映射示例

```python
SFZ_MAPPING = {
    'sample_id': '<group>name',
    'file_info.processed_file_path': 'sample',
    'audio_features.pitch.note_name': 'pitch_keycenter',
    'audio_features.dynamics.dynamic_level': 'amp_velcurve'
}
```

这个元数据规范为整个采样器系统提供了完整的数据标准，确保了数据的一致性、可追溯性和互操作性。
