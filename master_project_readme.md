# Sign Language Detection System

A complete end-to-end system for collecting, processing, and classifying sign language gestures using sensor data and machine learning. This project combines hardware sensor collection, deep learning classification, and web deployment into a unified solution.

## System Architecture

This system consists of three main components working together to enable sign language recognition:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Data          │    │   ML Model       │    │   Web Application   │
│   Collection    │───▶│   Training       │───▶│   Deployment        │
│                 │    │                  │    │                     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## Project Components

### 📊 [Data Collection System](collect_data/sensor_data_collection_readme.md)

**Location**: `collect_data/`

A comprehensive sensor data collection system using ESP32S3 devices with flex sensors and IMU units.

**Key Features**:
- Dual ESP32S3 setup with ESP-NOW wireless communication
- Real-time sensor data visualization
- labeling system with Thai language support
- Synchronized video recording
- Auto-advance and random mode data collection

**Hardware Components**:
- 2x ESP32S3 development boards
- 4x ADS1015 ADC modules  
- 2x QMI8658 IMU sensors
- 10x flex sensors (5 per hand)
- USB camera for video capture

**Quick Start**:
```bash
cd collect_data/collect_app/app/
python read_serial_data_V2_1.py
```

---

### 🤖 [Machine Learning Model](model/README.md)

**Location**: `model/`

CNN-based time series classifier for recognizing sign language gestures from sensor data.

**Key Features**:
- Custom CNN architecture optimized for time series
- Advanced data preprocessing and augmentation
- Multiple training schedulers and early stopping
- Comprehensive evaluation with visualization
- Model persistence and loading utilities

**Architecture**:
- Input: Time series data (50 × 28 features)
- CNN layers with batch normalization
- Dense layers with dropout regularization
- Multi-class classification output

**Performance**:
- Current accuracy: ~45% on real-world data
- Training time: ~3 minutes on GPU GTX1650
- Model size: Lightweight for deployment

**Quick Start**:
```bash
cd model/
python train.py  # Train new model
python load_model.py  # Load and test existing model
```

---

### 🌐 [Web Application](fastapi_docker_webapp/README.md)

**Location**: `fastapi_docker_webapp/`

Multi-modal FastAPI web application integrating sign language detection with ASR, LLM, and TTS capabilities.

**Key Features**:
- Sign language gesture prediction via trained CNN model
- Multi-modal AI integration (ASR, LLM, TTS)
- Docker containerization for easy deployment
- Interactive web interface with real-time predictions
- Comprehensive logging system with web viewer

**Integrated AI Models**:
- **ASR**: Thonburain Whisper (Thai speech recognition)
- **LLM**: KhanommThan 1B (Thai language model)  
- **TTS**: Vachanatts (Thai text-to-speech)
- **Hand-Sign**: Custom CNN time series classifier

**Project Structure**:
```
app/
├── asset/
│   ├── model/              # All AI models storage
│   ├── config.yaml         # Model paths configuration
│   ├── utils.py           # Utility functions
│   └── rollback.json      # Prediction to text mapping
└── static/
    ├── index.html         # Interactive web interface
    ├── logs.html          # Log viewer template
    ├── script.js          # JavaScript functionality
    └── styles.css         # Web interface styling
```

**Quick Start**:
```bash
cd fastapi_docker_webapp/
docker compose up --build
# Available at http://localhost:8000/
```

**Alternative - Direct Uvicorn**:
```bash
pip install uvicorn
uvicorn main:app --host 0.0.0.0 --port 80 --reload
```

**API Endpoints**:
- `GET /` - Interactive prediction interface
- `POST /predict` - JSON API for predictions `{"text": "..."}`
- `POST /predict-form` - Form POST endpoint
- `GET /logs` - Web-based log viewer (last 200 lines)

## Data Flow Pipeline

### 1. Data Collection Phase
```
ESP32 Sensors → Serial Communication → Python GUI → Labeled CSV + Video
```
- Collect sensor data using dual ESP32 setup
- Label gestures in real-time using Python application
- Export synchronized sensor data and video recordings

### 2. Model Training Phase  
```
CSV Data → Preprocessing → CNN Training → Model Evaluation → Saved Model
```
- Load and preprocess collected sensor data
- Train CNN time series classifier
- Evaluate performance and generate reports
- Save trained model for deployment

### 3. Deployment Phase
```
Saved Model → FastAPI → Docker Container → Web Service → Predictions
```
- Load trained model into FastAPI application
- Containerize with Docker for deployment
- Serve predictions via REST API
- Provide web interface for testing

## Quick Setup Guide

### Prerequisites
- Python 3.8+
- Arduino IDE or PlatformIO
- Docker (for web deployment)
- ESP32 development boards
- Required sensors and components

### Complete System Setup

1. **Hardware Setup**:
   ```bash
   # Flash ESP32 firmware
   # Connect sensors according to wiring diagram
   # See collect_data/sensor_data_collection_readme.md
   ```

2. **Data Collection**:
   ```bash
   cd collect_data/collect_app/app/
   pip install -r requirements.txt
   python read_serial_data_V2_1.py
   ```

3. **Model Training**:
   ```bash
   cd model/
   pip install -r requirements.txt
   python train.py
   ```

4. **Web Deployment**:
   ```bash
   cd fastapi_docker_webapp/
   docker-compose up --build
   ```

## Repository Structure

```
sign-language-detection/
├── collect_data/                    # Data collection system
│   ├── collect_app/app/            # Python GUI application
│   ├── codeFlexHardWare/           # ESP32 firmware
│   └── sensor_data_collection_readme.md
├── model/                          # Machine learning components
│   ├── model.py                    # CNN architecture
│   ├── train.py                    # Training script
│   ├── load_model.py              # Model loading utilities
│   └── README.md
├── fastapi_docker_webapp/          # Web application
│   ├── app/                       # FastAPI application
│   ├── Dockerfile                 # Container configuration
│   ├── docker-compose.yml         # Multi-service setup
│   └── README.md
└── README.md                       # This file
```

## Development Workflow

### Adding New Gestures
1. Update label list in data collection GUI
2. Collect training data for new gestures
3. Retrain model with expanded dataset
4. Update web application model

### Improving Model Performance
1. Collect more diverse training data
2. Experiment with different CNN architectures
3. Implement data augmentation techniques
4. Fine-tune hyperparameters

### Scaling Deployment
1. Optimize model size and inference speed
2. Implement model versioning
3. Add monitoring and logging
4. Deploy to cloud infrastructure

## Performance Metrics

### Current System Performance
- **Data Collection Rate**: 20 Hz sensor sampling
- **Model Accuracy**: 60-70% on real-world data (drops from 89% controlled to ~43% production)
- **Inference Time**: <100ms per prediction
- **System Latency**: End-to-end <500ms

### Multi-Modal AI Performance
- **ASR (Thonburain Whisper)**: Limited by compute resources
- **LLM (KhanommThan 1B)**: Speed constrained by GPU availability
- **TTS (Vachanatts)**: Performance depends on compute unit
- **Hand-Sign Detection**: Custom CNN model accuracy varies with real-world conditions

### Known Limitations
- **Accuracy Drop**: Significant performance degradation from controlled (89%) to production (43%) environments
- **Data Quality**: Training data not representative of real-world usage scenarios
- **Compute Resources**: AI model speeds limited by insufficient GPU power
- **Sensor Positioning**: Hand-sign detection sensitive to device placement and user variation
- **Environmental Factors**: Real-world conditions affect all model components

### Critical Performance Notes
> **Accuracy Dependencies**: Model performance heavily depends on data type, model architecture, and model size. Production accuracy can drop significantly when real-world conditions differ from training data.

> **Speed Dependencies**: Processing speed is constrained by available compute units (GPU). Language model, TTS, and ASR components require substantial computational resources for optimal performance.

> **Development Recommendations**: For continued development, prioritize GPU upgrades, comprehensive data collection plans, and real-world testing scenarios. These infrastructure improvements will enable easier project continuation and better performance.

## Contributing

1. Fork the repository
2. Create feature branch for specific component
3. Follow component-specific development guidelines
4. Test with complete system integration
5. Submit pull request with detailed description

## Troubleshooting

### Common Issues
- **Hardware Connection**: Check ESP32 wiring and power supply
- **Model Performance**: Increase training data diversity
- **Web Deployment**: Verify Docker installation and port availability
- **Integration**: Ensure model format compatibility between components

### Support Resources
- Component-specific README files
- Hardware wiring diagrams
- Model architecture documentation
- API endpoint specifications

## License

This project is for research and educational purposes. Please ensure compliance with local regulations regarding data collection and deployment.

## Future Roadmap

### Immediate Priorities (Infrastructure)
- [ ] **GPU Hardware Upgrade**: Deploy dedicated GPU resources for AI model acceleration
- [ ] **Data Collection Enhancement**: Expand real-world training datasets to bridge accuracy gap
- [ ] **Environmental Testing**: Comprehensive evaluation across diverse usage conditions
- [ ] **Model Optimization**: Architecture improvements for production deployment efficiency


### Development Prerequisites
> **Critical Requirements**: Before continuing development, address hardware limitations (GPU), create comprehensive data collection plans, and establish real-world testing protocols. These foundational improvements will significantly ease future development and improve system reliability.

---
