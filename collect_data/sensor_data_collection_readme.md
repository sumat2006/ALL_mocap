# Sensor Data Collection & Labeling System

A comprehensive sensor data collection system for sign language gesture recognition using ESP32S3 devices with flex sensors and IMU, paired with a Python GUI application for real-time data visualization and labeling.

## 🎯 System Overview

This system consists of three main components:
1. **Python GUI Application** - Data collection, labeling, and video recording interface
2. **ESP32S3 Master Device (Imortal Joe)** - Main hand sensor unit with ESP-NOW communication
3. **ESP32S3 Slave Device (Warboy)** - Secondary hand sensor unit

## 🏗️ Hardware Architecture

### Components
- **2x ESP32S3S3 Development Boards**
- **4x ADS1015 ADC Modules** (2 per ESP32S3)
- **2x QMI8658 IMU Sensors** (1 per ESP32S3)
- **10x Flex Sensors** (5 per hand)
- **USB Camera** for video recording
- **Serial Communication** via USB

### Sensor Configuration
Each ESP32S3 device reads:
- **5 Flex Sensors** via ADS1015 ADC modules
- **6-axis IMU Data** (accelerometer + gyroscope) from QMI8658
- **Calculated Angles** (X, Y, Z orientations)

### Data Output Format
```
timestamp_ms,ax_slave,ay_slave,az_slave,gx_slave,gy_slave,gz_slave,angle_x_slave,angle_y_slave,angle_z_slave,flex_slave_0,flex_slave_1,flex_slave_2,flex_slave_3,flex_slave_4,ax_master,ay_master,az_master,gx_master,gy_master,gz_master,angle_x_master,angle_y_master,angle_z_master,flex_master_0,flex_master_1,flex_master_2,flex_master_3,flex_master_4,[Label]['gesture_name']
```

## 📁 Project Structure

```
collect_data/
├── collect_app/app/
│   └── read_serial_data_V2_1.py    # Main GUI application
├── codeFlexHardWare/
│   ├── Immaltol_joe_main_hand/src/
│   │   └── main.cpp                # Master ESP32S3 code
│   └── warboy_sub_hand/src/
│       └── main.cpp                # Slave ESP32S3 code
└── out_data/                       # Output directory for collected data
    └── YYYY-MM-DD/
        ├── data/                   # CSV sensor data files
        └── video/                  # MP4 video recordings
```

## 📋 System Requirements & Dependencies

### Software Requirements

**Python Environment:**
- Python 3.7+ (recommended: Python 3.8 or higher)
- Operating System: Windows 10/11, macOS, or Linux

**Python Dependencies:**
- `tkinter` - GUI framework (pre-installed with Python)
- `opencv-python` (cv2) - Video capture and image processing
- `pillow` (PIL) - Image manipulation and font handling
- `pyserial` - Serial communication with ESP32S3
- `numpy` - Numerical operations and array handling

**Arduino Development:**
- Arduino IDE 2.0+ or PlatformIO
- ESP32S3 Board Package v2.0+
- USB drivers for ESP32S3 (CP210x or CH340)

### Hardware Requirements

**Essential Components:**
- 2x ESP32S3S3 Development Boards
- 4x ADS1015 12-bit ADC modules (2 per ESP32S3)
- 2x QMI8658 6-axis IMU sensors
- 10x Flex sensors (resistive type)
- USB Camera (720p or higher recommended)
- USB cables for ESP32S3 programming and communication
- Breadboards and jumper wires for prototyping

**Power Requirements:**
- 5V power supply for ESP32S3 boards (USB or external)
- Stable power recommended for accurate sensor readings

## 🚀 Quick Start Guide

### 1. Hardware Setup

**Wiring Configuration:**
```
ESP32S3 Pin Connections:
- SDA: GPIO 15
- SCL: GPIO 14
- IRQ: GPIO 4

ADS1015 Addresses:
- ADS1015_1: 0x48
- ADS1015_2: 0x49

Flex Sensor Mapping:
- Sensor 0: ADS1015_2, Channel 0
- Sensor 1: ADS1015_1, Channel 3
- Sensor 2: ADS1015_1, Channel 2
- Sensor 3: ADS1015_1, Channel 1
- Sensor 4: ADS1015_1, Channel 0
```

### 2. ESP32S3 Programming

**Arduino IDE Setup:**
1. Install ESP32S3 board package in Arduino IDE
2. Select "ESP32S3S3 Dev Module" as board type
3. Install required Arduino libraries

**Required Arduino Libraries:**
```cpp
// Core ESP32S3 libraries
#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

// External libraries (install via Library Manager)
#include <ArduinoJson.h>        // JSON parsing and serialization
#include <Adafruit_ADS1X15.h>   // ADS1015/ADS1115 ADC driver
#include <QMI8658.h>           // QMI8658 IMU sensor driver
```

**Library Installation:**
1. **ArduinoJson** by Benoit Blanchon - Install via Library Manager
2. **Adafruit ADS1X15** by Adafruit - Install via Library Manager  
3. **QMI8658** - May need manual installation from manufacturer

**Flash the firmware to both ESP32S3 devices:**
1. Master device: `Immaltol_joe_main_hand/src/main.cpp`
2. Slave device: `warboy_sub_hand/src/main.cpp`

**Update MAC addresses in the code:**
```cpp
// In master code - update slave address
uint8_t slaveAddress[] = {0xfc, 0x01, 0x2c, 0xd9, 0x3d, 0x90};

// In slave code - update master address  
uint8_t masterAddress[] = {0xfc, 0x01, 0x2c, 0xd9, 0x35, 0x0c};
```

### 3. Python Application Setup

**Install Required Libraries:**
```bash
# Install all dependencies at once
pip install opencv-python pillow pyserial numpy

# Or install individually:
pip install opencv-python    # For video capture and image processing
pip install pillow          # For image manipulation and fonts
pip install pyserial        # For ESP32S3 serial communication
pip install numpy           # For numerical operations
```

**Alternative Installation Methods:**
```bash
# Using conda
conda install opencv pillow pyserial numpy

# Using pipenv
pipenv install opencv-python pillow pyserial numpy

# From requirements.txt (create this file with the dependencies)
pip install -r requirements.txt
```

**Font Installation (for Thai language support):**
- Download `tahoma.ttf` font file
- Place in the same directory as `read_serial_data_V2_1.py`
- Or update `THAI_FONT_PATH` variable in the code to point to your font location

**Run the Application:**
```bash
python read_serial_data_V2_1.py
```

**Verify Installation:**
```python
# Run this to check if all libraries are properly installed
python -c "import tkinter, cv2, serial, numpy, PIL; print('All libraries installed successfully!')"
```

## 🎛️ GUI Application Features

### Connection Management
- **Port Detection**: Automatic serial port scanning
- **Connection Status**: Real-time connection monitoring
- **Device Communication**: Bi-directional serial communication

### Data Collection
- **Real-time Sensor Visualization**: Live sensor data display
- **Video Recording**: Synchronized camera feed recording
- **Timestamp Synchronization**: Precise timing alignment
- **Session Management**: Organized data storage by date

### Labeling System
- **Custom Label Sets**: User-defined gesture categories
- **Group Labeling**: Multi-gesture sequences
- **Random Mode**: Randomized gesture prompts
- **Auto-Advance**: Automated label progression
- **Break Management**: Configurable rest periods between gestures

### Advanced Features
- **Error Handling**: Redo functionality for incorrect gestures
- **Cooldown Periods**: Prevention of rapid label changes
- **Thai Font Support**: Multi-language gesture names
- **Keyboard Shortcuts**: 
  - `a`: Toggle recording
  - `s`: Start/stop labeling
  - `d`: Flag error and redo

## 📊 Data Collection Workflow

### Standard Collection Process
1. **Connect Hardware**: Power on ESP32S3 devices and connect master via USB
2. **Calibrate Sensors**: Run `flex_cal` command for accurate readings
3. **Configure Labels**: Set up gesture categories and grouping
4. **Start Recording**: Begin data collection session
5. **Label Gestures**: Use keyboard shortcuts or GUI controls
6. **Review Data**: Check collected samples and video recordings

### Calibration Procedure
**Flex Sensor Calibration:**
1. Type `flex_cal` in serial command field
2. Follow on-screen instructions:
   - **Phase 1**: Hold hand in relaxed/straight position
   - **Phase 2**: Make a tight fist/bent position
3. System automatically calculates calibration values
4. Calibration applies to both master and slave devices

## ⚙️ Configuration Options

### Label Configuration
```python
SELECTABLE_LABELS_LIST = [
    "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", 
    "หก", "เจ็ด", "แปด", "เก้า", "สิบ"
]
```

### System Settings
```python
# Communication
BAUD_RATE = 9600
SERIAL_TIMEOUT = 1
BOARD_RESET_DELAY_MS = 2000

# Video Recording
VIDEO_FPS = 20.0

# Labeling
REDO_COOLDOWN_S = 10
DEFAULT_UNLABELED_VALUE = "nothing"
```

## 🔧 Hardware Commands

### Available Serial Commands
- `help` - Display available commands
- `flex_cal` - Start flex sensor calibration on both devices
- `reset` - Restart ESP32S3 devices
- `reset_time` - Reset CSV timestamp baseline

### ESP-NOW Communication
The system uses ESP-NOW for wireless communication between ESP32S3 devices:
- **Master Device**: Collects data from both devices and sends to PC
- **Slave Device**: Sends sensor data to master wirelessly
- **Automatic Reconnection**: Handles connection drops gracefully
- **Status Monitoring**: Real-time connection status display

## 📈 Data Output

### File Structure
```
out_data/2024-01-15/
├── data/
│   └── 20240115_143022_session_sensor.txt
└── video/
    └── 20240115_143022_session.mp4
```

### Data Format
Each line contains 30 sensor values plus label:
- **Timestamp**: Milliseconds since recording start
- **Slave Data**: 9 IMU values + 5 flex sensor values
- **Master Data**: 9 IMU values + 5 flex sensor values
- **Label**: Current gesture being performed

## 🛠️ Troubleshooting

### Common Issues

**ESP32S3 Connection Problems:**
- Verify correct COM port selection
- Check USB cable and power supply
- Ensure proper baud rate (9600)
- Try pressing reset button on ESP32S3

**Sensor Calibration Issues:**
- Ensure stable hand positioning during calibration
- Complete full calibration sequence for both phases
- Verify sensor wiring connections
- Check ADS1015 I2C addresses

**ESP-NOW Communication Problems:**
- Verify MAC addresses in both ESP32S3 codes
- Ensure both devices are on same WiFi channel
- Check for interference from other WiFi devices
- Power cycle both ESP32S3 devices

**Video Recording Issues:**
- Verify camera is properly connected
- Check camera permissions in system settings
- Ensure sufficient disk space for recordings
- Try different camera index if multiple cameras present

### Performance Tips
- Use USB 3.0 ports for stable data transfer
- Ensure adequate power supply for ESP32S3 devices
- Close other serial applications before connecting
- Monitor system resources during long recording sessions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch for hardware or software components
3. Test with actual hardware setup
4. Submit pull request with detailed description

## 📄 License

This project is for research and educational purposes. Please ensure compliance with local regulations regarding data collection.

## 🙏 Acknowledgments

- Built for SKB Co. data collection project
- ESP32S3 development using Arduino framework
- GUI application using Python tkinter
- Thai language support for gesture recognition

