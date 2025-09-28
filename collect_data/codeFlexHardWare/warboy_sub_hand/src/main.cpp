// RECEIVER CODE (e.g., for "warboy") - FINAL WITH LOG FORWARDING
#include <Arduino.h>
#include <Wire.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <WiFi.h>
#include <ArduinoJson.h>
#include <Adafruit_ADS1X15.h>
#include <QMI8658.h>
#include <stdarg.h> // Required for variadic function

// ──── PIN DEFINITIONS ───────────────────────────────────────────────────────────
#ifndef SENSOR_SCL
#define SENSOR_SCL  14
#endif
#ifndef SENSOR_SDA
#define SENSOR_SDA  15
#endif
#ifndef SENSOR_IRQ
#define SENSOR_IRQ  4
#endif

// ──── PERIPHERAL ADDRESSES & CONSTANTS ─────────────────────────────────────────
const uint8_t ADS1015_1_ADDRESS = 0x48;
const uint8_t ADS1015_2_ADDRESS = 0x49;
const int NUM_FLEX_SENSORS = 5;
const int CALIBRATION_SAMPLES = 1000;
#define WIFI_CHANNEL 1

// ──── PERIPHERAL OBJECTS ───────────────────────────────────────────────────────
Adafruit_ADS1015 ads1015_1;
Adafruit_ADS1015 ads1015_2;
QMI8658          imu;

// ──── CALIBRATION & SENSOR VARIABLES ───────────────────────────────────────────
float mean_acd_max[NUM_FLEX_SENSORS] = {1800.0f, 1800.0f, 1800.0f, 1800.0f, 1800.0f};
float mean_acd_low[NUM_FLEX_SENSORS] = {900.0f, 900.0f, 900.0f, 900.0f, 900.0f};
float low_value = 0.0f;
float high_value = 1000.0f;
float flex_raw_value[NUM_FLEX_SENSORS];
float flex_cal[NUM_FLEX_SENSORS];

// ──── PROTOTYPES ────────────────────────────────────────────────────────────────
void flexCalibration();
void readAllFlexSensors(float* raw_values);
void ADS1015_INITIALIZATION();
void QMI8658_INITIALIZATION();
void checkCommand();
void LogToMaster(const char* format, ...);

// ──── ESP-NOW & COMMUNICATION ───────────────────────────────────────────────────
uint8_t masterAddress[] = {0xfc, 0x01, 0x2c, 0xd9, 0x35, 0x0c};
String payload;
JsonDocument jsonDoc;

struct CommandData {
  volatile int imotan_command = 0;
};
CommandData incomingCommand;

// This function prints a message locally AND sends it to the master.
void LogToMaster(const char* format, ...) {
    char buffer[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    // 1. Print locally for direct debugging
    Serial.print(buffer);

    // 2. Send to master
    JsonDocument logDoc;
    String logPayload;
    logDoc["log_msg"] = buffer;
    serializeJson(logDoc, logPayload);
    esp_now_send(masterAddress, (uint8_t*)logPayload.c_str(), logPayload.length());
}

void OnDataSent(const uint8_t* mac_addr, esp_now_send_status_t status) {
    if (status != ESP_NOW_SEND_SUCCESS) {
        // Avoid spamming logs on failure
    }
}

void RegisterPeer() {
    esp_now_peer_info_t peerInfo;
    memset(&peerInfo, 0, sizeof(peerInfo));
    memcpy(peerInfo.peer_addr, masterAddress, 6);
    peerInfo.channel = WIFI_CHANNEL;
    peerInfo.encrypt = false;
    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        LogToMaster("Failed to add peer\n");
    }
}

void OnMessageReceived(const uint8_t* mac_addr, const uint8_t* incomingData, int len) {
    JsonDocument doc;
    if (deserializeJson(doc, incomingData, len) != DeserializationError::Ok) {
        return;
    }
    incomingCommand.imotan_command = doc["imotan_command"] | 0;
}

void InitEspNow() {
    if (esp_now_init() != ESP_OK) {
        LogToMaster("Error initializing ESP-NOW\n");
        return;
    }
    esp_now_register_recv_cb(OnMessageReceived);
    esp_now_register_send_cb(OnDataSent);
    RegisterPeer();
}

void SendSensorData() {
    serializeJson(jsonDoc, payload);
    esp_now_send(masterAddress, (uint8_t*)payload.c_str(), payload.length());
    payload.clear();
    jsonDoc.clear();
}

void checkCommand() {
    if (incomingCommand.imotan_command != 0) {
        LogToMaster("[Warboy] Received command code: %d\n", incomingCommand.imotan_command);
        switch (incomingCommand.imotan_command) {
            case 1:
                flexCalibration();
                break;
            case 3:
                // LogToMaster("[Warboy] Rebooting on command...\n");
                // delay(100);
                // ESP.restart();
                break;
            default:
                LogToMaster("[Warboy] Unknown command code: %d\n", incomingCommand.imotan_command);
                break;
        }
        incomingCommand.imotan_command = 0;
    }
}

void setup() {
    Wire.begin(SENSOR_SDA, SENSOR_SCL);
    Serial.begin(115200);
    while (!Serial) { delay(10); }

    WiFi.mode(WIFI_AP_STA);
    if (esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE) != ESP_OK) {
        Serial.println("Error setting WiFi channel"); // Cannot use LogToMaster yet
        return;
    }
    
    // Init ESP-NOW early so we can send logs during the rest of setup
    InitEspNow(); 
    LogToMaster("\nWarboy ESP32 starting...\n");

    ADS1015_INITIALIZATION();
    QMI8658_INITIALIZATION();
    
    LogToMaster("[Warboy] Initialization complete. Ready for commands and sending data.\n");
}

void loop() {
    checkCommand();
    readAllFlexSensors(flex_raw_value);

    for (int i = 0; i < NUM_FLEX_SENSORS; i++) {
        if (mean_acd_max[i] != mean_acd_low[i]) {
            flex_cal[i] = ((flex_raw_value[i] - mean_acd_low[i]) * (high_value - low_value) / (mean_acd_max[i] - mean_acd_low[i])) + low_value;
        } else { flex_cal[i] = low_value; }
        flex_cal[i] = constrain(flex_cal[i], low_value, high_value);
    }

    QMI8658_Data d;
    if (imu.readSensorData(d)) {
        jsonDoc["ax"] = d.accelX; jsonDoc["ay"] = d.accelY; jsonDoc["az"] = d.accelZ;
        jsonDoc["gx"] = d.gyroX; jsonDoc["gy"] = d.gyroY; jsonDoc["gz"] = d.gyroZ;
    }

    for (int i = 0; i < NUM_FLEX_SENSORS; i++) {
      String key = "flex_" + String(i);
      jsonDoc[key] = flex_cal[i];
    }
    jsonDoc["slav_online"] = 1;

    SendSensorData();
    delay(50);
}

void readAllFlexSensors(float* raw_values) {
    raw_values[0] = ads1015_2.readADC_SingleEnded(0);
    raw_values[1] = ads1015_1.readADC_SingleEnded(3);
    raw_values[2] = ads1015_1.readADC_SingleEnded(2);
    raw_values[3] = ads1015_1.readADC_SingleEnded(1);
    raw_values[4] = ads1015_1.readADC_SingleEnded(0);
}

void flexCalibration() {
  for (int i = 0; i < NUM_FLEX_SENSORS; i++) {
    mean_acd_max[i] = 0.0f;
    mean_acd_low[i] = 0.0f;
  }
  LogToMaster("\n");
  LogToMaster("[Warboy] ┌──────────────────────────────────────────────────┐\n");
  LogToMaster("[Warboy] │           🎯 Starting Flex calibration           │\n");
  LogToMaster("[Warboy] └──────────────────────────────────────────────────┘\n");
  delay(1000);
  LogToMaster("[Warboy] Follow instructions for best calibration results.\n");

  for (int i = 3; i > 0; i--) {
      LogToMaster("[Warboy] Cal starting in %d...\n", i);
      delay(1000);
  }

  float temp_sensor_values[NUM_FLEX_SENSORS];
  const char* phases[] = {"มือปกติ (normal hand / straight)", "กำหมัด (fist / bent)"};
  for (int phase = 0; phase < 2; phase++) {
    LogToMaster("[Warboy] %s - Hold position. Starting in 5 sec...\n", phases[phase]);
    for (int i = 5; i > 0; i--) {
      LogToMaster("[Warboy] %d...\n", i);
      delay(1000);
    }

    LogToMaster("[Warboy] 📈 Collecting %d samples...\n", CALIBRATION_SAMPLES);
    LogToMaster("[Warboy] ");

    for (int j = 0; j < CALIBRATION_SAMPLES; j++) {
      readAllFlexSensors(temp_sensor_values);
      for (int k = 0; k < NUM_FLEX_SENSORS; k++) {
        if (phase == 0) { mean_acd_low[k] += temp_sensor_values[k]; } 
        else { mean_acd_max[k] += temp_sensor_values[k]; }
      }
      if ((j % 50) == 0 && j > 0) LogToMaster(".");
      delay(5);
    }
    LogToMaster(" Done.\n");

    for (int k = 0; k < NUM_FLEX_SENSORS; k++) {
      if (phase == 0) { mean_acd_low[k] /= CALIBRATION_SAMPLES; } 
      else { mean_acd_max[k] /= CALIBRATION_SAMPLES; }
    }
  }
  LogToMaster("[Warboy] ✅ Flex Sensor Calibration complete.\n");
  for(int i=0; i<NUM_FLEX_SENSORS; ++i) {
      LogToMaster("[Warboy] Sensor %d: Low Avg: %.2f, High Avg: %.2f\n", i, mean_acd_low[i], mean_acd_max[i]);
  }
}

void ADS1015_INITIALIZATION(){
  LogToMaster("\n");
  LogToMaster("[Warboy] Initializing ADS1015 #1 (0x48)...");
  if (!ads1015_1.begin(ADS1015_1_ADDRESS)) { LogToMaster("❌ FAILED!\n"); while (1) delay(1000); }
  LogToMaster("✅ OK.\n");
  LogToMaster("[Warboy] Initializing ADS1015 #2 (0x49)...");
  if (!ads1015_2.begin(ADS1015_2_ADDRESS)) { LogToMaster("❌ FAILED!\n"); while (1) delay(1000); }
  LogToMaster("✅ OK.\n");
}

void QMI8658_INITIALIZATION(){
  LogToMaster("\n");
  LogToMaster("[Warboy] Initializing QMI8658 IMU...");
  if (!imu.begin(SENSOR_SDA, SENSOR_SCL)) { LogToMaster("❌ FAILED!\n"); while (1) delay(1000); }
  LogToMaster("✅ OK.\n");
  imu.setAccelRange(QMI8658_ACCEL_RANGE_2G);
  imu.setAccelODR(QMI8658_ACCEL_ODR_1000HZ);
  imu.setGyroRange(QMI8658_GYRO_RANGE_256DPS);
  imu.setGyroODR(QMI8658_GYRO_ODR_1000HZ);
  imu.setAccelUnit_mps2(true);
  imu.setGyroUnit_rads(true);
  imu.enableSensors(QMI8658_ENABLE_ACCEL | QMI8658_ENABLE_GYRO);
}