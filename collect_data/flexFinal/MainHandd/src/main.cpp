#include <Arduino.h>
#include <Wire.h>
#include <BLEDevice.h>
#include <BLEClient.h>
#include <BLEScan.h>
#include <Adafruit_ADS1X15.h>
#include <QMI8658.h>
#include <WiFi.h> // For ESP32/ESP8266
#include <HTTPClient.h> // For ESP32/ESP8266
#include <ArduinoJson.h>
#include <esp_wifi.h>
#include <WiFiClient.h>

const char* ssid = "xxxxxxxxx";
const char* password = "xxxxxxxxx";
const char* serverUrl = "http://xxxxxxxxxxxx.app/predict_hand";
bool CONLECTOR = false;

// ──── PIN DEFINITIONS ────────────────────────────────────────────────────────────
#ifndef SENSOR_SCL
#define SENSOR_SCL  14
#endif
#ifndef SENSOR_SDA
#define SENSOR_SDA  15
#endif

// ──── PERIPHERAL ADDRESSES & CONSTANTS ─────────────────────────────────────────
const uint8_t ADS1015_1_ADDRESS = 0x48;
const uint8_t ADS1015_2_ADDRESS = 0x49;
const int NUM_FLEX_SENSORS = 5;
const int CALIBRATION_SAMPLES = 1000;

// UUIDs must match the sender
#define SERVICE_UUID "91bad492-b950-4226-aa2b-4ede9fa42f59"
#define CHARACTERISTIC_UUID "8beb1071-e483-4c71-a0ff-891272254a99"
#define STATUS_CHARACTERISTIC_UUID "b1234567-8ed3-4bdf-8a39-a01bebede296"

static BLEClient* pClient = nullptr;
static BLERemoteCharacteristic* pRemoteCharacteristic = nullptr;
static BLERemoteCharacteristic* pStatusCharacteristic = nullptr;
static boolean doConnect = false;
static boolean connected = false;
static boolean doScan = false;
static BLEAdvertisedDevice* myDevice = nullptr;

// ──── PERIPHERAL OBJECTS ───────────────────────────────────────────────────────
Adafruit_ADS1015 ads1015_1;
Adafruit_ADS1015 ads1015_2;
QMI8658          imu;
QMI8658_Data d;

// ──── CALIBRATION & FILTER VARIABLES ───────────────────────────────────────────
// Local (receiver) calibration
float mean_acd_max[NUM_FLEX_SENSORS] = {1800.0f, 1800.0f, 1800.0f, 1800.0f, 1800.0f};
float mean_acd_low[NUM_FLEX_SENSORS] = {900.0f, 900.0f, 900.0f, 900.0f, 900.0f};
// Sender calibration
float mean_acd_max_sender[NUM_FLEX_SENSORS] = {1800.0f, 1800.0f, 1800.0f, 1800.0f, 1800.0f};
float mean_acd_low_sender[NUM_FLEX_SENSORS] = {900.0f, 900.0f, 900.0f, 900.0f, 900.0f};

float low_value = 0.0f;
float high_value = 1000.0f;
float flex_raw_value[NUM_FLEX_SENSORS];
float flex_calibrated_sender[NUM_FLEX_SENSORS];

// Simple low-pass filter state for local sensors
static const float alpha = 0.5f;
float filtered_ax = 0.0f, filtered_ay = 0.0f, filtered_az = 0.0f;
float filtered_ax_slav = 0.0f, filtered_ay_slav = 0.0f, filtered_az_slav = 0.0f;

// CSV Timestamp
unsigned long start_timestamp = 0;

// Structure for received data from SENDER
struct ReceivedData {
  float ax, ay, az, gx, gy, gz;
  float angle_x, angle_y, angle_z;
  float flex_raw[NUM_FLEX_SENSORS];
};
ReceivedData senderData;

// Structure for LOCAL sensor data
struct LocalSensorData {
  float ax, ay, az, gx, gy, gz;
  float angle_x, angle_y, angle_z;
  float flex[NUM_FLEX_SENSORS];  // Calibrated values
};
LocalSensorData localData;

// ──── PROTOTYPES ────────────────────────────────────────────────────────────────
void checkCommand();
void readAllFlexSensors(float* raw_values);
void flexCalibration();
void resetCsvTimestamp();
void printCountdown(int seconds, const char* message_prefix = "[Receiver]");
void sendHttpPostRequest();
void initializeDoc();
void addSampleToDoc();

// Callback when data is received from sender
static void notifyCallback(BLERemoteCharacteristic* pBLERemoteCharacteristic, 
                           uint8_t* pData, size_t length, bool isNotify) {
  
  if (length == 56) {
    int offset = 0;
    
    // Extract all 14 float values from SENDER
    memcpy(&senderData.ax, pData + offset, 4); offset += 4;
    memcpy(&senderData.ay, pData + offset, 4); offset += 4;
    memcpy(&senderData.az, pData + offset, 4); offset += 4;
    memcpy(&senderData.gx, pData + offset, 4); offset += 4;
    memcpy(&senderData.gy, pData + offset, 4); offset += 4;
    memcpy(&senderData.gz, pData + offset, 4); offset += 4;
    memcpy(&senderData.angle_x, pData + offset, 4); offset += 4;
    memcpy(&senderData.angle_y, pData + offset, 4); offset += 4;
    memcpy(&senderData.angle_z, pData + offset, 4); offset += 4;
    memcpy(&senderData.flex_raw[0], pData + offset, 4); offset += 4;
    memcpy(&senderData.flex_raw[1], pData + offset, 4); offset += 4;
    memcpy(&senderData.flex_raw[2], pData + offset, 4); offset += 4;
    memcpy(&senderData.flex_raw[3], pData + offset, 4); offset += 4;
    memcpy(&senderData.flex_raw[4], pData + offset, 4); offset += 4;
    
    // Apply filtering to sender's accelerometer data
    // filtered_ax_slav = alpha * senderData.ax + (1.0f - alpha) * filtered_ax_slav;
    // filtered_ay_slav = alpha * senderData.ay + (1.0f - alpha) * filtered_ay_slav;
    // filtered_az_slav = alpha * senderData.az + (1.0f - alpha) * filtered_az_slav;
  }
}

// Callback when status is received
static void statusCallback(BLERemoteCharacteristic* pBLERemoteCharacteristic, 
                           uint8_t* pData, size_t length, bool isNotify) {
  String status = "";
  for (int i = 0; i < length; i++) {
    status += (char)pData[i];
  }
  Serial.print("[STATUS] ");
  Serial.println(status);
}

// Client callbacks for connection monitoring
class MyClientCallback : public BLEClientCallbacks {
  void onConnect(BLEClient* pclient) {
    Serial.println("[BLE] Client connected");
  }

  void onDisconnect(BLEClient* pclient) {
    connected = false;
    Serial.println("[BLE] Disconnected from sender");
    Serial.println("[BLE] Will attempt to reconnect...");
    doScan = true;
  }
};

// Connect to sender
bool connectToServer() {
  Serial.print("[BLE] Connecting to sender...");
  
  if (pClient == nullptr) {
    pClient = BLEDevice::createClient();
    pClient->setClientCallbacks(new MyClientCallback());
    Serial.println(" Client created");
  }
  
  if (!pClient->connect(myDevice)) {
    Serial.println(" Failed to connect!");
    return false;
  }
  Serial.println(" Connected!");

  BLERemoteService* pRemoteService = pClient->getService(SERVICE_UUID);
  if (pRemoteService == nullptr) {
    Serial.println("[BLE] Service not found");
    pClient->disconnect();
    return false;
  }
  Serial.println("[BLE] Service found");

  pRemoteCharacteristic = pRemoteService->getCharacteristic(CHARACTERISTIC_UUID);
  if (pRemoteCharacteristic == nullptr) {
    Serial.println("[BLE] Data characteristic not found");
    pClient->disconnect();
    return false;
  }
  Serial.println("[BLE] Data characteristic found");

  pStatusCharacteristic = pRemoteService->getCharacteristic(STATUS_CHARACTERISTIC_UUID);
  if (pStatusCharacteristic == nullptr) {
    Serial.println("[BLE] Status characteristic not found");
    pClient->disconnect();
    return false;
  }
  Serial.println("[BLE] Status characteristic found");

  if(pRemoteCharacteristic->canNotify()) {
    pRemoteCharacteristic->registerForNotify(notifyCallback);
    Serial.println("[BLE] Subscribed to data notifications");
  }
  
  if(pStatusCharacteristic->canNotify()) {
    pStatusCharacteristic->registerForNotify(statusCallback);
    Serial.println("[BLE] Subscribed to status notifications");
  }

  connected = true;
  return true;
}

// Callback when sender is found
class MyAdvertisedDeviceCallbacks: public BLEAdvertisedDeviceCallbacks {
  void onResult(BLEAdvertisedDevice advertisedDevice) {
    if (advertisedDevice.haveName() && advertisedDevice.getName() == "ESP32_Sender") {
      BLEDevice::getScan()->stop();
      
      if (myDevice != nullptr) {
        delete myDevice;
      }
      
      myDevice = new BLEAdvertisedDevice(advertisedDevice);
      doConnect = true;
      doScan = false;
      Serial.println("[BLE] Found ESP32_Sender!");
    }
  }
};

// ──── SENSOR INITIALIZATION ────────────────────────────────────────────────────
void ADS1015_INITIALIZATION(){
  Serial.println();
  Serial.print("[Receiver] Initializing ADS1015 #1 (0x48)...");
  if (!ads1015_1.begin(ADS1015_1_ADDRESS)) { 
    Serial.println("❌ FAILED!"); 
    while (1) delay(1000); 
  }
  Serial.println("✅ OK.");
  
  Serial.print("[Receiver] Initializing ADS1015 #2 (0x49)...");
  if (!ads1015_2.begin(ADS1015_2_ADDRESS)) { 
    Serial.println("❌ FAILED!"); 
    while (1) delay(1000); 
  }
  Serial.println("✅ OK.");
}

void QMI8658_INITIALIZATION(){
  Serial.println();
  Serial.print("[Receiver] Initializing QMI8658 IMU...");
  if (!imu.begin(SENSOR_SDA, SENSOR_SCL)) { 
    Serial.println("❌ FAILED!"); 
    while (1) delay(1000); 
  }
  Serial.println("✅ OK.");
  imu.setAccelRange(QMI8658_ACCEL_RANGE_2G);
  imu.setAccelODR(QMI8658_ACCEL_ODR_1000HZ);
  imu.setGyroRange(QMI8658_GYRO_RANGE_256DPS);
  imu.setGyroODR(QMI8658_GYRO_ODR_1000HZ);
  imu.setAccelUnit_mps2(true);
  imu.setGyroUnit_rads(true);
  imu.enableSensors(QMI8658_ENABLE_ACCEL | QMI8658_ENABLE_GYRO);
}

// Read local flex sensors
void readAllFlexSensors(float* raw_values) {
    raw_values[4] = ads1015_2.readADC_SingleEnded(0);
    raw_values[0] = ads1015_1.readADC_SingleEnded(3);
    raw_values[1] = ads1015_1.readADC_SingleEnded(2);
    raw_values[2] = ads1015_1.readADC_SingleEnded(1);
    raw_values[3] = ads1015_1.readADC_SingleEnded(0);
}

// Flex sensor calibration - BOTH LOCAL AND SENDER
void flexCalibration() {
    if (!connected) {
        Serial.println("[Receiver] ⚠️ ERROR: Sender not connected! Connect sender first.");
        return;
    }

    // Reset all calibration values
    for (int i = 0; i < NUM_FLEX_SENSORS; i++) {
        mean_acd_max[i] = 0.0f;
        mean_acd_low[i] = 0.0f;
        mean_acd_max_sender[i] = 0.0f;
        mean_acd_low_sender[i] = 0.0f;
    }
    
    Serial.println();
    Serial.println(F("[Receiver] ╔══════════════════════════════════════════════════════╗"));
    Serial.println(F("[Receiver] ║       🎯 DUAL CALIBRATION - LOCAL + SENDER          ║"));
    Serial.println(F("[Receiver] ╚══════════════════════════════════════════════════════╝"));
    delay(1000);
    Serial.println(F("[Receiver] Both devices will be calibrated simultaneously."));
    Serial.println(F("[Receiver] Make sure BOTH hands follow the same positions!"));
    printCountdown(3, "[Receiver] Cal starting in");

    float temp_local_values[NUM_FLEX_SENSORS];
    const char* phases[] = {"มือปกติ (BOTH hands normal/straight)", "กำหมัด (BOTH hands fist/bent)"};
    
    for (int phase = 0; phase < 2; phase++) {
        Serial.println();
        Serial.println(F("[Receiver] ════════════════════════════════════════════════"));
        Serial.printf("[Receiver] Phase %d/2: %s\n", phase + 1, phases[phase]);
        Serial.println(F("[Receiver] ════════════════════════════════════════════════"));
        Serial.println("[Receiver] ⚠️ IMPORTANT: Position BOTH hands correctly!");
        Serial.println("[Receiver] Hold position steady. Starting in 5 sec...");
        printCountdown(5, "[Receiver]");
        
        Serial.printf("[Receiver] 📈 Collecting %d samples from BOTH devices...\n", CALIBRATION_SAMPLES);
        Serial.print("[Receiver] Progress: ");
        
        for (int j = 0; j < CALIBRATION_SAMPLES; j++) {
            // Read LOCAL sensors
            readAllFlexSensors(temp_local_values);
            
            // Accumulate local data
            for (int k = 0; k < NUM_FLEX_SENSORS; k++) {
                if (phase == 0) { 
                    mean_acd_low[k] += temp_local_values[k];
                    mean_acd_low_sender[k] += senderData.flex_raw[k]; 
                } else { 
                    mean_acd_max[k] += temp_local_values[k];
                    mean_acd_max_sender[k] += senderData.flex_raw[k]; 
                }
            }
            
            if ((j % 50) == 0 && j > 0) Serial.print("█");
            delay(5);
        }
        Serial.println(" ✓ Done!");

        // Calculate averages
        for (int k = 0; k < NUM_FLEX_SENSORS; k++) {
            if (phase == 0) { 
                mean_acd_low[k] /= CALIBRATION_SAMPLES;
                mean_acd_low_sender[k] /= CALIBRATION_SAMPLES;
            } else { 
                mean_acd_max[k] /= CALIBRATION_SAMPLES;
                mean_acd_max_sender[k] /= CALIBRATION_SAMPLES;
            }
        }
    }
    
    Serial.println();
    Serial.println(F("[Receiver] ╔══════════════════════════════════════════════════════╗"));
    Serial.println(F("[Receiver] ║        ✅ DUAL CALIBRATION COMPLETE!                 ║"));
    Serial.println(F("[Receiver] ╚══════════════════════════════════════════════════════╝"));
    Serial.println();
    
    // Print LOCAL calibration results
    Serial.println(F("[Receiver] 📊 LOCAL SENSOR CALIBRATION RESULTS:"));
    Serial.println(F("[Receiver] ────────────────────────────────────────────"));
    for(int i = 0; i < NUM_FLEX_SENSORS; ++i) {
        Serial.printf("[Receiver]   Sensor %d: Low=%.2f, High=%.2f (Range: %.2f)\n", 
            i, mean_acd_low[i], mean_acd_max[i], mean_acd_max[i] - mean_acd_low[i]);
    }
    
    Serial.println();
    // Print SENDER calibration results
    Serial.println(F("[Receiver] 📡 SENDER SENSOR CALIBRATION RESULTS:"));
    Serial.println(F("[Receiver] ────────────────────────────────────────────"));
    for(int i = 0; i < NUM_FLEX_SENSORS; ++i) {
        Serial.printf("[Receiver]   Sensor %d: Low=%.2f, High=%.2f (Range: %.2f)\n", 
            i, mean_acd_low_sender[i], mean_acd_max_sender[i], mean_acd_max_sender[i] - mean_acd_low_sender[i]);
    }
    Serial.println();
}

void printCountdown(int seconds, const char* message_prefix) {
    for (int i = seconds; i > 0; i--) {
        Serial.printf("%s %d...\n", message_prefix, i);
        delay(1000);
    }
}

void resetCsvTimestamp(){
  Serial.println("[Receiver] Resetting CSV Timestamp...");
  start_timestamp = millis();
}

void checkCommand() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        Serial.printf("[CMD] Received: '%s'\n", command.c_str());

        if (command.equalsIgnoreCase("help")) {
            Serial.println("\n[CMD] Available commands:");
            Serial.println("  help       - Show this help message");
            Serial.println("  reset      - Reboot device");
            Serial.println("  flex_cal   - Start flex sensor calibration");
            Serial.println("  reset_time - Reset the CSV time baseline");
        } else if (command.equalsIgnoreCase("reset")) {
            Serial.println("[CMD] ➤ Rebooting...");
            delay(1000);
            ESP.restart();
        } else if (command.equalsIgnoreCase("flex_cal")) {
            Serial.println("[CMD] ➤ Initiating Flex Sensor Calibration.");
            flexCalibration();
        } else if (command.equalsIgnoreCase("reset_time")) {
            Serial.println("[CMD] ➤ Resetting CSV timestamp.");
            resetCsvTimestamp();
        } else {
            Serial.printf("[CMD] Unknown command: '%s'\n", command.c_str());
        }
    }
}

void setupWiFi() {
  if(!(CONLECTOR)){
    Serial.print("Connecting to WiFi");
    WiFi.begin(ssid, password);
    esp_wifi_set_max_tx_power(40);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n✓ WiFi connected!");
        Serial.print("IP address: ");
        Serial.println(WiFi.localIP());
        Serial.print("Signal strength: ");
        Serial.print(WiFi.RSSI());
        Serial.println(" dBm");
    } else {
        Serial.println("\n✗ WiFi connection failed!");
        ESP.restart();
    }
  }
}

void setup() {
  // Connect to WiFi
  Serial.begin(9600);
  delay(2000); // Give serial time to initialize
  Wire.begin(SENSOR_SDA, SENSOR_SCL);
  // Setup WiFi
  setupWiFi();
  Serial.println();
  Serial.println(F("██████╗░███████╗░█████╗░███████╗██╗██╗░░░██╗███████╗██████╗░"));
  Serial.println(F("██╔══██╗██╔════╝██╔══██╗██╔════╝██║██║░░░██║██╔════╝██╔══██╗"));
  Serial.println(F("██████╔╝█████╗░░██║░░╚═╝█████╗░░██║╚██╗░██╔╝█████╗░░██████╔╝"));
  Serial.println(F("██╔══██╗██╔══╝░░██║░░██╗██╔══╝░░██║░╚████╔╝░██╔══╝░░██╔══██╗"));
  Serial.println(F("██║░░██║███████╗╚█████╔╝███████╗██║░░╚██╔╝░░███████╗██║░░██║"));
  Serial.println(F("╚═╝░░╚═╝╚══════╝░╚════╝░╚══════╝╚═╝░░░╚═╝░░░╚══════╝╚═╝░░╚═╝"));
  Serial.println();

  Serial.print("[Receiver] "); Serial.println(F("┌──────────────────────────────────────────────────────────┐"));
  Serial.print("[Receiver] "); Serial.println(F("│                SKB Co. | Data Collection                 │"));
  Serial.print("[Receiver] "); Serial.println(F("│           Device: ESP32S3 SKB (Receiver Edition)         │"));
  Serial.print("[Receiver] "); Serial.println(F("│        Status: Initialized and Ready to Collect          │"));
  Serial.print("[Receiver] "); Serial.println(F("└──────────────────────────────────────────────────────────┘"));
  Serial.println();

  // Initialize local sensors
  ADS1015_INITIALIZATION();
  QMI8658_INITIALIZATION();

  // Initialize BLE
  BLEDevice::init("");
  // Start initial scan
  resetCsvTimestamp();
  doScan = true;
  
  Serial.println("\n[Receiver] Setup complete. Scanning for sender...");
  Serial.println("[Receiver] Type 'help' for a list of commands.");
}

// Global variables for stacking data
StaticJsonDocument<8192> doc;  // Increased size for multiple rows
JsonArray feature;
int sampleCount = 0;
const int MAX_SAMPLES = 30;  // Number of samples to collect before sending
bool docInitialized = false;

void loop() {
  checkCommand();
  if (WiFi.status() != WL_CONNECTED){
    setupWiFi();
  }
  // Restart scanning if needed
  if (doScan) {
    Serial.println("[BLE] Scanning for ESP32_Sender...");
    BLEScan* pBLEScan = BLEDevice::getScan();
    pBLEScan->clearResults();
    pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
    pBLEScan->setInterval(1349);
    pBLEScan->setWindow(449);
    pBLEScan->setActiveScan(true);
    pBLEScan->start(0, false);
    doScan = false;
  }

  // Connect to sender when found
  if (doConnect == true) {
    if (connectToServer()) {
      Serial.println("[BLE] Ready to receive data!");
    } else {
      Serial.println("[BLE] Failed to connect, will retry...");
      delay(1000);
      doScan = true;
    }
    doConnect = false;
  }

  // Check if still connected
  if (connected) {
    if (!pClient->isConnected()) {
      connected = false;
      Serial.println("[BLE] Connection lost!");
      delay(1000);
      doScan = true;
    }
  }
  // ──── READ LOCAL SENSORS ────────────────────────────────────────────────────
  if ((connected && (WiFi.status() == WL_CONNECTED)) || (connected && CONLECTOR)) {
    // Read raw flex values
    readAllFlexSensors(flex_raw_value);
    
    // Apply calibration mapping for LOCAL sensors (raw -> 0-1000)
    for (int i = 0; i < NUM_FLEX_SENSORS; i++) {
        if (mean_acd_max[i] != mean_acd_low[i]) {
            localData.flex[i] = ((flex_raw_value[i] - mean_acd_low[i]) * (high_value - low_value) / 
                                 (mean_acd_max[i] - mean_acd_low[i])) + low_value;
        } else { 
            localData.flex[i] = low_value; 
        }
        localData.flex[i] = constrain(localData.flex[i], low_value, high_value);
    }
    
    // Apply calibration mapping for SENDER sensors (raw -> 0-1000)
    for (int i = 0; i < NUM_FLEX_SENSORS; i++) {
        if (mean_acd_max_sender[i] != mean_acd_low_sender[i]) {
            flex_calibrated_sender[i] = ((senderData.flex_raw[i] - mean_acd_low_sender[i]) * (high_value - low_value) / 
                                         (mean_acd_max_sender[i] - mean_acd_low_sender[i])) + low_value;
        } else { 
            flex_calibrated_sender[i] = low_value; 
        }
        flex_calibrated_sender[i] = constrain(flex_calibrated_sender[i], low_value, high_value);
    }
    
    // Read IMU
    if (imu.readSensorData(d)) {
        localData.ax = d.accelX; 
        localData.ay = d.accelY; 
        localData.az = d.accelZ;
        localData.gx = d.gyroX; 
        localData.gy = d.gyroY; 
        localData.gz = d.gyroZ;
    }
    
    // Apply filtering and calculate angles
    filtered_ax = alpha * localData.ax + (1.0f - alpha) * filtered_ax;
    filtered_ay = alpha * localData.ay + (1.0f - alpha) * filtered_ay;
    filtered_az = alpha * localData.az + (1.0f - alpha) * filtered_az;
    // Calculate angles from filtered data
    localData.angle_x = atan2f(filtered_ay, sqrtf(filtered_ax * filtered_ax + filtered_az * filtered_az)) * RAD_TO_DEG;
    localData.angle_y = atan2f(filtered_ax, sqrtf(filtered_ay * filtered_ay + filtered_az * filtered_az)) * RAD_TO_DEG;
    localData.angle_z = atan2f(sqrtf(filtered_ax * filtered_ax + filtered_ay * filtered_ay), filtered_az) * RAD_TO_DEG;

    // Calculate sender angles from filtered data
    // senderData.angle_x = atan2f(filtered_ay_slav, sqrtf(filtered_ax_slav * filtered_ax_slav + filtered_az_slav * filtered_az_slav)) * RAD_TO_DEG;
    // senderData.angle_y = atan2f(filtered_ax_slav, sqrtf(filtered_ay_slav * filtered_ay_slav + filtered_az_slav * filtered_az_slav)) * RAD_TO_DEG;
    // senderData.angle_z = atan2f(sqrtf(filtered_ax_slav * filtered_ax_slav + filtered_ay_slav * filtered_ay_slav), filtered_az_slav) * RAD_TO_DEG;
    
    // Print combined data (CSV format) - NOW WITH CALIBRATED SENDER FLEX VALUES
    static char outBuf[1024];
    snprintf(outBuf, sizeof(outBuf),
      "%lu,"
      "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
      "%.2f,%.2f,%.2f,%.2f,%.2f,"
      "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
      "%.2f,%.2f,%.2f,%.2f,%.2f",
      millis() - start_timestamp,
      senderData.ax, senderData.ay, senderData.az, senderData.gx, senderData.gy, senderData.gz, 
      senderData.angle_x, senderData.angle_y, senderData.angle_z,
      flex_calibrated_sender[0], flex_calibrated_sender[1], flex_calibrated_sender[2], flex_calibrated_sender[3], flex_calibrated_sender[4],
      localData.ax, localData.ay, localData.az, localData.gx, localData.gy, localData.gz, 
      localData.angle_x, localData.angle_y, localData.angle_z,
      localData.flex[0], localData.flex[1], localData.flex[2], localData.flex[3], localData.flex[4]
    );
    Serial.printf("[Sensor] %s\n", outBuf);
    if(!(CONLECTOR)){
      addSampleToDoc();
      if (WiFi.status() == WL_CONNECTED) {
        // sendHttpPostRequest();
        // Send when enough samples collected
        if (sampleCount >= MAX_SAMPLES) {
          sendHttpPostRequest();
        }
      } else {
        Serial.println("WiFi disconnected");
      }   
    }
    delay(10);
  } else {
    delay(30);
  }
}


void initializeDoc() {
  doc.clear();
  doc["Id"] = "1595123198513";
  doc["Status"] = 1;
  feature = doc.createNestedArray("feature");
  sampleCount = 0;
  docInitialized = true;
}

void addSampleToDoc() {
  if (!docInitialized) {
    initializeDoc();
  }
  
  // Your sensor data array
  float temps[] = {(float)(millis() - start_timestamp),
                   senderData.ax, senderData.ay, senderData.az, 
                   senderData.gx, senderData.gy, senderData.gz, 
                   senderData.angle_x, senderData.angle_y, senderData.angle_z,
                   flex_calibrated_sender[0], flex_calibrated_sender[1], 
                   flex_calibrated_sender[2], flex_calibrated_sender[3], 
                   flex_calibrated_sender[4],
                   localData.ax, localData.ay, localData.az, 
                   localData.gx, localData.gy, localData.gz, 
                   localData.angle_x, localData.angle_y, localData.angle_z,
                   localData.flex[0], localData.flex[1], localData.flex[2], 
                   localData.flex[3], localData.flex[4]};
  
  // Add the temps array as a new row
  JsonArray row = feature.createNestedArray();
  for (int i = 0; i < 29; i++) {
    row.add(temps[i]);
  }
  
  sampleCount++;
}

void sendHttpPostRequest() {
  if (sampleCount == 0) {
    Serial.println("No samples to send");
    return;
  }
  
  HTTPClient http;
  WiFiClient client;
  
  http.begin(client, serverUrl);
  
  // Set headers
  http.addHeader("Content-Type", "application/json");
  
  // Set timeout (ngrok can be slow)
  http.setTimeout(10000); // 10 seconds
  
  // Serialize JSON to string
  String jsonString;
  serializeJson(doc, jsonString);
  
  Serial.println("Sending JSON with " + String(sampleCount) + " samples");
  Serial.println("JSON: " + jsonString);
  
  // Send POST request
  int httpResponseCode = http.POST(jsonString);
  
  // Handle response
  if (httpResponseCode > 0) {
    Serial.print("HTTP Response code: ");
    Serial.println(httpResponseCode);
    
    String response = http.getString();
    Serial.println("Response: " + response);
  } else {
    Serial.print("Error code: ");
    Serial.println(httpResponseCode);
    Serial.print("Error: ");
    Serial.println(http.errorToString(httpResponseCode));
  }
  
  // Free resources
  http.end();
  
  // Reset for next batch
  initializeDoc();
}

// void sendHttpPostRequest() {
//   HTTPClient http;
//   WiFiClient client;
  
//   // client.setInsecure();
  
//   http.begin(client, serverUrl);
  
//   // Set headers
//   http.addHeader("Content-Type", "application/json");
  
//   // Set timeout (ngrok can be slow)
//   http.setTimeout(10000); // 10 seconds
  
//   // Create JSON document (large enough for all sensor data)
//   StaticJsonDocument<2048> doc;
  
//   // Your sensor data array
//   float temps[] = {(float)(millis() - start_timestamp),
//                    senderData.ax, senderData.ay, senderData.az, 
//                    senderData.gx, senderData.gy, senderData.gz, 
//                    senderData.angle_x, senderData.angle_y, senderData.angle_z,
//                    flex_calibrated_sender[0], flex_calibrated_sender[1], 
//                    flex_calibrated_sender[2], flex_calibrated_sender[3], 
//                    flex_calibrated_sender[4],
//                    localData.ax, localData.ay, localData.az, 
//                    localData.gx, localData.gy, localData.gz, 
//                    localData.angle_x, localData.angle_y, localData.angle_z,
//                    localData.flex[0], localData.flex[1], localData.flex[2], 
//                    localData.flex[3], localData.flex[4]};
  
//   doc["Id"] = "1595123198513";      // Replace with your actual ID
//   doc["Status"] = 1;    // Replace with your actual status
//   // Create "feature" as List[List[float]]
//   JsonArray feature = doc.createNestedArray("feature");
  
//   // Add the temps array as a single row: [[value1, value2, value3, ...]]
//   JsonArray row = feature.createNestedArray();
//   for (int i = 0; i < 29; i++) {
//     row.add(temps[i]);
//   }
  
//   // Serialize JSON to string
//   String jsonString;
//   serializeJson(doc, jsonString);
  
//   Serial.println("Sending JSON: " + jsonString);
  
//   // Send POST request
//   int httpResponseCode = http.POST(jsonString);
  
//   // Handle response
//   if (httpResponseCode > 0) {
//     Serial.print("HTTP Response code: ");
//     Serial.println(httpResponseCode);
    
//     String response = http.getString();
//     Serial.println("Response: " + response);
//   } else {
//     Serial.print("Error code: ");
//     Serial.println(httpResponseCode);
//     Serial.print("Error: ");
//     Serial.println(http.errorToString(httpResponseCode));
//   }
  
//   // Free resources
//   http.end();
// }