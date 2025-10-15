#include <Arduino.h>
#include <Wire.h>
#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <Adafruit_ADS1X15.h>
#include <QMI8658.h>
#include <ArduinoJson.h>

// ──── PIN DEFINITIONS ────────────────────────────────────────────────────────────
#ifndef SENSOR_SCL
#define SENSOR_SCL  14
#endif
#ifndef SENSOR_SDA
#define SENSOR_SDA  15
#endif

// Hardware UART Pins (Serial1 บน ESP32-S3)
#define RXD_PIN 17  // ไม่ได้ใช้จริง
#define TXD_PIN 18  // ส่งข้อมูลไป main_hub GPIO18
#define UART_BAUD 230400  // Hardware UART รองรับ baud rate สูง

// ──── PERIPHERAL ADDRESSES & CONSTANTS ─────────────────────────────────────────
const uint8_t ADS1015_1_ADDRESS = 0x48;
const uint8_t ADS1015_2_ADDRESS = 0x49;
const int NUM_FLEX_SENSORS = 5;
const int CALIBRATION_SAMPLES = 1000;

// ──── DATA COLLECTION CONFIGURATION ────────────────────────────────────────────
const int MAX_SAMPLES = 30;  // Number of samples before sending to hub
bool COLLECTOR_MODE = false; // true = CSV only, false = Send to hub
bool TEST_MODE = false;      // false = Use real sensor data

// ──── PERIPHERAL OBJECTS ───────────────────────────────────────────────────────
Adafruit_ADS1015 ads1015_1;
Adafruit_ADS1015 ads1015_2;
QMI8658          imu;
QMI8658_Data d;

// Hardware UART (Serial1) - no need to declare, it's built-in

// ──── CALIBRATION & FILTER VARIABLES ───────────────────────────────────────────
// Local (main_hand) calibration
float mean_acd_max[NUM_FLEX_SENSORS] = {1800.0f, 1800.0f, 1800.0f, 1800.0f, 1800.0f};
float mean_acd_low[NUM_FLEX_SENSORS] = {900.0f, 900.0f, 900.0f, 900.0f, 900.0f};
// Slave_hand calibration
float mean_acd_max_slave[NUM_FLEX_SENSORS] = {1800.0f, 1800.0f, 1800.0f, 1800.0f, 1800.0f};
float mean_acd_low_slave[NUM_FLEX_SENSORS] = {900.0f, 900.0f, 900.0f, 900.0f, 900.0f};

float low_value = 0.0f;
float high_value = 1000.0f;
float flex_raw_value[NUM_FLEX_SENSORS];
float flex_calibrated_slave[NUM_FLEX_SENSORS];

// Simple low-pass filter state for local sensors
static const float alpha = 0.5f;
float filtered_ax = 0.0f, filtered_ay = 0.0f, filtered_az = 0.0f;

// CSV Timestamp
unsigned long start_timestamp = 0;

// Structure for received data from SLAVE_HAND (via ESP-NOW)
struct ReceivedData {
  float ax, ay, az, gx, gy, gz;
  float angle_x, angle_y, angle_z;
  float flex_raw[NUM_FLEX_SENSORS];
};
ReceivedData slaveData;
bool slaveDataReceived = false;

// Structure for LOCAL sensor data
struct LocalSensorData {
  float ax, ay, az, gx, gy, gz;
  float angle_x, angle_y, angle_z;
  float flex[NUM_FLEX_SENSORS];  // Calibrated values
};
LocalSensorData localData;

// ──── JSON DATA BATCHING ────────────────────────────────────────────────────────
JsonDocument doc;
JsonArray feature;
int sampleCount = 0;
bool docInitialized = false;

// ──── ESP-NOW STATISTICS ────────────────────────────────────────────────────────
unsigned long espnowReceiveCount = 0;
unsigned long lastStatsTime = 0;

// ──── FUNCTION PROTOTYPES ───────────────────────────────────────────────────────
void printCountdown(int seconds, const char* message_prefix);

// ════════════════════════════════════════════════════════════════════════════════
// ESP-NOW Callback เมื่อรับข้อมูลจาก slave_hand (รองรับทั้ง ESP-IDF v4 และ v5)
// ════════════════════════════════════════════════════════════════════════════════
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
// ESP-IDF v5.x (ใหม่) - ใช้ esp_now_recv_info struct
void OnDataRecv(const esp_now_recv_info *recv_info, const uint8_t *incomingData, int len) {
#else
// ESP-IDF v4.x (เก่า) - ใช้ MAC address โดยตรง
void OnDataRecv(const uint8_t *mac_addr, const uint8_t *incomingData, int len) {
#endif
  // Debug: Show first 5 packet lengths
  static int packetDebugCount = 0;
  if (packetDebugCount < 5) {
    Serial.printf("[ESP-NOW] Packet received: len=%d (expected 56)\n", len);
    packetDebugCount++;
  }

  if (len == 56) {  // 14 floats × 4 bytes = 56 bytes
    int offset = 0;

    // Extract all 14 float values from SLAVE_HAND
    memcpy(&slaveData.ax, incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.ay, incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.az, incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.gx, incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.gy, incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.gz, incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.angle_x, incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.angle_y, incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.angle_z, incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.flex_raw[0], incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.flex_raw[1], incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.flex_raw[2], incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.flex_raw[3], incomingData + offset, 4); offset += 4;
    memcpy(&slaveData.flex_raw[4], incomingData + offset, 4); offset += 4;

    slaveDataReceived = true;
    espnowReceiveCount++;

    // แสดง message ทุกครั้งที่รับ (first 10, then every 100)
    static unsigned long totalReceived = 0;
    totalReceived++;
    if (totalReceived <= 10 || totalReceived % 100 == 0) {
      Serial.printf("[ESP-NOW] ✓ Received #%lu (%.1f, %.1f, %.1f)\n",
                    totalReceived, slaveData.ax, slaveData.ay, slaveData.az);
    }

    // แสดง statistics ทุก 5 วินาที
    if (millis() - lastStatsTime > 5000) {
      Serial.printf("[ESP-NOW] Received packets: %lu (%.1f Hz)\n",
                    espnowReceiveCount, espnowReceiveCount / 5.0f);
      espnowReceiveCount = 0;
      lastStatsTime = millis();
    }
  } else {
    // Wrong packet size
    static int wrongSizeCount = 0;
    if (wrongSizeCount < 5) {
      Serial.printf("[ESP-NOW] ⚠️ Wrong size: got %d bytes, expected 56\n", len);
      wrongSizeCount++;
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════════
// ADS1015 Initialization
// ════════════════════════════════════════════════════════════════════════════════
void ADS1015_INITIALIZATION(){
  Serial.println();
  Serial.print("[Main_Hand] Initializing ADS1015 #1 (0x48)...");
  if (!ads1015_1.begin(ADS1015_1_ADDRESS)) {
    Serial.println("❌ FAILED!");
    while (1) delay(1000);
  }
  Serial.println("✅ OK.");

  Serial.print("[Main_Hand] Initializing ADS1015 #2 (0x49)...");
  if (!ads1015_2.begin(ADS1015_2_ADDRESS)) {
    Serial.println("❌ FAILED!");
    while (1) delay(1000);
  }
  Serial.println("✅ OK.");
}

// ════════════════════════════════════════════════════════════════════════════════
// QMI8658 IMU Initialization
// ════════════════════════════════════════════════════════════════════════════════
void QMI8658_INITIALIZATION(){
  Serial.println();
  Serial.print("[Main_Hand] Initializing QMI8658 IMU...");
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

// ════════════════════════════════════════════════════════════════════════════════
// อ่านค่า Flex Sensors ทั้ง 5 ตัว (LOCAL)
// ════════════════════════════════════════════════════════════════════════════════
void readAllFlexSensors(float* raw_values) {
    raw_values[4] = ads1015_2.readADC_SingleEnded(0);  // นิ้วก้อย
    raw_values[0] = ads1015_1.readADC_SingleEnded(3);  // นิ้วหัวแม่มือ
    raw_values[1] = ads1015_1.readADC_SingleEnded(2);  // นิ้วชี้
    raw_values[2] = ads1015_1.readADC_SingleEnded(1);  // นิ้วกลาง
    raw_values[3] = ads1015_1.readADC_SingleEnded(0);  // นิ้วนาง
}

// ════════════════════════════════════════════════════════════════════════════════
// Flex Sensor Calibration - BOTH LOCAL AND SLAVE
// ════════════════════════════════════════════════════════════════════════════════
void flexCalibration() {
    if (!slaveDataReceived) {
        Serial.println("[Main_Hand] ⚠️ ERROR: Slave not connected! Connect slave_hand first.");
        return;
    }

    // Reset all calibration values
    for (int i = 0; i < NUM_FLEX_SENSORS; i++) {
        mean_acd_max[i] = 0.0f;
        mean_acd_low[i] = 0.0f;
        mean_acd_max_slave[i] = 0.0f;
        mean_acd_low_slave[i] = 0.0f;
    }

    Serial.println();
    Serial.println(F("[Main_Hand] ╔══════════════════════════════════════════════════════╗"));
    Serial.println(F("[Main_Hand] ║       🎯 DUAL CALIBRATION - LOCAL + SLAVE           ║"));
    Serial.println(F("[Main_Hand] ╚══════════════════════════════════════════════════════╝"));
    delay(1000);
    Serial.println(F("[Main_Hand] Both devices will be calibrated simultaneously."));
    Serial.println(F("[Main_Hand] Make sure BOTH hands follow the same positions!"));
    printCountdown(3, "[Main_Hand] Cal starting in");

    float temp_local_values[NUM_FLEX_SENSORS];
    const char* phases[] = {"มือปกติ (BOTH hands normal/straight)", "กำหมัด (BOTH hands fist/bent)"};

    for (int phase = 0; phase < 2; phase++) {
        Serial.println();
        Serial.println(F("[Main_Hand] ════════════════════════════════════════════════"));
        Serial.printf("[Main_Hand] Phase %d/2: %s\n", phase + 1, phases[phase]);
        Serial.println(F("[Main_Hand] ════════════════════════════════════════════════"));
        Serial.println("[Main_Hand] ⚠️ IMPORTANT: Position BOTH hands correctly!");
        Serial.println("[Main_Hand] Hold position steady. Starting in 5 sec...");
        printCountdown(5, "[Main_Hand]");

        Serial.printf("[Main_Hand] 📈 Collecting %d samples from BOTH devices...\n", CALIBRATION_SAMPLES);
        Serial.print("[Main_Hand] Progress: ");

        for (int j = 0; j < CALIBRATION_SAMPLES; j++) {
            // Read LOCAL sensors
            readAllFlexSensors(temp_local_values);

            // Accumulate local data
            for (int k = 0; k < NUM_FLEX_SENSORS; k++) {
                if (phase == 0) {
                    mean_acd_low[k] += temp_local_values[k];
                    mean_acd_low_slave[k] += slaveData.flex_raw[k];
                } else {
                    mean_acd_max[k] += temp_local_values[k];
                    mean_acd_max_slave[k] += slaveData.flex_raw[k];
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
                mean_acd_low_slave[k] /= CALIBRATION_SAMPLES;
            } else {
                mean_acd_max[k] /= CALIBRATION_SAMPLES;
                mean_acd_max_slave[k] /= CALIBRATION_SAMPLES;
            }
        }
    }

    Serial.println();
    Serial.println(F("[Main_Hand] ╔══════════════════════════════════════════════════════╗"));
    Serial.println(F("[Main_Hand] ║        ✅ DUAL CALIBRATION COMPLETE!                 ║"));
    Serial.println(F("[Main_Hand] ╚══════════════════════════════════════════════════════╝"));
    Serial.println();

    // Print LOCAL calibration results
    Serial.println(F("[Main_Hand] 📊 LOCAL SENSOR CALIBRATION RESULTS:"));
    Serial.println(F("[Main_Hand] ────────────────────────────────────────────"));
    for(int i = 0; i < NUM_FLEX_SENSORS; ++i) {
        Serial.printf("[Main_Hand]   Sensor %d: Low=%.2f, High=%.2f (Range: %.2f)\n",
            i, mean_acd_low[i], mean_acd_max[i], mean_acd_max[i] - mean_acd_low[i]);
    }

    Serial.println();
    // Print SLAVE calibration results
    Serial.println(F("[Main_Hand] 📡 SLAVE SENSOR CALIBRATION RESULTS:"));
    Serial.println(F("[Main_Hand] ────────────────────────────────────────────"));
    for(int i = 0; i < NUM_FLEX_SENSORS; ++i) {
        Serial.printf("[Main_Hand]   Sensor %d: Low=%.2f, High=%.2f (Range: %.2f)\n",
            i, mean_acd_low_slave[i], mean_acd_max_slave[i], mean_acd_max_slave[i] - mean_acd_low_slave[i]);
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
  Serial.println("[Main_Hand] Resetting CSV Timestamp...");
  start_timestamp = millis();
}

// ════════════════════════════════════════════════════════════════════════════════
// Serial Command Handler
// ════════════════════════════════════════════════════════════════════════════════
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

// ════════════════════════════════════════════════════════════════════════════════
// Initialize JSON Document for batching
// ════════════════════════════════════════════════════════════════════════════════
void initializeDoc() {
  doc.clear();
  doc["Id"] = "1595123198513";
  doc["Status"] = 1;
  feature = doc["feature"].to<JsonArray>();
  sampleCount = 0;
  docInitialized = true;
}

// ════════════════════════════════════════════════════════════════════════════════
// Add Sample to JSON Document
// ════════════════════════════════════════════════════════════════════════════════
void addSampleToDoc() {
  if (!docInitialized) {
    initializeDoc();
  }

  // Your sensor data array
  float temps[] = {(float)(millis() - start_timestamp),
                   slaveData.ax, slaveData.ay, slaveData.az,
                   slaveData.gx, slaveData.gy, slaveData.gz,
                   slaveData.angle_x, slaveData.angle_y, slaveData.angle_z,
                   flex_calibrated_slave[0], flex_calibrated_slave[1],
                   flex_calibrated_slave[2], flex_calibrated_slave[3],
                   flex_calibrated_slave[4],
                   localData.ax, localData.ay, localData.az,
                   localData.gx, localData.gy, localData.gz,
                   localData.angle_x, localData.angle_y, localData.angle_z,
                   localData.flex[0], localData.flex[1], localData.flex[2],
                   localData.flex[3], localData.flex[4]};

  // Add the temps array as a new row
  JsonArray row = feature.add<JsonArray>();
  for (int i = 0; i < 29; i++) {
    row.add(temps[i]);
  }

  sampleCount++;
}

// ════════════════════════════════════════════════════════════════════════════════
// Send JSON Data via UART to main_hub
// ════════════════════════════════════════════════════════════════════════════════
void sendViaUART() {
  if (sampleCount == 0) {
    Serial.println("[UART] No samples to send");
    return;
  }

  // Serialize JSON to string
  String jsonString;
  serializeJson(doc, jsonString);

  unsigned long startTime = millis();
  Serial.printf("[UART] Sending %d samples (%d bytes)...\n", sampleCount, jsonString.length());

  // ส่งข้อมูลผ่าน Hardware UART (Serial1) พร้อม newline เป็น delimiter
  Serial1.println(jsonString);
  Serial1.flush();  // รอให้ส่งเสร็จ

  unsigned long duration = millis() - startTime;
  Serial.printf("[UART] ✓ Sent in %lu ms\n", duration);

  // Reset for next batch
  initializeDoc();
}

// ════════════════════════════════════════════════════════════════════════════════
// SETUP
// ════════════════════════════════════════════════════════════════════════════════
void setup() {
  // Start I2C and Serial
  Wire.begin(SENSOR_SDA, SENSOR_SCL);
  Serial.begin(115200);  // USB Serial for debugging (ต้องเร็วกว่า SoftwareSerial)

  // รอ Serial Monitor สูงสุด 3 วินาที (ถ้าไม่เสียบ USB ก็ข้ามไป)
  unsigned long serialStart = millis();
  while (!Serial && (millis() - serialStart < 3000)) {
    delay(10);
  }

  // Start Hardware UART (Serial1) for communication with main_hub
  // ESP32-S3: Serial1 uses GPIO 17 (RX) and GPIO 18 (TX) by default
  Serial1.begin(UART_BAUD, SERIAL_8N1, RXD_PIN, TXD_PIN);
  delay(100);

  Serial.println();
  Serial.println(F("██████╗░███████╗░█████╗░███████╗██╗██╗░░░██╗███████╗██████╗░"));
  Serial.println(F("██╔══██╗██╔════╝██╔══██╗██╔════╝██║██║░░░██║██╔════╝██╔══██╗"));
  Serial.println(F("██████╔╝█████╗░░██║░░╚═╝█████╗░░██║╚██╗░██╔╝█████╗░░██████╔╝"));
  Serial.println(F("██╔══██╗██╔══╝░░██║░░██╗██╔══╝░░██║░╚████╔╝░██╔══╝░░██╔══██╗"));
  Serial.println(F("██║░░██║███████╗╚█████╔╝███████╗██║░░╚██╔╝░░███████╗██║░░██║"));
  Serial.println(F("╚═╝░░╚═╝╚══════╝░╚════╝░╚══════╝╚═╝░░░╚═╝░░░╚══════╝╚═╝░░╚═╝"));
  Serial.println();
  Serial.println(F("┌────────────────────────────────────────────────────────┐"));
  Serial.println(F("│          SKB Co. | Data Collection                     │"));
  Serial.println(F("│       Device: ESP32S3 SKB (Main Hand Edition)          │"));
  Serial.println(F("│       Protocol: ESP-NOW RX + UART TX                   │"));
  Serial.println(F("└────────────────────────────────────────────────────────┘"));
  Serial.println();

  // Initialize sensors
  ADS1015_INITIALIZATION();
  QMI8658_INITIALIZATION();

  Serial.println();
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println(F("         ⚡ ESP-NOW INITIALIZATION                       "));
  Serial.println(F("════════════════════════════════════════════════════════"));

  // Set WiFi mode (ไม่ต้องเชื่อมต่อ WiFi จริง)
  WiFi.mode(WIFI_STA);  // ESP-NOW requires WIFI_STA only, not WIFI_AP_STA
  WiFi.disconnect();
  delay(100);  // Wait for WiFi to initialize

  // แสดง MAC Address ของตัวเอง
  Serial.print("[Main_Hand] My MAC Address: ");
  Serial.println(WiFi.macAddress());
  Serial.println("[Main_Hand] Copy this MAC to slave_hand's MAIN_HAND_MAC_STRING!");

  // Init ESP-NOW ก่อน (ต้อง init ก่อนค่อย set channel)
  Serial.print("[ESP-NOW] Initializing...");
  if (esp_now_init() != ESP_OK) {
    Serial.println(" ❌ FAILED!");
    ESP.restart();
  }
  Serial.println(" ✅ OK");

  // Set WiFi channel หลัง ESP-NOW init
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);

  // Verify channel
  uint8_t primary_channel;
  wifi_second_chan_t secondary_channel;
  esp_wifi_get_channel(&primary_channel, &secondary_channel);
  Serial.printf("[Main_Hand] WiFi Channel: %d (verified)\n", primary_channel);

  // Set TX power (40 = 10dBm, ค่าที่ stable กว่า 84)
  esp_wifi_set_max_tx_power(40);
  int8_t power;
  esp_wifi_get_max_tx_power(&power);
  Serial.printf("[Main_Hand] TX Power: %d (= %.1f dBm)\n", power, power * 0.25f);

  // Register receive callback
  Serial.print("[ESP-NOW] Registering receive callback...");
  esp_now_register_recv_cb(OnDataRecv);
  Serial.println(" ✅ OK");

  Serial.println();
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println(F("         ⚡ HARDWARE UART INITIALIZATION                 "));
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.printf("[UART] Using Serial1 (Hardware UART)\n");
  Serial.printf("[UART] TX Pin: GPIO%d → main_hub GPIO18\n", TXD_PIN);
  Serial.printf("[UART] RX Pin: GPIO%d (not used)\n", RXD_PIN);
  Serial.printf("[UART] Baud Rate: %d\n", UART_BAUD);
  Serial.println("[UART] ✅ Hardware UART ready (DMA-based, non-blocking)");

  Serial.println();
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println(F("         ✅ SETUP COMPLETE - READY TO RECEIVE!          "));
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println("[Info] Waiting for data from slave_hand via ESP-NOW");
  Serial.println("[Info] Data will be sent to main_hub via UART");
  Serial.println("[Info] Type 'help' for a list of commands.");
  Serial.println();

  resetCsvTimestamp();
  lastStatsTime = millis();
}

// ════════════════════════════════════════════════════════════════════════════════
// MAIN LOOP
// ════════════════════════════════════════════════════════════════════════════════
void loop() {
  checkCommand();

  // ──── READ LOCAL SENSORS (แต่ไม่บล็อค ESP-NOW callback) ────────────────────
  if (slaveDataReceived) {
    // Read raw flex values (ช้า ~15ms)
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

    // Apply calibration mapping for SLAVE sensors (raw -> 0-1000)
    for (int i = 0; i < NUM_FLEX_SENSORS; i++) {
        if (mean_acd_max_slave[i] != mean_acd_low_slave[i]) {
            flex_calibrated_slave[i] = ((slaveData.flex_raw[i] - mean_acd_low_slave[i]) * (high_value - low_value) /
                                         (mean_acd_max_slave[i] - mean_acd_low_slave[i])) + low_value;
        } else {
            flex_calibrated_slave[i] = low_value;
        }
        flex_calibrated_slave[i] = constrain(flex_calibrated_slave[i], low_value, high_value);
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

    // Print combined data (CSV format)
    static char outBuf[1024];
    snprintf(outBuf, sizeof(outBuf),
      "%lu,"
      "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
      "%.2f,%.2f,%.2f,%.2f,%.2f,"
      "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
      "%.2f,%.2f,%.2f,%.2f,%.2f",
      millis() - start_timestamp,
      slaveData.ax, slaveData.ay, slaveData.az, slaveData.gx, slaveData.gy, slaveData.gz,
      slaveData.angle_x, slaveData.angle_y, slaveData.angle_z,
      flex_calibrated_slave[0], flex_calibrated_slave[1], flex_calibrated_slave[2], flex_calibrated_slave[3], flex_calibrated_slave[4],
      localData.ax, localData.ay, localData.az, localData.gx, localData.gy, localData.gz,
      localData.angle_x, localData.angle_y, localData.angle_z,
      localData.flex[0], localData.flex[1], localData.flex[2], localData.flex[3], localData.flex[4]
    );
    // Serial.printf("[Sensor] %s\n", outBuf);

    // Add to JSON batch and send if needed
    if (!COLLECTOR_MODE) {
      addSampleToDoc();

      // Debug: แสดง sample count
      if (sampleCount == 1 || sampleCount % 10 == 0) {
        Serial.printf("[JSON] Collected %d/%d samples\n", sampleCount, MAX_SAMPLES);
      }

      if (sampleCount >= MAX_SAMPLES) {
        sendViaUART();
      }
    }

    // ลด delay เพื่อไม่ให้พลาด ESP-NOW packets
    delay(1);  // แค่ให้ yield เท่านั้น
  } else {
    // Waiting for slave_hand connection
    delay(50);
  }
}
