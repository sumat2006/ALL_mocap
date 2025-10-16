#include <Arduino.h>
#include <Wire.h>
#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <Adafruit_ADS1X15.h>
#include <QMI8658.h>

// ──── PIN DEFINITIONS ────────────────────────────────────────────────────────────
#ifndef SENSOR_SCL
#define SENSOR_SCL  14
#endif
#ifndef SENSOR_SDA
#define SENSOR_SDA  15
#endif

// ──── MAC Address Configuration ─────────────────────────────────────────────────
// MAC Address ของ main_hand (กรอกเป็น hex array โดยตรง)
// วิธีหา: upload main_hand แล้วดู Serial Monitor จะแสดง MAC Address
uint8_t mainHandMacAddress[6] = {0xB8, 0xF8, 0x62, 0xE9, 0x65, 0x78};  // เปลี่ยนตรงนี้!!!

// ──── PERIPHERAL ADDRESSES & CONSTANTS ─────────────────────────────────────────
const uint8_t ADS1015_1_ADDRESS = 0x48;
const uint8_t ADS1015_2_ADDRESS = 0x49;
const int NUM_FLEX_SENSORS = 5;

// Timer variables
unsigned long lastTime = 0;
unsigned long timerDelay = 50; // Send every 50ms (20 Hz) - เพื่อความ stable

// ──── PERIPHERAL OBJECTS ───────────────────────────────────────────────────────
Adafruit_ADS1015 ads1015_1;
Adafruit_ADS1015 ads1015_2;
QMI8658          imu;
QMI8658_Data d;

// RAW sensor data structure
struct RawSensorData {
  float ax, ay, az, gx, gy, gz;
  float angle_x, angle_y, angle_z;  // Basic angles calculated from raw accel
  float flex_raw[NUM_FLEX_SENSORS]; // RAW ADC values
};
RawSensorData MyData;

// Simple low-pass filter state
static const float alpha = 0.5f;
float filtered_ax = 0.0f, filtered_ay = 0.0f, filtered_az = 0.0f;

// Statistics
unsigned long sendSuccessCount = 0;
unsigned long sendFailCount = 0;
bool sendInProgress = false;  // ป้องกันการส่งทับซ้อน

// ════════════════════════════════════════════════════════════════════════════════
// ESP-NOW Callback เมื่อส่งข้อมูลสำเร็จ/ล้มเหลว
// ════════════════════════════════════════════════════════════════════════════════
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  sendInProgress = false;  // ส่งเสร็จแล้ว

  if (status == ESP_NOW_SEND_SUCCESS) {
    sendSuccessCount++;
    // Serial.println("[ESP-NOW] ✓ Send Success");
  } else {
    sendFailCount++;
    // Serial.println("[ESP-NOW] ✗ Send Failed");  // ลด log เพื่อไม่ให้ช้า
  }

  // แสดง statistics ทุก 1000 packets
  if ((sendSuccessCount + sendFailCount) % 1000 == 0) {
    float successRate = (float)sendSuccessCount / (sendSuccessCount + sendFailCount) * 100.0f;
    Serial.printf("[Stats] Sent: %lu | Success: %lu (%.1f%%) | Failed: %lu\n",
                  sendSuccessCount + sendFailCount, sendSuccessCount, successRate, sendFailCount);
  }
}

// ════════════════════════════════════════════════════════════════════════════════
// ADS1015 Initialization
// ════════════════════════════════════════════════════════════════════════════════
void ADS1015_INITIALIZATION(){
  Serial.println();
  Serial.print("[Sender] Initializing ADS1015 #1 (0x48)...");
  if (!ads1015_1.begin(ADS1015_1_ADDRESS)) {
    Serial.println("❌ FAILED!");
    while (1) delay(1000);
  }
  Serial.println("✅ OK.");

  Serial.print("[Sender] Initializing ADS1015 #2 (0x49)...");
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
  Serial.print("[Sender] Initializing QMI8658 IMU...");
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
// อ่านค่า Flex Sensors ทั้ง 5 ตัว
// ════════════════════════════════════════════════════════════════════════════════
void readAllFlexSensors(float* raw_values) {
    raw_values[4] = ads1015_2.readADC_SingleEnded(0);  // นิ้วก้อย
    raw_values[0] = ads1015_1.readADC_SingleEnded(3);  // นิ้วหัวแม่มือ
    raw_values[1] = ads1015_1.readADC_SingleEnded(2);  // นิ้วชี้
    raw_values[2] = ads1015_1.readADC_SingleEnded(1);  // นิ้วกลาง
    raw_values[3] = ads1015_1.readADC_SingleEnded(0);  // นิ้วนาง
}

// ════════════════════════════════════════════════════════════════════════════════
// SETUP
// ════════════════════════════════════════════════════════════════════════════════
void setup() {
  // เริ่ม I2C และ Serial
  Wire.begin(SENSOR_SDA, SENSOR_SCL);
  Serial.begin(9600);

  // รอ Serial Monitor สูงสุด 3 วินาที (ถ้าไม่เสียบ USB ก็ข้ามไป)
  unsigned long serialStart = millis();
  while (!Serial && (millis() - serialStart < 3000)) {
    delay(10);
  }

  Serial.println();
  Serial.println(F("░██████╗███████╗███╗░░██╗██████╗░███████╗██████╗░"));
  Serial.println(F("██╔════╝██╔════╝████╗░██║██╔══██╗██╔════╝██╔══██╗"));
  Serial.println(F("╚█████╗░█████╗░░██╔██╗██║██║░░██║█████╗░░██████╔╝"));
  Serial.println(F("░╚═══██╗██╔══╝░░██║╚████║██║░░██║██╔══╝░░██╔══██╗"));
  Serial.println(F("██████╔╝███████╗██║░╚███║██████╔╝███████╗██║░░██║"));
  Serial.println(F("╚═════╝░╚══════╝╚═╝░░╚══╝╚═════╝░╚══════╝╚═╝░░╚═╝"));
  Serial.println();
  Serial.println(F("┌────────────────────────────────────────────────────┐"));
  Serial.println(F("│          SKB Co. | Data Collection                │"));
  Serial.println(F("│       Device: ESP32S3 SKB (Sender Edition)        │"));
  Serial.println(F("│       Protocol: ESP-NOW (RAW DATA MODE)           │"));
  Serial.println(F("└────────────────────────────────────────────────────┘"));
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
  delay(100);

  // แสดง MAC Address ของตัวเอง
  Serial.print("[Sender] My MAC Address: ");
  Serial.println(WiFi.macAddress());

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
  Serial.printf("[Sender] WiFi Channel: %d (verified)\n", primary_channel);

  // Set TX power (40 = 10dBm, ค่าที่ stable กว่า 84)
  esp_wifi_set_max_tx_power(40);
  int8_t power;
  esp_wifi_get_max_tx_power(&power);
  Serial.printf("[Sender] TX Power: %d (= %.1f dBm)\n", power, power * 0.25f);

  // แสดง Target MAC Address (ไม่ต้อง parse เพราะเป็น hex array โดยตรงแล้ว)
  Serial.print("[Sender] Target MAC: ");
  for (int i = 0; i < 6; i++) {
    Serial.printf("%02X", mainHandMacAddress[i]);
    if (i < 5) Serial.print(":");
  }
  Serial.println();

  // Register callback
  Serial.print("[ESP-NOW] Registering send callback...");
  esp_now_register_send_cb(OnDataSent);
  Serial.println(" ✅ OK");

  // Add peer (main_hand)
  Serial.print("[ESP-NOW] Adding peer (main_hand)...");
  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  memcpy(peerInfo.peer_addr, mainHandMacAddress, 6);
  peerInfo.channel = 1;  // Must match WiFi channel
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println(" ❌ FAILED!");
    Serial.println("[ERROR] Could not add peer!");
    Serial.println("[ERROR] Make sure main_hand is powered on and MAC is correct.");
    ESP.restart();
  }
  Serial.println(" ✅ OK");

  Serial.println();
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println(F("         ✅ SETUP COMPLETE - READY TO SEND!            "));
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println("[Info] Sending data at 20 Hz (every 50ms)");
  Serial.println("[Info] Data packet size: 56 bytes (14 floats)");
  Serial.println();

  delay(1000);  // Wait 1 second before starting to send
}

// ════════════════════════════════════════════════════════════════════════════════
// MAIN LOOP
// ════════════════════════════════════════════════════════════════════════════════
void loop() {
  // อ่าน Flex Sensors (RAW ADC values - ไม่มี calibration)
  readAllFlexSensors(MyData.flex_raw);

  // อ่าน IMU (RAW values)
  if (imu.readSensorData(d)) {
      MyData.ax = d.accelX;
      MyData.ay = d.accelY;
      MyData.az = d.accelZ;
      MyData.gx = d.gyroX;
      MyData.gy = d.gyroY;
      MyData.gz = d.gyroZ;
  }

  // คำนวณมุมจาก raw accelerometer data + low-pass filter
  filtered_ax = alpha * MyData.ax + (1.0f - alpha) * filtered_ax;
  filtered_ay = alpha * MyData.ay + (1.0f - alpha) * filtered_ay;
  filtered_az = alpha * MyData.az + (1.0f - alpha) * filtered_az;

  MyData.angle_x = atan2f(filtered_ay, sqrtf(filtered_ax * filtered_ax + filtered_az * filtered_az)) * RAD_TO_DEG;
  MyData.angle_y = atan2f(filtered_ax, sqrtf(filtered_ay * filtered_ay + filtered_az * filtered_az)) * RAD_TO_DEG;
  MyData.angle_z = atan2f(sqrtf(filtered_ax * filtered_ax + filtered_ay * filtered_ay), filtered_az) * RAD_TO_DEG;

  // ส่งข้อมูลผ่าน ESP-NOW ทุก 50ms (และต้องรอให้ส่งครั้งก่อนเสร็จก่อน)
  if ((millis() - lastTime) >= timerDelay && !sendInProgress) {
    sendInProgress = true;  // ล็อคไม่ให้ส่งซ้ำ

    // Pack ข้อมูล 14 floats เป็น byte array (56 bytes)
    uint8_t data[56];
    int offset = 0;

    memcpy(data + offset, &MyData.ax, 4); offset += 4;
    memcpy(data + offset, &MyData.ay, 4); offset += 4;
    memcpy(data + offset, &MyData.az, 4); offset += 4;
    memcpy(data + offset, &MyData.gx, 4); offset += 4;
    memcpy(data + offset, &MyData.gy, 4); offset += 4;
    memcpy(data + offset, &MyData.gz, 4); offset += 4;
    memcpy(data + offset, &MyData.angle_x, 4); offset += 4;
    memcpy(data + offset, &MyData.angle_y, 4); offset += 4;
    memcpy(data + offset, &MyData.angle_z, 4); offset += 4;
    memcpy(data + offset, &MyData.flex_raw[0], 4); offset += 4;
    memcpy(data + offset, &MyData.flex_raw[1], 4); offset += 4;
    memcpy(data + offset, &MyData.flex_raw[2], 4); offset += 4;
    memcpy(data + offset, &MyData.flex_raw[3], 4); offset += 4;
    memcpy(data + offset, &MyData.flex_raw[4], 4); offset += 4;

    // ส่งข้อมูล
    esp_err_t result = esp_now_send(mainHandMacAddress, data, sizeof(data));

    if (result != ESP_OK) {
      // Print error code for first 10 errors
      static int errorCount = 0;
      if (errorCount < 10) {
        Serial.printf("[ESP-NOW] ⚠️ Send error code: %d (", result);
        if (result == ESP_ERR_ESPNOW_NOT_INIT) Serial.print("NOT_INIT");
        else if (result == ESP_ERR_ESPNOW_ARG) Serial.print("INVALID_ARG");
        else if (result == ESP_ERR_ESPNOW_INTERNAL) Serial.print("INTERNAL_ERROR");
        else if (result == ESP_ERR_ESPNOW_NO_MEM) Serial.print("NO_MEMORY");
        else if (result == ESP_ERR_ESPNOW_NOT_FOUND) Serial.print("PEER_NOT_FOUND");
        else if (result == ESP_ERR_ESPNOW_IF) Serial.print("INVALID_INTERFACE");
        else Serial.print("UNKNOWN");
        Serial.println(")");
        errorCount++;
      }
      sendInProgress = false;  // ถ้าส่งไม่สำเร็จเลย ปลดล็อค
    }

    lastTime = millis();
  }

  // No delay - let loop run as fast as possible to maintain timing
}
