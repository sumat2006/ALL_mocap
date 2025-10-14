#include <Arduino.h>
#include <Wire.h>
#include <esp_now.h>
#include <WiFi.h>
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
// MAC Address ของ main_hand (กรอกเป็น String แล้วแปลงอัตโนมัติใน setup)
// วิธีหา: upload main_hand แล้วดู Serial Monitor จะแสดง MAC Address
const char* MAIN_HAND_MAC_STRING = "XX:XX:XX:XX:XX:XX";  // เปลี่ยนตรงนี้!!!
uint8_t mainHandMacAddress[6];  // จะถูกแปลงจาก String ใน setup()

// ──── PERIPHERAL ADDRESSES & CONSTANTS ─────────────────────────────────────────
const uint8_t ADS1015_1_ADDRESS = 0x48;
const uint8_t ADS1015_2_ADDRESS = 0x49;
const int NUM_FLEX_SENSORS = 5;

// Timer variables
unsigned long lastTime = 0;
unsigned long timerDelay = 5; // Send every 5ms (200 Hz)

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

// ════════════════════════════════════════════════════════════════════════════════
// ฟังก์ชันแปลง MAC String เป็น byte array
// ════════════════════════════════════════════════════════════════════════════════
bool parseMacAddress(const char* macStr, uint8_t* macArray) {
  int values[6];
  if (sscanf(macStr, "%x:%x:%x:%x:%x:%x",
             &values[0], &values[1], &values[2],
             &values[3], &values[4], &values[5]) == 6) {
    for (int i = 0; i < 6; i++) {
      macArray[i] = (uint8_t)values[i];
    }
    return true;
  }
  return false;
}

// ════════════════════════════════════════════════════════════════════════════════
// ESP-NOW Callback เมื่อส่งข้อมูลสำเร็จ/ล้มเหลว
// ════════════════════════════════════════════════════════════════════════════════
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  if (status == ESP_NOW_SEND_SUCCESS) {
    sendSuccessCount++;
    // Serial.println("[ESP-NOW] ✓ Send Success");
  } else {
    sendFailCount++;
    Serial.println("[ESP-NOW] ✗ Send Failed");
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
  while (!Serial) { delay(10); }

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
  WiFi.mode(WIFI_STA);

  // แสดง MAC Address ของตัวเอง
  Serial.print("[Sender] My MAC Address: ");
  Serial.println(WiFi.macAddress());

  // แปลง MAC Address String เป็น byte array
  Serial.println("[Sender] Parsing Target MAC Address...");
  Serial.print("[Sender] Target MAC: ");
  Serial.println(MAIN_HAND_MAC_STRING);

  if (!parseMacAddress(MAIN_HAND_MAC_STRING, mainHandMacAddress)) {
    Serial.println();
    Serial.println(F("╔═══════════════════════════════════════════════════╗"));
    Serial.println(F("║               ❌ ERROR: INVALID MAC!              ║"));
    Serial.println(F("╚═══════════════════════════════════════════════════╝"));
    Serial.println(F("[ERROR] Invalid MAC Address format!"));
    Serial.println(F("[ERROR] Format should be: XX:XX:XX:XX:XX:XX"));
    Serial.println(F("[ERROR] Example: A0:B1:C2:D3:E4:F5"));
    Serial.println(F("[ERROR] "));
    Serial.println(F("[ERROR] HOW TO FIX:"));
    Serial.println(F("[ERROR] 1. Upload main_hand code"));
    Serial.println(F("[ERROR] 2. Check Serial Monitor for MAC Address"));
    Serial.println(F("[ERROR] 3. Copy MAC and paste to MAIN_HAND_MAC_STRING"));
    Serial.println(F("[ERROR] 4. Re-upload this code"));
    Serial.println();
    while(1) {
      delay(1000);  // หยุดไม่ให้ทำงานต่อ
    }
  }

  // แสดง MAC ที่แปลงแล้ว
  Serial.print("[Sender] ✓ Parsed MAC: ");
  for (int i = 0; i < 6; i++) {
    Serial.printf("%02X", mainHandMacAddress[i]);
    if (i < 5) Serial.print(":");
  }
  Serial.println();

  // Init ESP-NOW
  Serial.print("[ESP-NOW] Initializing...");
  if (esp_now_init() != ESP_OK) {
    Serial.println(" ❌ FAILED!");
    ESP.restart();
  }
  Serial.println(" ✅ OK");

  // Register callback
  Serial.print("[ESP-NOW] Registering send callback...");
  esp_now_register_send_cb(OnDataSent);
  Serial.println(" ✅ OK");

  // Add peer (main_hand)
  Serial.print("[ESP-NOW] Adding peer (main_hand)...");
  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  memcpy(peerInfo.peer_addr, mainHandMacAddress, 6);
  peerInfo.channel = 0;
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
  Serial.println("[Info] Sending data at 200 Hz (every 5ms)");
  Serial.println("[Info] Data packet size: 56 bytes (14 floats)");
  Serial.println();
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

  // ส่งข้อมูลผ่าน ESP-NOW ทุก 5ms
  if ((millis() - lastTime) > timerDelay) {
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
      Serial.printf("[ESP-NOW] ⚠️ Send error: %d\n", result);
    }

    lastTime = millis();
  }

  delay(10);
}
