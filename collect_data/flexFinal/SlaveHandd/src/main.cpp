/*********
  SENDER - Only reads RAW sensor data and sends to receiver
  Now with automatic reconnection support
*********/
#include <Arduino.h>
#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <Adafruit_ADS1X15.h>
#include <QMI8658.h>

// ──── PIN DEFINITIONS ────────────────────────────────────────────────────────────
#ifndef SENSOR_SCL
#define SENSOR_SCL  14
#endif
#ifndef SENSOR_SDA
#define SENSOR_SDA  15
#endif

//BLE server name
#define bleServerName "ESP32_Sender"

// Timer variables
unsigned long lastTime = 0;
unsigned long timerDelay = 5; // Send every 5ms

bool deviceConnected = false;
bool oldDeviceConnected = false;  // NEW: Track previous connection state

// Service UUID
#define SERVICE_UUID "91bad492-b950-4226-aa2b-4ede9fa42f59"

// Characteristic for RAW sensor data (14 floats = 56 bytes)
BLECharacteristic sensorCharacteristic("8beb1071-e483-4c71-a0ff-891272254a99", BLECharacteristic::PROPERTY_NOTIFY);
BLEDescriptor sensorDescriptor(BLEUUID((uint16_t)0x2902));

// Characteristic for sending status to receiver
BLECharacteristic statusCharacteristic("b1234567-8ed3-4bdf-8a39-a01bebede296", BLECharacteristic::PROPERTY_NOTIFY);
BLEDescriptor statusDescriptor(BLEUUID((uint16_t)0x2904));

// NEW: Store server pointer for reconnection
BLEServer *pServer = nullptr;

// ──── PERIPHERAL ADDRESSES & CONSTANTS ─────────────────────────────────────────
const uint8_t ADS1015_1_ADDRESS = 0x48;
const uint8_t ADS1015_2_ADDRESS = 0x49;
const int NUM_FLEX_SENSORS = 5;

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
float filtered_ax_slav = 0.0f, filtered_ay_slav = 0.0f, filtered_az_slav = 0.0f;

//Setup callbacks onConnect and onDisconnect
class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("[SENDER] Receiver connected");
    // Send status
    String status = "CONNECTED:Sender online";
    statusCharacteristic.setValue(status.c_str());
    statusCharacteristic.notify();
  };
  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("[SENDER] Receiver disconnected");
    // Don't restart advertising here - do it in loop()
  }
};

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

void readAllFlexSensors(float* raw_values) {
    raw_values[4] = ads1015_2.readADC_SingleEnded(0);
    raw_values[0] = ads1015_1.readADC_SingleEnded(3);
    raw_values[1] = ads1015_1.readADC_SingleEnded(2);
    raw_values[2] = ads1015_1.readADC_SingleEnded(1);
    raw_values[3] = ads1015_1.readADC_SingleEnded(0);
}

void setup() {
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
  Serial.println("[Sender] RAW DATA MODE - No processing");
  Serial.println();
  
  ADS1015_INITIALIZATION();
  QMI8658_INITIALIZATION();

  // Initialize BLE
  Serial.println("[SENDER] Initializing BLE...");
  BLEDevice::init(bleServerName);

  // Create BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create BLE Service
  BLEService *sensorService = pServer->createService(SERVICE_UUID);

  // Create BLE Characteristic for RAW sensor data
  sensorService->addCharacteristic(&sensorCharacteristic);
  sensorDescriptor.setValue("RAW Sensor Data (14 floats)");
  sensorCharacteristic.addDescriptor(&sensorDescriptor);
  
  // Create BLE Characteristic for sending status
  sensorService->addCharacteristic(&statusCharacteristic);
  statusDescriptor.setValue("Status updates");
  statusCharacteristic.addDescriptor(&statusDescriptor);
  
  // Start the service
  sensorService->start();

  // Start advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pServer->getAdvertising()->start();
  Serial.println("[SENDER] Waiting for receiver connection...");
}

void loop() {
  if (!deviceConnected && oldDeviceConnected) {
    delay(500); // Give the Bluetooth stack time to get ready
    pServer->startAdvertising(); // Restart advertising
    Serial.println("[SENDER] Restarting advertising...");
    oldDeviceConnected = deviceConnected;
  }
  
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }
  
  // Read RAW flex sensors (no calibration, just ADC values)
  readAllFlexSensors(MyData.flex_raw);
  
  // Read RAW IMU data
  if (imu.readSensorData(d)) {
      MyData.ax = d.accelX; 
      MyData.ay = d.accelY; 
      MyData.az = d.accelZ;
      MyData.gx = d.gyroX; 
      MyData.gy = d.gyroY; 
      MyData.gz = d.gyroZ;
  }
  
  // Calculate basic angles from raw accelerometer (no filtering)
  filtered_ax = alpha * MyData.ax + (1.0f - alpha) * filtered_ax;
  filtered_ay = alpha * MyData.ay + (1.0f - alpha) * filtered_ay;
  filtered_az = alpha * MyData.az + (1.0f - alpha) * filtered_az;
  MyData.angle_x = atan2f(filtered_ay, sqrtf(filtered_ax * filtered_ax + filtered_az * filtered_az)) * RAD_TO_DEG;
  MyData.angle_y = atan2f(filtered_ax, sqrtf(filtered_ay * filtered_ay + filtered_az * filtered_az)) * RAD_TO_DEG;
  MyData.angle_z = atan2f(sqrtf(filtered_ax * filtered_ax + filtered_ay * filtered_ay), filtered_az) * RAD_TO_DEG;
  
  // Send via BLE if connected
  if (deviceConnected) {
    if ((millis() - lastTime) > timerDelay) {
      // Pack all 14 float values into byte array (14 floats x 4 bytes = 56 bytes)
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
      
      sensorCharacteristic.setValue(data, 56);
      sensorCharacteristic.notify();
      
      lastTime = millis();
    }
  }
  
  // Print RAW data to Serial (for debugging)
  // static unsigned long lastPrint = 0;
  // if (millis() - lastPrint > 1000) {
  //   Serial.printf("[SENDER-RAW] Flex:[%.0f,%.0f,%.0f,%.0f,%.0f] IMU:[%.2f,%.2f,%.2f]\n",
  //     MyData.flex_raw[0], MyData.flex_raw[1], MyData.flex_raw[2], MyData.flex_raw[3], MyData.flex_raw[4],
  //     MyData.ax, MyData.ay, MyData.az
  //   );
  //   lastPrint = millis();
  // }
  
  delay(10);
}