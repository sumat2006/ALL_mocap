#include <esp_now.h>
#include <esp_wifi.h>
#include <WiFi.h>
#include <Wire.h>
// #include <QMI8658.h>
#include <Arduino.h>

#define WIFI_CHANNEL 1
#define ONE_G (9.807f)

// MAC address of the Main/Receiver board
uint8_t receiver_mac[] = {0xB8, 0xF8, 0x62, 0xE7, 0x09, 0xE0};

// --- I2C Addresses ---
const uint8_t MUX_ADDRESS = 0x70;
const uint8_t SENSOR_ADDRESS = 0x6B;

// --- QMI8658C Register Addresses ---
// QMI8658          imu; // For 6th sensor
const uint8_t QMI8658C_WHO_AM_I = 0x00;
const uint8_t QMI8658C_CTRL1 = 0x02;
const uint8_t QMI8658C_CTRL2 = 0x03;
const uint8_t QMI8658C_CTRL3 = 0x04;
const uint8_t QMI8658C_CTRL7 = 0x08;
const uint8_t QMI8658C_OUTPUT_REG_START = 0x35;

const int NUM_SENSORS = 5;

// Simple low-pass filter state
static const float alpha = 0.5f;
float filtered_ax = 0.0f, filtered_ay = 0.0f, filtered_az = 0.0f;

// This struct holds a complete sensor dataset for one board
struct SensorData {
  float ax1, ay1, az1, gx1, gy1, gz1;
  float ax2, ay2, az2, gx2, gy2, gz2;
  float ax3, ay3, az3, gx3, gy3, gz3;
  float ax4, ay4, az4, gx4, gy4, gz4;
  float ax5, ay5, az5, gx5, gy5, gz5;
  bool isValid = false;
  volatile bool slav_online = false;
};

SensorData localData;

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  if (status != ESP_NOW_SEND_SUCCESS) {
    Serial.println("Packet delivery failed");
  }
}

void RegisterPeer() {
    esp_now_peer_info_t peerInfo;
    memset(&peerInfo, 0, sizeof(peerInfo));
    memcpy(peerInfo.peer_addr, receiver_mac, 6);
    peerInfo.channel = WIFI_CHANNEL;
    peerInfo.encrypt = false;
    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        Serial.println("Failed to add peer");
    }
}

void InitEspNow() {
    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        return;
    }
    esp_now_register_send_cb(OnDataSent);
    RegisterPeer();
}

void selectMuxChannel(uint8_t bus) {
  if (bus > 7) return;
  Wire.beginTransmission(MUX_ADDRESS);
  Wire.write(1 << bus);
  Wire.endTransmission();
}

void writeRegister(uint8_t reg, uint8_t data) {
  Wire.beginTransmission(SENSOR_ADDRESS);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission();
}

bool initQMI8658C() {
  Wire.beginTransmission(SENSOR_ADDRESS);
  Wire.write(QMI8658C_WHO_AM_I);
  Wire.endTransmission(false);
  Wire.requestFrom(SENSOR_ADDRESS, (uint8_t)1);
  if (Wire.read() != 0x05) {
    return false;
  }
  writeRegister(QMI8658C_CTRL1, 0x40);
  writeRegister(QMI8658C_CTRL2, 0b00100101); // Accel: +/- 8g
  writeRegister(QMI8658C_CTRL3, 0b01100101); // Gyro: +/- 2048dps
  writeRegister(QMI8658C_CTRL7, 0b00000011);
  return true;
}

void setup() {
  Serial.begin(1843200);
  while (!Serial);
  Serial.println("\n[Slave] QMI8658C + TCA9548A Initialization");
  Wire.begin(15, 14);

  for (int i = 0; i < NUM_SENSORS; i++) {
    selectMuxChannel(i);
    Serial.print("Initializing sensor on MUX channel ");
    Serial.print(i);
    if (initQMI8658C()) {
      Serial.println("... Success!");
    } else {
      Serial.println("... Failed!");
    }
    delay(100);
  }
  
  WiFi.mode(WIFI_AP_STA);
  if (esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE) != ESP_OK) {
      Serial.println("Error setting WiFi channel");
      return;
  }
  InitEspNow();
}

void loop() {
  bool all_sensors_ok = true;

  for (int i = 0; i < NUM_SENSORS; i++) {
    selectMuxChannel(i);

    uint8_t dataBuffer[12];
    Wire.beginTransmission(SENSOR_ADDRESS);
    Wire.write(QMI8658C_OUTPUT_REG_START);
    Wire.endTransmission(false);
    Wire.requestFrom(SENSOR_ADDRESS, (uint8_t)12);

    if (Wire.available() == 12) {
      for (int k = 0; k < 12; k++) {
        dataBuffer[k] = Wire.read();
      }

      int16_t rawAccX = (dataBuffer[1] << 8) | dataBuffer[0];
      int16_t rawAccY = (dataBuffer[3] << 8) | dataBuffer[2];
      int16_t rawAccZ = (dataBuffer[5] << 8) | dataBuffer[4];
      int16_t rawGyroX = (dataBuffer[7] << 8) | dataBuffer[6];
      int16_t rawGyroY = (dataBuffer[9] << 8) | dataBuffer[8];
      int16_t rawGyroZ = (dataBuffer[11] << 8) | dataBuffer[10];
      
      float ax = (float)(rawAccX * ONE_G) / 4096.0f;
      float ay = (float)(rawAccY * ONE_G) / 4096.0f;
      float az = (float)(rawAccZ * ONE_G) / 4096.0f;
      float gx = (float)(rawGyroX * 0.01745f) / 16.0f;
      float gy = (float)(rawGyroY * 0.01745f) / 16.0f;
      float gz = (float)(rawGyroZ * 0.01745f) / 16.0f;

      switch(i) {
        case 0: localData.ax1 = ax; localData.ay1 = ay; localData.az1 = az; localData.gx1 = gx; localData.gy1 = gy; localData.gz1 = gz; break;
        case 1: localData.ax2 = ax; localData.ay2 = ay; localData.az2 = az; localData.gx2 = gx; localData.gy2 = gy; localData.gz2 = gz; break;
        case 2: localData.ax3 = ax; localData.ay3 = ay; localData.az3 = az; localData.gx3 = gx; localData.gy3 = gy; localData.gz3 = gz; break;
        case 3: localData.ax4 = ax; localData.ay4 = ay; localData.az4 = az; localData.gx4 = gx; localData.gy4 = gy; localData.gz4 = gz; break;
        case 4: localData.ax5 = ax; localData.ay5 = ay; localData.az5 = az; localData.gx5 = gx; localData.gy5 = gy; localData.gz5 = gz; break;
      }
    } else {
      all_sensors_ok = false;
    }
  }
  localData.isValid = all_sensors_ok;
  localData.slav_online = true;
  
  esp_now_send(receiver_mac, (uint8_t *) &localData, sizeof(localData));
  
  delay(10); 
}