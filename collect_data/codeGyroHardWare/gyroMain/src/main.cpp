#include <esp_now.h>
#include <esp_wifi.h>
#include <WiFi.h>
#include <Wire.h>
#include <Arduino.h>
#include "esp_task_wdt.h" 

#define WIFI_CHANNEL 1
#define SLAVE_TIMEOUT_MS 1000
#define ONE_G (9.807f)
// MAC Address of the Slave/Sender board
uint8_t sender_mac[] = {0xFC, 0x01, 0x2C, 0xD9, 0x3B, 0x5C};

// --- I2C Addresses ---
const uint8_t MUX_ADDRESS = 0x70;
const uint8_t SENSOR_ADDRESS = 0x6B;

// --- QMI8658C Register Addresses ---
const uint8_t QMI8658C_WHO_AM_I = 0x00;
const uint8_t QMI8658C_CTRL1 = 0x02;
const uint8_t QMI8658C_CTRL2 = 0x03;
const uint8_t QMI8658C_CTRL3 = 0x04;
const uint8_t QMI8658C_CTRL7 = 0x08;
const uint8_t QMI8658C_OUTPUT_REG_START = 0x35;
unsigned long last_slave_message_time = 0;
const int NUM_SENSORS = 5;

static const float alpha = 0.5f;
float filtered_ax = 0.0f, filtered_ay = 0.0f, filtered_az = 0.0f;

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
SensorData receivedData;

void RegisterPeer() {
    esp_now_peer_info_t peerInfo;
    memset(&peerInfo, 0, sizeof(peerInfo));
    memcpy(peerInfo.peer_addr, sender_mac, 6);
    peerInfo.channel = WIFI_CHANNEL;
    peerInfo.encrypt = false;
    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        Serial.println("Failed to add peer");
    }
}

void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  if (len == sizeof(receivedData)) {
    memcpy(&receivedData, incomingData, sizeof(receivedData));
    
    if(receivedData.slav_online){
      if(!localData.slav_online){
        Serial.println("\n[Main] Slave connected! Data logging started.");
      }
      localData.slav_online = true;
      last_slave_message_time = millis(); 
    } else {
      Serial.println("\n[Main] Slave reported offline. Clearing data.");
      memset(&receivedData, 0, sizeof(receivedData));
      localData.slav_online = false;
    }
  } else {
    Serial.println("Received packet with wrong size.");
  }
}

void InitEspNow() {
    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        return;
    }
    esp_now_register_recv_cb(OnDataRecv);
    RegisterPeer();
}

void selectMuxChannel(uint8_t bus) {
  if (bus > 7) return;
  Wire.beginTransmission(MUX_ADDRESS);
  Wire.write(1 << bus);
  uint8_t result = Wire.endTransmission();
  if (result != 0) {
    Serial.printf("[Error] MUX channel %d selection failed: %d\n", bus, result);
  }
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
  uint8_t result = Wire.endTransmission(false);
  if (result != 0) return false;
  
  Wire.requestFrom(SENSOR_ADDRESS, (uint8_t)1);
  if (!Wire.available() || Wire.read() != 0x05) {
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
  Serial.println("\n[Main] QMI8658C + TCA9548A Initialization");
  
  // Initialize I2C with timeout settings
  Wire.begin(15, 14);
  Wire.setTimeout(100); // 100ms timeout for I2C operations
  Wire.setClock(100000); // Set I2C clock to 100kHz

  // Configure watchdog timer - CRITICAL FOR PREVENTING CRASHES
  esp_task_wdt_init(10, true);
  esp_task_wdt_add(NULL);
  Serial.println("[Main] Watchdog configured (10s timeout)");

  memset(&localData, 0, sizeof(localData));
  memset(&receivedData, 0, sizeof(receivedData));

  for (int i = 0; i < NUM_SENSORS; i++) {
    esp_task_wdt_reset();
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

  Serial.println("\nInitialization complete. Waiting for slave...");
  WiFi.mode(WIFI_AP_STA);
  if (esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE) != ESP_OK) {
      Serial.println("Error setting WiFi channel");
      return;
  }
  InitEspNow();
}

void loop() {
  esp_task_wdt_reset();

  if (localData.slav_online && (millis() - last_slave_message_time > SLAVE_TIMEOUT_MS)){
    localData.slav_online = false;
    memset(&receivedData, 0, sizeof(receivedData));
    Serial.println("\n[Main] Slave connection lost (timeout).");
  }

  if (localData.slav_online) {
    bool all_local_sensors_ok = true;
    
    // Read sensors with timeout protection
    for (int i = 0; i < NUM_SENSORS; i++) {
      // Feed watchdog before each sensor read - PREVENTS TIMEOUT
      esp_task_wdt_reset();
      
      selectMuxChannel(i);
      uint8_t dataBuffer[12];
      
      Wire.beginTransmission(SENSOR_ADDRESS);
      Wire.write(QMI8658C_OUTPUT_REG_START);
      uint8_t transmit_result = Wire.endTransmission(false);
      
      // Check for I2C transmission errors
      if (transmit_result != 0) {
        Serial.printf("[Error] Sensor %d I2C transmit failed: %d\n", i, transmit_result);
        all_local_sensors_ok = false;
        continue; // Skip this sensor and continue with next
      }
      
      uint8_t bytes_received = Wire.requestFrom(SENSOR_ADDRESS, (uint8_t)12);
      
      if (bytes_received == 12 && Wire.available() == 12) {
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
            case 0: localData.ax1 = ax; localData.ay1 = ay; localData.az1 = az; 
                   localData.gx1 = gx; localData.gy1 = gy; localData.gz1 = gz; break;
            case 1: localData.ax2 = ax; localData.ay2 = ay; localData.az2 = az; 
                   localData.gx2 = gx; localData.gy2 = gy; localData.gz2 = gz; break;
            case 2: localData.ax3 = ax; localData.ay3 = ay; localData.az3 = az; 
                   localData.gx3 = gx; localData.gy3 = gy; localData.gz3 = gz; break;
            case 3: localData.ax4 = ax; localData.ay4 = ay; localData.az4 = az; 
                   localData.gx4 = gx; localData.gy4 = gy; localData.gz4 = gz; break;
            case 4: localData.ax5 = ax; localData.ay5 = ay; localData.az5 = az; 
                   localData.gx5 = gx; localData.gy5 = gy; localData.gz5 = gz; break;
        }
      } else {
        Serial.printf("[Error] Sensor %d data read failed. Expected 12 bytes, got %d\n", i, bytes_received);
        all_local_sensors_ok = false;
      }
      
      // Small delay between sensor reads to prevent I2C bus overload
      delay(5);
    }
    
    localData.isValid = all_local_sensors_ok;
    
    // Feed watchdog before potentially long snprintf operation
    esp_task_wdt_reset();
    
    static char outBuf[512];
    snprintf(outBuf, sizeof(outBuf),
        "%lu"
        ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
        ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
        ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
        ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
        ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
        ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
        ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
        ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
        ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
        ",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"
        ,
        millis(),
        localData.ax1, localData.ay1, localData.az1, localData.gx1, localData.gy1, localData.gz1,
        localData.ax2, localData.ay2, localData.az2, localData.gx2, localData.gy2, localData.gz2,
        localData.ax3, localData.ay3, localData.az3, localData.gx3, localData.gy3, localData.gz3,
        localData.ax4, localData.ay4, localData.az4, localData.gx4, localData.gy4, localData.gz4,
        localData.ax5, localData.ay5, localData.az5, localData.gx5, localData.gy5, localData.gz5,
        receivedData.ax1, receivedData.ay1, receivedData.az1, receivedData.gx1, receivedData.gy1, receivedData.gz1,
        receivedData.ax2, receivedData.ay2, receivedData.az2, receivedData.gx2, receivedData.gy2, receivedData.gz2,
        receivedData.ax3, receivedData.ay3, receivedData.az3, receivedData.gx3, receivedData.gy3, receivedData.gz3,
        receivedData.ax4, receivedData.ay4, receivedData.az4, receivedData.gx4, receivedData.gy4, receivedData.gz4,
        receivedData.ax5, receivedData.ay5, receivedData.az5, receivedData.gx5, receivedData.gy5, receivedData.gz5
      );
    Serial.printf("[Sensor] %s\n", outBuf);
  } else {
    Serial.print(".");
    delay(500);
  }
  
  // Ensure minimum loop time but allow task switching
  delay(10); 
}
  // "[timestamp]%lu"
  // ",[ax1]%.4f,[ay1]%.4f,[az1]%.4f,[gx1]%.4f,[gy1]%.4f,[gz1]%.4f"
  // ",[ax2]%.4f,[ay2]%.4f,[az2]%.4f,[gx2]%.4f,[gy2]%.4f,[gz2]%.4f"
  // ",[ax3]%.4f,[ay3]%.4f,[az3]%.4f,[gx3]%.4f,[gy3]%.4f,[gz3]%.4f"
  // ",[ax4]%.4f,[ay4]%.4f,[az4]%.4f,[gx4]%.4f,[gy4]%.4f,[gz4]%.4f"
  // ",[ax5]%.4f,[ay5]%.4f,[az5]%.4f,[gx5]%.4f,[gy5]%.4f,[gz5]%.4f"
  // ",[ax1_slav]%.4f,[ay1_slav]%.4f,[az1_slav]%.4f,[gx1_slav]%.4f,[gy1_slav]%.4f,[gz1_slav]%.4f"
  // ",[ax2_slav]%.4f,[ay2_slav]%.4f,[az2_slav]%.4f,[gx2_slav]%.4f,[gy2_slav]%.4f,[gz2_slav]%.4f"
  // ",[ax3_slav]%.4f,[ay3_slav]%.4f,[az3_slav]%.4f,[gx3_slav]%.4f,[gy3_slav]%.4f,[gz3_slav]%.4f"
  // ",[ax4_slav]%.4f,[ay4_slav]%.4f,[az4_slav]%.4f,[gx4_slav]%.4f,[gy4_slav]%.4f,[gz4_slav]%.4f"
  // ",[ax5_slav]%.4f,[ay5_sl