// SENDER CODE (e.g., for "imortal joe") - FINAL
#include <Arduino.h>
#include <Wire.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <WiFi.h>
#include <ArduinoJson.h>
#include <Adafruit_ADS1X15.h>
#include <QMI8658.h>

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
#define SLAVE_TIMEOUT_MS 2000

// ──── PERIPHERAL OBJECTS ───────────────────────────────────────────────────────
Adafruit_ADS1015 ads1015_1;
Adafruit_ADS1015 ads1015_2;
QMI8658          imu;

// ──── CALIBRATION & FILTER VARIABLES ───────────────────────────────────────────
float mean_acd_max[NUM_FLEX_SENSORS] = {1800.0f, 1800.0f, 1800.0f, 1800.0f, 1800.0f};
float mean_acd_low[NUM_FLEX_SENSORS] = {900.0f, 900.0f, 900.0f, 900.0f, 900.0f};
float low_value = 0.0f;
float high_value = 1000.0f;
float flex_raw_value[NUM_FLEX_SENSORS];
unsigned long start_timestamp = 0;

// Simple low-pass filter state
static const float alpha = 0.5f;
float filtered_ax = 0.0f, filtered_ay = 0.0f, filtered_az = 0.0f;
float filtered_ax_slav = 0.0f, filtered_ay_slav = 0.0f, filtered_az_slav = 0.0f;

// ──── PROTOTYPES ────────────────────────────────────────────────────────────────
void checkCommand();
void readAllFlexSensors(float* raw_values);
void flexCalibration();
void ADS1015_INITIALIZATION();
void QMI8658_INITIALIZATION();
void resetCsvTimestamp();
void printCountdown(int seconds, const char* message_prefix = "[Arduino]");

// Combined data struct for master and slave
struct SensorData {
  float ax_slav, ay_slav, az_slav, gx_slav, gy_slav, gz_slav;
  float angle_x_slav, angle_y_slav, angle_z_slav;
  float flex_slav[NUM_FLEX_SENSORS];
  float ax, ay, az, gx, gy, gz;
  float angle_x, angle_y, angle_z;
  float flex[NUM_FLEX_SENSORS];
  volatile bool slav_online = false;
};
SensorData MyData;

// ──── ESP-NOW & COMMUNICATION ───────────────────────────────────────────────────
uint8_t slaveAddress[] = {0xfc, 0x01, 0x2c, 0xd9, 0x3d, 0x90}; // Address of "warboy"
String payload;
JsonDocument jsonDoc;
unsigned long last_slave_message_time = 0;


// Callback when data is sent (for commands)
void OnDataSent(const uint8_t* mac_addr, esp_now_send_status_t status) {
    if (status != ESP_NOW_SEND_SUCCESS) {
        Serial.println("[Imotan Joe] Command delivery failed");
    }
}

// Function to register the slave as a peer
void RegisterPeer() {
    esp_now_peer_info_t peerInfo;
    memset(&peerInfo, 0, sizeof(peerInfo));
    memcpy(peerInfo.peer_addr, slaveAddress, 6);
    peerInfo.channel = WIFI_CHANNEL;
    peerInfo.encrypt = false;
    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        Serial.println("Failed to add peer");
    }
}

// Callback for received data from slave
void OnMessageReceived(const uint8_t* mac_addr, const uint8_t* incomingData, int len) {
    JsonDocument doc;
    DeserializationError error = deserializeJson(doc, incomingData, len);
    if (error) {
        return; 
    }

    // DISPATCHER: Check if packet is a log message or sensor data
    if (doc.containsKey("log_msg")) {
        // It's a log message from the slave. Print it.
        // The slave sends newline chars, so we use print() not println().
        const char* logMsg = doc["log_msg"];
        Serial.print(logMsg); 
        
    } else if (doc.containsKey("ax")) {
        // It's a sensor data packet. Process it as before.
        MyData.ax_slav = doc["ax"]; MyData.ay_slav = doc["ay"]; MyData.az_slav = doc["az"];
        MyData.gx_slav = doc["gx"]; MyData.gy_slav = doc["gy"]; MyData.gz_slav = doc["gz"];
        for (int i = 0; i < NUM_FLEX_SENSORS; i++) {
            String key = "flex_" + String(i);
            MyData.flex_slav[i] = doc[key];
        }
        
        // Handle the heartbeat
        if (doc["slav_online"] == 1) {
            if (!MyData.slav_online) { 
              Serial.println("\n[Imotan Joe] Warboy connected! Data logging started.");
            }
            MyData.slav_online = true;
            last_slave_message_time = millis(); 
        }
    }
}


void InitEspNow() {
    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        return;
    }
    esp_now_register_recv_cb(OnMessageReceived);
    esp_now_register_send_cb(OnDataSent);
    RegisterPeer();
}

void SendCommand(int command_code) {
    jsonDoc.clear();
    jsonDoc["imotan_command"] = command_code;
    serializeJson(jsonDoc, payload);
    esp_now_send(slaveAddress, (uint8_t*)payload.c_str(), payload.length());
    payload.clear();
}

void setup() {
    Wire.begin(SENSOR_SDA, SENSOR_SCL);
    Serial.begin(115200);
    while (!Serial) { delay(10); }

    Serial.println();
    Serial.println(F("░██████╗██╗░░██╗██████╗░"));
    Serial.println(F("██╔════╝██║░██╔╝██╔══██╗"));
    Serial.println(F("╚█████╗░█████═╝░██████╦╝"));
    Serial.println(F("░╚═══██╗██╔═██╗░██╔══██╗"));
    Serial.println(F("██████╔╝██║░╚██╗██████╦╝"));
    Serial.println(F("╚═════╝░╚═╝░░╚═╝╚═════╝░"));
    Serial.println();

    Serial.print("[Arduino] "); Serial.println(F("┌──────────────────────────────────────────────────────────┐"));
    Serial.print("[Arduino] "); Serial.println(F("│                SKB Co. | Data Collection                 │"));
    Serial.print("[Arduino] "); Serial.println(F("│           Device: ESP32S3 SKB (Optimized Edition)        │"));
    Serial.print("[Arduino] "); Serial.println(F("│        Status: Initialized and Ready to Collect          │"));
    Serial.print("[Arduino] "); Serial.println(F("└──────────────────────────────────────────────────────────┘"));
    Serial.println();

    Serial.println("Imortal Joe ESP32 starting...");
    
    ADS1015_INITIALIZATION();
    QMI8658_INITIALIZATION();

    WiFi.mode(WIFI_AP_STA);
    if (esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE) != ESP_OK) {
        Serial.println("Error setting WiFi channel");
        return;
    }
    InitEspNow();
    resetCsvTimestamp();
    Serial.println("\n[Imotan Joe] Setup complete. Waiting for Warboy connection...");
    Serial.println("[Imotan Joe] Type 'help' for a list of commands.");
}

void loop() {
    checkCommand(); 

    // Robust slave connection check using a timeout
    if (MyData.slav_online && (millis() - last_slave_message_time > SLAVE_TIMEOUT_MS)) {
        MyData.slav_online = false;
        Serial.println("\n[Imotan Joe] Warboy connection lost (timeout).");
    }

    if (MyData.slav_online) {
        readAllFlexSensors(flex_raw_value);
        for (int i = 0; i < NUM_FLEX_SENSORS; i++) {
            if (mean_acd_max[i] != mean_acd_low[i]) {
                MyData.flex[i] = ((flex_raw_value[i] - mean_acd_low[i]) * (high_value - low_value) / (mean_acd_max[i] - mean_acd_low[i])) + low_value;
            } else { MyData.flex[i] = low_value; }
            MyData.flex[i] = constrain(MyData.flex[i], low_value, high_value);
        }

        QMI8658_Data d;
        if (imu.readSensorData(d)) {
            MyData.ax = d.accelX; MyData.ay = d.accelY; MyData.az = d.accelZ;
            MyData.gx = d.gyroX; MyData.gy = d.gyroY; MyData.gz = d.gyroZ;
        }

        filtered_ax = alpha * MyData.ax + (1.0f - alpha) * filtered_ax;
        filtered_ay = alpha * MyData.ay + (1.0f - alpha) * filtered_ay;
        filtered_az = alpha * MyData.az + (1.0f - alpha) * filtered_az;
        filtered_ax_slav = alpha * MyData.ax_slav + (1.0f - alpha) * filtered_ax_slav;
        filtered_ay_slav = alpha * MyData.ay_slav + (1.0f - alpha) * filtered_ay_slav;
        filtered_az_slav = alpha * MyData.az_slav + (1.0f - alpha) * filtered_az_slav;

        MyData.angle_x = atan2f(filtered_ay, sqrtf(filtered_ax * filtered_ax + filtered_az * filtered_az)) * RAD_TO_DEG;
        MyData.angle_y = atan2f(filtered_ax, sqrtf(filtered_ay * filtered_ay + filtered_az * filtered_az)) * RAD_TO_DEG;
        MyData.angle_z = atan2f(sqrtf(filtered_ax * filtered_ax + filtered_ay * filtered_ay), filtered_az) * RAD_TO_DEG;
        MyData.angle_x_slav = atan2f(filtered_ay_slav, sqrtf(filtered_ax_slav * filtered_ax_slav + filtered_az_slav * filtered_az_slav)) * RAD_TO_DEG;
        MyData.angle_y_slav = atan2f(filtered_ax_slav, sqrtf(filtered_ay_slav * filtered_ay_slav + filtered_az_slav * filtered_az_slav)) * RAD_TO_DEG;
        MyData.angle_z_slav = atan2f(sqrtf(filtered_ax_slav * filtered_ax_slav + filtered_ay_slav * filtered_ay_slav), filtered_az_slav) * RAD_TO_DEG;

        static char outBuf[1024];
        snprintf(outBuf, sizeof(outBuf),
          "%lu,"
          "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
          "%.2f,%.2f,%.2f,%.2f,%.2f,"
          "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
          "%.2f,%.2f,%.2f,%.2f,%.2f",
          millis() - start_timestamp,
          MyData.ax_slav, MyData.ay_slav, MyData.az_slav, MyData.gx_slav, MyData.gy_slav, MyData.gz_slav, MyData.angle_x_slav, MyData.angle_y_slav, MyData.angle_z_slav,
          MyData.flex_slav[0], MyData.flex_slav[1], MyData.flex_slav[2], MyData.flex_slav[3], MyData.flex_slav[4],
          MyData.ax, MyData.ay, MyData.az, MyData.gx, MyData.gy, MyData.gz, MyData.angle_x, MyData.angle_y, MyData.angle_z,
          MyData.flex[0], MyData.flex[1], MyData.flex[2], MyData.flex[3], MyData.flex[4]
        );
        Serial.printf("[Sensor] %s\n", outBuf);
        delay(50);
    } else {
        delay(1000); 
    }
}

void checkCommand() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        Serial.printf("[CMD] Received: '%s'\n", command.c_str());

        if (command.equalsIgnoreCase("help")) {
            Serial.println("\n[CMD] Available commands:");
            Serial.println("  help       - Show this help message");
            Serial.println("  reset      - Reboot devices");
            Serial.println("  flex_cal   - Start flex sensor calibration on both devices");
            Serial.println("  reset_time - Reset the CSV time baseline on this device");
        } else if (command.equalsIgnoreCase("reset")) {
            // Serial.println("[CMD] ➤ Sending reboot command to Warboy and rebooting self...");
            // SendCommand(3); 
            // delay(1000);
            ESP.restart();
        } else if (command.equalsIgnoreCase("flex_cal")) {
            Serial.println("[CMD] ➤ Initiating Flex Sensor Calibration on both units.");
            SendCommand(1);
            flexCalibration();
        } else if (command.equalsIgnoreCase("reset_time")) {
            Serial.println("[CMD] ➤ Resetting CSV timestamp.");
            resetCsvTimestamp();
        } else {
            Serial.printf("[CMD] Unknown command: '%s'\n", command.c_str());
        }
    }
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
    Serial.println();
    Serial.println(F("[Imotan Joe] ┌──────────────────────────────────────────────────┐"));
    Serial.println(F("[Imotan Joe] │           🎯 Starting Flex calibration           │"));
    Serial.println(F("[Imotan Joe] └──────────────────────────────────────────────────┘"));
    delay(1000);
    Serial.println(F("[Imotan Joe] Follow instructions for best calibration results."));
    printCountdown(3, "[Imotan Joe] Cal starting in");

    float temp_sensor_values[NUM_FLEX_SENSORS];
    const char* phases[] = {"มือปกติ (normal hand / straight)", "กำหมัด (fist / bent)"};
    for (int phase = 0; phase < 2; phase++) {
        Serial.printf("[Imotan Joe] %s - Hold position. Starting in 5 sec...\n", phases[phase]);
        printCountdown(5, "[Imotan Joe]");
        Serial.printf("[Imotan Joe] 📈 Collecting %d samples...\n", CALIBRATION_SAMPLES);
        Serial.print("[Imotan Joe] ");
        for (int j = 0; j < CALIBRATION_SAMPLES; j++) {
            readAllFlexSensors(temp_sensor_values);
            for (int k = 0; k < NUM_FLEX_SENSORS; k++) {
                if (phase == 0) { mean_acd_low[k] += temp_sensor_values[k]; } 
                else { mean_acd_max[k] += temp_sensor_values[k]; }
            }
            if ((j % 50) == 0 && j > 0) Serial.print(".");
            delay(5);
        }
        Serial.println(" Done.");

        for (int k = 0; k < NUM_FLEX_SENSORS; k++) {
            if (phase == 0) { mean_acd_low[k] /= CALIBRATION_SAMPLES; } 
            else { mean_acd_max[k] /= CALIBRATION_SAMPLES; }
        }
    }
    Serial.println(F("[Imotan Joe] ✅ Flex Sensor Calibration complete."));
    for(int i=0; i<NUM_FLEX_SENSORS; ++i) {
        Serial.printf("[Imotan Joe] Sensor %d: Low Avg: %.2f, High Avg: %.2f\n", i, mean_acd_low[i], mean_acd_max[i]);
    }
}


void printCountdown(int seconds, const char* message_prefix) {
    for (int i = seconds; i > 0; i--) {
        Serial.printf("%s %d...\n", message_prefix, i);
        delay(1000);
    }
}

void ADS1015_INITIALIZATION(){
  Serial.println();
  Serial.print("[Imotan Joe] Initializing ADS1015 #1 (0x48)...");
  if (!ads1015_1.begin(ADS1015_1_ADDRESS)) { Serial.println("❌ FAILED!"); while (1) delay(1000); }
  Serial.println("✅ OK.");
  Serial.print("[Imotan Joe] Initializing ADS1015 #2 (0x49)...");
  if (!ads1015_2.begin(ADS1015_2_ADDRESS)) { Serial.println("❌ FAILED!"); while (1) delay(1000); }
  Serial.println("✅ OK.");
}

void QMI8658_INITIALIZATION(){
  Serial.println();
  Serial.print("[Imotan Joe] Initializing QMI8658 IMU...");
  if (!imu.begin(SENSOR_SDA, SENSOR_SCL)) { Serial.println("❌ FAILED!"); while (1) delay(1000); }
  Serial.println("✅ OK.");
  imu.setAccelRange(QMI8658_ACCEL_RANGE_2G);
  imu.setAccelODR(QMI8658_ACCEL_ODR_1000HZ);
  imu.setGyroRange(QMI8658_GYRO_RANGE_256DPS);
  imu.setGyroODR(QMI8658_GYRO_ODR_1000HZ);
  imu.setAccelUnit_mps2(true);
  imu.setGyroUnit_rads(true);
  imu.enableSensors(QMI8658_ENABLE_ACCEL | QMI8658_ENABLE_GYRO);
}

void resetCsvTimestamp(){
  Serial.println("[Imotan Joe] Resetting CSV Timestamp...");
  start_timestamp = millis();
}