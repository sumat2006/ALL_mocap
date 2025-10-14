#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include <esp_wifi.h>

// ──── WiFi Configuration ────────────────────────────────────────────────────────
const char* ssid = "YOUR_WIFI_SSID";          // เปลี่ยนเป็น WiFi ของคุณ
const char* password = "YOUR_WIFI_PASSWORD";   // เปลี่ยนเป็น Password ของคุณ

// ──── API Configuration ─────────────────────────────────────────────────────────
const char* serverUrl = "http://YOUR_SERVER_IP:8000/predict_hand";  // เปลี่ยนเป็น URL ของคุณ

// ──── UART Configuration ────────────────────────────────────────────────────────
#define UART_RX_PIN 16  // รับข้อมูลจาก main_hand GPIO17
#define UART_TX_PIN 17  // ไม่ได้ใช้จริง แต่ต้องมี
#define UART_BAUD 115200

// ──── LED Status (Optional) ─────────────────────────────────────────────────────
#define LED_BUILTIN 2  // Built-in LED for status indication
#define LED_BLINK_WIFI_CONNECTING 500   // Blink every 500ms when connecting WiFi
#define LED_BLINK_SENDING 100           // Fast blink when sending data

// ──── Statistics ────────────────────────────────────────────────────────────────
unsigned long httpSuccessCount = 0;
unsigned long httpFailCount = 0;
unsigned long lastStatsTime = 0;

// ──── Buffer for incoming UART data ─────────────────────────────────────────────
String uartBuffer = "";
const int UART_BUFFER_SIZE = 16384;  // 16KB buffer for JSON data

// ────────────────────────────────────────────────────────────────────────────────
// LED Control Functions
// ────────────────────────────────────────────────────────────────────────────────
void ledOn() {
  digitalWrite(LED_BUILTIN, HIGH);
}

void ledOff() {
  digitalWrite(LED_BUILTIN, LOW);
}

void ledBlink(int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    ledOn();
    delay(delayMs);
    ledOff();
    delay(delayMs);
  }
}

// ════════════════════════════════════════════════════════════════════════════════
// WiFi Setup
// ════════════════════════════════════════════════════════════════════════════════
void setupWiFi() {
  Serial.println();
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println(F("         📡 WiFi CONNECTION                             "));
  Serial.println(F("════════════════════════════════════════════════════════"));

  Serial.print("[WiFi] Connecting to: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  esp_wifi_set_max_tx_power(78);  // Max power for better range

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    ledBlink(1, 100);
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("[WiFi] ✓ Connected!");
    Serial.print("[WiFi] IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("[WiFi] Signal strength: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    ledOn();
    delay(1000);
    ledOff();
  } else {
    Serial.println();
    Serial.println("[WiFi] ✗ Connection failed!");
    Serial.println("[WiFi] Please check your WiFi credentials.");
    Serial.println("[WiFi] Device will restart in 10 seconds...");
    ledBlink(10, 500);
    ESP.restart();
  }
}

// ════════════════════════════════════════════════════════════════════════════════
// Check if URL is HTTPS
// ════════════════════════════════════════════════════════════════════════════════
bool isHttpsUrl(const char* url) {
  return (strncmp(url, "https://", 8) == 0);
}

// ════════════════════════════════════════════════════════════════════════════════
// Send HTTP/HTTPS POST Request
// ════════════════════════════════════════════════════════════════════════════════
bool sendHttpPostRequest(const String& jsonString) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[HTTP] ⚠️ WiFi not connected!");
    return false;
  }

  HTTPClient http;
  bool useHttps = isHttpsUrl(serverUrl);

  // สร้าง client ตามประเภท
  if (useHttps) {
    WiFiClientSecure *secureClient = new WiFiClientSecure();

    // Skip certificate verification for ngrok and self-signed certs
    // ⚠️ Not secure for production, but necessary for ngrok
    secureClient->setInsecure();

    Serial.println("[HTTPS] Using secure connection (certificate verification disabled)");
    http.begin(*secureClient, serverUrl);
  } else {
    WiFiClient *client = new WiFiClient();
    Serial.println("[HTTP] Using standard connection");
    http.begin(*client, serverUrl);
  }

  http.addHeader("Content-Type", "application/json");
  http.setTimeout(15000); // 15 seconds timeout

  Serial.printf("[HTTP] Sending %d bytes to server...\n", jsonString.length());
  Serial.printf("[HTTP] Target: %s\n", serverUrl);

  ledBlink(3, 50);  // Fast blink when sending

  int httpResponseCode = http.POST(jsonString);

  bool success = false;
  if (httpResponseCode > 0) {
    Serial.print("[HTTP] Response code: ");
    Serial.println(httpResponseCode);

    if (httpResponseCode == 200) {
      String response = http.getString();
      Serial.println("[HTTP] ✓ Success!");
      Serial.print("[HTTP] Response: ");
      Serial.println(response);
      httpSuccessCount++;
      success = true;
      ledOn();
      delay(50);
      ledOff();
    } else {
      Serial.printf("[HTTP] ⚠️ Unexpected status code: %d\n", httpResponseCode);
      httpFailCount++;
    }
  } else {
    Serial.print("[HTTP] ✗ Error code: ");
    Serial.println(httpResponseCode);
    Serial.print("[HTTP] Error: ");
    Serial.println(http.errorToString(httpResponseCode));
    httpFailCount++;
    ledBlink(5, 100);  // Error blink pattern
  }

  http.end();

  // แสดง statistics ทุก 10 requests
  if ((httpSuccessCount + httpFailCount) % 10 == 0) {
    float successRate = 0.0f;
    if ((httpSuccessCount + httpFailCount) > 0) {
      successRate = (float)httpSuccessCount / (httpSuccessCount + httpFailCount) * 100.0f;
    }
    Serial.printf("[Stats] HTTP Requests: %lu | Success: %lu (%.1f%%) | Failed: %lu\n",
                  httpSuccessCount + httpFailCount, httpSuccessCount, successRate, httpFailCount);
  }

  return success;
}

// ════════════════════════════════════════════════════════════════════════════════
// Process Incoming UART Data
// ════════════════════════════════════════════════════════════════════════════════
void processUartData() {
  while (Serial1.available()) {
    char c = Serial1.read();

    if (c == '\n') {
      // ได้ JSON string ครบแล้ว
      if (uartBuffer.length() > 0) {
        Serial.printf("[UART] Received %d bytes\n", uartBuffer.length());

        // ตรวจสอบว่าเป็น JSON ที่ถูกต้องหรือไม่
        StaticJsonDocument<256> testDoc;  // Small doc just for validation
        DeserializationError error = deserializeJson(testDoc, uartBuffer);

        if (error) {
          Serial.print("[UART] ⚠️ Invalid JSON: ");
          Serial.println(error.c_str());
          Serial.println("[UART] Data: " + uartBuffer.substring(0, 200) + "...");
        } else {
          // JSON ถูกต้อง ส่งไปยัง API
          Serial.println("[UART] ✓ Valid JSON received");

          // Retry logic: พยายามส่ง 3 ครั้ง
          bool sent = false;
          for (int retry = 0; retry < 3 && !sent; retry++) {
            if (retry > 0) {
              Serial.printf("[HTTP] Retry attempt %d/3...\n", retry + 1);
              delay(1000 * retry);  // Exponential backoff
            }
            sent = sendHttpPostRequest(uartBuffer);
          }

          if (!sent) {
            Serial.println("[HTTP] ✗ Failed to send after 3 attempts");
            ledBlink(10, 100);  // Error pattern
          }
        }

        // Clear buffer
        uartBuffer = "";
      }
    } else {
      // สะสม character
      if (uartBuffer.length() < UART_BUFFER_SIZE) {
        uartBuffer += c;
      } else {
        Serial.println("[UART] ⚠️ Buffer overflow! Clearing buffer.");
        uartBuffer = "";
      }
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════════
// Check WiFi Connection and Reconnect if Needed
// ════════════════════════════════════════════════════════════════════════════════
void checkWiFiConnection() {
  static unsigned long lastCheck = 0;
  if (millis() - lastCheck > 10000) {  // Check every 10 seconds
    lastCheck = millis();

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[WiFi] Connection lost! Reconnecting...");
      ledBlink(5, 200);
      setupWiFi();
    }
  }
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
      Serial.println("  stats      - Show statistics");
      Serial.println("  wifi       - Show WiFi info");
    } else if (command.equalsIgnoreCase("reset")) {
      Serial.println("[CMD] ➤ Rebooting...");
      delay(1000);
      ESP.restart();
    } else if (command.equalsIgnoreCase("stats")) {
      float successRate = 0.0f;
      if ((httpSuccessCount + httpFailCount) > 0) {
        successRate = (float)httpSuccessCount / (httpSuccessCount + httpFailCount) * 100.0f;
      }
      Serial.println("\n[Stats] ════════════════════════════════");
      Serial.printf("  Total Requests: %lu\n", httpSuccessCount + httpFailCount);
      Serial.printf("  Success: %lu (%.1f%%)\n", httpSuccessCount, successRate);
      Serial.printf("  Failed: %lu\n", httpFailCount);
      Serial.println("════════════════════════════════════════");
    } else if (command.equalsIgnoreCase("wifi")) {
      Serial.println("\n[WiFi] ════════════════════════════════");
      Serial.print("  Status: ");
      Serial.println(WiFi.status() == WL_CONNECTED ? "Connected ✓" : "Disconnected ✗");
      if (WiFi.status() == WL_CONNECTED) {
        Serial.print("  SSID: ");
        Serial.println(WiFi.SSID());
        Serial.print("  IP: ");
        Serial.println(WiFi.localIP());
        Serial.print("  Signal: ");
        Serial.print(WiFi.RSSI());
        Serial.println(" dBm");
      }
      Serial.println("════════════════════════════════════════");
    } else {
      Serial.printf("[CMD] Unknown command: '%s'\n", command.c_str());
    }
  }
}

// ════════════════════════════════════════════════════════════════════════════════
// SETUP
// ════════════════════════════════════════════════════════════════════════════════
void setup() {
  // Initialize LED
  pinMode(LED_BUILTIN, OUTPUT);
  ledOff();

  // Start Serial for debugging
  Serial.begin(9600);
  delay(2000);
  while (!Serial) { delay(10); }

  // Start UART for communication with main_hand
  Serial1.begin(UART_BAUD, SERIAL_8N1, UART_RX_PIN, UART_TX_PIN);
  delay(100);

  Serial.println();
  Serial.println(F("██╗░░██╗██╗░░░██╗██████╗░"));
  Serial.println(F("██║░░██║██║░░░██║██╔══██╗"));
  Serial.println(F("███████║██║░░░██║██████╦╝"));
  Serial.println(F("██╔══██║██║░░░██║██╔══██╗"));
  Serial.println(F("██║░░██║╚██████╔╝██████╦╝"));
  Serial.println(F("╚═╝░░╚═╝░╚═════╝░╚═════╝░"));
  Serial.println();
  Serial.println(F("┌────────────────────────────────────────────────────────┐"));
  Serial.println(F("│          SKB Co. | Data Collection                     │"));
  Serial.println(F("│       Device: ESP32S3 (Main Hub Edition)               │"));
  Serial.println(F("│       Protocol: UART RX + WiFi + HTTP POST             │"));
  Serial.println(F("└────────────────────────────────────────────────────────┘"));
  Serial.println();

  // Setup WiFi
  setupWiFi();

  Serial.println();
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println(F("         ⚡ UART INITIALIZATION                          "));
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.printf("[UART] RX Pin: GPIO%d, TX Pin: GPIO%d (not used)\n", UART_RX_PIN, UART_TX_PIN);
  Serial.printf("[UART] Baud Rate: %d\n", UART_BAUD);
  Serial.println("[UART] Ready to receive from main_hand");

  Serial.println();
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println(F("         ✅ SETUP COMPLETE - READY TO FORWARD!          "));
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println("[Info] Waiting for data from main_hand via UART");
  Serial.println("[Info] Data will be forwarded to API via HTTP POST");
  Serial.println("[Info] Server: " + String(serverUrl));
  Serial.println("[Info] Type 'help' for a list of commands.");
  Serial.println();

  uartBuffer.reserve(UART_BUFFER_SIZE);
  lastStatsTime = millis();

  ledBlink(3, 200);  // Startup complete signal
}

// ════════════════════════════════════════════════════════════════════════════════
// MAIN LOOP
// ════════════════════════════════════════════════════════════════════════════════
void loop() {
  checkCommand();
  checkWiFiConnection();
  processUartData();

  delay(1);  // Small delay to prevent watchdog issues
}
