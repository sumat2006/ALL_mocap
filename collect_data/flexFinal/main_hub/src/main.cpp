#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
#include <esp_wifi.h>

// ──── WiFi Configuration ────────────────────────────────────────────────────────
const char* ssid = "xxxxxxxxxx_2.4G";          // เปลี่ยนเป็น WiFi ของคุณ
const char* password = "xxxxxxxxxx";           // เปลี่ยนเป็น Password ของคุณ

// ──── API Configuration ─────────────────────────────────────────────────────────
const char* serverUrl = "https://a6751c7cdec1.ngrok-free.app/predict_hand";  // เปลี่ยนเป็น URL ของคุณ

// ──── Hardware UART Configuration ───────────────────────────────────────────────
#define RXD_PIN 18  // รับข้อมูลจาก main_hand GPIO18 (main_hand TX -> hub RX)
#define TXD_PIN 17  // ไม่ได้ใช้จริง
#define UART_BAUD 230400  // Hardware UART รองรับ baud rate สูง

// ──── Queue Configuration ───────────────────────────────────────────────────────
#define MAX_QUEUE_SIZE 5           // เก็บได้สูงสุด 5 payloads (~35KB RAM)
#define MAX_PAYLOAD_SIZE 8192      // ขนาด payload ต่อ 1 item (8KB)

// ──── LED Status (Optional) ─────────────────────────────────────────────────────
#define LED_BUILTIN 2  // Built-in LED for status indication

// ──── Circular Queue Structure ─────────────────────────────────────────────────
struct PayloadQueue {
  String items[MAX_QUEUE_SIZE];
  int head;         // ตำแหน่งที่จะ dequeue
  int tail;         // ตำแหน่งที่จะ enqueue
  int count;        // จำนวน items ใน queue

  PayloadQueue() : head(0), tail(0), count(0) {}

  bool isEmpty() {
    return count == 0;
  }

  bool isFull() {
    return count >= MAX_QUEUE_SIZE;
  }

  bool enqueue(const String& payload) {
    if (isFull()) {
      return false;  // Queue full
    }

    items[tail] = payload;
    tail = (tail + 1) % MAX_QUEUE_SIZE;
    count++;
    return true;
  }

  String dequeue() {
    if (isEmpty()) {
      return "";
    }

    String payload = items[head];
    items[head] = "";  // Clear to free memory
    head = (head + 1) % MAX_QUEUE_SIZE;
    count--;
    return payload;
  }

  String peek() {
    if (isEmpty()) {
      return "";
    }
    return items[head];
  }
};

// ──── Global Variables ──────────────────────────────────────────────────────────
PayloadQueue sendQueue;
String uartBuffer = "";
const int UART_BUFFER_SIZE = 16384;  // 16KB buffer for incoming UART data

// Hardware UART (Serial1) - no need to declare, it's built-in

// Statistics
unsigned long httpSuccessCount = 0;
unsigned long httpFailCount = 0;
unsigned long queueDropCount = 0;
unsigned long totalReceivedCount = 0;
unsigned long lastStatsTime = 0;

// State management
bool isSending = false;
unsigned long lastSendAttempt = 0;
int currentRetryCount = 0;
const int MAX_RETRIES = 3;

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
// Send HTTP/HTTPS POST Request (Non-blocking attempt)
// ════════════════════════════════════════════════════════════════════════════════
bool sendHttpPostRequest(const String& jsonString) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[HTTP] ⚠️ WiFi not connected!");
    return false;
  }

  HTTPClient http;
  bool useHttps = isHttpsUrl(serverUrl);
  bool success = false;

  // สร้าง client ตามประเภท (ใช้ stack allocation แทน heap)
  if (useHttps) {
    WiFiClientSecure secureClient;
    secureClient.setInsecure();

    if (!http.begin(secureClient, serverUrl)) {
      Serial.println("[HTTP] ✗ Failed to begin HTTPS connection");
      return false;
    }

    http.addHeader("Content-Type", "application/json");
    http.setTimeout(15000); // 15 seconds timeout

    Serial.printf("[HTTP] Sending %d bytes (Queue: %d/%d)...\n",
                  jsonString.length(), sendQueue.count, MAX_QUEUE_SIZE);

    ledBlink(2, 30);  // Quick blink when sending

    int httpResponseCode = http.POST(jsonString);

    if (httpResponseCode > 0) {
      if (httpResponseCode == 200) {
        String response = http.getString();
        Serial.printf("[HTTP] ✓ Success! Response: %s\n", response.c_str());
        httpSuccessCount++;
        success = true;
        ledOn();
        delay(50);
        ledOff();
      } else {
        Serial.printf("[HTTP] ⚠️ Status %d\n", httpResponseCode);
        httpFailCount++;
      }
    } else {
      Serial.printf("[HTTP] ✗ Error: %s\n", http.errorToString(httpResponseCode).c_str());
      httpFailCount++;
    }

    http.end();
    // secureClient จะถูก destroy อัตโนมัติ (stack cleanup)

  } else {
    WiFiClient client;

    if (!http.begin(client, serverUrl)) {
      Serial.println("[HTTP] ✗ Failed to begin HTTP connection");
      return false;
    }

    http.addHeader("Content-Type", "application/json");
    http.setTimeout(15000);

    Serial.printf("[HTTP] Sending %d bytes (Queue: %d/%d)...\n",
                  jsonString.length(), sendQueue.count, MAX_QUEUE_SIZE);

    ledBlink(2, 30);

    int httpResponseCode = http.POST(jsonString);

    if (httpResponseCode > 0) {
      if (httpResponseCode == 200) {
        String response = http.getString();
        Serial.printf("[HTTP] ✓ Success! Response: %s\n", response.c_str());
        httpSuccessCount++;
        success = true;
        ledOn();
        delay(50);
        ledOff();
      } else {
        Serial.printf("[HTTP] ⚠️ Status %d\n", httpResponseCode);
        httpFailCount++;
      }
    } else {
      Serial.printf("[HTTP] ✗ Error: %s\n", http.errorToString(httpResponseCode).c_str());
      httpFailCount++;
    }

    http.end();
    // client จะถูก destroy อัตโนมัติ (stack cleanup)
  }

  // แสดง statistics ทุก 10 requests
  if ((httpSuccessCount + httpFailCount) % 10 == 0 && (httpSuccessCount + httpFailCount) > 0) {
    float successRate = (float)httpSuccessCount / (httpSuccessCount + httpFailCount) * 100.0f;
    Serial.printf("[Stats] Sent: %lu | Success: %lu (%.1f%%) | Failed: %lu | Dropped: %lu\n",
                  httpSuccessCount + httpFailCount, httpSuccessCount, successRate,
                  httpFailCount, queueDropCount);
  }

  return success;
}

// ════════════════════════════════════════════════════════════════════════════════
// Process Queue (Background Sender)
// ════════════════════════════════════════════════════════════════════════════════
void processQueue() {
  // ถ้ากำลังส่งอยู่ ข้ามไป
  if (isSending) {
    return;
  }

  // ถ้า queue ว่าง ไม่ต้องทำอะไร
  if (sendQueue.isEmpty()) {
    return;
  }

  // เริ่มส่ง
  isSending = true;
  String payload = sendQueue.peek();  // ดูแต่ยังไม่เอาออก

  bool sent = sendHttpPostRequest(payload);

  if (sent) {
    // สำเร็จ - เอาออกจาก queue
    sendQueue.dequeue();
    currentRetryCount = 0;
  } else {
    // ล้มเหลว - retry
    currentRetryCount++;

    if (currentRetryCount >= MAX_RETRIES) {
      // เกิน retry limit - drop และส่งต่อไป
      Serial.printf("[Queue] ⚠️ Max retries reached, dropping payload\n");
      sendQueue.dequeue();
      queueDropCount++;
      currentRetryCount = 0;
      ledBlink(5, 100);  // Error pattern
    } else {
      // รอก่อน retry
      Serial.printf("[Queue] Retry %d/%d in 1s...\n", currentRetryCount, MAX_RETRIES);
      delay(1000);
    }
  }

  isSending = false;
}

// ════════════════════════════════════════════════════════════════════════════════
// Process Incoming Hardware UART Data (Non-blocking)
// ════════════════════════════════════════════════════════════════════════════════
void processUartData() {
  while (Serial1.available()) {
    char c = Serial1.read();

    if (c == '\n') {
      // ได้ JSON string ครบแล้ว
      if (uartBuffer.length() > 0) {
        totalReceivedCount++;
        Serial.printf("[UART] Received payload #%lu (%d bytes)\n",
                      totalReceivedCount, uartBuffer.length());

        // ตรวจสอบ JSON format (แค่ครั้งแรก)
        static int debugCount = 0;
        if (debugCount < 1) {
          Serial.printf("[DEBUG] First char: '%c' (0x%02X)\n", uartBuffer.charAt(0), (uint8_t)uartBuffer.charAt(0));
          Serial.printf("[DEBUG] Last char: '%c' (0x%02X)\n",
                        uartBuffer.charAt(uartBuffer.length() - 1),
                        (uint8_t)uartBuffer.charAt(uartBuffer.length() - 1));
          debugCount++;
        }

        // Format ถูกต้อง - ใส่เข้า queue
        if (sendQueue.isFull()) {
          // Queue เต็ม - drop oldest
          String dropped = sendQueue.dequeue();
          queueDropCount++;
          Serial.println("[Queue] ⚠️ Queue full! Dropping oldest payload.");
          ledBlink(3, 50);
        }

        // Enqueue
        if (sendQueue.enqueue(uartBuffer)) {
          Serial.printf("[Queue] ✓ Enqueued (Queue: %d/%d)\n",
                        sendQueue.count, MAX_QUEUE_SIZE);
        } else {
          Serial.println("[Queue] ✗ Enqueue failed!");
          queueDropCount++;
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
        queueDropCount++;
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
      Serial.println("  queue      - Show queue status");
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
      float dropRate = 0.0f;
      if (totalReceivedCount > 0) {
        dropRate = (float)queueDropCount / totalReceivedCount * 100.0f;
      }
      Serial.println("\n[Stats] ════════════════════════════════════════");
      Serial.printf("  Total Received:    %lu\n", totalReceivedCount);
      Serial.printf("  HTTP Success:      %lu (%.1f%%)\n", httpSuccessCount, successRate);
      Serial.printf("  HTTP Failed:       %lu\n", httpFailCount);
      Serial.printf("  Dropped:           %lu (%.1f%%)\n", queueDropCount, dropRate);
      Serial.printf("  Queue Size:        %d/%d\n", sendQueue.count, MAX_QUEUE_SIZE);
      Serial.println("═══════════════════════════════════════════════");
    } else if (command.equalsIgnoreCase("queue")) {
      Serial.println("\n[Queue] ════════════════════════════════════════");
      Serial.printf("  Current Size:      %d/%d\n", sendQueue.count, MAX_QUEUE_SIZE);
      Serial.printf("  Head:              %d\n", sendQueue.head);
      Serial.printf("  Tail:              %d\n", sendQueue.tail);
      Serial.printf("  Is Full:           %s\n", sendQueue.isFull() ? "Yes" : "No");
      Serial.printf("  Is Empty:          %s\n", sendQueue.isEmpty() ? "Yes" : "No");
      Serial.printf("  Is Sending:        %s\n", isSending ? "Yes" : "No");
      Serial.println("═══════════════════════════════════════════════");
    } else if (command.equalsIgnoreCase("wifi")) {
      Serial.println("\n[WiFi] ═════════════════════════════════════════");
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
      Serial.println("═══════════════════════════════════════════════");
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

  // Start Serial for debugging (ต้องเร็วกว่า SoftwareSerial)
  Serial.begin(115200);
  delay(1000);  // รอให้ Serial พร้อม

  // รอ Serial Monitor สูงสุด 3 วินาที (ถ้าไม่เสียบ USB ก็ข้ามไป)
  unsigned long serialStart = millis();
  while (!Serial && (millis() - serialStart < 3000)) {
    delay(10);
  }

  // Debug: แสดงว่า boot สำเร็จ
  Serial.println("\n\n========== BOOT START ==========");
  Serial.println("main_hub is starting...");

  // Start Hardware UART (Serial1) for communication with main_hand
  // ESP32-S3: Serial1 uses GPIO 18 (RX) and GPIO 17 (TX)
  Serial1.begin(UART_BAUD, SERIAL_8N1, RXD_PIN, TXD_PIN);
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
  Serial.println(F("│       Protocol: UART RX + Queue + WiFi + HTTP          │"));
  Serial.println(F("└────────────────────────────────────────────────────────┘"));
  Serial.println();

  // Setup WiFi
  setupWiFi();

  Serial.println();
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println(F("         ⚡ QUEUE SYSTEM INITIALIZATION                  "));
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.printf("[Queue] Max queue size:     %d payloads\n", MAX_QUEUE_SIZE);
  Serial.printf("[Queue] Max payload size:   %d bytes\n", MAX_PAYLOAD_SIZE);
  Serial.printf("[Queue] Total RAM usage:    ~%d KB\n", (MAX_QUEUE_SIZE * MAX_PAYLOAD_SIZE) / 1024);
  Serial.println("[Queue] Drop policy:        Drop oldest when full");
  Serial.println("[Queue] Retry policy:       3 attempts with 1s delay");

  Serial.println();
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println(F("         ⚡ HARDWARE UART INITIALIZATION                 "));
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.printf("[UART] Using Serial1 (Hardware UART)\n");
  Serial.printf("[UART] RX Pin: GPIO%d ← main_hand GPIO18\n", RXD_PIN);
  Serial.printf("[UART] TX Pin: GPIO%d (not used)\n", TXD_PIN);
  Serial.printf("[UART] Baud Rate: %d\n", UART_BAUD);
  Serial.println("[UART] ✅ Hardware UART ready (DMA-based, non-blocking)");

  Serial.println();
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println(F("         ✅ SETUP COMPLETE - READY TO FORWARD!          "));
  Serial.println(F("════════════════════════════════════════════════════════"));
  Serial.println("[Info] Waiting for data from main_hand via Hardware UART");
  Serial.println("[Info] Data will be queued and sent to API");
  Serial.println("[Info] Server: " + String(serverUrl));
  Serial.println("[Info] Expected speed: ~278ms per 6400-byte packet");
  Serial.println("[Info] Type 'help' for a list of commands.");
  Serial.println();

  uartBuffer.reserve(UART_BUFFER_SIZE);
  lastStatsTime = millis();

  ledBlink(3, 200);  // Startup complete signal
}

// ════════════════════════════════════════════════════════════════════════════════
// MAIN LOOP (Non-blocking)
// ════════════════════════════════════════════════════════════════════════════════
void loop() {
  checkCommand();          // ตรวจสอบ Serial commands
  checkWiFiConnection();   // ตรวจสอบ WiFi ทุก 10 วินาที
  processUartData();       // รับข้อมูลจาก UART (non-blocking)
  processQueue();          // ส่งข้อมูลจาก queue (non-blocking)

  delay(1);  // Small delay to prevent watchdog issues
}
