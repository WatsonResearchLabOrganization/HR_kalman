#include <Adafruit_AS7341.h>
#include <Wire.h>
#include "Adafruit_TLC59711.h"

Adafruit_AS7341 as7341;

#define NUM_TLC59711 1
#define DATA_PIN 10
#define CLOCK_PIN 8

// ----------------------------------------
// Adjustable Parameters
// ----------------------------------------
int pdNum = 10;       // Number of PD channels
int ledNum = 9;       // Number of LEDs (TLC channels)
int pwmValue = 5000;  // PWM Value for LEDs
int atime = 29;       // AS7341 ATIME setting
int astep = 50;       // AS7341 ASTEP setting
int stabTime = 10;
int delayTime = 5;
int portNum = 115200;
int disconnectDelay = 1000;
as7341_gain_t gain = AS7341_GAIN_256X; // Gain setting for AS7341

// ----------------------------------------
// LED Brightness
// ----------------------------------------
int ledBrightness = 3000; 

// ----------------------------------------
// Create TLC59711 instance
// ----------------------------------------
Adafruit_TLC59711 tlc = Adafruit_TLC59711(NUM_TLC59711, CLOCK_PIN, DATA_PIN);

// ----------------------------------------
// Sensor readings array
// ----------------------------------------
uint16_t readings[10] = {0};
unsigned long startTime = 0;
String combinedData = "";

// ----------------------------------------
// Simple On/Off Switches for each LED
// If 'true' -> that LED is ON
// If 'false' -> that LED is OFF
// ledSwitch[0] corresponds to LED #1
// ledSwitch[1] corresponds to LED #2
// ...
// ledSwitch[8] corresponds to LED #9
// ----------------------------------------
bool ledSwitch[9] = {
  false,   // LED #1 ON
  false,  // LED #2 OFF
  false,   // LED #3 ON
  false,   // LED #4 ON
  false,  // LED #5 OFF
  false,   // LED #6 ON
  true,   // LED #7 ON
  false,  // LED #8 OFF
  false    // LED #9 ON
};

// Optional band-pass filter variables (if needed)
float filteredReading = 0;
float prevReading = 0;
float alpha = 0.5;
float lowFreqCutoff = 0.1;
float highFreqCutoff = 2;
float samplingInterval = 0.05;  // 50ms -> 20Hz

// ----------------------------------------
// Helper function to combine data
// ----------------------------------------
String combineData() {
  String pddata = "";
  for (int i = 0; i < pdNum; i++) {
    pddata += String(readings[i]);
    if (i < pdNum - 1) {
      pddata += ",";
    }
  }
  return pddata;
}

// ----------------------------------------
// Setup
// ----------------------------------------
void setup() {
  Serial.begin(portNum);

  // AS7341 Setup
  if (!as7341.begin()) {
    Serial.println("Could not find AS7341");
    while (1) {
      delay(disconnectDelay);
    }
  }
  as7341.setATIME(atime);
  as7341.setASTEP(astep);
  as7341.setGain(gain);

  // Initialize TLC
  tlc.begin();

  // Optional: Turn on each LED as per ledSwitch[] in setup
  // so we can visually confirm which ones are ON/OFF at startup
  for (int i = 0; i < ledNum; i++) {
    if (ledSwitch[i]) {
      tlc.setPWM(i, ledBrightness);
    } else {
      tlc.setPWM(i, 0);
    }
  }
  tlc.write();

  delay(2000); // Wait 2 seconds to observe initial state

  startTime = millis();
  // Print header, if desired
  // Serial.println("415nm,445nm,480nm,515nm,555nm,590nm,630nm,680nm,NIR,Clear");
}

// ----------------------------------------
// Main Loop
// ----------------------------------------
void loop() {
  unsigned long currentTime = millis();

  // ------------------------------------------------
  // Turn LED(s) on/off based on ledSwitch[] each time
  // ------------------------------------------------
  for (int i = 0; i < ledNum; i++) {
    if (ledSwitch[i]) {
      tlc.setPWM(i, ledBrightness);
    } else {
      tlc.setPWM(i, 0);
    }
  }
  tlc.write();
  
  // Short delay to let the sensor stabilize
  delay(stabTime);

  // ------------------------------------
  // Read all channels from the AS7341
  // ------------------------------------
  if (!as7341.readAllChannels()) {
    Serial.println("Error reading all channels!");
    return;
  }
  readings[0] = as7341.getChannel(AS7341_CHANNEL_415nm_F1);
  readings[1] = as7341.getChannel(AS7341_CHANNEL_445nm_F2);
  readings[2] = as7341.getChannel(AS7341_CHANNEL_480nm_F3);
  readings[3] = as7341.getChannel(AS7341_CHANNEL_515nm_F4);
  readings[4] = as7341.getChannel(AS7341_CHANNEL_555nm_F5);
  readings[5] = as7341.getChannel(AS7341_CHANNEL_590nm_F6);
  readings[6] = as7341.getChannel(AS7341_CHANNEL_630nm_F7);
  readings[7] = as7341.getChannel(AS7341_CHANNEL_680nm_F8);
  readings[8] = as7341.getChannel(AS7341_CHANNEL_NIR);
  readings[9] = as7341.getChannel(AS7341_CHANNEL_CLEAR);

  // ------------------------------------------------
  // Print sensor readings
  // ------------------------------------------------
  String pdData = combineData();
  Serial.println(pdData);

  // --------------------------------------
  // Small delay until next reading
  // --------------------------------------
  //delay(1000);
  
  // (Optional) Additional code, filters, etc.
}
