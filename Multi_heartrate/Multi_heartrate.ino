#include <Adafruit_AS7341.h>
#include <Wire.h>
#include "Adafruit_TLC59711.h"

Adafruit_AS7341 as7341;

#define NUM_TLC59711 1
#define data 10
#define clock 8

// Customizable Variables
int pdNum = 10;       // Number of PD channels
int ledNum = 9;       // Number of LEDs
int pwmValue = 5000;  // PWM Value for LEDs
int atime = 29;       // AS7341 ATIME setting
int astep = 50;       // AS7341 ASTEP setting
int stabTime = 10;
int delayTime = 5;
int portNum = 115200;
int disconnectDelay = 1000;
as7341_gain_t gain = AS7341_GAIN_256X; // Gain setting for AS7341

// LED brightness for each LED (adjust as needed)
int ledBrightness = 3000;

Adafruit_TLC59711 tlc = Adafruit_TLC59711(NUM_TLC59711, clock, data);
uint16_t readings[10] = {0};
unsigned long startTime = 0;
String combinedData = "";

// (Unused band-pass filter variables in your script)
float filteredReading = 0;
float prevReading = 0;
float alpha = 0.5;
float lowFreqCutoff = 0.1;
float highFreqCutoff = 2;
float samplingInterval = 0.05;  // 50ms -> 20Hz

String combineData() {
  String pddata = "";
  // Remove the time—only append sensor readings
  for (int i = 0; i < pdNum; i++) {
      pddata += String(readings[i]);
      // Add a comma *only* if it’s not the last reading
      if (i < pdNum - 1) {
          pddata += ",";
      }
  }
  return pddata;
}


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

  tlc.begin();

  // Turn on ALL LEDs in setup (for a quick test)
  for (int i = 0; i < ledNum; i++) {
      tlc.setPWM(i, ledBrightness); 
  }
  tlc.write();
  delay(2000);

  // (Optional) Turn off LEDs once, if desired — but we leave them on for now
  // for (int i = 0; i < ledNum; i++) {
  //     tlc.setPWM(i, 0);
  // }
  // tlc.write();
  // delay(2000);

  startTime = millis();

  // Print header for data columns
  //Serial.println("Time(ms),415nm,445nm,480nm,515nm,555nm,590nm,630nm,680nm,NIR");
}

void loop() {
  unsigned long currentTime = millis();

  // Keep ALL LEDs ON constantly:
  for (int i = 0; i < ledNum; ++i) {
      tlc.setPWM(i, ledBrightness);
  }
  tlc.write();
  delay(stabTime);

  // Read all channels
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

  // Print all readings in one line
  String pdData = combineData(); // Use '999' (or any label) as LEDIndex if desired
  Serial.println(pdData);

  // *********** IMPORTANT CHANGE ***********
  // Comment out or remove the lines that turn off the LEDs
  // for (int i = 0; i < ledNum; ++i) {
  //     tlc.setPWM(i, 0);
  // }
  // tlc.write();
  // ***************************************

  // Delay before the next loop iteration
  //delay(2000);

  // (Optional) Additional code can go here for baseline correction or filtering
}
