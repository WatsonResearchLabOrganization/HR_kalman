#include <Adafruit_AS7341.h>
#include <Wire.h>
#include "Adafruit_TLC59711.h"

Adafruit_AS7341 as7341;
#define NUM_TLC59711 1
#define DATA_PIN 10
#define CLOCK_PIN 8

Adafruit_TLC59711 tlc = Adafruit_TLC59711(NUM_TLC59711, CLOCK_PIN, DATA_PIN);

// Customizable Variables
int pdNum = 10;       // Number of PD channels
int ledNum = 9;       // Number of LEDs
int pwmValue = 500;  // PWM Value for LEDs
int atime = 50;      // AS7341 ATIME setting, the number of integration cycles
int astep = 499;      // AS7341 ASTEP setting, the duration of each cycle
int stabTime = 10;
int delayTime = 5;
int portNum = 115200;
int disconnectDelay = 1000;
as7341_gain_t gain = AS7341_GAIN_256X; // Gain setting for AS7341

// Band-pass filter variables
float filteredReading = 0;
float prevReading = 0;
float alpha = 0.1;  // Smoothing factor
float lowFreqCutoff = 0.6;
float highFreqCutoff = 2.0;
float samplingInterval = 0.05;  // 50ms -> 20Hz

void setup() {
  Serial.begin(portNum);

  // Initialize the AS7341 sensor
  if (!as7341.begin()) {
    Serial.println("Failed to initialize AS7341 sensor!");
    while (1);
  }

  // Set AS7341 integration time and gain
  as7341.setATIME(atime);
  as7341.setASTEP(astep);
  as7341.setGain(gain);

  // Initialize TLC59711 LED driver
  tlc.begin();
  tlc.write();

  // Set initial LED8 brightness to 0 (off)
  tlc.setPWM(8, 0);
  tlc.write();
}

void loop() {
  // Set LED8 intensity using PWM
  tlc.setPWM(7, pwmValue);
  tlc.write();

  // Get NIR channel reading from AS7341 sensor
  uint16_t nirReading = as7341.getChannel(AS7341_CHANNEL_680nm_F8);

  // Apply a band-pass filter to the NIR reading
  float newReading = (float)nirReading;
  filteredReading = alpha * (newReading - prevReading) + (1 - alpha) * filteredReading;
  prevReading = newReading;

  // Print filtered data to Serial Plotter
  //Serial.println(filteredReading);
  Serial.println(nirReading);

  delay(samplingInterval * 1000);  // Delay based on sampling interval (50 ms)
}
