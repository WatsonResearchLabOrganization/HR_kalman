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
int atime = 29;      // AS7341 ATIME setting, the number of integration cycles
int astep = 50;      // AS7341 ASTEP setting, the duration of each cycle
int stabTime = 10;
int delayTime = 5;
int portNum = 115200;
int disconnectDelay = 1000;
as7341_gain_t gain = AS7341_GAIN_256X; // Gain setting for AS7341

// LED brightness for each LED (adjust as needed)
int ledBrightness = 3500;

Adafruit_TLC59711 tlc = Adafruit_TLC59711(NUM_TLC59711, clock, data);
uint16_t readings[10] = {0};
unsigned long startTime = 0;
String combinedData = "";

// Band-pass filter variables
float filteredReading = 0;
float prevReading = 0;
float alpha = 0.5;  // Smoothing factor
float lowFreqCutoff = 0.1;
float highFreqCutoff = 2;
float samplingInterval = 0.05;  // 50ms -> 20Hz

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
    as7341.setGain(gain);  // Set gain correctly

    tlc.begin();

    // LED Array Setup

    // for (int i = 0; i < ledNum; i++) {
    //     tlc.setPWM(i, pwmValue); // Set initial LED brightness
    // }
    // tlc.write();
    // delay(2000);

    // Turn off LEDs after the initial setup
    // for (int i = 0; i < ledNum; i++) {
    //     tlc.setPWM(i, 0);
    // }
    // tlc.write();
    // delay(2000);

    startTime = millis();

    // Print header for data columns
    //Serial.println("Time(ms),415nm,445nm,480nm,515nm,555nm,590nm,630nm,680nm,NIR,CLEAR,LEDNum");
}

void loop() {
    unsigned long currentTime = millis(); 
    //for (int i = 0; i < ledNum; ++i) {
        // Set LED brightness dynamically
        //tlc.setPWM(7, ledBrightness);
        tlc.setPWM(8, ledBrightness);// Use individual brightness values
        tlc.write();
        //delay(stabTime); // Stabilize sensor after changing LED state

        // Read channels three times and average
        if (!as7341.readAllChannels()) {
            Serial.println("Error reading all channels!");
            return;
        }
        //readings[0] = as7341.getChannel(AS7341_CHANNEL_415nm_F1);
       // readings[1] = as7341.getChannel(AS7341_CHANNEL_445nm_F2);
        //readings[2] = as7341.getChannel(AS7341_CHANNEL_480nm_F3);
        //readings[3] = as7341.getChannel(AS7341_CHANNEL_515nm_F4);
        //readings[4] = as7341.getChannel(AS7341_CHANNEL_555nm_F5);
        //readings[5] = as7341.getChannel(AS7341_CHANNEL_590nm_F6);
        //readings[6] = as7341.getChannel(AS7341_CHANNEL_630nm_F7);
        //readings[7] = as7341.getChannel(AS7341_CHANNEL_680nm_F8);
        readings[8] = as7341.getChannel(AS7341_CHANNEL_NIR);
        //readings[9] = as7341.getChannel(AS7341_CHANNEL_CLEAR);
        //delay(delayTime); // Short delay between readings

        // Combine data for output
        //String pdData = combineData(currentTime, i);
        //combinedData += pdData;

        // Turn off the LED after the reading
        //tlc.setPWM(i, 0);
        //tlc.write();
        
        // Print the combined data over Serial
        //Serial.println(readings[7]);
        //  combinedData = "";

         // Perform baseline subtraction for readings[2] and readings[7]
        //static float baseline2 = 0;
        //static float baseline7 = 0;
        //static int baselineCount = 0;
        //const int baselineSamples = 100;
        //if (baselineCount < baselineSamples) {
          //baseline2 += (float)readings[2];
          //baseline7 += (float)readings[7];
          //baselineCount++;
          //if (baselineCount == baselineSamples) {
            //baseline2 /= baselineSamples;  // Calculate average baseline for reading 2 after 100 readings
            //baseline7 /= baselineSamples;  // Calculate average baseline for reading 7 after 100 readings
          //}
        //}
        //float baselineCorrectedReading2 = (float)readings[2] - baseline2;
        //float baselineCorrectedReading7 = (float)readings[7] - baseline7;

  
 //float newReading = (float)readings[2]-(float)readings[7];
        //filteredReading = alpha * (newReading - prevReading) + (1 - alpha) * filteredReading;
        //prevReading = newReading;

         //Print filtered data to Serial Plotter
        Serial.println(readings[8]);
        
    }

    // Optionally delay before the next loop iteration
    // delay(2000);


//String combineData(unsigned long time, int ledNum) {
    //String pddata = "";
    //pddata += time;
    //pddata += ",";
    //for (int i = 0; i < pdNum; i++) {
        //pddata += String(readings[i]);
        //pddata += ",";
    //}
    //pddata += String(ledNum);
    //return pddata;
//}
