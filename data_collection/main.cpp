#include <Wire.h>
#include <MPU6050.h>
#include "SerialDataExporter.h"

int bufferSizes[] = {100, 4, 4};
SerialDataExporter exporter = SerialDataExporter(Serial, bufferSizes);

MPU6050 mpu;

bool collecting = false;  // Toggle for data collection
char input;
int tremor = 0;


void setup() {
    Serial.begin(9600);  // Start Serial communication
    Wire.begin();          // Initialize I2C
    mpu.initialize();      // Initialize MPU6050

    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed!");
        while (1);  // Halt if the sensor is not found
    }

    Serial.println("MPU6050 initialized.");
}

void loop() {
    // Check for Serial command to toggle data collection
    char command;
    if (Serial.available() > 0) {
        command = Serial.read();
        if (command == 's') {
            collecting = !collecting;
            Serial.print("Data Collection: ");
            Serial.println(collecting ? "STARTED" : "STOPPED");
        }
         if(command == 'y'){
        tremor = 1;
      }
      if(command == 'n'){
          tremor = 0;
    }
    }



    if (collecting) {
        int16_t accelX, accelY, accelZ, gyroX, gyroY, gyroZ;
        mpu.getAcceleration(&accelX, &accelY, &accelZ);
        mpu.getRotation(&gyroX, &gyroY, &gyroZ);
        // Print sensor data in CSV format
        Serial.print(accelX);
        Serial.print(",");
        Serial.print(accelY);
        Serial.print(",");
        Serial.print(accelZ);
        Serial.print(",");
        exporter.add("x", accelX);
        exporter.add("y", accelY);
        exporter.add("z", accelZ);

        if(tremor == 1){
            Serial.println(1);
            exporter.add("tremor", 1);
        }else{
            Serial.println(0);
            exporter.add("tremor", 0);
        }

        exporter.exportJSON();
        delay(100);  // Adjust sampling rate
    }
}