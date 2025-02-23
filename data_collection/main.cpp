#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;
bool collecting = false;
int tremor = 0;  // Default: no tremor

void setup() {
    Serial.begin(115200);  // Higher baud rate for faster transfer
    Wire.begin();
    mpu.initialize();

    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed!");
        while (1);  // Stop if sensor not found
    }

    Serial.println("MPU6050 initialized.");
    Serial.println("ax,ay,az,gx,gy,gz,tremor");  // CSV Header
}

void loop() {
    if (Serial.available() > 0) {
        char command = Serial.read();
        if (command == 's') {
            collecting = !collecting;
            Serial.println(collecting ? "STARTED" : "STOPPED");
        } 
        if (command == 'y') tremor = 1;
        if (command == 'n') tremor = 0;
    }

    if (collecting) {
        int16_t ax, ay, az, gx, gy, gz;
        mpu.getAcceleration(&ax, &ay, &az);
        mpu.getRotation(&gx, &gy, &gz);

        Serial.print(ax); Serial.print(",");
        Serial.print(ay); Serial.print(",");
        Serial.print(az); Serial.print(",");
        Serial.print(gx); Serial.print(",");
        Serial.print(gy); Serial.print(",");
        Serial.print(gz); Serial.print(",");
        Serial.println(tremor);

        delay(50);  // Adjust for sampling rate
    }
}