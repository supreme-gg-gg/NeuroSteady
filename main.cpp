#include <Wire.h>
#include <MPU6050.h>
#include <Servo.h>

Servo servo1;  
Servo servo2; 

const int filterWindow = 5; // Size of the sliding window for filtering
int tremorHistory[filterWindow] = {0}; // Store last `filterWindow` tremor values
int tremorIndex = 0;
bool movedTo90 = false; // Flag to track if servos have already moved
const int servoSpeed = 3; // Smaller = smoother, slower movement

MPU6050 mpu;
bool collecting = false;

void setup() {
    // Initialize Serial communication
    Serial.begin(115200);
    // Wait for Serial connection on boards with native USB
    while (!Serial) {
      ; // wait for serial port to connect.
    }

    // MPU6050 setup
    Wire.begin();
    mpu.initialize();
    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed!");
        while (1);  // Stop if sensor not found
    }
    Serial.println("MPU6050 initialized.");

    // Servo setup
    servo1.attach(9);  // Attach servo1 to pin 9
    servo2.attach(10); // Attach servo2 to pin 10

    servo1.write(0);  // Start servo1 at 0 degrees
    servo2.write(90); // Start servo2 at 90 degrees (opposite)
} 

void moveServoSmooth(Servo &servo, int startPos, int endPos) {
  int step = (startPos < endPos) ? 1 : -1; // Determine direction
  for (int pos = startPos; pos != endPos; pos += step) {
    servo.write(pos);
    delay(servoSpeed); // Control speed of movement
  }
  servo.write(endPos); // Ensure final position is reached
}

// Function to filter out random spikes (stray 1s or 0s)
int getFilteredTremor(int newTremor) {
  tremorHistory[tremorIndex] = newTremor; // Store new value in history
  tremorIndex = (tremorIndex + 1) % filterWindow; // Update index in circular buffer

  // Count occurrences of 1s and 0s in the window
  int ones = 0, zeros = 0;
  for (int i = 0; i < filterWindow; i++) {
    if (tremorHistory[i] == 1) ones++;
    else zeros++;
  }

  // If most recent values are 1s, return 1, otherwise return 0
  return (ones > zeros) ? 1 : 0;
}

unsigned long previousMillis = 0;
const long interval = 100; // 50ms interval for sampling does not work likely due to bandwidth of the Serial cable/port

void loop() {
  unsigned long currentMillis = millis();

  if (Serial.available() > 0) {
    char tremorRaw = Serial.read();  // Read single byte instead of full integer
    int tremor = getFilteredTremor(tremorRaw - '0');  // Convert from ASCII to int

    if (tremor == 1 && !movedTo90) {
      moveServoSmooth(servo1, servo1.read(), 90);
      moveServoSmooth(servo2, servo2.read(), 0);
      movedTo90 = true;
    } else if (tremor == 0 && movedTo90) {
      moveServoSmooth(servo1, servo1.read(), 0);
      moveServoSmooth(servo2, servo2.read(), 90);
      movedTo90 = false;
    }
  }

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    if (Serial) {
      int16_t ax, ay, az;
      mpu.getAcceleration(&ax, &ay, &az);
      Serial.print(ax); Serial.print(",");
      Serial.print(ay); Serial.print(",");
      Serial.println(az);
    }
  }
}
