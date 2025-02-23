#include <Servo.h>

Servo servo1;  
Servo servo2;  

const int filterWindow = 5; // Size of the sliding window for filtering
int tremorHistory[filterWindow] = {0}; // Store last `filterWindow` tremor values
int tremorIndex = 0;
bool movedTo90 = false; // Flag to track if servos have already moved
const int servoSpeed = 3; // Smaller = smoother, slower movement

void setup() {
  Serial.begin(9600); // For receiving ML model tremor values
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

void loop() {
  if (Serial.available() > 0) { // If ML model sends data
    int tremorRaw = Serial.parseInt(); // Read tremor value (1 or 0)
    int tremor = getFilteredTremor(tremorRaw); // Apply filtering

    if (tremor == 1 && !movedTo90) {
      // Move servo1 to 90° and servo2 to 0° (inverted)
      moveServoSmooth(servo1, servo1.read(), 90);
      moveServoSmooth(servo2, servo2.read(), 0);
      movedTo90 = true;
    } 
    else if (tremor == 0 && movedTo90) {
      // Move servo1 to 0° and servo2 to 90° (inverted)
      moveServoSmooth(servo1, servo1.read(), 0);
      moveServoSmooth(servo2, servo2.read(), 90);
      movedTo90 = false;
    }
  }

  delay(50); // Match 20Hz signal updates
}
