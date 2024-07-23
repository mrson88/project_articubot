
#ifdef USE_SERVOS

// Constructor
SweepServo::SweepServo()
{
  this->currentPositionDegrees = 0;
  this->targetPositionDegrees = 0;
  this->lastSweepCommand = 0;
}


// Init
void SweepServo::initServo(
  int servoPin,
  int stepDelayMs,
  int initPosition)
{

  this->stepDelayMs = stepDelayMs;
  this->currentPositionDegrees = initPosition;
  this->targetPositionDegrees = initPosition;
  this->lastSweepCommand = millis();
}



// Perform Sweep
void SweepServo::doSweep(int servo, int angle)
{

  // Get ellapsed time
  int delta = millis() - this->lastSweepCommand;

  // Check if time for a step
  if (delta > this->stepDelayMs) {
    // Check step direction
    if (this->targetPositionDegrees > this->currentPositionDegrees) {
      this->currentPositionDegrees++;
      pwm.setPWM(servo, 0, angleToPulse(this->currentPositionDegrees));
    }
    else if (this->targetPositionDegrees < this->currentPositionDegrees) {
      this->currentPositionDegrees--;
      pwm.setPWM(servo, 0, angleToPulse(this->currentPositionDegrees));
    }
    // if target == current position, do nothing

    // reset timer
    this->lastSweepCommand = millis();
  }
}


// Set a new target position
void SweepServo::setTargetPosition(int servo, int servo_position)
{
  pwm.setPWM(servo, 0, angleToPulse(servo_position) );
  servoCurrentPosition[servo]=servo_position;
}


// Accessor for servo object
int SweepServo::getServo(int servo_Pin)
{
  int position_pin;
  position_pin=servoCurrentPosition[servo_Pin];
  return position_pin;
}

int angleToPulse(int ang) {
  int pulse = map(ang, 0, 180, SERVOMIN, SERVOMAX); // map angle of 0 to 180 to Servo min and Servo max
     Serial.print("Angle: ");Serial.print(ang);
     Serial.print(" pulse: ");Serial.println(pulse);
  return pulse;
}



#endif
