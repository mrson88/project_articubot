#ifdef USE_SERVOS

#include <Adafruit_PWMServoDriver.h>

// Constructor
SweepServo::SweepServo()
{
    this->currentPositionDegrees = 0;
    this->targetPositionDegrees = 0;
    this->lastUpdateTime = 0;
    this->currentSpeed = 0;
    this->maxSpeed = 300; // degrees per second
    this->acceleration = 200; // degrees per second^2
}

// Init
void SweepServo::initServo(int servoPin, int stepDelayMs, int initPosition)
{
    this->servoPin = servoPin;
    this->currentPositionDegrees = initPosition;
    this->targetPositionDegrees = initPosition;
    this->lastUpdateTime = millis();
    pwm.setPWM(servoPin, 0, angleToPulse(initPosition));
}

// Perform Smooth Movement
void SweepServo::doSweep(int servo, int angle)
{
    unsigned long currentTime = millis();
    float deltaTime = (currentTime - this->lastUpdateTime) / 1000.0f; // Convert to seconds
    this->lastUpdateTime = currentTime;

    // Calculate direction
    float direction = (this->targetPositionDegrees > this->currentPositionDegrees) ? 1.0f : -1.0f;

    // Update speed
    if (abs(this->targetPositionDegrees - this->currentPositionDegrees) > 0.1f) {
        this->currentSpeed += this->acceleration * deltaTime * direction;
    } else {
        this->currentSpeed = 0;
        this->currentPositionDegrees = this->targetPositionDegrees; // Snap to target
    }

    // Limit speed
    this->currentSpeed = constrain(this->currentSpeed, -this->maxSpeed, this->maxSpeed);

    // Calculate new position
    float newPosition = this->currentPositionDegrees + this->currentSpeed * deltaTime;

    // Check if we've reached or overshot the target
    if ((direction > 0 && newPosition >= this->targetPositionDegrees) || 
        (direction < 0 && newPosition <= this->targetPositionDegrees)) {
        newPosition = this->targetPositionDegrees;
        this->currentSpeed = 0;
    }

    // Update position
    this->currentPositionDegrees = constrain(newPosition, 0, 180);
    pwm.setPWM(servo, 0, angleToPulse(round(this->currentPositionDegrees)));
}

// Set a new target position
void SweepServo::setTargetPosition(int servo, int servo_position)
{
    this->targetPositionDegrees = constrain(servo_position, 0, 180);
    servoCurrentPosition[servo] = servo_position;
}

// Accessor for servo object
int SweepServo::getServo(int servo_Pin)
{
    return round(this->currentPositionDegrees);
}

int angleToPulse(int ang) {
    return map(ang, 0, 180, SERVOMIN, SERVOMAX);
}

#endif