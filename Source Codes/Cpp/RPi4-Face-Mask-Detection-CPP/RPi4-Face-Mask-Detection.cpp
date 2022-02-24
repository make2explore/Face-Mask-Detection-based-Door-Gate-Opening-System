// ---------------------------------- make2explore.com-----------------------------------------------------------//
// Project           - No Mask, No Entry. RPi4 Face Mask Detection based Door/Gate Opening System 
// Created By        - info@make2explore.com
// Version - 1.0
// Last Modified     - 24/02/2022 15:00:00 @admin
// Software          - C/C++, CodeBlocks IDE, Standard C/C++ Libraries, OpenCV, Libraries - TensorFlow Lite, Keras. etc.
// Hardware          - Raspberry Pi 4 model B, Logitech C270 webcam, EM-18 RFID Reader, Level Converter, SG-90 Servo
// Sensors Used      - EM-18 RFID Reader, Logitech C270 webcam
// code Reference    - Q-Engineering
// Source Repo       - https://github.com/make2explore/Face-Mask-Detection-based-Door-Gate-Opening-System
// --------------------------------------------------------------------------------------------------------------//

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <wiringPi.h>
#include <wiringPiI2C.h>
#include <wiringSerial.h>


#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include <cmath>

using namespace cv;
using namespace std;

int model_width;
int model_height;
int model_channels;

// Define some device parameters
#define I2C_ADDR   0x3f // I2C device address

// Define some device constants
#define LCD_CHR  1 // Mode - Sending data
#define LCD_CMD  0 // Mode - Sending command

#define LINE1  0x80 // 1st line
#define LINE2  0xC0 // 2nd line

#define LCD_BACKLIGHT   0x08  // On
// LCD_BACKLIGHT = 0x00  # Off

#define ENABLE  0b00000100 // Enable bit

void lcd_init(void);
void lcd_byte(int bits, int mode);
void lcd_toggle_enable(int bits);

// added by Lewis
void typeInt(int i);
void typeFloat(float myFloat);
void lcdLoc(int line); //move cursor
void ClrLcd(void); // clr LCD return home
void typeln(const char *s);
void typeChar(char val);
int fd;  // seen by all subroutines

void lcd_init(void); // setup LCD
void init_servo(void); //initialize Servo
void init_buzzer(void); // Init Buzzer
void init_relay(void);

int wiringPi_init(void);
int serialbegin(int);
void setup(void);
int get_card(void);
int detect_mask(void);

void maskDetected(void);
void maskNotDetected(void);

// Pin number declarations. We're using the Broadcom chip pin numbers.
const int Relay1Pin = 5;
const int Relay2Pin = 6;
const int buzzerPin = 26;
const int ServoPin = 12;

int sp;
int count1=0,count2=0,count3=0,count4=0,count5=0;
char ch;
char rfid[13];
int j=0;
int maskDetectIndex=0;
int flag = 0;

std::unique_ptr<tflite::Interpreter> interpreter;

//-----------------------------------------------------------------------------------------------------------------------
void GetImageTFLite(float* out, Mat &src)
{
    int i,Len;
    float f;
    uint8_t *in;
    static Mat image;

    // copy image to input as input tensor
    cv::resize(src, image, Size(model_width,model_height),INTER_NEAREST);

    //model posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite runs from -1.0 ... +1.0
    //model multi_person_mobilenet_v1_075_float.tflite                 runs from  0.0 ... +1.0
    in=image.data;
    Len=image.rows*image.cols*image.channels();
    for(i=0;i<Len;i++){
        f     =in[i];
        out[i]=(f - 127.5f) / 127.5f;
    }
}
//-----------------------------------------------------------------------------------------------------------------------
void detect_from_video(Mat &src)
{
    Mat image;
    int cam_width =src.cols;
    int cam_height=src.rows;

    // copy image to input as input tensor
    GetImageTFLite(interpreter->typed_tensor<float>(interpreter->inputs()[0]), src);

    interpreter->AllocateTensors();
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);      //quad core

    interpreter->Invoke();      // run your model

    const float* detection_locations = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* detection_classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    const float* detection_scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    const int    num_detections = *interpreter->tensor(interpreter->outputs()[3])->data.f;

    //there are ALWAYS 10 detections no matter how many objects are detectable
    //cout << "number of detections: " << num_detections << "\n";

    const float confidence_threshold = 0.5;
    for(int i = 0; i < num_detections; i++){
        if(detection_scores[i] > confidence_threshold){
            int  det_index = (int)detection_classes[i];
            float y1=detection_locations[4*i  ]*cam_height;
            float x1=detection_locations[4*i+1]*cam_width;
            float y2=detection_locations[4*i+2]*cam_height;
            float x2=detection_locations[4*i+3]*cam_width;

            Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
            if(det_index==0){
                rectangle(src,rec, Scalar(0, 255, 0), 2, 8, 0);
                putText(src,"Mask Detected", Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.7, Scalar(0, 255, 0), 1, 8, 0);
                maskDetectIndex = 0;
                flag++;
            }
            if(det_index==1){
                rectangle(src,rec, Scalar(0, 0, 255), 2, 8, 0);
                putText(src,"No Mask", Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.7, Scalar(0, 0, 255), 1, 8, 0);
                maskDetectIndex = 1;

            }
            if(det_index==2){
                rectangle(src,rec, Scalar(0, 127, 255), 2, 8, 0);
                putText(src,"Weared Incorrectly", Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.7, Scalar(0, 127, 255), 1, 8, 0);
                maskDetectIndex = 2;

            }
        }
    }
}


int detect_mask()
{
    float f;
    float FPS[16];
    int i;
    int Fcnt=0;
    Mat frame;
    chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

    // Load model
    //std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("ssd_mobilenet_v2_fpnlite.tflite");
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("ssdlite_mobilenet_v2.tflite");

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    interpreter->AllocateTensors();

    // Get input dimension from the input tensor metadata
    // Assuming one input only
    int In = interpreter->inputs()[0];
    model_height   = interpreter->tensor(In)->dims->data[1];
    model_width    = interpreter->tensor(In)->dims->data[2];
    model_channels = interpreter->tensor(In)->dims->data[3];

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        ClrLcd();
        lcdLoc(LINE1);
        typeln("Err- Unable to");
        lcdLoc(LINE2);
        typeln("Open the Camera");
        delay(250);
        return 0;
    }

        ClrLcd();
        lcdLoc(LINE1);
        typeln("Starting Camera");
        lcdLoc(LINE2);
        typeln("Camera - ON");
        delay(1000);

        ClrLcd();
        lcdLoc(LINE1);
        typeln("Look Straight");
        lcdLoc(LINE2);
        typeln("into the Camera");
        delay(250);

    while(1){
//        frame=imread("Kapje_2.jpg");  //need to refresh frame before dnn class detection
        cap >> frame;
        if (frame.empty()) {
            cerr << "End of movie" << endl;
            break;
        }

        detect_from_video(frame);

        Tend = chrono::steady_clock::now();
        //calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();

        Tbegin = chrono::steady_clock::now();

        FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(frame, format("FPS %0.2f",f/16),Point(10,20),FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 0, 255));

        //show output
        imshow("make2explore - RPi4 Face Mask Detection", frame);

        if(maskDetectIndex == 0 && (flag>10))
          {
              flag = 0;
              cout << "Closing the camera" << endl;
              // When everything done, release the video capture and write object
              cap.release();
              destroyAllWindows();
              cout << "Bye!" << endl;
              maskDetected();
          }
         else
         {
              maskNotDetected();
         }

        char esc = waitKey(5);
        if(esc == 27) break;
    }

    cout << "Closing the camera" << endl;

    // When everything done, release the video capture and write object
    cap.release();

    destroyAllWindows();
    cout << "Bye!" << endl;

    return 0;
}



//-----------------------------------------------------------------------------------------------------------------------
int main(int argc,char ** argv)
{
    cout << "RPi4 Face Mask Detection: " << "\n";
    setup();
    while(1)
      {
        digitalWrite(buzzerPin, HIGH); // Turn LED ON
        ClrLcd();
        lcdLoc(LINE1);
        typeln("Place Your RFID");
        lcdLoc(LINE2);
        typeln("On the Reader");
        delay(1000);

        get_card();
        delay(200);
        ClrLcd();
        lcdLoc(LINE1);
        typeln("Scanning Card...");
        lcdLoc(LINE2);
        typeln("Checking ID >>>");
        delay(1000);

        if(strncmp(rfid,"0900963100AE",12)==0)
        {
          count1++;
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Authorized User");
          digitalWrite(Relay2Pin, LOW);
          delay(1000);
          digitalWrite(Relay2Pin, HIGH);
          detect_mask();
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Hello Adhya");
          lcdLoc(LINE2);
          typeln("Welcome!");
          delay(1000);
        }
        else if(strncmp(rfid,"88001964699C",12)==0)
        {
          count2++;
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Authorized User");
          digitalWrite(Relay2Pin, LOW);
          delay(1000);
          digitalWrite(Relay2Pin, HIGH);
          detect_mask();
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Hello Samihan");
          lcdLoc(LINE2);
          typeln("Welcome!");
          delay(1000);
        }
        else if(strncmp(rfid,"880013E5235D",12)==0)
        {
          count3++;
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Authorized User");
          digitalWrite(Relay2Pin, LOW);
          delay(1000);
          digitalWrite(Relay2Pin, HIGH);
          detect_mask();
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Hello Raj");
          lcdLoc(LINE2);
          typeln("Welcome!");
          delay(1000);
        }
        else if(strncmp(rfid,"880013E5225C",12)==0)
        {
          count4++;
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Authorized User");
          digitalWrite(Relay2Pin, LOW);
          delay(1000);
          digitalWrite(Relay2Pin, HIGH);
          detect_mask();
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Hello Vaidehi");
          lcdLoc(LINE2);
          typeln("Welcome!");
          delay(1000);
        }
        else if(strncmp(rfid,"880019646A9F",12)==0)
        {
          count5++;
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Authorized User");
          digitalWrite(Relay2Pin, LOW);
          delay(1000);
          digitalWrite(Relay2Pin, HIGH);
          detect_mask();
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Hello Mahesh");
          lcdLoc(LINE2);
          typeln("Welcome!");
          delay(1000);
        }
        else
        {
          ClrLcd();
          lcdLoc(LINE1);
          typeln("Invalid RFID");
          lcdLoc(LINE2);
          typeln("Unauthorized");
          delay(1000);
          ClrLcd();
          lcdLoc(LINE1);
          typeln("    Access");
          lcdLoc(LINE2);
          typeln("    Denied");
          delay(1000);
        }
      }

    return 0;
}
//-----------------------------------------------------------------------------------------------------------------------


// float to string
void typeFloat(float myFloat)   {
  char buffer[20];
  sprintf(buffer, "%4.2f",  myFloat);
  typeln(buffer);
}

// int to string
void typeInt(int i)   {
  char array1[20];
  sprintf(array1, "%d",  i);
  typeln(array1);
}

// clr lcd go home loc 0x80
void ClrLcd(void)   {
  lcd_byte(0x01, LCD_CMD);
  lcd_byte(0x02, LCD_CMD);
}

// go to location on LCD
void lcdLoc(int line)   {
  lcd_byte(line, LCD_CMD);
}

// out char to LCD at current position
void typeChar(char val)   {

  lcd_byte(val, LCD_CHR);
}


// this allows use of any size string
void typeln(const char *s)   {

  while ( *s ) lcd_byte(*(s++), LCD_CHR);

}

void lcd_byte(int bits, int mode)   {

  //Send byte to data pins
  // bits = the data
  // mode = 1 for data, 0 for command
  int bits_high;
  int bits_low;
  // uses the two half byte writes to LCD
  bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT ;
  bits_low = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT ;

  // High bits
  wiringPiI2CReadReg8(fd, bits_high);
  lcd_toggle_enable(bits_high);

  // Low bits
  wiringPiI2CReadReg8(fd, bits_low);
  lcd_toggle_enable(bits_low);
}

void lcd_toggle_enable(int bits)   {
  // Toggle enable pin on LCD display
  delayMicroseconds(500);
  wiringPiI2CReadReg8(fd, (bits | ENABLE));
  delayMicroseconds(500);
  wiringPiI2CReadReg8(fd, (bits & ~ENABLE));
  delayMicroseconds(500);
}


void lcd_init()   {
  // Initialise display
  lcd_byte(0x33, LCD_CMD); // Initialise
  lcd_byte(0x32, LCD_CMD); // Initialise
  lcd_byte(0x06, LCD_CMD); // Cursor move direction
  lcd_byte(0x0C, LCD_CMD); // 0x0F On, Blink Off
  lcd_byte(0x28, LCD_CMD); // Data length, number of lines, font size
  lcd_byte(0x01, LCD_CMD); // Clear display
  delayMicroseconds(500);
}

void init_servo()   {
  pinMode(ServoPin,PWM_OUTPUT);
  pwmSetMode(PWM_MODE_MS);
  pwmSetRange(200);
  pwmSetClock(1920);
}

void init_buzzer()  {
  pinMode(buzzerPin, OUTPUT);     // Set Buzzer Pin as output
  digitalWrite(buzzerPin, HIGH); // Turn Buzzer OFF
}

void init_relay()   {
  pinMode(Relay1Pin, OUTPUT);     // Set Relay1 Pin as output
  pinMode(Relay2Pin, OUTPUT);     // Set Relay2 Pin as output
  digitalWrite(Relay1Pin, HIGH); // Turn Relay1 OFF
  digitalWrite(Relay2Pin, HIGH); // Turn Relay2 OFF
}

int wiringPi_init()
{
  if (wiringPiSetup () == -1)                                   /* initializes wiringPi setup */
  {
    fprintf (stdout, "Unable to start wiringPi: %s\n", strerror (errno));
    return 1;
  }
  return 0;
}

int serialbegin(int baud)
{
  if ((sp = serialOpen ("/dev/ttyAMA0",baud)) < 0)
  {
    fprintf (stderr, "Unable to open serial device: %s\n", strerror (errno));
    return 1;
  }
  return 0;
}

void setup()
{
  wiringPi_init();

  fd = wiringPiI2CSetup(I2C_ADDR);
  wiringPiSetupGpio();

  serialbegin(9600);
  lcd_init();
  init_relay();
  init_buzzer();
  init_servo();
  delay(1000);

  lcdLoc(LINE1);
  typeln("Welcome To");
  lcdLoc(LINE2);
  typeln("make2explore.com");
  delay(2000);

  ClrLcd();
  lcdLoc(LINE1);
  typeln("Project - RPi4");
  lcdLoc(LINE2);
  typeln("based Face");
  delay(2000);

  ClrLcd();
  lcdLoc(LINE1);
  typeln("Mask Detection");
  lcdLoc(LINE2);
  typeln("System");
  delay(2000);
}

int get_card()
{
   j=0;
   while(j<12)
   {
     while(serialDataAvail(sp))
     {
       ch = serialGetchar(sp);
       rfid[j] = ch;
       fflush(stdout);
       j++;
     }
   }
  rfid[j]='\0';
  return 0;
}

//mask Detected
void maskDetected(){
    ClrLcd();
    lcdLoc(LINE1);
    typeln("Mask Detected");

    digitalWrite(buzzerPin, LOW); // Turn LED ON
    delay(100);
    digitalWrite(buzzerPin, HIGH); // Turn LED OFF
    delay(100);
    digitalWrite(buzzerPin, LOW); // Turn LED ON
    delay(100);
    digitalWrite(buzzerPin, HIGH); // Turn LED OFF

    digitalWrite(Relay1Pin, LOW);
    pwmWrite(ServoPin,120);
    delay(2000);
    lcdLoc(LINE2);
    typeln("Gate : Unlocked");
    pwmWrite(ServoPin,10);
    digitalWrite(Relay1Pin, HIGH);
}

//mask Not Detected
void maskNotDetected(){
    ClrLcd();
    lcdLoc(LINE1);
    typeln("NoMask No Entry");
    lcdLoc(LINE2);
    typeln("Gate : LOCKED");
    digitalWrite(buzzerPin, LOW); // Turn LED ON
    delay(200);
    digitalWrite(buzzerPin, HIGH); // Turn LED OFF
    delay(200);
}

// ---------------------------------- make2explore.com-----------------------------------------------------------//