// Autofocus control code of Canon EF-S18-55mm f/3.5-5.6 IS II
// helpful instructions:
// https://docplayer.net/20752779-How-to-move-canon-ef-lenses-yosuke-bando.html
// https://github.com/crescentvenus/EF-Lens-CONTROL/blob/master/EF-Lens-control.ino
// https://gist.github.com/marcan/858c242db2fc595da1e0bb70a05192fc
// Contact: Suyeon Choi (suyeon@stanford.edu)


#include <SPI.h>
#include <EEPROM.h>

const int HotShoe_Pin = 8;
const int HotShoe_Gnd = 9;
const int LogicVDD_Pin = 10;
const int Cam2Lens_Pin = 11;
const int Clock_Pin = 13;

// manually set for 11 planes (the last one is dummy)
// *** Calibrate these focus values for your own lens ***
// Inf to 3D (0D, 0.3D, ... 3D) gotta correspond to some range in your physical setup
const int focus[12] = {2047, 1975, 1896, 1830, 1795, 1750, 1655, 1580, 1560, 1510, 1460, 0};

const int current_focus = 0; 
#define INPUT_SIZE 30

void init_lens() {
  // Before sending signal to lens you should do this
  
  SPI.transfer(0x0A);
  delay(30);
  SPI.transfer(0x00);
  delay(30);
  SPI.transfer(0x0A);
  delay(30);
  SPI.transfer(0x00);
  delay(30);
}

void setup() // initialization
{
  Serial.begin(9600);

  pinMode(LogicVDD_Pin, OUTPUT);
  digitalWrite(LogicVDD_Pin, HIGH);
  pinMode(Cam2Lens_Pin, OUTPUT);
  pinMode(Clock_Pin, OUTPUT);
  digitalWrite(Clock_Pin, HIGH);
  SPI.beginTransaction(SPISettings(9600, MSBFIRST, SPI_MODE3));
  move_focus_infinity();
}

void loop() {
  char input[INPUT_SIZE + 1];
  byte size = Serial.readBytes(input, INPUT_SIZE);
  // Add the final 0 to end the C string
  input[size] = 0;

  // Read each command
  char* command = strtok(input, ",");
  while (command != 0)
  {
    // input command is assumed to be an integer.
    int idx_plane = atoi(command);
    move_focus(idx_plane);

    // Find the next command in input string
    command = strtok(0, ",");

  }
}

void move_focus(int idx_plane) {
  // Move focus state of lens with index of plane (values for each plane are predefined)
  
  if (idx_plane > 10) {
    Serial.print("  - wrong idx");
    return;
  }
  //else if (idx_plane == 0){ // commenting this out is and using relative is indeed more stable
  // Serial.print("  - move to infinity idx");
  // move_focus_infinity(); 
  //}
  else {
    // Below print cmds are for python
    
    Serial.print("  - from arduino: moving to the ");
    Serial.print(idx_plane);
    Serial.print("th plane");      
    int offset = focus[idx_plane] - read_int_EEPROM(current_focus);
    Serial.print(offset); 
    if (offset != 0){
      ///////////////////////////////////
      // This is what you send to lens //
      ///////////////////////////////////
      byte HH = highByte(offset);
      byte LL = lowByte(offset);
      init_lens();
      SPI.transfer(0x44);       delay(10);
      SPI.transfer(HH);         delay(10);
      SPI.transfer(LL);         delay(10);
      write_int_EEPROM(current_focus, focus[idx_plane]);  
    }
  }
  delay(100);

}


void move_focus_value(int value) {
  // Move focus state of lens with exact value
  
  int offset = value - read_int_EEPROM(current_focus);
  byte HH = highByte(offset);
  byte LL = lowByte(offset);
  init_lens();
  SPI.transfer(0x44);      delay(10);
  SPI.transfer(HH);        delay(10);
  SPI.transfer(LL);        delay(10);

  write_int_EEPROM(current_focus, value);
}

void move_focus_infinity(){
  init_lens();
  SPI.transfer(0x05);       delay(10);
  
  write_int_EEPROM(current_focus, focus[0]);
}

void write_int_EEPROM(int address, int number)
{ 
  EEPROM.write(address, number >> 8);
  EEPROM.write(address + 1, number & 0xFF);
}

int read_int_EEPROM(int address)
{
  return (EEPROM.read(address) << 8) + EEPROM.read(address + 1);
}
