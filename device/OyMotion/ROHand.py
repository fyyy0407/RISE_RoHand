import asyncio
import os
import signal
import sys
import time
import math

from pymodbus import FramerType
from pymodbus.client import ModbusSerialClient

from lib_gforce import gforce
from lib_gforce.gforce import EmgRawDataConfig, SampleResolution

from roh_registers_v1 import *

class ROHand:
    """
    API for Robot Hand Control

    Args:
    COM_PORT,NODE_ID,NUM_FINGERS: ROHand Configuration
    """
    def __init__(self,COM_PORT="/dev/ttyUSB0",NODE_ID=2,NUM_FINGERS=6):
        super(ROHand,self).__init__()
        self.NODE_ID = NODE_ID
        self.NUM_FINGERS = NUM_FINGERS
        self.client = ModbusSerialClient(COM_PORT,FramerType.RTU,115200)

    def connect(self):
        self.client.connect()
    
    def disconnect(self):
        self.client.close()

    def get_cali_max(self):
        '''
        Return the maximum calibration value for each finger 
        '''
        clib_max = self.client.read_holding_registers(ROH_CALI_END0,self.NUM_FINGERS,self.NODE_ID)
        return clib_max.registers

    def get_cali_min(self):
        '''
        Return the minimum calibration value for each finger 
        '''
        clib_min = self.client.read_holding_registers(ROH_CALI_START0,self.NUM_FINGERS,self.NODE_ID)
        return clib_min.registers

    def get_status(self):
        '''
        Return the status of each finger
        '''
        status = self.client.read_holding_registers(ROH_FINGER_STATUS0,self.NUM_FINGERS,self.NODE_ID)
        return status.registers

    def get_force_limit(self):
        '''
        Return the force limit for each finger
        '''
        force_limit = self.client.read_holding_registers(ROH_FINGER_FORCE_LIMIT0,self.NUM_FINGERS,self.NODE_ID)
        return status.registers
    
    def set_force(self,registerID,force):
        '''
        Set the forces for fingers
        Args: 
        registerID: Range from ROH_FINGER_FORCE0 to ROH_FINGER_FORCE4
        force: Size can be 1 to 5, setting a singer finger to all fingers
        '''
        resp = self.client.write_registers(registerID,force,self.NODE_ID)
        return resp
    
    def get_speed(self):
        '''
        Return the speed of each finger
        '''
        speed = self.client.read_holding_registers(ROH_FINGER_SPEED0,self.NUM_FINGERS,self.NODE_ID)
        return speed.registers


    def set_speed(self,registerID,speed):
        '''
        Set the speeds for fingers
        Args: 
        registerID: range from ROH_FINGER_SPEED0 to ROH_FINGER_SPEED5
        speed: Size can be 1 to 6, setting a singer finger to all fingers.
               The last value represents the rotation speed of thumb.

        '''
        resp = self.client.write_registers(registerID,speed,self.NODE_ID)
        return resp

    def set_finger_pos(self,registerID,pos):
        '''
        Set the positions for fingers
        Args: 
        registerID: range from ROH_FINGER_POS_TARGET0 to ROH_FINGER_POS_TARGET5
        pos: Size can be 1 to 6, setting a singer finger to all fingers.
               The last value represents the rotation position of thumb.

        '''
        resp = self.client.write_registers(registerID,pos,self.NODE_ID)
        return resp


    def get_current_pos(self):
        '''
        Return current position for each finger
        '''
        current_pos = self.client.read_holding_registers(ROH_FINGER_POS0,self.NUM_FINGERS,self.NODE_ID)
        return current_pos.registers    
        
    def set_finger_angle(self,registerID,angle):
        '''
        Set the angles for fingers
        Args: 
        registerID: range from ROH_FINGER_ANGLE_TARGET0 to ROH_FINGER_ANGLE_TARGET5
        pos: Size can be 1 to 6, setting a singer finger to all fingers.
               The last value represents the rotation angle of thumb.

        '''
        resp = self.client.write_registers(registerID,angle,self.NODE_ID)
        return resp

    def get_current_angle(self):
        '''
        Return current angle for each finger
        '''
        current_angle = self.client.read_holding_registers(ROH_FINGER_ANGLE0,self.NUM_FINGERS,self.NODE_ID)
        return current_angle.registers   
    
    def reset(self):
        '''
        Initialize positions for fingers
        '''
        finger_data = [0,0,0,0,0,0]
        resp = self.client.write_registers(ROH_FINGER_POS_TARGET0, finger_data,self.NODE_ID)