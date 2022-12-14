import pygame
import numpy as np
import time
import sys


class XboxController:
    def __init__(self,scales = [1.0,1.0,1.0,1.0]):
        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        
        self.lStick = LeftStick(self.controller.get_axis(0),
                                   self.controller.get_axis(1))
        self.rStick = RightStick(self.controller.get_axis(4),
                                     self.controller.get_axis(3))
        dPadDirs = getDirs(self.controller)
        
        self.dPad = DPad(dPadDirs)
        self.trigger = Trigger(self.controller.get_axis(2))
        self.inUse = [False,False,False,False]

        length = len(scales)
        self.offsets = np.zeros(length)
        self.uScale = np.ones(length)
        self.lScale = np.ones(length)
        self.driftLimit = .05
        self.calibrate()
        self.scales = np.array(scales)
        time.sleep(1)
        self.calibrate()
        
    def getUpdates(self):
        for event in pygame.event.get(): # User did something
            if event.type == pygame.JOYBUTTONDOWN and self.controller.get_button(7) == 1.0: # If user clicked close
                return None
        state = self.getControllerState()
        updates = self.convert(state)
        result = updates * self.scales
        return result
        
    def calibrate(self):
        save_stdout = sys.stdout
        sys.stdout = open('trash', 'w') 
        # Calibrate sticks, reset offsets and scaling factors 
        length = len(self.offsets)
        self.offsets = np.zeros(length)
        self.uScale = np.ones(length)
        self.lScale = np.ones(length)
        
        state = self.getControllerState()      
        self.offsets = self.convert(state)    
        self.uScale = abs(1/(np.ones(length)-self.offsets))
        self.lScale = abs(1/(-np.ones(length)-self.offsets))
        sys.stdout = save_stdout
        
    def convert(self,state):
        left_stick_horizontal = state['left_stick'][0]
        left_stick_vertical = state['left_stick'][1]
        right_stick_horizontal = state['right_stick'][0] 
        right_stick_vertical = state['right_stick'][1]

        # Offset
        updates = np.array([left_stick_horizontal, left_stick_vertical,
                    right_stick_horizontal, right_stick_vertical]) - self.offsets
        
        # Scale upper and lower bounds 
        for i in range(0,len(updates)):
            if updates[i] > 0:
                updates[i] = updates[i]*self.uScale[i]
            else:
                updates[i] = updates[i]*self.lScale[i]
            if abs(updates[i]) < self.driftLimit:
                updates[i] = 0
        return updates
        

    def getControllerState(self):
        pygame.event.clear()
        self.update()
        state = {'left_stick':self.lStick.getPos(),
                     'right_stick':self.rStick.getPos(),
                     'd_pad':self.dPad.getPos(),
                     'trigger':self.trigger.getPos()
                     }
        return state
            
    def update(self):
        self.lStick.setCurrent(self.controller.get_axis(0),
                                   self.controller.get_axis(1))
        self.rStick.setCurrent(self.controller.get_axis(3),
                                     self.controller.get_axis(4))
        self.dPad.setCurrent(getDirs(self.controller))
        self.trigger.setCurrent(self.controller.get_axis(2))

    def isInUse(self):
        self.inUse = [self.lStick.isInUse(), self.rStick.isInUse(),
                      self.dPad.isInUse(), self.trigger.isInUse()]
        for thing in self.inUse:
            if thing:
                return thing

        return False

    def override(self):
        return self.controller.get_button(9) == 1.0


def getDirs(controller):
    # Returns dirs in up down left right order
    dPadDirs = [
                controller.get_button(0),
                controller.get_button(1),
                controller.get_button(2),
                controller.get_button(3)
    ]
    return dPadDirs


class LeftStick:
    def __init__(self, axis0, axis1):
        self.initA0 = axis0
        self.initA1 = axis1

        self.a0 = self.initA0
        self.a1 = self.initA1

    def getPos(self):
        return np.array([self.a0, self.a1])

    def setCurrent(self, a0, a1):
        self.a0 = a0
        self.a1 = a1
        return self.getPos()

    def isInUse(self):
        return (self.a0!=self.initA0 or self.a1!=self.initA1)


class RightStick:
    def __init__(self, axis0, axis1):
        self.initA0 = axis0
        self.initA1 = axis1

        self.a0 = self.initA0
        self.a1 = self.initA1

    def getPos(self):
        return np.array([self.a0, self.a1])

    def setCurrent(self, a0, a1):
        self.a0 = a0
        self.a1 = a1
        return self.getPos()

    def isInUse(self):
        return (self.a0!=self.initA0 or self.a1!=self.initA1)

class DPad:
    def __init__(self, dirs):
        hat = self.conv(dirs)
        self.initH = dirs
        self.h = self.initH

    def getPos(self):
        return self.h

    def setCurrent(self, dirs):
        self.h = self.conv(dirs)
        return self.getPos()

    def conv(self, dirs):
        up, down, left, right = dirs
        x, y = 0,0
        if left == 1.0:
            x = -1.0
        elif right == 1.0:
            x = 1.0
        if up == 1.0:
            y = 1.0
        elif down == 1.0:
            y = -1.0
        return x,y
    
    def isInUse(self):
        return self.h!=self.initH

class Trigger:
    def __init__(self, axis0):
        self.initA0 = axis0
        self.a0 = self.initA0

    def getPos(self):
        return self.a0

    def setCurrent(self, a0):
        self.a0 = a0
        return self.getPos()

    def isInUse(self):
        return self.a0!=self.initA0

if __name__ == '__main__':
    controller = XboxController()
    for i in range(10000):
        time.sleep(0.1)
        state = controller.getControllerState()
        print("State")
        print(state)