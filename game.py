from pyglet.gl import *
from pyglet.window import key
import math
import numpy as np
import sys
import argparse

CAMERA_DISTANCE = 2.5
CAMERA_HEIGHT = 0.5
SEA_COLOR_1 = ('c3f', (0.2, 0.2, 0.9,) * 4)
SEA_COLOR_2 = ('c3f', (0.3, 0.3, 1.0,) * 4)
RUNWAY_COLOR = ('c3f', (0.3, 0.3, 0.3,) * 4)
RUNWAY_STRIPE_COLOR = ('c3f', (0.8, 0.8, 0.8,) * 4)
LIGHT_COLOR = ('c3f', (1.0, 0.0, 0.0,) * 4)
MASS = 3
INERTIA = 10
IS_JOYSTICK = False
IS_AUTOPILOT = False
g = 9.81/10
f_g = np.array([0, 0, MASS*g])
MAX_SPEED = 4
CRUISE_SPEED = MAX_SPEED
CRUISE_ALTITUDE = 20
CRITICAL_STALL_ANGLE = np.pi/12
CRASH_SPEED = 2
UPDATE = 0

Jx = 0.8244 #kg m^2
Jy = 1.135
Jz = 1.759
Jxz = 0.1204

gamma = Jx * Jz - (Jxz**2)
gamma1 = (Jxz * (Jx - Jy + Jz)) / gamma
gamma2 = (Jz * (Jz - Jy) + (Jxz**2)) / gamma
gamma3 = Jz / gamma
gamma4 = Jxz / gamma
gamma5 = (Jz - Jx) / Jy
gamma6 = Jxz / Jy
gamma7 = ((Jx - Jy) * Jx + (Jxz**2)) / gamma
gamma8 = Jx / gamma

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def Rzyx(phi, theta, psi):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.vstack([
        np.hstack([cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth]),
        np.hstack([spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi]),
        np.hstack([-sth, cth*sphi, cth*cphi])
    ])

def Tzyx(phi, theta):
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)
 
    if cth==0:
        raise ValueError('Tzyx is singular for theta = +-90 degrees')

    return np.vstack([
        np.hstack([1, sphi*sth/cth, cphi*sth/cth]),
        np.hstack([0, cphi, -sphi]),
        np.hstack([0, sphi/cth, cphi/cth])
    ])

def Jzyx(phi, theta, psi):
    R = Rzyx(phi, theta, psi)
    T = Tzyx(phi, theta)
    J12 = np.zeros((3,3))

    return np.block([
        [R, J12],
        [J12, T]
    ])

def princip(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi

W,H=800,600
class Airplane:
    def __init__(self):
        self.batch = pyglet.graphics.Batch()

        self.thruster = 0
        self.elevator = 0
        self.rudder_angle = 0
        self.flaps = 0
        self.aileron = 0
        self.f_drag = 0
        self.f_lift = 0
        self.f_thruster = 0

        if (IS_JOYSTICK):
            try:
                self.controller = pygame.joystick.Joystick(0)
                self.controller.init()
            except:
                sys.exit(1)

        self.eta = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.nu = np.array([10, 0, 0, 0, 0, 0], dtype=np.float64)
        self.nu_dot = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.eta_dot = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.integrators = {}
        self.last_val = {}
        self.derivatives = {}
        self.landed = self.eta[2] == 0

    def draw(self, frame):
        x,y,z,phi,theta,psi = self.eta
        x = -x
        y = -y
        z = -z

        phi_deg = (phi)*180/np.pi
        theta_deg = (theta)*180/np.pi
        psi_deg = (psi)*180/np.pi

        cam_x = -y-CAMERA_DISTANCE*np.sin(psi)
        cam_z = -x-CAMERA_DISTANCE*np.cos(psi)
        cam_y = -z-CAMERA_HEIGHT
        
        glRotatef(-psi_deg, 0, 1, 0)
        glTranslatef(cam_x,cam_y,cam_z) 

        counter = 0
        TILE_WIDTH = 20
        TILE_LENGTH = 20
        TILE_NUM = max(5, int(np.sqrt(np.sqrt(abs(z/10))))*5)

        bgxlower =  TILE_WIDTH*int(y//TILE_WIDTH) - TILE_NUM*TILE_WIDTH
        bgxupper =  TILE_WIDTH*int(y//TILE_WIDTH) + TILE_NUM*TILE_WIDTH
        bgylower =  TILE_WIDTH*int(x//TILE_LENGTH) - TILE_NUM*TILE_LENGTH
        bgyupper =  TILE_WIDTH*int(x//TILE_LENGTH) + TILE_NUM*TILE_LENGTH

        for bgxi, bgx in enumerate(range(bgxlower, bgxupper, TILE_WIDTH)):
            for bgyi, bgy in enumerate(range(bgylower, bgyupper, TILE_LENGTH)):
                #color = SEA_COLOR_1 if (bgxi+bgyi) % 2 == 0 else SEA_COLOR_2
                color = SEA_COLOR_1 if (bgx//TILE_WIDTH+bgy//TILE_LENGTH) % 2 == 0 else SEA_COLOR_2

                #d = np.sqrt((y-bgx)**2 + (x-bgy)**2)
                #ang = np.arctan(abs(z)/d)

                #print(z, d, ang)

                #if d < max(5, np.sqrt(abs(z)))*20:
                pyglet.graphics.draw(4, GL_QUADS,
                    ('v3f',(
                        bgx-TILE_WIDTH,0,bgy-TILE_LENGTH, 
                        bgx-TILE_WIDTH,0,bgy+TILE_LENGTH, 
                        bgx+TILE_WIDTH,0,bgy+TILE_LENGTH, 
                        bgx+TILE_WIDTH,0,bgy-TILE_LENGTH, 
                    )),color
                )

        #runway
        pyglet.graphics.draw(4, GL_QUADS,
            ('v3f',(
                -1,0,bgylower-TILE_LENGTH, 
                1,0,bgylower-TILE_LENGTH, 
                1,0,bgyupper+TILE_LENGTH, 
                -1,0,bgyupper+TILE_LENGTH, 
            )),RUNWAY_COLOR
        )

        for bgyi, bgy in enumerate(range(bgylower, bgyupper, int(TILE_LENGTH//5))):
            pyglet.graphics.draw(4, GL_QUADS,
                ('v3f',(
                    -0.01,0,bgy, 
                    0.01,0,bgy, 
                    0.01,0,bgy+int(TILE_LENGTH//5)*0.5, 
                    -0.01,0,bgy+int(TILE_LENGTH//5)*0.5, 
                )),RUNWAY_STRIPE_COLOR
            )

        #lights
        if ((frame//25) % 2 == 0):
            for bgyi, bgy in enumerate(range(bgylower, bgyupper, TILE_LENGTH)):
                pyglet.graphics.draw(4, GL_QUADS,
                    ('v3f',(
                        -1.1,0,bgy, 
                        -1.1,0.1,bgy, 
                        -1.0,0.1,bgy, 
                        -1.0,0,bgy, 
                    )),LIGHT_COLOR
                )
                pyglet.graphics.draw(4, GL_QUADS,
                    ('v3f',(
                        1.1,0,bgy, 
                        1.1,0.1,bgy, 
                        1.0,0.1,bgy, 
                        1.0,0,bgy, 
                    )),LIGHT_COLOR
                )

        dbg = [cam_x, cam_y, cam_z, x, y, z, phi_deg, theta_deg, psi_deg]
        #print(('{:.2f}, '*len(dbg)).format(*dbg))

        v0 = np.array([0, -0.5, 0])
        v1 = np.array([0, 0.5, 0])
        v4 = np.array([0, 0, 0])
        v5 = np.array([0, -0.1, 0])
        v5r = np.array([0, 0.1, 0])

        v2 = np.array([-1, 0.1, 0])
        v3 = np.array([-1, -0.1, 0])
        v6 = np.array([-1, 0, 0])
        vc = np.array([-0.5, 0, 0])
        
        vt = np.array([0, 0, 0.2])
        vt1 = np.array([0, 0.1, 0.2])
        vt2 = np.array([0, -0.1, 0.2])

        v0r = Rzyx(phi, theta, psi).dot(v0)
        v1r = Rzyx(phi, theta, psi).dot(v1)
        v2r = Rzyx(phi, theta, psi).dot(v2)
        v3r = Rzyx(phi, theta, psi).dot(v3)
        v4r = Rzyx(phi, theta, psi).dot(v4)
        v5rr = Rzyx(phi, theta, psi).dot(v5r)
        v5r = Rzyx(phi, theta, psi).dot(v5)
        v6r = Rzyx(phi, theta, psi).dot(v6)
        vcr = Rzyx(phi, theta, psi).dot(vc)
        vtr = Rzyx(phi, theta, psi).dot(vt)
        vt1r = Rzyx(phi, theta, psi).dot(vt1)
        vt2r = Rzyx(phi, theta, psi).dot(vt2)

        # shadow
        pyglet.graphics.draw(4, GL_QUADS,
            ('v3f',(
                y+v0r[1],0,x+v0r[0], 
                y+v4r[1],0,x+v4r[0], 
                y+v4r[1],0,x+v4r[0], 
                y+v6r[1],0,x+v6r[0]
            )),('c3f', (0.0, 0.0, 0.0,) * 4)
        )
        pyglet.graphics.draw(4, GL_QUADS,
            ('v3f',(
                y+v1r[1],0,x+v1r[0], 
                y+v4r[1],0,x+v4r[0], 
                y+v4r[1],0,x+v4r[0], 
                y+v6r[1],0,x+v6r[0]
            )),('c3f', (0.0, 0.0, 0.0,) * 4)
        )

        #glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        pyglet.graphics.draw(4, GL_QUADS,
            ('v3f',(
                y+v0r[1],z+v0r[2],x+v0r[0], 
                y+v4r[1],z+v4r[2],x+v4r[0], 
                y+v4r[1],z+v4r[2],x+v4r[0], 
                y+v6r[1],z+v6r[2],x+v6r[0]
            )),('c3f', (1.0, 1.0, 1.0,) * 4)
        )
        pyglet.graphics.draw(4, GL_QUADS,
            ('v3f',(
                y+v1r[1],z+v1r[2],x+v1r[0], 
                y+v4r[1],z+v4r[2],x+v4r[0], 
                y+v4r[1],z+v4r[2],x+v4r[0], 
                y+v6r[1],z+v6r[2],x+v6r[0]
            )),('c3f', (1.0, 1.0, 1.0,) * 4)
        )
        pyglet.graphics.draw(4, GL_QUADS,
            ('v3f',(
                y+vtr[1],z+vtr[2],x+vtr[0],
                y+v5r[1],z+v5r[2],x+v5r[0], 
                y+v5rr[1],z+v5rr[2],x+v5rr[0], 
                y+vtr[1],z+vtr[2],x+vtr[0]
            )),('c3f', (0.0, 0.3, 1.0,) * 4)
        )

        #glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        

    def update(self,dt,keys):

        if (self.eta[2] > 0):
            if (abs(self.eta_dot[2]) > CRASH_SPEED or
                abs(self.eta_dot[1]) > CRASH_SPEED or 
                abs(self.eta[3]) > np.pi/24 or
                abs(self.eta[4]) > np.pi/24):
                raise ValueError('Crash! Score was: ' + str(int(self.eta[0])))
            else:
                self.eta[3] = 0
                self.eta[4] = 0
                self.eta[2] = 0
                self.nu[2] = 0
                self.nu[1] = 0
                self.nu[4] = 0
                self.nu[3] = 0
                self.nu[5] = 0
                self.landed = True
        else:
            self.landed = False

        x,y,z,phi,theta,psi = self.eta
        x_d, y_d, z_d, phi_d, theta_d, psi_d = self.eta_dot
        u,v,w,p,q,r = self.nu

        V_a = np.sqrt(u**2+v**2+w**2)

        alpha = np.arctan(w/u)
        beta = np.arcsin(v/V_a)
        a0 = CRITICAL_STALL_ANGLE
        M = 30
        sig = (1 + np.exp(-M*(alpha-a0)) + np.exp(M*(alpha+a0)))/((1 + np.exp(-M*(alpha-a0)))*(1 + np.exp(M*(alpha+a0))))
        C_L_0 = 0.5
        C_L_a = 50
        C_L_lin = (C_L_0 + C_L_a*alpha)
        C_L_stall = (2*np.sign(alpha)*np.sin(alpha)**2*np.cos(alpha))
        C_L = (1-sig)*C_L_lin + sig*C_L_stall

        C_D_0 = 2
        C_D_a = 1

        self.f_lift = (1+1*self.flaps)*max(0, 0.05*V_a**2*C_L)
        self.f_drag = (1+1*self.flaps)*0.001*V_a**2 * (C_D_0 + C_D_a*alpha)**2

        self.f_thruster = 0.016*max(0, (MAX_SPEED**2*self.thruster - u**2))
        self.l = 0.01*V_a**2 * (-0.1*beta - 30*p/V_a - 0*r/V_a - self.aileron)
        self.m = 0.05*V_a**2 * (-3*alpha - 50*q/V_a + self.elevator)
        self.n = 0.001*V_a**2 * (5*beta - 10*p/V_a - 100*r/V_a + self.rudder_angle)

        f_x = np.cos(alpha)*(-self.f_drag) - np.sin(alpha)*(-self.f_lift)
        f_z = np.sin(alpha)*(-self.f_drag) + np.cos(alpha)*(-self.f_lift)

        f_g_b = (Rzyx(phi, theta, 0).T).dot(f_g)

        self.nu_dot[0] = ((self.f_thruster + f_g_b[0]+f_x)/MASS + r*v - q*w)
        self.nu_dot[1] = (f_g_b[1]/MASS + p*w - r*u)
        self.nu_dot[2] = ((f_g_b[2]+f_z)/MASS + q*u - p*v)
        self.nu_dot[3] = gamma1*p*q - gamma2*q*r + gamma3*self.l + gamma4*self.n
        self.nu_dot[4] = gamma5*p*r - gamma6*(p**2 - r**2) + 1/Jy*self.m
        self.nu_dot[5] = gamma7*p*q - gamma1*q*r + gamma4*self.l + gamma8*self.n

        self.nu[0] += dt*self.nu_dot[0]
        if (not self.landed):
            self.nu[1] += dt*self.nu_dot[1]
        if (not self.landed and self.nu_dot[2] > 0):
            self.nu[2] += dt*self.nu_dot[2]
        if (not self.landed):
            self.nu[3] += dt*self.nu_dot[3]
        self.nu[4] += dt*self.nu_dot[4]
        if (not self.landed):
            self.nu[5] += dt*self.nu_dot[5]

        self.eta_dot = Jzyx(phi, theta, psi).dot(self.nu)
        self.eta += dt*self.eta_dot
        self.eta[3:] = princip(self.eta[3:])

        if (IS_AUTOPILOT):
            if not 'e_chi' in self.integrators: self.integrators['e_chi'] = 0
            if not 'e_chi' in self.last_val: self.last_val['e_chi'] = 0
            if not 'e_chi' in self.derivatives: self.derivatives['e_chi'] = 0
            if not 'e_phi' in self.integrators: self.integrators['e_phi'] = 0
            if not 'e_phi' in self.last_val: self.last_val['e_phi'] = 0
            if not 'e_phi' in self.derivatives: self.derivatives['e_phi'] = 0
            if not 'e_z' in self.integrators: self.integrators['e_z'] = 0
            if not 'e_z' in self.last_val: self.last_val['e_z'] = 0
            if not 'e_z' in self.derivatives: self.derivatives['e_z'] = 0
            if not 'e_speed' in self.integrators: self.integrators['e_speed'] = 0
            if not 'e_speed' in self.last_val: self.last_val['e_speed'] = 0
            if not 'e_speed' in self.derivatives: self.derivatives['e_speed'] = 0

            z_d = -CRUISE_ALTITUDE
            e_z = z_d - z
            
            chi = np.arctan2(self.eta_dot[1], self.eta_dot[0])
            chi_weight = sigmoid(e_z)
            chi_d = chi_weight*np.pi + (1-chi_weight)*chi

            e_speed = CRUISE_SPEED-u - e_z - self.integrators['e_z']
            d_e_speed = e_speed - self.last_val['e_speed']
            if (self.landed):
                self.thruster = 5
            else:
                self.thruster = np.clip(2*e_speed - 10*self.derivatives['e_speed'] + 1*self.integrators['e_speed'], 0, 3)
            self.integrators['e_speed'] += e_speed*dt
            self.derivatives['e_speed'] = 0.9*self.derivatives['e_speed'] + 0.1*d_e_speed
            self.last_val['e_speed'] = e_speed

            self.elevator = -0.038*e_z + -10*self.derivatives['e_z']
            if (z < -1):
                self.elevator = np.clip(self.elevator, -np.pi/24, np.pi/24)

            d_e_phi = e_z - self.last_val['e_z']
            self.integrators['e_z'] += e_z*dt
            self.derivatives['e_z'] = 0.9*self.derivatives['e_z'] + 0.1*d_e_phi
            self.last_val['e_z'] = e_z
            
            e_chi = chi_d - chi
            d_e_chi = e_chi - self.last_val['e_chi']

            self.integrators['e_chi'] += e_chi*dt
            self.derivatives['e_chi'] = 0.9*self.derivatives['e_chi'] + 0.1*d_e_chi

            phi_d = np.clip(1*e_chi + 1*self.derivatives['e_chi'] + 0.01*self.integrators['e_chi'], -np.pi/6, np.pi/6)
            e_phi = phi_d - phi
            d_e_phi = e_phi - self.last_val['e_phi']

            self.integrators['e_phi'] += e_phi*dt
            self.derivatives['e_phi'] = 0.9*self.derivatives['e_phi'] + 0.1*d_e_phi

            self.aileron = np.clip(-1*e_phi - 100*self.derivatives['e_phi'] + 0*self.integrators['e_phi'], -1, 1)

            self.last_val['e_chi'] = e_chi
            self.last_val['e_phi'] = e_phi

        else:
            if (IS_JOYSTICK):
                event = pygame.event.poll()
                if (event.type != pygame.NOEVENT):
                    pass
                    #print('Event, ', event)
                if (event.type == pygame.JOYAXISMOTION):
                    # if (event.axis == 0):
                    #     self.nu[5] = round(event.value, 2)
                    if (event.axis == 1):
                        #pass
                        # up down
                        self.thruster = -10*round(event.value, 2)
                    elif (event.axis == 2):
                        # left up
                        self.aileron = round(event.value, 2)
                    elif (event.axis == 3):
                        self.elevator = round(event.value, 2)

                elif (event.type == pygame.JOYBUTTONUP):
                    if (event.button == 4):
                        self.rudder_angle = 0
                    elif (event.button == 5):
                        self.rudder_angle = 0
                    if (event.button == 6):
                        self.flaps = 0

                elif (event.type == pygame.JOYBUTTONDOWN):
                    if (event.button == 4):
                        self.rudder_angle = 1
                    elif (event.button == 5):
                        self.rudder_angle = -1
                    if (event.button == 6):
                        self.flaps = 1
            else:
                if keys[key.UP]: self.elevator = -1
                elif keys[key.DOWN]: self.elevator = 1
                else: self.elevator = 0
                if keys[key.LEFT]: self.aileron = -1
                elif keys[key.RIGHT]: self.aileron = 1
                else: self.aileron = 0

class Window(pyglet.window.Window):

    def Projection(self): glMatrixMode(GL_PROJECTION); glLoadIdentity()
    def Model(self): glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    def set2d(self):self.Projection();gluOrtho2D(0,self.width,0,self.height); self.Model()
    def set3d(self):self.Projection();gluPerspective(70,self.width/self.height,0.05,1000);self.Model()


    def setLock(self,state): self.lock = state; self.set_exclusive_mouse(state)

    lock = False;mouse_lock = property(lambda self:self.lock,setLock)
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.set_minimum_size(400,200)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        pyglet.clock.schedule(self.update)
        self.airplane = Airplane()
        self.frame = 0

        # The label that is displayed in the top left of the canvas.
        self.labels = [pyglet.text.Label('', font_name='Arial', font_size=10,
            x=10, y=self.height - 20*(i+1), anchor_x='left', anchor_y='top',
            color=(0, 0, 0, 255)) for i in range(20)]

        self.stall_warning = pyglet.text.Label('', font_size=50,
            x=self.width-200, y=self.height - 40*(0+1), anchor_x='left', anchor_y='top',
            color=(255, 0, 0, 255))

    def update(self,dt):
        self.frame += 1
        self.airplane.update(dt,self.keys)
    def on_draw(self):
        self.clear()
        self.set3d()
        self.airplane.draw(self.frame)
        self.set2d()
        self.draw_label()

    def draw_label(self):
        """ Draw the label in the top left of the screen.
        """
        x, y, z, phi, theta, psi = self.airplane.eta
        u, v, w, p, q, r = self.airplane.nu
        u_dot, v_dot, w_dot, p_dot, q_dot, r_dot = self.airplane.nu_dot
        alpha = np.arctan(w/u)
        V_a = np.sqrt(u**2+v**2+w**2)
        beta = np.arcsin(v/V_a)

        self.labels[0].text = 'Roll [deg]: %.2f' % (phi*180/np.pi,)
        self.labels[0].draw()
        self.labels[1].text = 'Pitch [deg]: %.2f' % (theta*180/np.pi,)
        self.labels[1].draw()
        self.labels[3].text = 'Pos: (%.2f, %.2f, %.2f)' % (x, y, z)
        self.labels[3].draw()
        self.labels[4].text = 'Speed: %.2f (%.2f, %.2f, %.2f)' % (V_a, u, v, w)
        self.labels[4].draw()
        self.labels[5].text = 'Acceleration: (%.2f, %.2f, %.2f)' % (u_dot, v_dot, w_dot)
        self.labels[5].draw()
        self.labels[6].text = 'Angle of attack: %.2f' % (alpha,)
        self.labels[6].draw()
        self.labels[7].text = 'Sideslip angle: %.2f' % (beta,)
        self.labels[7].draw()

        self.labels[9].text = 'Drag: %.2f' % (self.airplane.f_drag,)
        self.labels[9].draw()
        self.labels[10].text = 'Lift: %.2f' % (self.airplane.f_lift,)
        self.labels[10].draw()
        self.labels[11].text = 'Thruster: %.2f' % (self.airplane.f_thruster,)
        self.labels[11].draw()
        self.labels[12].text = 'Elevators: %.2f' % (self.airplane.elevator,)
        self.labels[12].draw()
        self.labels[13].text = 'Ailerons: %.2f' % (self.airplane.aileron,)
        self.labels[13].draw()
        self.labels[14].text = 'Rudder angle: %.2f' % (self.airplane.rudder_angle,)
        self.labels[14].draw()
        self.labels[15].text = 'Flaps: %.2f' % (self.airplane.flaps,)
        self.labels[15].draw()

        if (alpha > CRITICAL_STALL_ANGLE):
            self.stall_warning.text = 'Stall!'
            self.stall_warning.draw()


        

def setup():
    """ Basic OpenGL configuration.
    """
    # Set the color of "clear", i.e. the sky, in rgba.
    glClearColor(0.25, 0.34, 0.5, 1)
    # Enable culling (not rendering) of back-facing facets -- facets that aren't
    # visible to you.
    glEnable(GL_CULL_FACE)
    # Set the texture minification/magnification function to GL_NEAREST (nearest
    # in Manhattan distance) to the specified texture coordinates. GL_NEAREST
    # "is generally faster than GL_LINEAR, but it can produce textured images
    # with sharper edges because the transition between texture elements is not
    # as smooth."
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Flight simulator!!!.')
    parser.add_argument('--joystick', action='store_true', help='If you have a ps4-controller connected through USB')
    parser.add_argument('--autopilot', action='store_true', help='Use autopilot controller')
    #parser.add_argument('--speed', type=int, help='Initial airplane speed', default=5)
    args = parser.parse_args()
    IS_JOYSTICK = args.joystick
    IS_AUTOPILOT = args.autopilot

    if (IS_JOYSTICK):
        import pygame
        pygame.init()
        pygame.joystick.init()

    setup()
    window = Window(width=W,height=H, caption='Flight Simulator',resizable=True)
    window.maximize()
    pyglet.gl.glClearColor(0.5,0.7,1,1)
    pyglet.app.run()
