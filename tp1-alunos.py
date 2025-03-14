import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = True
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
RENDER_MODE = None #seleccione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000
SHOW_ALL = False

env = gym.make("LunarLander-v3", render_mode =RENDER_MODE, 
    continuous=True, gravity=GRAVITY, 
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
    turbulence_power=TURBULENCE_POWER)


def check_successful_landing(observation):
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        if SHOW_ALL:
            print("✅ Aterragem bem sucedida!")
        return True

    if SHOW_ALL:
        print("⚠️ Aterragem falhada!")        
    return False      
        
def simulate(steps=1000,seed=None, policy = None):    
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ)

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success



#Perceptions
##TODO: Defina as suas perceções aqui
def getX(observation):
    return observation[0]

def getY(observation):
    return observation[1]

def getVx(observation):
    return observation[2]

def getVy(observation):
    return observation[3]

def geto(observation):
    return observation[4]

def getVo(observation):
    return observation[5]

def getLLT(observation):
    return observation[6]

def getRLT(observation):
    return observation[7]


#Actions
##TODO: Defina as suas ações aqui
def goUp(action):
    action =+ np.array([1,0])
    return action

def leftThrust(action):
    action =+ np.array([0,-1])
    return action

def rightThrust(action):
    action =+ np.array([0,1])
    return action

def reactive_agent(observation):

    #print (observation[2])

    action = [0, 0]
    
    if (ENABLE_WIND == False):
        if (getLLT(observation) and getRLT(observation) and abs(getX(observation)) < 0.19):
            action = [0,0]
            return action

        #if falling too fast slow down
        elif (getVy(observation) < -0.19):#and abs(geto(observation)) < np.deg2rad(45)):
            return  goUp(action)

        #avoid having too much angular speed
        elif (getVo(observation) < -0.1):
            return leftThrust(action)  
        elif (getVo(observation) > 0.1):
            return rightThrust(action) 

        #avoid having too much horizontal speed
        elif (getVx(observation) < -0.08):
            return rightThrust(action) + goUp(action)
        elif (getVx(observation) > 0.08):
            return leftThrust(action) + goUp(action)

        #try not to rotate too much between the flags
        elif (geto(observation) > np.deg2rad(0) and abs(getX(observation)) < 0.19):
            return rightThrust(action)
        elif (geto(observation) < -np.deg2rad(0) and abs(getX(observation)) < 0.19):
            return leftThrust(action)

        #if not in flags area move to flags area
        elif (getX(observation) < -0.19 and geto(observation) < np.deg2rad(8) and getVy(observation) < 0.05):
            return leftThrust(action) + goUp(action)
        elif (getX(observation) < -0.19 and geto(observation) > np.deg2rad(8) and getVy(observation) < 0.05):
            return rightThrust(action) + goUp(action)
        elif (getX(observation) > 0.19 and geto(observation) < np.deg2rad(8) and getVy(observation) < 0.05):
            return leftThrust(action) + goUp(action)
        elif (getX(observation) > 0.19 and geto(observation) > np.deg2rad(8) and getVy(observation) < 0.05):
            return rightThrust(action) + goUp(action)
        
    else:
        if (getLLT(observation) and getRLT(observation) and abs(getX(observation)) < 0.19):
            action = [0,0]
            return action
        
        elif (getX(observation) < -0.19 and geto(observation) > -np.deg2rad(8) and getVy(observation) < 0):
            return rightThrust(action) + goUp(action)
        elif (getX(observation) < -0.19 and geto(observation) < -np.deg2rad(8) and getVy(observation) < 0):
            return leftThrust(action) + goUp(action)
        elif (getX(observation) > 0.19 and geto(observation) > np.deg2rad(8) and getVy(observation) < 0):
            return rightThrust(action) + goUp(action)
        elif (getX(observation) > 0.19 and geto(observation) < np.deg2rad(8) and getVy(observation) < 0):
            return leftThrust(action) + goUp(action)
        
        elif (geto(observation) > np.deg2rad(0) and abs(getX(observation)) < 0.19):
            return rightThrust(action)
        elif (geto(observation) < -np.deg2rad(0) and abs(getX(observation)) < 0.19):
            return leftThrust(action)
        
        elif (-0.18 < getVy(observation) < 0):
            return goUp()
        
        #if falling too fast slow down
        elif (getVy(observation) < -0.19 and abs(getX(observation)) < 0.19):
            return  goUp(action)
        
    return action 
    
    
def keyboard_agent(observation):
    action = [0,0] 
    keys = pygame.key.get_pressed()
    
    #print('observação:',observation)

    if keys[pygame.K_UP]:  
        action =+ np.array([1,0])
    if keys[pygame.K_LEFT]:  
        action =+ np.array( [0,-1])
    if keys[pygame.K_RIGHT]: 
        action =+ np.array([0,1])

    return action
    

success = 0.0
steps = 0.0
for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy= reactive_agent)
    if su:
        steps += st
    success += su
    
    if su>0:
        finalSteps = steps/(su*(i+1))*100
        if SHOW_ALL:
            print('Média de passos das aterragens bem sucedidas:', steps/(su*(i+1))*100)
    finalRate = success/(i+1)*100
    if SHOW_ALL:
        print('Taxa de sucesso:', success/(i+1)*100)
        

print('\nWind Active: ', ENABLE_WIND)
print('\nMédia de passos das aterragens bem sucedidas:', finalSteps)
print('Taxa de sucesso:', finalRate) 