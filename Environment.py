import numpy as np
import math
import random

class Point():
    def __init__(self,pt):
        self.x = float(pt[0])
        self.y = float(pt[1])

    def __add__(self,point):
        return Point((self.x+point.x,self.y+point.y))

    def __sub__(self,point):
        return Point((self.x-point.x,self.y-point.y))

    def __mul__(self,value):
        return Point((value*self.x,value*self.y))

    def distance(self,point):
        return math.sqrt((self.x-point.x)**2 + (self.y-point.y)**2)

    def __str__(self):
        return 'P('+'{0:.2f}'.format(self.x)+','+'{0:.2f}'.format(self.y)+')'

class Vector():
    def __init__(self,p1,p2):
        self.point = p1
        self.end_point = p2
        self.vector = p2-p1

    def length(self):
        return math.sqrt(self.vector.x**2 + self.vector.y**2)

    def angle(self):
        if self.vector.x==0:
            return 0.0
        return math.atan2(self.vector.y,self.vector.x)

    def __add__(self,vec):
        return Vector(self.point,self.vector+vec.vector)

    def __sub__(self,vec):
        return Vector(self.point,self.vector-vec.vector)

    def __mul__(self,value):
        return Vector(self.point,self.point+(self.vector*value))

    def dot(self,vec):
        return (self.vector.x*vec.vector.x + self.vector.y*vec.vector.y)

    def cross(self,vec):
        return (self.vector.x*vec.vector.y - self.vector.y*vec.vector.x)

    def perpendicular(self):
        return Vector(self.point,self.point+Point((-self.vector.y,self.vector.x)))

    def unit_vector(self):
        return Vector(self.point,self.point+(self.vector*(1.0/self.length())))

    def in_segment(self,point):
        if self.vector.x != 0: # Not Vertical
            if ((self.point.x <= point.x) and (point.x <= self.point.x+self.vector.x)):
                return True
            if ((self.point.x >= point.x) and (point.x >= self.point.x+self.vector.x)):
                return True
        else:
            if ((self.point.y <= point.y) and (point.y <= self.point.y+self.vector.y)):
                return True
            if ((self.point.y >= point.y) and (point.y >= self.point.y+self.vector.y)):
                return True
        return False

    def intersection(self,vec):
        # http://geomalgorithms.com/a05-_intersect-1.html#intersect2D_2Segments()
        # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
        dp = Vector(vec.point,self.point)
        denom = self.cross(vec)
        if denom == 0: # Parallel segments
            return None
        X_point = (vec*(self.cross(dp)/denom)).vector + vec.point
        if (self.in_segment(X_point)==True and vec.in_segment(X_point)==True):
            return X_point
        else:
            return None

    def perpendicular_base_length(self,point):
        l_2 = self.vector.x**2 + self.vector.y**2
        l = math.sqrt(l_2)
        d = (l_2 + (self.point.x-point.x)**2 + (self.point.y-point.y)**2 - (self.end_point.x-point.x)**2 - (self.end_point.y-point.y)**2 )/(2*l)
        return d

    def draw_points(self):
        return (self.point.x,self.point.x+self.vector.x),(self.point.y,self.point.y+self.vector.y)

    def __str__(self):
        return 'V:'+str(self.vector.x)+'i'+ ('+' if self.vector.y>=0 else '-') +str(abs(self.vector.y))+'j'

class Car():
    # Bicycle model (Not to be mistaken for Ackerman model)
    # Current version: Robotics, Vision and Control by Peter Croke - 4.2 Car-like Mobile Robots
    # Another version: http://www.me.berkeley.edu/~frborrel/pdfpub/IV_KinematicMPC_jason.pdf
    def __init__(self,car_detail=None):
        self.detail = car_detail
        # Controls
        self.gamma = 0 # Steering angle
        self.v = 0 # Heading velocity
        # States
        self.theta = self.detail['state'][2] # Heading angle
        self.x = self.detail['state'][0] # Position x
        self.y = self.detail['state'][1] # Position y
        self.state = 'running' # or collided
        # Sensing
        self.sensor_reading = None
        # Timing
        self.steps = 0
        # Scoring
        self.score = 0
        self.prev_score = 0
        self.delta_state = np.array([0,0,0])

    def constrain(self,quantity,c_min,c_max):
        return min(max(quantity,c_min),c_max)

    def wrap_angle(self,val):
        return( ( val + np.pi) % (2 * np.pi ) - np.pi )

    def encode_angle(self,val):
        return [math.sin(val),math.cos(val)]

    def update(self,dt):
        self.x += dt*self.v*math.cos(self.theta)
        self.y += dt*self.v*math.sin(self.theta)
        self.theta += (dt*self.v/self.detail['L'])*math.tan(self.gamma)
        self.theta = self.wrap_angle(self.theta)

    def set_steering(self,angle):
        self.gamma = self.constrain(angle,-self.detail['gamma_limit'],self.detail['gamma_limit'])

    def get_steering(self):
        return self.gamma

    def set_velocity(self,velocity):
        self.v = self.constrain(velocity,-self.detail['v_limit'],self.detail['v_limit'])

    def get_velocity(self):
        return self.v

    def set_state(self,state):
        self.x,self.y,self.theta = state

    def get_state(self):
        return self.x,self.y,self.theta

    def set_sensor_reading(self,reading):
        self.sensor_reading = reading

    def get_sensor_reading(self):
        return self.sensor_reading

    def get_delta_state(self):
        return self.delta_state

    def random_state(self,mean,variance,angle_variance):
        self.set_state([mean[0]+random.uniform(-variance,variance), mean[1]+random.uniform(-variance,variance), mean[2]+random.uniform(-angle_variance,angle_variance)])

    def scale(self,val,valmin,valmax,mi,ma):
        return ((val-valmin) * float(ma-mi)/(valmax-valmin))+mi

    def get_state_to_train(self,delta_limit):
        s,c = self.encode_angle(self.delta_state[2])
        dist = self.scale(math.sqrt(self.delta_state[0]**2 + self.delta_state[1]**2),0,delta_limit,0,1)
        dstate = np.array([dist,s,c])
        return dstate

    def reset(self):
        self.gamma = 0 # Steering angle
        self.v = 0 # Heading velocity
        self.theta = self.detail['state'][2] # Heading angle
        self.x = self.detail['state'][0] # Position x
        self.y = self.detail['state'][1] # Position y
        self.state = 'running' # or collided
        self.steps = 0
        self.score = 0
        self.prev_score = 0
        self.delta_state = np.array([0,0,0])

class Environment():
    def __init__(self,environment_details,max_steps):
        self.max_steps = max_steps
        self.start_angle = environment_details['start_angle']
        self.border_segments = []
        points = environment_details['points'][:]
        points.append(points[0])
        for i in range(1,len(points)):
            self.border_segments.append(Vector(Point(points[i-1]),Point(points[i])))
        self.destination = Point(environment_details['dest'])
        self.destination_radius = environment_details['dest_radius']
        points_np = np.array(points)
        self.limits = (points_np[:,0].min(),points_np[:,0].max(),points_np[:,1].min(),points_np[:,1].max())
        self.max_delta = math.sqrt((self.limits[1]-self.limits[0])**2 + (self.limits[3]-self.limits[2])**2)

    def rotation_matrix(self,theta):
        ct = math.cos(theta)
        st = math.sin(theta)
        R = np.array([[ct,-st],[st,ct]])
        return R

    def find_range(self,pos,angle,length):
        R = self.rotation_matrix(angle)
        points = np.array([ [0,0],[length,0] ]).T
        points = np.dot(R,points)
        points[0,:] += pos.x
        points[1,:] += pos.y
        sensor = Vector(Point((points[0][0],points[1][0])),Point((points[0][1],points[1][1])))
        intersections = []
        for seg in self.border_segments:
            X = sensor.intersection(seg)
            if X is not None:
                intersections.append(X)
        dist = np.array([pos.distance(p) for p in intersections])
        return pos.distance(intersections[np.argmin(dist)]) if len(dist)>0 else length

    def point_orientation_wrt_segment(self,point,segment):
        # From https://www.cs.cmu.edu/~quake/robust.html
        return True if ((segment[0][0]-point[0])*(segment[1][1]-point[1]) - (segment[1][0]-point[0])*(segment[0][1]-point[1]))>0 else False

    def point_projection_on_segment(self,point,segment):
        # https://stackoverflow.com/questions/17581738/check-if-a-point-projected-on-a-line-segment-is-not-outside-it
        dx = segment[1][0] - segment[0][0]
        dy = segment[1][1] - segment[0][1]
        innerProduct = (point[0] - segment[0][0])*dx + (point[1] - segment[0][1])*dy
        return ((0<=innerProduct) and (innerProduct<=(dx*dx+dy*dy)))

    def check_point_inside_polygon(self,point,polygon):
        # From http://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
        pos_segment = Vector(point, Point((1000,point.y)))
        intercepts = [pos_segment.intersection(seg) for seg in polygon]
        inside = len([i for i in intercepts if i is None])%2 != 0
        return inside

    def compute_interaction(self,agents):
        for agent in agents:
            x,y,theta = agent.get_state()
            car_pos = Point((x,y))
            # Sensor update
            sensor_values = []
            for s in agent.detail['sensors']:
                sensor_values.append(self.find_range(car_pos,theta+s['angle'],s['range']))
            agent.set_sensor_reading(sensor_values)
            # Calculate euclidian distance from destination
            agent.prev_score = agent.score
            agent.score = car_pos.distance(self.destination)
            # Collision update
            inside = self.check_point_inside_polygon(car_pos,self.border_segments)
            # Delta state
            delta = Vector(car_pos,self.destination)
            agent.delta_state = np.array([delta.vector.x,delta.vector.y,delta.angle()-theta])
            # Update agent condition
            if not inside:
                agent.state = 'collided'
            # Timing
            elif agent.steps > self.max_steps:
                agent.state = 'timeup'
            # Check destination reached
            elif agent.score<self.destination_radius:
                agent.state = 'destination'
            if agent.state=='running':
                agent.steps += 1

    def set_max_steps(self,value):
        self.max_steps = value

    def randomize(self):
        self.destination.x,self.destination.y = random.uniform(self.limits[0],self.limits[1]),random.uniform(self.limits[2],self.limits[3])

def env_generator(env_dict,env_select):
    env_dict['path'] = env_dict[env_select][:]
    env_dict['points'] = env_dict[env_select][:]
    env_dict['start_angle'] = 0.0
