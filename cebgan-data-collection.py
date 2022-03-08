#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import csv
import weakref

try:
    sys.path.append(glob.glob('../carla-simulator/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.behavior_agent import BehaviorAgent

from buffered_saver_lc import BufferedImageSaver

import random
from PIL import Image
import scipy.misc as misc
import math
import logging

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_m
    from pygame.locals import K_o
    from pygame.locals import K_a
    from pygame.locals import K_l
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_j
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_t
    from pygame.locals import K_u
    from pygame.locals import K_b
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_DOWN
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, vehicle, autopilot_enabled=True):
        self.vehicle = vehicle
        self.autopilot_enabled = autopilot_enabled
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0
        self.start_data_collection = False
        self.get_waypoint = False
        

    def parse_events(self, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == K_m:
                    self.autopilot_enabled = False
                    self.vehicle.set_autopilot(self.autopilot_enabled)
                    print('Autopilot Off')
                if event.key == K_o:
                    self.autopilot_enabled = True
                    self.vehicle.set_autopilot(self.autopilot_enabled)
                    print('Autopilot On; Lane change deactivated!')
                if event.key == K_c:
                    self.start_data_collection = True
                    print('Starting data collection')
                if event.key == K_p:
                    self.start_data_collection = False
                    print('Pausing data collection')
                if event.key == K_g:
                    print("getting waypoint")
                    self.get_waypoint = True

            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

        if not self.autopilot_enabled:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self.control.reverse = self.control.gear < 0          
            self.vehicle.apply_control(self.control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self.control.throttle = 1.0 if keys[K_UP] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT]:
            if self.steer_cache > 0:
                self.steer_cache = 0
            else:
                self.steer_cache -= steer_increment
        elif keys[K_RIGHT]:
            if self.steer_cache < 0:
                self.steer_cache = 0
            else:
                self.steer_cache += steer_increment
        else:
            self.steer_cache = 0.0
        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))
        self.control.steer = round(self.steer_cache, 1)
        self.control.brake = 1.0 if keys[K_DOWN] else 0.0
        self.control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        #self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        #self.hud.notification('Crossed line %s' % ' and '.join(text))
        print('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# ------------------------------------------------------------------------------
# ==============================================================================

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))
    
    return image.raw_data, array

def draw_image2(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    """
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0)) # (800, 0)
    """
    return image.raw_data, array


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.
        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=10.0)


def main():
    # change this to your local path where you want to save data
    data_path = '/home/apoorva/Research/CEBGAN/data/town01-lf-data/'
    trace_number = 0
    BIS = BufferedImageSaver(data_path, 100, 800, 600, 3, 'CameraRGB', trace_number)
    actor_list = []
    pygame.init()
    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.load_world('Town01') #get_world()# 
    world.set_weather(carla.WeatherParameters.Default)
    tm = client.get_trafficmanager(3000)
    tm.set_synchronous_mode(True)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    tot_target_reached = 0
    num_min_waypoints = 5
    obstacle = None
    
    try:
        m = world.get_map()
        spawn_points = m.get_spawn_points()
        start_pose = spawn_points[6]
        end_location = spawn_points[43].location
        #random.choice(spawn_points).location 
        blueprint_library = world.get_blueprint_library()
        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.audi.a2')),
            start_pose)
        actor_list.append(vehicle)
        world.player = vehicle
        tm.ignore_lights_percentage(vehicle,100)
        tm.auto_lane_change(vehicle, False)
        agent = BasicAgent(vehicle)
        agent.set_destination([end_location.x, end_location.y, end_location.z])
        #agent = BehaviorAgent(vehicle, ignore_traffic_light=True, behavior='cautious')
        #agent.set_destination(agent.vehicle.get_location(), end_location, clean=True)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=2.5, z=1.5)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_top_view = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8),
                            carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_top_view)

        controller = KeyboardControl(world, vehicle, True)
        # Create a synchronous mode context.
        steering_list = []
        with CarlaSyncMode(world, camera_rgb, camera_top_view, fps=30) as sync_mode:
            while True:
                if controller.parse_events(clock):
                    return
                #agent.update_information(world)
                clock.tick()
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_topview = sync_mode.tick(timeout=2.0)
                # Draw the display.
                raw, img = draw_image2(display, image_rgb) #draw_image2
                raw2, img2 = draw_image(display, image_topview)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                if vehicle.is_at_traffic_light():
                    traffic_light = vehicle.get_traffic_light()
                    if traffic_light.get_state() == carla.TrafficLightState.Red:
                        traffic_light.set_state(carla.TrafficLightState.Green)
                        traffic_light.set_green_time(10.0)

                """
                # Set new destination when target has been reached
                if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints:
                    agent.reroute(spawn_points)
                    tot_target_reached += 1
                    print("ReRouting")
                """         

                if controller.autopilot_enabled:
                    control = agent.run_step()
                    #steering_list.append(control.steer)
                    vehicle.apply_control(control)

                if controller.get_waypoint:
                    # to just the vehicle 10meters ahead usually to move fast in functions
                    location = vehicle.get_location()
                    ego_vehicle_wp = m.get_waypoint(location)
                    next_loc = list(ego_vehicle_wp.next(10))[0].transform.location
                    vehicle.set_location(next_loc)
                    controller.get_waypoint = False

                v = vehicle.get_velocity()
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)), (500, 10)) # 8, 10
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)), (500, 28)) # 8, 28
                display.blit(
                    font.render('% 5f speed (ego-car)' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)), True, (255, 255, 255)), (500, 48)) # 8, 48
                display.blit(
                    font.render('% 5f steering ' % control.steer, True, (255, 255, 255)), (500, 68)) # 8, 68
                pygame.display.flip()
    

                if controller.start_data_collection:
                    if BIS.index % 20 == 0:
                        print(BIS.index)
                    BIS.add_image(raw,
                                  control.steer,
                                  0,
                                  0,
                                  0,
                                  0,
                                  1000,
                                  'CameraRGB')

    finally:
        print('destroying actors.')
        print(steering_list)
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
