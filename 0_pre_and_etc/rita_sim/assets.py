"""Fetch only referenced model assets, with pinned revisions and SHA-256 records."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'sim_models'
MENAGERIE = '370c43965323c0944b433b202aac5058b5f907a9'
KORTEX = 'c3280d8a6a6b5d96590d51538e3f265710a6c77d'
ROBOT_IDS = ('kinova', 'g1', 'misty', 'go2')


def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'Rita-Sim/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        temporary = destination.with_suffix(destination.suffix + '.part')
        temporary.write_bytes(payload)
        temporary.replace(destination)
    payload = destination.read_bytes()
    return {'path': destination.relative_to(MODELS).as_posix(), 'url': url,
            'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def setup():
    import mujoco
    records = []
    for robot, folder, filename in [('g1', 'unitree_g1', 'g1.xml'),
                                    ('go2', 'unitree_go2', 'go2.xml')]:
        target = MODELS / robot
        base = f'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{MENAGERIE}/{folder}/'
        for name in (filename, 'LICENSE', 'README.md'):
            records.append(fetch(base + name, target / name))
        xml = ET.parse(target / filename)
        meshfiles = sorted({element.get('file') for element in xml.findall('.//asset/mesh')})
        with ThreadPoolExecutor(max_workers=4) as pool:
            records.extend(pool.map(lambda name: fetch(base + 'assets/' + name,
                                                       target / 'assets' / name), meshfiles))
        prepare_menagerie(xml, target / 'model.xml', robot)

    target = MODELS / 'kinova'
    target.mkdir(parents=True, exist_ok=True)
    urdf = ET.parse(ROOT / 'data/robot/GEN3-6DOF_VISION_URDF_ARM_V01.urdf')
    meshfiles = sorted({Path(e.get('filename')).name for e in urdf.findall('.//mesh')})
    base = (f'https://raw.githubusercontent.com/Kinovarobotics/ros_kortex/{KORTEX}/'
            'kortex_description/arms/gen3/6dof/meshes/')
    with ThreadPoolExecutor(max_workers=4) as pool:
        records.extend(pool.map(lambda name: fetch(base + name, target / 'assets' / name), meshfiles))
    shutil.copyfile(ROOT / 'data/robot/KINOVA_LICENSE', target / 'LICENSE')
    for element in urdf.findall('.//mesh'):
        element.set('filename', Path(element.get('filename')).name)
    extension = ET.SubElement(urdf.getroot(), 'mujoco')
    ET.SubElement(extension, 'compiler', meshdir='assets', discardvisual='false', fusestatic='false')
    source = target / 'source.urdf'
    urdf.write(source, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(source))
    converted = target / 'model.xml'
    mujoco.mj_saveLastXML(str(converted), model)
    xml = ET.parse(converted)
    add_scene(xml.getroot(), 'kinova')
    xml.write(converted, encoding='utf-8', xml_declaration=True)

    make_misty(MODELS / 'misty/model.xml')
    manifest = {'menagerie_commit': MENAGERIE, 'kortex_commit': KORTEX,
                'misty': 'Local simplified 5-DOF head/arms model; not an official model.',
                'files': records, 'download_bytes': sum(r['bytes'] for r in records)}
    (MODELS / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Downloaded model sources: {manifest['download_bytes']/1024**2:.1f} MiB")


def add_scene(root, robot):
    # All four environments are deliberately kinematic, with contact disabled.
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('gravity', '0 0 0')
    flag = option.find('flag')
    if flag is None:
        flag = ET.SubElement(option, 'flag')
    flag.set('contact', 'disable')
    for tag in ('actuator', 'sensor', 'keyframe', 'contact'):
        for element in root.findall(tag):
            root.remove(element)
    for geom in root.findall('.//geom'):
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', name='rita_key', pos='1 -1 3', dir='-0.3 0.3 -1', diffuse='.85 .85 .85')
    ET.SubElement(world, 'light', name='rita_fill', pos='-1 1 2', dir='.3 -.3 -1', diffuse='.45 .45 .45')
    ET.SubElement(world, 'geom', name='rita_floor', type='plane', size='3 3 .1',
                  rgba='.16 .19 .24 1', contype='0', conaffinity='0')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='960')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.6 .6 .6')


def prepare_menagerie(xml, destination, robot):
    root = xml.getroot()
    key = root.find('keyframe/key')
    home = [float(x) for x in key.get('qpos').split()]
    # Pin the floating root; no locomotion/balance controller is needed.
    base = root.find('worldbody/body')
    base.set('pos', ' '.join(map(str, home[:3])))
    base.set('quat', ' '.join(map(str, home[3:7])))
    for parent in root.iter():
        for element in list(parent):
            if element.tag == 'freejoint' or (element.tag == 'joint' and element.get('type') == 'free'):
                parent.remove(element)
            elif element.tag == 'geom' and element.get('class') in ('collision', 'foot'):
                parent.remove(element)
    add_scene(root, robot)
    xml.write(destination, encoding='utf-8', xml_declaration=True)


def make_misty(destination):
    # Dimensions are approximate; joint ranges follow the Misty API degree table.
    xml = '''<mujoco model="misty_simplified_5dof">
  <compiler angle="degree"/>
  <default><geom rgba=".88 .91 .94 1" contype="0" conaffinity="0"/></default>
  <worldbody>
    <body name="base" pos="0 0 .065">
      <geom type="ellipsoid" size=".115 .09 .055" rgba=".13 .15 .18 1"/>
      <geom type="cylinder" pos="0 .09 -.025" euler="90 0 0" size=".04 .012" rgba=".08 .09 .11 1"/>
      <geom type="cylinder" pos="0 -.09 -.025" euler="90 0 0" size=".04 .012" rgba=".08 .09 .11 1"/>
      <body name="torso" pos="0 0 .095">
        <geom type="ellipsoid" size=".075 .07 .10"/>
        <geom type="box" pos=".068 0 0" size=".008 .037 .035" rgba=".14 .17 .20 1"/>
        <body name="left_arm" pos="0 .077 .025">
          <joint name="left_arm" axis="0 1 0" range="-29 90"/>
          <geom type="capsule" fromto="0 0 0 .09 0 0" size=".016"/>
        </body>
        <body name="right_arm" pos="0 -.077 .025">
          <joint name="right_arm" axis="0 1 0" range="-29 90"/>
          <geom type="capsule" fromto="0 0 0 .09 0 0" size=".016"/>
        </body>
        <body name="neck_yaw" pos="0 0 .105">
          <joint name="head_yaw" axis="0 0 1" range="-81 81"/>
          <inertial mass=".01" pos="0 0 0" diaginertia=".00001 .00001 .00001"/>
          <body name="neck_roll">
            <joint name="head_roll" axis="1 0 0" range="-40 40"/>
            <inertial mass=".01" pos="0 0 0" diaginertia=".00001 .00001 .00001"/>
            <body name="head" pos="0 0 .025">
              <joint name="head_pitch" axis="0 1 0" range="-40 26"/>
              <geom type="ellipsoid" size=".072 .079 .055"/>
              <geom type="box" pos=".063 0 0" size=".008 .055 .033" rgba=".025 .035 .05 1"/>
              <geom type="sphere" pos=".073 .026 .004" size=".010" rgba=".1 .8 1 1"/>
              <geom type="sphere" pos=".073 -.026 .004" size=".010" rgba=".1 .8 1 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>'''
    root = ET.fromstring(xml)
    add_scene(root, 'misty')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)
