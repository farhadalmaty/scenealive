"""The catalogue of everything the system can name: English name (used as the detector prompt), Russian name, category, severity
and whether it moves. One table drives the vocabulary, the page, the alerts and the agent.

severity:  danger  - warn at once (weapons, fire, open manhole, predators, broken glass ...)
           caution - worth a mention when close (vehicles, big animals, holes, ice, stairs, curbs ...)
           info    - just a fact"""
from __future__ import annotations

# name | russian | category | severity | moves(1/0)
_TABLE = """
person | человек | people | info | 1
car | машина | vehicle | caution | 1
bus | автобус | vehicle | caution | 1
truck | грузовик | vehicle | caution | 1
van | фургон | vehicle | caution | 1
motorcycle | мотоцикл | vehicle | caution | 1
bicycle | велосипед | vehicle | caution | 1
scooter | самокат | vehicle | caution | 1
tram | трамвай | vehicle | caution | 1
train | поезд | vehicle | caution | 1
stroller | коляска | vehicle | info | 1
wheelchair | инвалидная коляска | vehicle | info | 1
shopping cart | тележка | vehicle | info | 1
dog | собака | animal | caution | 1
cat | кошка | animal | info | 1
bird | птица | animal | info | 1
chicken | курица | animal | info | 1
duck | утка | animal | info | 1
horse | лошадь | animal | caution | 1
cow | корова | animal | caution | 1
goat | коза | animal | caution | 1
sheep | овца | animal | info | 1
pig | свинья | animal | info | 1
deer | олень | animal | caution | 1
rat | крыса | animal | caution | 1
snake | змея | animal | danger | 1
bear | медведь | animal | danger | 1
wolf | волк | animal | danger | 1
wall | стена | structure | info | 0
building | здание | structure | info | 0
fence | забор | structure | info | 0
door | дверь | structure | info | 0
glass door | стеклянная дверь | structure | caution | 0
window | окно | structure | info | 0
stairs | лестница | structure | caution | 0
curb | бордюр | structure | caution | 0
ramp | пандус | structure | info | 0
railing | перила | structure | info | 0
pillar | колонна | structure | info | 0
bridge | мост | structure | info | 0
elevator | лифт | structure | info | 0
escalator | эскалатор | structure | caution | 0
curtain | штора | indoor | info | 0
sofa | диван | indoor | info | 0
chair | стул | indoor | info | 0
table | стол | indoor | info | 0
bed | кровать | indoor | info | 0
tree | дерево | nature | info | 0
bush | куст | nature | info | 0
hedge | живая изгородь | nature | info | 0
branch | ветка | nature | caution | 0
rock | камень | nature | caution | 0
log | бревно | nature | caution | 0
flower bed | клумба | nature | info | 0
pole | столб | furniture | info | 0
traffic light | светофор | furniture | info | 0
street sign | дорожный знак | furniture | info | 0
stop sign | знак «стоп» | furniture | info | 0
bollard | столбик | furniture | info | 0
barrier | ограждение | furniture | info | 0
fire hydrant | гидрант | furniture | info | 0
bench | скамейка | furniture | info | 0
trash can | урна | furniture | info | 0
mailbox | почтовый ящик | furniture | info | 0
parking meter | паркомат | furniture | info | 0
kiosk | киоск | furniture | info | 0
billboard | рекламный щит | furniture | info | 0
construction cone | дорожный конус | obstacle | caution | 0
scaffolding | строительные леса | obstacle | caution | 0
dumpster | мусорный контейнер | obstacle | info | 0
barricade | заграждение | obstacle | caution | 0
chain | цепь | obstacle | caution | 0
cable | кабель | obstacle | caution | 0
snow pile | сугроб | obstacle | info | 0
speed bump | лежачий полицейский | obstacle | info | 0
open manhole | открытый люк | hazard | danger | 0
hole | яма | hazard | caution | 0
pothole | выбоина | hazard | caution | 0
ditch | канава | hazard | caution | 0
puddle | лужа | hazard | caution | 0
ice | лёд | hazard | caution | 0
mud | грязь | hazard | caution | 0
broken glass | разбитое стекло | hazard | danger | 0
syringe | шприц | hazard | danger | 0
fire | огонь | hazard | danger | 0
smoke | дым | hazard | danger | 0
exposed wire | оголённый провод | hazard | danger | 0
hammer | молоток | hazard | caution | 0
crowbar | лом | hazard | caution | 0
scissors | ножницы | hazard | caution | 0
knife | нож | weapon | danger | 0
gun | огнестрельное оружие | weapon | danger | 0
weapon | оружие | weapon | danger | 0
axe | топор | weapon | danger | 0
chainsaw | бензопила | weapon | danger | 0
baseball bat | бита | weapon | caution | 0
trash | мусор | trash | info | 0
garbage bag | мусорный пакет | trash | info | 0
plastic bag | пластиковый пакет | trash | info | 0
litter | мелкий мусор | trash | info | 0
bottle | бутылка | trash | info | 0
can | банка | trash | info | 0
cardboard box | картонная коробка | trash | info | 0
pile of trash | куча мусора | trash | caution | 0
debris | обломки | trash | caution | 0
construction waste | строительный мусор | trash | caution | 0
road | дорога | surface | info | 0
sidewalk | тротуар | surface | info | 0
asphalt | асфальт | surface | info | 0
cracked road | потрескавшаяся дорога | surface | caution | 0
paving tiles | плитка | surface | info | 0
tactile paving | тактильная плитка | surface | info | 0
crosswalk | пешеходный переход | surface | info | 0
gravel | гравий | surface | info | 0
dirt road | грунтовая дорога | surface | info | 0
sand | песок | surface | info | 0
grass | трава | surface | info | 0
manhole cover | люк | surface | info | 0
tram tracks | трамвайные пути | surface | caution | 0
"""

CATEGORY_RU = {"people": "люди", "vehicle": "транспорт", "animal": "животные", "structure": "конструкции", "indoor": "в помещении",
               "nature": "растения и камни", "furniture": "уличные объекты", "obstacle": "препятствия", "hazard": "опасности",
               "weapon": "оружие и острые предметы", "trash": "мусор", "surface": "покрытие"}

ENTRIES: list[tuple[str, str, str, str, bool]] = []
for _line in _TABLE.strip().splitlines():
    _n, _ru, _cat, _sev, _mv = [x.strip() for x in _line.split("|")]
    assert _cat in CATEGORY_RU and _sev in ("danger", "caution", "info"), _line
    ENTRIES.append((_n, _ru, _cat, _sev, _mv == "1"))

NAMES = [e[0] for e in ENTRIES]
RU_NAME = {e[0]: e[1] for e in ENTRIES}
CATEGORY = {e[0]: e[2] for e in ENTRIES}
SEVERITY = {e[0]: e[3] for e in ENTRIES}
MOVES = {e[0] for e in ENTRIES if e[4]}
assert len(set(NAMES)) == len(NAMES), "duplicate names in the catalogue"


def names_in_category(cat: str) -> list[str]:
    return [n for n in NAMES if CATEGORY[n] == cat]


def names_with_severity(*sev: str) -> list[str]:
    return [n for n in NAMES if SEVERITY[n] in sev]


# --------------------------------------------------------------------------------------- surfaces (looked at as regions of the ground)
# name | russian | quality (good / fair / bad)
_SURFACES = """
asphalt | асфальт | good
concrete pavement | бетон | good
paving tiles | плитка | good
tactile paving | тактильная плитка | good
crosswalk | пешеходный переход | good
indoor floor | пол в помещении | good
carpet | ковёр | good
wet asphalt | мокрый асфальт | fair
cobblestone | брусчатка | fair
gravel | гравий | fair
dirt | грунт | fair
sand | песок | fair
grass | трава | fair
cracked asphalt | потрескавшийся асфальт | bad
pothole | яма на дороге | bad
puddle | лужа | bad
mud | грязь | bad
snow | снег | bad
ice | лёд | bad
stairs | лестница | bad
"""
SURFACES: list[tuple[str, str, str]] = [tuple(x.strip() for x in l.split("|")) for l in _SURFACES.strip().splitlines()]
SURFACE_RU = {s[0]: s[1] for s in SURFACES}
SURFACE_QUALITY = {s[0]: s[2] for s in SURFACES}
QUALITY_RU = {"good": "хорошее", "fair": "среднее", "bad": "плохое"}
