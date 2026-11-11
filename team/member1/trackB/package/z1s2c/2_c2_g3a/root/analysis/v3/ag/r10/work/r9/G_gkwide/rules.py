import re, math, statistics

INF = float('inf')
G = 9.81
GEXC = (r'\b(toy|toys|model|models|miniature|mini|lego|doll|dolls|dollhouse|statue|sculpture|figurine|cartoon|drawing|picture|'
        r'poster|painting|photo|image|sign|signs|lettering|text|logo|shadow|reflection|replica|plush|stuffed|cake|cutout|robot|rc|'
        r'inflatable|origami|paper|icon|emoji|animated|virtual|hologram|projection|mural)\b')

PERSONH = r'(person|man|woman|pedestrian|human|walker|lady|guy|gentleman|shooter|passer|receiver|player|athlete|adult|teacher|student|people)'
CARH = r'(car|cars|sedan|taxi|cab|suv|hatchback|coupe|jeep)'

T = [
 ('person', ('size',), {'height': (1.40, 2.10), 'width': (0.30, 0.65)}, PERSONH, None,
  r'child|kid|boy|girl|baby|toddler|infant|sitting|seated|kneel|crouch|squat|lying|riding|scooter|bike|bicycle|horse|shoulder|arm|leg|foot|feet|hand|head|astronaut|jump|dive|ladder|stilts',
  'adult standing height 1.40-2.10 m, shoulder width 0.30-0.65 m'),
 ('baby', ('size',), {'height': (0.40, 1.10), 'length': (0.40, 1.10)}, r'(baby|infant|toddler)', None, None, 'newborn ~0.5 m, toddler <= ~1.0 m'),
 ('person_walk', ('speed', 'avgspeed'), {None: (0.4, 2.5)}, PERSONH, r'walk|pedestrian|stroll', r'run|jog|sprint|ride|riding|cycl|skat|swim|ski\b|scooter|fall|jump|car\b|treadmill',
  'walking speed 0.4 (stroll) - 2.5 m/s (race walking pace below)'),
 ('person_swim', ('speed', 'avgspeed'), {None: (0.05, 2.5)}, PERSONH + r'|swimmer', r'swim', None, 'swimming speed <= 2.5 m/s (elite sprint ~2.4 m/s)'),
 ('person_v', ('speed', 'avgspeed'), {None: (0.0, 12.5)}, PERSONH + r'|runner|jogger', None, r'ride|riding|cycl|bike|bicycle|skat|ski\b|scooter|car\b|fall|jump|dive|slid|vehicle|train|horse|throw|ball|hand',
  'human running speed <= 12.5 m/s (Bolt peak ~12.4 m/s)'),
 ('person_a', ('accel',), {None: (0.0, 10.0)}, PERSONH + r'|runner|swimmer', None, r'fall|jump|dive|ride|riding|car\b|vehicle|ski|astronaut', 'sprint start acceleration <= ~10 m/s^2'),
 ('car', ('size',), {'length': (2.5, 6.5), 'width': (1.4, 2.3), 'height': (1.1, 2.1)}, CARH, None,
  r'cart|carpet|card|forklift|train|lane|road|light|cable|truck|bus|race|kart|wheel|tire|tyre|door|window|seat|roof|trunk|hood|park',
  'passenger car: length 2.5 (microcar) - 6.5 m (pickup), width 1.4 (kei) - 2.3 m (with mirrors), height 1.1 (sports) - 2.1 m (van/SUV)'),
 ('car_v', ('speed', 'avgspeed'), {None: (0.0, 70.0)}, CARH, None, r'race|f1|formula|kart|train|cart|roller|toy', 'road car speed <= 70 m/s (250 km/h)'),
 ('car_a', ('accel',), {None: (0.0, 15.0)}, CARH + r'|vehicle|truck|bus|van|forklift', None, r'race|f1|formula|crash|collision|impact|gravit|fall|drop',
  'road vehicle acceleration / braking <= ~1.5 g'),
 ('truck', ('size',), {'length': (4.5, 25.0), 'width': (1.7, 2.7), 'height': (1.6, 4.6)}, r'(truck|lorry|firetruck)', None,
  r'forklift|wheel|tire|tyre|cargo|box|container|bed\b|door|cab\b', 'truck: pickup 4.5 m .. semi 25 m long, 1.7-2.6 m wide (+mirrors), 1.6-4.6 m tall'),
 ('truck_v', ('speed', 'avgspeed'), {None: (0.0, 45.0)}, r'(truck|lorry|firetruck)', None, r'toy|forklift|hand', 'trucks <= 45 m/s (160 km/h)'),
 ('bus', ('size',), {'length': (6.0, 25.0), 'width': (2.0, 2.7), 'height': (2.4, 4.6)}, r'(bus|coach)', None, r'stop|wheel|door|window', 'bus 6-25 m long, 2.0-2.6 m wide, 2.4-4.6 m tall'),
 ('motorcycle', ('size',), {'length': (1.4, 2.7), 'height': (0.8, 1.6)}, r'(motorcycle|motorbike|cycle)', r'motor', None, 'motorcycle 1.4-2.7 m long, 0.8-1.6 m tall'),
 ('motorcycle_v', ('speed', 'avgspeed'), {None: (0.0, 90.0)}, r'(motorcycle|motorbike)', None, None, 'motorcycle <= 90 m/s'),
 ('bicycle', ('size',), {'length': (0.9, 2.3), 'height': (0.55, 1.35)}, r'(bicycle|bike)s?', None, r'motor|wheel|tire|tyre|sign|lane|rack|person|rider|man|woman',
  'bicycle (child 16 in .. tandem excl.) 0.9-2.3 m long, 0.55-1.35 m tall'),
 ('bicycle_v', ('speed', 'avgspeed'), {None: (0.3, 20.0)}, r'(bicycle|bike|cyclist)s?', None, r'motor|e-?bike|electric', 'cycling 1-70 km/h (downhill), > 0.3 m/s to stay upright in motion'),
 ('scooter', ('size',), {'height': (0.6, 1.6), 'width': (0.3, 0.9)}, r'scooter', None, r'person|rider|man|woman', 'kick / motor scooter 0.6-1.6 m tall, 0.3-0.9 m wide'),
 ('forklift', ('size',), {'height': (1.8, 3.2), 'length': (2.0, 4.8), 'width': (0.8, 2.2)}, r'forklift', None, r'fork tips|fork\b|mast raised|load|pallet',
  'counterbalance forklift: overhead guard / lowered mast 1.8-3.2 m, length incl. forks 2.0-4.8 m, width 0.8-2.2 m'),
 ('forklift_v', ('speed', 'avgspeed'), {None: (0.0, 7.0)}, r'forklift', None, None, 'forklift top speed ~20-25 km/h'),
 ('bike_wheel', ('size',), {'diameter': (0.30, 0.76)}, r'(wheel|tire|tyre)s?', r'bicycle|bike', r'motor|steering|car\b|truck|aircraft|landing', '12-29 in bicycle wheel 0.30-0.76 m'),
 ('car_wheel', ('size',), {'diameter': (0.50, 0.85)}, r'(wheel|tire|tyre)s?', r'\b(car|cars|taxi|sedan|suv|van)\b', r'bicycle|bike|truck|steering|aircraft|landing|toy|kart|cart',
  'passenger car tyre 0.50-0.85 m'),
 ('truck_wheel', ('size',), {'diameter': (0.70, 1.30)}, r'(wheel|tire|tyre)s?', r'truck|lorry', r'bicycle|bike|steering|toy', 'truck tyre 0.70-1.30 m'),
 ('gear_wheel', ('size',), {'diameter': (0.35, 1.60)}, r'(wheel|tire|tyre)s?', r'aircraft|airplane|landing gear', r'nose', 'aircraft main-gear tyre 0.35 (light aircraft) - 1.6 m (A380 ~1.4 m)'),
 ('airplane', ('size',), {'length': (6.0, 85.0), 'height': (1.8, 25.0), 'wingspan': (7.0, 90.0)}, r'(airplane|aeroplane|plane|jet|airliner|aircraft)', None,
  r'paper|glider|drone|model|toy|wing\b|window|door|landing', 'manned fixed-wing aircraft: length 6 (Cessna 150 ~7 m) - 85 m (An-225), height 1.8-25 m, span 7-90 m'),
 ('airplane_v', ('speed', 'avgspeed'), {None: (0.0, 350.0)}, r'(airplane|aeroplane|plane|jet|airliner|aircraft)', None, r'paper|model|toy|drone|glider', 'subsonic aircraft < 350 m/s'),
 ('helicopter', ('size',), {'length': (6.0, 41.0)}, r'helicopter', None, None, 'helicopter length incl. rotor 6-41 m (Mi-26 40 m)'),
 ('balloon_hot', ('size',), {'height': (10.0, 40.0), 'width': (8.0, 30.0)}, r'balloon', r'hot air', None, 'hot-air balloon envelope 10-40 m tall, 8-30 m wide'),
 ('canoe', ('size',), {'length': (2.5, 20.0), 'width': (0.5, 1.1)}, r'(canoe|kayak)', None, None, 'canoe / kayak 2.5 m .. war / outrigger canoe, dragon boat 12-20 m long, 0.5-1.1 m wide'),
 ('canoe_v', ('speed', 'avgspeed'), {None: (0.0, 7.0)}, r'(canoe|kayak)', None, None, 'canoe sprint ~6 m/s'),
 ('speedboat', ('size',), {'length': (3.0, 20.0)}, r'(speedboat|motorboat|speed boat)', None, None, 'speedboat 3-20 m'),
 ('boat_v', ('speed', 'avgspeed'), {None: (0.0, 50.0)}, r'(boat|speedboat|sailboat|ship|canoe|yacht|ferry)', None, r'toy|model|paper|rc', 'boats < 50 m/s (180 km/h)'),
 ('cat', ('size',), {'length': (0.30, 1.25), 'height': (0.15, 0.60)}, r'(cat|kitten)s?', None, r'big cat|lion|tiger|leopard|cheetah|jaguar|kitten',
  'domestic cat body 0.3-0.8 m (to 1.2 m with tail), height 0.15-0.6 m'),
 ('cat_v', ('speed', 'avgspeed'), {None: (0.0, 14.0)}, r'(cat|kitten)s?', None, r'big cat|lion|tiger|leopard|cheetah|jaguar', 'domestic cat sprint <= 14 m/s'),
 ('dog', ('size',), {'length': (0.15, 1.9), 'height': (0.12, 1.1)}, r'(dog|puppy|dogs)', None, r'hot ?dog|house|bed\b', 'dogs: chihuahua .. great dane'),
 ('dog_v', ('speed', 'avgspeed'), {None: (0.0, 20.0)}, r'(dog|puppy|dogs)', None, None, 'greyhound <= 20 m/s'),
 ('horse', ('size',), {'length': (1.5, 3.0), 'height': (1.0, 2.6)}, r'(horse|horses|pony)', None, r'rocking|sea ?horse|carousel|toy',
  'horse / pony: body 1.5-3.0 m, withers 1.0-1.9 m, head up to 2.6 m'),
 ('horse_v', ('speed', 'avgspeed'), {None: (0.0, 20.0)}, r'(horse|horses|pony)', None, r'rocking|sea ?horse|carousel', 'gallop record ~19-20 m/s'),
 ('rabbit', ('size',), {'length': (0.20, 0.80), 'height': (0.12, 0.50)}, r'(rabbit|bunny|hare)s?', None, None, 'rabbit / hare 0.2-0.8 m long'),
 ('rabbit_v', ('speed', 'avgspeed'), {None: (0.0, 20.0)}, r'(rabbit|bunny|hare)s?', None, None, 'hare <= 20 m/s'),
 ('deer', ('size',), {'length': (0.8, 3.1), 'height': (0.35, 2.6)}, r'(deer|elk|moose|reindeer)', None, None, 'pudu 0.8 m .. moose 3.1 m long, shoulder <= 2.1 m (head higher)'),
 ('antelope', ('size',), {'length': (0.45, 3.5), 'height': (0.30, 2.2)}, r'(antelope|gazelle|impala)s?', None, None, 'dik-dik .. eland'),
 ('elephant', ('size',), {'height': (0.80, 4.2)}, r'elephants?', None, None, 'newborn elephant ~0.9 m .. African bull 4.0 m shoulder'),
 ('elephant_v', ('speed', 'avgspeed'), {None: (0.0, 7.5)}, r'elephants?', None, None, 'elephant top speed ~7 m/s'),
 ('raccoon', ('size',), {'length': (0.40, 1.05)}, r'raccoons?', None, None, 'raccoon 0.4-0.7 m body, <= 1.05 m with tail'),
 ('raccoon_v', ('speed', 'avgspeed'), {None: (0.0, 7.0)}, r'raccoons?', None, None, 'raccoon <= 7 m/s'),
 ('turtle', ('size',), {'length': (0.05, 2.2)}, r'(turtle|tortoise)s?', None, None, 'hatchling .. leatherback 2.2 m'),
 ('turtle_v', ('speed', 'avgspeed'), {None: (0.0, 10.0)}, r'(turtle|tortoise)s?', None, None, 'sea turtle burst <= ~10 m/s'),
 ('shark', ('size',), {'length': (0.15, 20.0)}, r'sharks?', None, None, 'dwarf lanternshark 0.2 m .. whale shark 18-20 m'),
 ('shark_v', ('speed', 'avgspeed'), {None: (0.0, 20.0)}, r'sharks?', None, None, 'shortfin mako <= ~20 m/s'),
 ('whale', ('size',), {'length': (2.5, 33.0)}, r'whales?', None, r'shark', 'dwarf sperm whale 2.7 m .. blue whale 30-33 m'),
 ('whale_v', ('speed', 'avgspeed'), {None: (0.0, 15.0)}, r'whales?', None, r'shark', 'fin whale <= ~13 m/s'),
 ('eagle', ('size',), {'length': (0.40, 1.10), 'wingspan': (1.0, 2.9)}, r'eagles?', None, None, 'eagles: body 0.4-1.1 m, span 1.0-2.9 m'),
 ('eagle_v', ('speed', 'avgspeed'), {None: (0.0, 70.0)}, r'eagles?', None, None, 'golden eagle stoop <= ~67 m/s'),
 ('bird', ('size',), {'length': (0.05, 2.8), 'wingspan': (0.08, 3.7)}, r'birds?', None, None, 'hummingbird .. ostrich / albatross'),
 ('bird_v', ('speed', 'avgspeed'), {None: (0.0, 90.0)}, r'birds?', None, None, 'peregrine stoop ~90 m/s'),
 ('butterfly', ('size',), {'wingspan': (0.01, 0.31), 'width': (0.01, 0.31), 'length': (0.005, 0.12)}, r'butterfl(y|ies)', None, None, 'butterfly span 1-31 cm (Queen Alexandra 28 cm)'),
 ('soccer_ball', ('size',), {'diameter': (0.135, 0.230)}, r'(ball|football|soccerball)', r'soccer|football', r'american|rugby|foosball|table',
  'soccer ball sizes 1-5: 14.0-22.3 cm'),
 ('basketball', ('size',), {'diameter': (0.150, 0.250)}, r'(basketball|ball)', r'basket ?ball', r'hoop|rim|court|stand|backboard|basket\b(?! ?ball)', 'basketball sizes 3-7 + youth: 17.8-24.5 cm'),
 ('volleyball', ('size',), {'diameter': (0.180, 0.220)}, r'(volleyball|ball)', r'volley ?ball', r'beach volleyball court|net', 'volleyball 19.5-21.3 cm (kids lighter balls ~ same size)'),
 ('tennis_ball', ('size',), {'diameter': (0.060, 0.070)}, r'(ball|tennis ball)', r'tennis', r'table|ping', 'ITF tennis ball 6.54-6.86 cm (stage 1-3 balls up to 7 cm)'),
 ('pingpong', ('size',), {'diameter': (0.037, 0.041)}, r'(ball|pingpong)', r'ping[- ]?pong|table[- ]tennis', None, 'table tennis ball 38-40 mm'),
 ('golf_ball', ('size',), {'diameter': (0.042, 0.044)}, r'ball', r'golf', None, 'golf ball >= 42.67 mm'),
 ('baseball', ('size',), {'diameter': (0.070, 0.078)}, r'(baseball|ball)', r'base ?ball', r'bat|glove|cap|field', 'baseball 7.3-7.6 cm'),
 ('billiard', ('size',), {'diameter': (0.047, 0.0625)}, r'ball', r'billiard|pool ball|snooker|cue ball', None, 'snooker 52.5 mm .. carom 61.5 mm'),
 ('bowling', ('size',), {'diameter': (0.10, 0.22)}, r'ball', r'bowling', None, 'candlepin 11 cm .. ten-pin 21.8 cm'),
 ('yoga_ball', ('size',), {'diameter': (0.30, 1.00)}, r'ball', r'yoga|exercise|gym ball|swiss|stability', None, 'exercise balls 35-95 cm'),
 ('bowling_pin', ('size',), {'height': (0.22, 0.42)}, r'pins?', r'bowling', None, 'duckpin 23 cm .. ten-pin 38 cm / candlepin 40 cm'),
 ('ball_v_pp', ('speed', 'avgspeed'), {None: (0.0, 35.0)}, r'(ball|pingpong)', r'ping[- ]?pong|table[- ]tennis', None, 'table-tennis smash record ~31 m/s'),
 ('ball_v_tennis', ('speed', 'avgspeed'), {None: (0.0, 75.0)}, r'ball', r'tennis', r'table|ping', 'tennis serve record ~73 m/s'),
 ('ball_v_soccer', ('speed', 'avgspeed'), {None: (0.0, 60.0)}, r'(ball|football)', r'soccer|football', None, 'hardest kick ~58 m/s'),
 ('ball_v_bb', ('speed', 'avgspeed'), {None: (0.0, 30.0)}, r'(basketball|ball)', r'basket ?ball', None, 'basketball throw / fall < 30 m/s'),
 ('free_a', ('accel',), {None: (0.0, 1.1 * G)}, r'(ball|basketball|football|acorn|apple|slice|toast|card|cup|mug|bottle|droplet|drop|stone|pebble|coin|egg|pingpong)', None,
  r'hit|kick|throw|thrown|bat|racket|paddle|bounce|collid|collision|impact|push|pull|rocket|motor|engine|spring|launch|string|rope|magnet|centrip|circular|spin|rotat|planet|moon|mars|jupiter|space|orbit|astronaut',
  'an unpowered body in flight / falling / rolling / sliding has |a| <= g (drag only reduces it); 1.1 g margin'),
 ('table', ('size',), {'length': (0.40, 4.0), 'width': (0.30, 1.6), 'height': (0.25, 1.15)}, r'(table|desk)s?', None,
  r'lamp|tennis|billiard|pool|snooker|leg|top|tablet|cloth|chair|mat|tray|tile|periodic|spread|timetable', 'tables / desks: coffee .. conference, height 0.25-1.15 m (bar), width 0.3-1.6 m'),
 ('billiard_table', ('size',), {'length': (1.8, 3.9), 'width': (0.9, 2.1)}, r'table', r'billiard|pool table|snooker', r'pocket', '7 ft pool table 2.0 m .. snooker 3.8 m external'),
 ('chair', ('size',), {'height': (0.40, 1.30), 'width': (0.35, 1.00)}, r'(chair|chairs|stool|armchair)', None, r'beach|deck|lounge|wheel|high ?chair|baby', 'chairs 0.4-1.3 m tall, 0.35-1.0 m wide'),
 ('deck_chair', ('size',), {'length': (0.8, 2.2), 'width': (0.45, 1.0)}, r'chairs?', r'beach|deck|lounge', None, 'beach / deck chair & lounger'),
 ('sofa', ('size',), {'length': (1.2, 3.8), 'width': (0.7, 3.8), 'height': (0.55, 1.15)}, r'(sofa|couch|loveseat|settee)', None, None, 'loveseat 1.2 m .. sectional 3.8 m'),
 ('bed', ('size',), {'length': (1.3, 2.4)}, r'beds?', None, r'flower|river|sea|truck|bunk', 'toddler 1.3 m .. king frame 2.3 m'),
 ('tv_stand', ('size',), {'length': (0.5, 2.6), 'height': (0.25, 1.3)}, r'stand', r'tv|television', None, 'TV stand 0.5-2.6 m long, 0.25-1.3 m tall'),
 ('tv', ('size',), {'length': (0.35, 2.2), 'width': (0.35, 2.2), 'height': (0.25, 1.3)}, r'(tv|television|monitor)', None, r'stand|remote', 'screens 19-85 in: width 0.42-1.9 m'),
 ('door', ('size',), {'height': (1.8, 3.0), 'width': (0.6, 2.0)}, r'doors?', None, r'car|cabinet|oven|fridge|cage|truck|bus|train|garage|gate|wardrobe|closet|cupboard|drawer|microwave|elevator|shower|toy|dog|cat',
  'building doors 1.8-3.0 m tall (80 in .. commercial 9-10 ft), 0.6-2.0 m wide (double)'),
 ('step', ('size',), {'height': (0.09, 0.23)}, r'(step|steps|stair|stairs|staircase)', None, r'foot|stepping|ladder|platform', 'stair riser 10-22 cm (building codes)'),
 ('curb', ('size',), {'height': (0.05, 0.35)}, r'(curb|kerb)s?', None, None, 'kerb 5-35 cm'),
 ('parking', ('size',), {'length': (4.0, 7.0)}, r'(space|spaces|bay|spot)', r'parking', None, 'parking bay long side 4.5-6.5 m'),
 ('streetlight', ('size',), {'height': (2.5, 20.0)}, r'(streetlight|streetlights|lamppost|lamp post|street lamp|light pole)', None, None, 'street light poles 3-15 m'),
 ('traffic_light', ('size',), {'height': (0.6, 8.0)}, r'light', r'traffic', None, 'signal head ~0.9 m .. pole 8 m'),
 ('hydrant', ('size',), {'height': (0.40, 1.20)}, r'hydrants?', None, None, 'fire hydrant 0.4-1.2 m'),
 ('booth', ('size',), {'height': (2.0, 2.8), 'width': (0.7, 1.3)}, r'booths?', r'telephone|phone', None, 'telephone booth (K6 2.4 m x 0.9 m)'),
 ('house', ('size',), {'height': (2.5, 25.0), 'width': (3.0, 50.0), 'length': (3.0, 50.0)}, r'(house|houses|home|cottage)', None,
  r'doll|toy|bird|dog|play|light|green|tree|model|ware|boat|club|court', 'houses (one to several storeys)'),
 ('room', ('size',), {'height': (2.1, 6.0)}, r'rooms?', None, r'bath|mush', 'ceiling height 2.1-6 m'),
 ('backboard', ('size',), {'width': (0.9, 1.9), 'height': (0.6, 1.2), 'length': (0.9, 1.9)}, r'backboards?', None, None, 'regulation 1.83 x 1.07 m, recreational smaller'),
 ('pitch', ('size',), {'width': (45.0, 90.0), 'length': (90.0, 120.0)}, r'(pitch|field)', r'football|soccer', None, 'FIFA pitch 90-120 x 45-90 m'),
 ('pickleball', ('size',), {'length': (13.0, 20.0)}, r'court', r'pickle', None, 'pickleball court 13.4 m (playing) .. ~19 m (with run-off)'),
 ('chessboard', ('size',), {'side': (0.15, 0.70), 'length': (0.15, 0.70), 'width': (0.15, 0.70)}, r'(chessboard|board)', r'chess', None, 'travel 0.2 m .. tournament 0.55 m, xiangqi ~0.6 m'),
 ('whiteboard', ('size',), {'length': (0.3, 5.0), 'width': (0.2, 5.0), 'height': (0.2, 1.6)}, r'(whiteboard|chalkboard|blackboard)', None, None, 'boards 0.3-5 m wide'),
 ('mirror_floor', ('size',), {'height': (1.0, 2.2), 'length': (1.0, 2.2), 'width': (0.25, 1.2)}, r'mirror', r'floor|full[- ]length|standing', None, 'full-length mirror 1.2-2 m'),
 ('bench', ('size',), {'length': (0.6, 8.0), 'height': (0.35, 1.1)}, r'(bench|benches)', None, r'work|press|test|mark', 'benches: seat 0.45 m, back <= 1.1 m, 0.6-8 m long'),
 ('book', ('size',), {'length': (0.08, 0.50), 'width': (0.06, 0.40)}, r'(book|books|notebook|note book|textbook)', None, r'face|shelf|case|end|mark|let', 'pocket .. atlas'),
 ('pen', ('size',), {'length': (0.09, 0.20)}, r'pens?', None, r'pencil|pig|play|light|drive|holder|cap', 'pens 9-20 cm'),
 ('pencil', ('size',), {'length': (0.04, 0.20)}, r'pencils?', None, r'case|bag|box|sharpen|holder|cup', 'new pencil 19 cm, worn ones shorter'),
 ('phone', ('size',), {'length': (0.09, 0.19), 'height': (0.09, 0.19), 'width': (0.045, 0.10), 'diagonal': (0.09, 0.20)}, r'(phone|iphone|smartphone|cellphone|mobile)', None,
  r'booth|box|stand|case|telephone|desk', 'smartphones: 4-7 in, 0.09-0.19 m long, 4.5-10 cm wide'),
 ('credit_card', ('size',), {'length': (0.084, 0.087), 'width': (0.053, 0.055)}, r'cards?', r'credit|bank|debit|id card', None, 'ISO/IEC 7810 ID-1 85.60 x 53.98 mm'),
 ('dollar', ('size',), {'length': (0.154, 0.158), 'width': (0.065, 0.067)}, r'(bill|note|banknote)', r'dollar', None, 'US bill 156 x 66.3 mm'),
 ('cup', ('size',), {'height': (0.04, 0.22), 'diameter': (0.035, 0.13)}, r'(cup|cups|mug|mugs|teacup|tea cup)', None,
  r'world cup|trophy|measuring|suction|bra|hub|cupboard|cupcake|plate|toothbrush', 'espresso cup 5 cm .. tall travel mug 20 cm, rim 3.5-13 cm'),
 ('cup_rim', ('size',), {'diameter': (0.035, 0.13)}, r'(rim|mouth|opening|top)', r'\b(cup|mug|glass)\b', r'wine', 'cup / mug / glass rim 3.5-13 cm'),
 ('wine_glass', ('size',), {'diameter': (0.045, 0.12)}, r'glass(es)?', r'wine', None, 'wine glass bowl 5-11 cm'),
 ('saucer', ('size',), {'diameter': (0.10, 0.22)}, r'saucers?', None, None, 'saucer 10-20 cm'),
 ('plate_d', ('size',), {'diameter': (0.12, 0.40)}, r'plates?', None, r'licen|number|tectonic|home|steel|hot', 'plates 12-35 cm'),
 ('bottle', ('size',), {'height': (0.06, 0.50), 'length': (0.06, 0.50), 'diameter': (0.02, 0.14)}, r'bottles?', None,
  r'cap|lid|label|neck|opener|brush|rack|tank|propane|gas|jug', '50 ml .. 2 L soda / magnum'),
 ('can', ('size',), {'height': (0.05, 0.25)}, r'cans?', r'soda|beer|coke|tin|aluminium|aluminum|drink', r'trash|garbage|waste|watering|spray|oil', 'drink / food cans 5-25 cm'),
 ('kettle', ('size',), {'height': (0.12, 0.35)}, r'kettles?', None, None, 'kettles 12-35 cm'),
 ('jar', ('size',), {'height': (0.04, 0.45), 'diameter': (0.03, 0.30)}, r'jars?', None, None, 'jars 4-45 cm'),
 ('vase', ('size',), {'height': (0.08, 1.30), 'diameter': (0.03, 0.70)}, r'vases?', None, None, 'bud vase .. floor vase'),
 ('trash_bin', ('size',), {'height': (0.18, 1.40)}, r'(bin|bins|can|cans|trashcan|dustbin|basket)', r'trash|garbage|waste|rubbish|recycl|dust', None, 'office 0.2 m .. wheelie bin 1.1 m'),
 ('lampshade', ('size',), {'diameter': (0.12, 0.70), 'height': (0.08, 0.50)}, r'(lampshade|shade)s?', r'lamp', None, 'table-lamp shades 12-70 cm'),
 ('table_lamp', ('size',), {'height': (0.18, 1.00)}, r'lamps?', r'table|desk', r'shade', 'table / desk lamps 0.18-1.0 m'),
 ('keyboard', ('size',), {'length': (0.22, 0.55), 'width': (0.08, 0.55)}, r'keyboards?', None, r'piano|music|synth', 'computer keyboards 60% .. full: 0.29-0.46 m (width ambiguous: depth or long side)'),
 ('spacebar', ('size',), {'length': (0.08, 0.17)}, r'(spacebar|space bar)', None, None, '5-7 key units'),
 ('laptop', ('size',), {'length': (0.24, 0.43), 'width': (0.15, 0.43)}, r'(laptop|macbook|notebook computer)', None, r'bag|case|sleeve|stand', '11-18 in laptops'),
 ('mouse', ('size',), {'length': (0.08, 0.14)}, r'mouse', r'magic|computer|wireless|apple', None, 'computer mouse 8-14 cm (Magic Mouse 11.35 cm)'),
 ('typewriter', ('size',), {'length': (0.25, 0.65), 'width': (0.20, 0.65), 'height': (0.07, 0.35)}, r'typewriters?', None, None, 'portable .. office typewriter'),
 ('toaster', ('size',), {'length': (0.18, 0.50)}, r'toasters?', None, None, '2-slice .. 4-slice long'),
 ('shoe', ('size',), {'length': (0.10, 0.37)}, r'(shoe|shoes|sneaker|boot)', None, r'horse|box|lace|rack|tree', 'child .. EU 52 adult'),
 ('suitcase', ('size',), {'length': (0.35, 0.90)}, r'(suitcase|luggage)', None, None, 'cabin .. large check-in'),
 ('wallet', ('size',), {'length': (0.08, 0.23)}, r'wallets?', None, r'short', 'bifold 11 cm .. long wallet 20 cm'),
 ('toothbrush', ('size',), {'length': (0.12, 0.26)}, r'toothbrush(es)?', None, r'cup|holder', 'child .. electric toothbrush'),
 ('tp_roll', ('size',), {'diameter': (0.08, 0.30)}, r'(roll|rolls)', r'toilet|paper', r'core|holder', 'household 11-13 cm .. jumbo 25+ cm'),
 ('pizza', ('size',), {'diameter': (0.15, 0.65)}, r'pizzas?', None, r'pepperoni|cutter|box|slice', '6-24 in pizza'),
 ('pepperoni', ('size',), {'diameter': (0.015, 0.09)}, r'(pepperoni|pepperonis|salami)', None, None, 'pepperoni slices 2-9 cm'),
 ('apple', ('size',), {'diameter': (0.04, 0.12)}, r'apples?', None, r'mouse|magic|logo|watch|iphone|macbook', 'apples 4-12 cm'),
 ('acorn', ('size',), {'height': (0.008, 0.06), 'width': (0.006, 0.04), 'length': (0.008, 0.06)}, r'acorns?', None, None, 'acorns 1-6 cm'),
 ('cookie', ('size',), {'diameter': (0.025, 0.20)}, r'(cookie|cookies|biscuit)', None, None, 'cookies 2.5-20 cm'),
 ('bread', ('size',), {'length': (0.07, 0.20)}, r'(toast|slice)', r'toast|bread', r'toaster', 'bread slice 7-20 cm'),
 ('baguette', ('size',), {'size': (0.25, 1.0), 'length': (0.25, 1.0)}, r'baguettes?', None, None, 'half .. full baguette'),
 ('avocado', ('size',), {'size': (0.06, 0.18), 'length': (0.06, 0.18)}, r'avocados?', None, None, 'avocados 6-18 cm long'),
 ('ice_cube', ('size',), {'height': (0.008, 0.07)}, r'(cube|cubes)', r'ice', None, 'ice cubes 1-7 cm'),
 ('swim_ring', ('size',), {'length': (0.35, 1.30), 'diameter': (0.35, 1.30)}, r'ring', r'swim', None, 'swim rings 0.35-1.3 m'),
 ('sword', ('size',), {'length': (0.40, 1.50)}, r'swords?', None, None, 'short sword .. two-handed'),
 ('tape', ('size',), {'diameter': (0.04, 0.16), 'width': (0.005, 0.08)}, r'(tape|tapes)', None, r'measur|caution|cassette|video', 'tape rolls 4-16 cm, 0.5-8 cm wide'),
 ('pencil_case', ('size',), {'length': (0.12, 0.35)}, r'(bag|case|pouch)', r'pencil', None, 'pencil cases 12-35 cm'),
 ('earphone_case', ('size',), {'length': (0.035, 0.12)}, r'case', r'earphone|earbud|airpod', None, 'earbud cases 4-12 cm'),
]

HEADCUT = re.compile(r"\s+(?:of|on|in|at|with|from|between|along|that|to|over|when|during|across|while|lying|riding|standing|formed|flying|hitting|"
                     r"crossing|sitting|moving|walking|running|near|next|under|above|behind|inside|for|which|who)\b.*$")
DIMS = ('height', 'width', 'length', 'diameter', 'wingspan', 'side', 'size', 'diagonal', 'thickness')


def _ok(z):
    try:
        z = float(z)
    except Exception:
        return None
    return z if math.isfinite(z) and z > 0 else None


def _usi(unit):
    u = str(unit or '').lower()
    if u.startswith('cm'): return 0.01
    if u.startswith('mm'): return 0.001
    if u.startswith('km/h'): return 1 / 3.6
    return 1.0


def _head(phrase):
    p = re.sub(r"[^a-z' -]", " ", str(phrase or '').lower()).strip()
    p = re.sub(r"^(the|a|an|each|every|this|that)\s+", "", p)
    core = HEADCUT.sub('', p).strip()
    w = core.split()
    if not w: return '', '', p
    return w[-1], ' '.join(w[:-1]), p


def _dimrange(dims, dim):
    if None in dims: return dims[None]
    if dim in dims: return dims[dim]
    if dim == 'radius' and 'diameter' in dims: lo, hi = dims['diameter']; return lo / 2, hi / 2
    if dim == 'circumference' and 'diameter' in dims: lo, hi = dims['diameter']; return lo * math.pi, hi * math.pi
    if dim in ('size', 'side') and dims:
        return min(v[0] for v in dims.values()), max(v[1] for v in dims.values())
    if dim == 'width' and 'diameter' in dims and len(dims) == 1: return dims['diameter']
    return None


def _match(head, mods, phrase, q, kind, dim):
    if kind not in ('size', 'speed', 'avgspeed', 'accel'): return None
    for nm, kinds, dims, hrx, need, exc, src in T:
        if kind not in kinds: continue
        if not re.fullmatch(hrx, head): continue
        ctx = (mods + ' ' + phrase + ' ' + q)
        if need and not re.search(need, ctx): continue
        if exc and re.search(exc, phrase + ' ' + q): continue
        rg = _dimrange(dims, dim if kind == 'size' else None)
        if rg is None: continue
        return nm, rg[0], rg[1], src
    return None


def _scaled_world(row):
    q = str(row.get('question', '')).lower()
    for p in ((row.get('parse') or {}).get('priors') or []):
        v = _ok(p.get('value_si')); k = p.get('kind')
        if v is None or k not in ('size', 'speed'): continue
        h, m, ph = _head(p.get('obj'))
        if not h: continue
        hit = _match(h, m, ph, '', k, p.get('dim') or 'size')
        if hit and not (hit[1] / 1.02 <= v <= hit[2] * 1.02): return True
    return False


def _prior_states_same(row, head, kind, dim):
    for p in ((row.get('parse') or {}).get('priors') or []):
        if p.get('kind') != ('size' if kind == 'size' else ('accel' if kind == 'accel' else 'speed')): continue
        h, _, _ = _head(p.get('obj'))
        if h and h.rstrip('s') == head.rstrip('s') and (kind != 'size' or p.get('dim') in (dim, None)):
            return True
    return False


def lookup(row):
    P = row.get('parse') or {}
    Q = P.get('q') or {}
    kind = Q.get('kind'); dim = Q.get('dim')
    if kind not in ('size', 'speed', 'avgspeed', 'accel'): return None
    if kind == 'size' and dim not in DIMS + ('radius', 'circumference'): return None
    q = str(row.get('question', '')).lower()
    head, mods, phrase = _head(Q.get('target') or (P.get('target') or {}).get('phrase'))
    if not head or re.search(GEXC, re.sub(r'toilet paper', '', phrase + ' ' + q)): return None
    hit = _match(head, mods, phrase, q, kind, dim)
    if hit is None: return None
    if _prior_states_same(row, head, kind, dim): return None
    if _scaled_world(row): return None
    if kind == 'accel' and re.search(r'gravit|\bg\s*=', str(row.get('prior', '')).lower()):
        m = re.search(r'([\d.]+)\s*m/s', str(row.get('prior', '')).lower())
        if hit[0] == 'free_a' and m:
            try:
                g = float(m.group(1))
                if g > 0: hit = (hit[0], 0.0, max(hit[2], 1.1 * g), hit[3])
            except ValueError:
                pass
    f = _usi(Q.get('unit'))
    nm, lo, hi, src = hit
    return nm, lo / f, hi / f, src


def own_cands(row):
    c = []
    for n, z in (('geo', row.get('s1_g')), ('geo_u', (row.get('geo_u') or {}).get('meas_value')), ('CaW', row.get('s1_cs')),
                 ('CaW_greedy', row.get('caw_full_greedy')), ('crop_greedy', row.get('caw_crop_greedy'))):
        z = _ok(z)
        if z is not None: c.append((n, z))
    for n, k in (('CaW16med', 'caw_full'), ('cropmed', 'caw_crop')):
        s = [x for x in (row.get(k) or []) if _ok(x)]
        if len(s) >= 4: c.append((n, float(statistics.median(s))))
    out, seen = [], []
    for n, z in c:
        if any(abs(math.log(z / s)) < 1e-6 for s in seen): continue
        seen.append(z); out.append((n, z))
    return out


def gkw(row, var):
    a = _ok(row.get('cur'))
    if a is None: return None
    hit = lookup(row)
    if hit is None: return None
    nm, lo, hi, src = hit
    kind = ((row.get('parse') or {}).get('q') or {}).get('kind')
    if var == 'W2' and kind != 'size': return None
    if lo <= a <= hi: return None
    ins = [(n, z) for n, z in own_cands(row) if lo <= z <= hi]
    if not ins: return None
    if var in ('W1', 'W2'):
        z = math.exp(sum(math.log(x) for _, x in ins) / len(ins))
        how = 'gm of own in-range estimates ' + ', '.join(f'{n} {x:.4g}' for n, x in ins)
    elif var == 'W3':
        n, z = min(ins, key=lambda t: abs(math.log(t[1] / a)))
        how = f'own in-range estimate closest to the answer: {n} {z:.4g}'
    else:
        raise ValueError(var)
    if abs(math.log(z / a)) < 1e-9: return None
    return z, f"r9_gkw {var}: {nm} range [{lo:.4g},{hi:.4g}] ({src}); own answer {a:.4g} outside -> {how} -> {z:.4g}"


def gkm(row, factor):
    a = _ok(row.get('cur'))
    if a is None: return None
    hit = lookup(row)
    if hit is None: return None
    nm, lo, hi, src = hit
    if lo / factor <= a <= hi * factor: return None
    ins = [(n, z) for n, z in own_cands(row) if lo <= z <= hi]
    if not ins: return None
    z = math.exp(sum(math.log(x) for _, x in ins) / len(ins))
    if abs(math.log(z / a)) < 1e-9: return None
    return z, (f"r9_gkm x{factor}: {nm} range [{lo:.4g},{hi:.4g}] ({src}); own answer {a:.4g} outside the range by > {factor}x -> "
               f"gm of own in-range estimates " + ', '.join(f'{n} {x:.4g}' for n, x in ins) + f" -> {z:.4g}")


def gkm_M1(row): return gkm(row, 1.25)
def gkm_M2(row): return gkm(row, 1.5)
def gkm_M3(row): return gkm(row, 2.0)


def gkw_W1(row): return gkw(row, 'W1')
def gkw_W2(row): return gkw(row, 'W2')
def gkw_W3(row): return gkw(row, 'W3')


ACCEPTED = [('gkm_M2', gkm_M2)]
