#!/usr/bin/env python
import sys, os, re, json
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
TAG = "rul336"
MARGIN = 1.5

S = "size"
TABLE = [
    ("tennis ball", r"tennis ball", "", (S, "diameter size width"), 0.0654, 0.0686, "ITF Rules of Tennis, Appendix I: diameter 6.54-6.86 cm"),
    ("table tennis ball", r"(?:ping[ -]?pong|table tennis) ball", "", (S, "diameter size width"), 0.038, 0.0402, "ITTF Handbook 2.3.1: 40 mm ball (38 mm before 2000)"),
    ("golf ball", r"golf ball", "", (S, "diameter size width"), 0.0411, 0.0430, "R&A / USGA equipment rules: diameter >= 42.67 mm (old British ball 41.15 mm)"),
    ("baseball", r"baseball", "", (S, "diameter size width"), 0.0729, 0.0748, "MLB Official Baseball Rules 3.01: circumference 9-9.25 in (22.9-23.5 cm)"),
    ("billiard ball", r"(?:billiard|pool|snooker|cue) ball", "", (S, "diameter size width"), 0.0525, 0.0615, "WPA ball 57.15 mm; snooker 52.5 mm; carom 61.5 mm"),
    ("basketball", r"basketball", "hoop|rim|net|court|backboard|stand|pole|board", (S, "diameter size width"), 0.18, 0.248, "FIBA equipment rules: size 7 circumference 749-780 mm (diam. 23.8-24.8 cm); mini size 3 ~ 18 cm"),
    ("soccer ball", r"soccer ball|football", "american|rugby|pitch|field|goal|player|stadium|table", (S, "diameter size width"), 0.137, 0.223, "IFAB Law 2: size 5 circumference 68-70 cm (diam. 21.6-22.3 cm); size 1 43-48 cm (diam. 13.7-15.3 cm)"),
    ("volleyball", r"volleyball", "net|court", (S, "diameter size width"), 0.18, 0.217, "FIVB: circumference 65-67 cm (diam. 20.7-21.3 cm), beach 66-68 cm; youth balls ~ 18 cm"),
    ("bowling ball", r"bowling ball", "", (S, "diameter size width"), 0.11, 0.2183, "USBC: ten-pin max 8.595 in (21.83 cm); candlepin / duckpin balls 11-12.7 cm"),
    ("exercise ball", r"(?:yoga|exercise|gym|swiss|stability|pilates) ball", "", (S, "diameter size width"), 0.23, 0.95, "retail exercise ball sizes 45-95 cm; mini pilates balls 23-25 cm"),
    ("bicycle wheel", r"(?:bicycle|bike|cycle) (?:wheel|tire|tyre)s?", "", (S, "diameter size"), 0.30, 0.76, "ISO 5775 bead seat diameters 203-635 mm (12 in to 29 in); outer diameter about 0.30-0.75 m"),
    ("car wheel", r"(?:car|truck|van|bus|suv|vehicle)(?:'s)? (?:wheel|tire|tyre)s?|(?:wheel|tire|tyre)s? of the (?:car|truck|van)", "", (S, "diameter size"), 0.50, 1.15, "passenger car tyres 0.53-0.81 m outer diameter (145/80R13 to 275/55R20); truck tyres up to about 1.1 m"),
    ("car", r"(?<!cable )(?<!train )(?<!race )(?<!racing )car", "toy|cart|race|racing|formula|f1|kart|cable|train|rail|tram|roller|police car model|go-kart|golf", (S, "length"), 2.3, 6.2, "production cars: Smart Fortwo 2.69 m to Chevrolet Suburban / Cadillac Escalade ESV 5.7-5.8 m"),
    ("car", r"(?<!cable )(?<!train )(?<!race )(?<!racing )car", "toy|cart|race|racing|formula|f1|kart|cable|train|rail|tram|roller|go-kart|golf", (S, "width"), 1.4, 2.2, "production cars: body width 1.5 m (Smart Fortwo 1.66 m, kei cars 1.48 m) to 2.2 m (Hummer H1)"),
    ("car", r"(?<!cable )(?<!train )(?<!race )(?<!racing )car", "toy|cart|race|racing|formula|f1|kart|cable|train|rail|tram|roller|go-kart|golf", (S, "height"), 1.0, 2.0, "production cars: height 1.07 m (Lamborghini Countach) to 1.97 m (Mercedes G-Class)"),
    ("truck", r"(?<!fire )(?<!toy )truck|lorry", "toy|fire|monster|mining|haul|dump|tow|forklift|hand", (S, "length"), 4.5, 25.0, "pickup trucks 4.7-6.4 m; tractor-trailer combinations up to about 23 m (US), 16.5-18.75 m (EU)"),
    ("truck", r"(?<!fire )(?<!toy )truck|lorry", "toy|fire|monster|mining|haul|dump|tow|forklift|hand", (S, "width"), 1.7, 2.6, "legal maximum vehicle width 2.55 m (EU 96/53/EC) / 2.6 m (US 102 in); pickups 1.9-2.1 m"),
    ("truck", r"(?<!fire )(?<!toy )truck|lorry", "toy|fire|monster|mining|haul|dump|tow|forklift|hand", (S, "height"), 1.7, 4.3, "legal maximum height 4.0 m (EU), about 4.1-4.3 m (US states); pickups about 1.8-2.0 m"),
    ("fire truck", r"fire (?:truck|engine)", "toy", (S, "length"), 6.0, 15.0, "fire apparatus: pumpers about 9-11 m, aerial ladder trucks up to about 15 m"),
    ("fire truck", r"fire (?:truck|engine)", "toy", (S, "width"), 2.2, 2.6, "fire apparatus within the legal width 2.55-2.6 m"),
    ("fire truck", r"fire (?:truck|engine)", "toy", (S, "height"), 2.5, 4.0, "fire apparatus height about 2.8-3.8 m"),
    ("bus", r"(?<!school )(?<!mini)bus|school bus|coach", "toy|minibus|bus stop|bus shelter|station", (S, "length"), 6.0, 25.0, "buses: midibus about 7 m, standard 12 m, articulated 18 m, bi-articulated about 25 m"),
    ("bus", r"(?<!mini)bus|coach", "toy|minibus|bus stop|bus shelter|station", (S, "width"), 2.0, 2.6, "legal maximum width 2.55 m (EU) / 2.6 m (US)"),
    ("bus", r"(?<!mini)bus|coach", "toy|minibus|bus stop|bus shelter|station", (S, "height"), 2.4, 4.6, "single-deck about 3 m; double-deck up to 4.4 m (UK no legal limit, typical 4.38 m)"),
    ("van", r"van", "toy|caravan|minivan", (S, "length"), 4.0, 7.5, "panel vans: Ford Transit 5.5-6.7 m, Mercedes Sprinter up to 7.4 m; small vans about 4.2 m"),
    ("forklift", r"fork ?lift(?: truck)?", "", (S, "height"), 1.8, 4.0, "counterbalance forklifts: overhead guard about 2.0-2.3 m, lowered mast up to about 3.5 m (Toyota 8-series, Linde spec sheets)"),
    ("forklift", r"fork ?lift(?: truck)?", "", (S, "length"), 1.8, 5.0, "counterbalance forklifts: length with forks about 2.8-4.5 m"),
    ("forklift", r"fork ?lift(?: truck)?", "", (S, "width"), 0.8, 2.0, "counterbalance forklifts: width about 1.0-1.5 m"),
    ("bicycle", r"(?:bicycle|bike|cycle)(?<!motorcycle)(?<!motorbike)", "motor|e-?bike wheel|wheel|tire|tyre|seat|lane|rack|path", (S, "length"), 1.0, 2.6, "adult bicycles about 1.6-1.9 m long, children's bicycles from about 1.0 m, tandems about 2.4-2.6 m"),
    ("bicycle", r"(?:bicycle|bike|cycle)(?<!motorcycle)(?<!motorbike)", "motor|wheel|tire|tyre|seat|lane|rack|path", (S, "height"), 0.6, 1.3, "bicycle height to handlebar / saddle about 0.9-1.2 m; children's bicycles from about 0.6 m"),
    ("motorcycle", r"motorcycle|motorbike|motor bike", "", (S, "length"), 1.5, 2.7, "motorcycles: mopeds about 1.6 m to touring bikes about 2.6 m (Honda Gold Wing 2.6 m)"),
    ("motorcycle", r"motorcycle|motorbike|motor bike", "", (S, "height"), 0.9, 1.6, "motorcycles: height about 1.0-1.5 m"),
    ("airplane", r"(?:air|aero)?plane|aircraft|airliner|jet", "paper|toy|model|rc|drone|inclined|plane surface|ramp|slope|glider|jet ski|jet of|water jet", (S, "length"), 6.0, 85.0, "fixed-wing aircraft: Cessna 152 7.3 m to Antonov An-225 84 m"),
    ("airplane", r"(?:air|aero)?plane|aircraft|airliner|jet", "paper|toy|model|rc|drone|inclined|plane surface|ramp|slope|glider|jet ski|jet of|water jet", (S, "wingspan"), 7.0, 90.0, "fixed-wing aircraft: wingspan Cessna 152 10.2 m (ultralights about 8 m) to An-225 88.4 m"),
    ("airplane", r"(?:air|aero)?plane|aircraft|airliner|jet", "paper|toy|model|rc|drone|inclined|plane surface|ramp|slope|glider|jet ski|jet of|water jet", (S, "height"), 1.8, 25.0, "fixed-wing aircraft: height Cessna 152 2.6 m to A380 24.1 m"),
    ("helicopter", r"helicopter", "toy|model|rc|drone", (S, "length"), 8.0, 40.0, "helicopters: Robinson R22 8.8 m to Mil Mi-26 40 m"),
    ("helicopter", r"helicopter", "toy|model|rc|drone", (S, "height"), 2.4, 8.5, "helicopters: Robinson R22 2.7 m to Mil Mi-26 8.1 m"),
    ("canoe", r"canoe", "", (S, "length"), 3.0, 8.0, "canoes about 4-6 m (solo canoes about 3.5 m)"),
    ("kayak", r"kayak", "", (S, "length"), 2.0, 6.5, "kayaks: whitewater about 2-2.5 m, sea kayaks up to about 6.4 m"),
    ("person", r"person|pedestrian|man|woman|walker|player|boy|girl|kid|child|lady|guy|runner|human|people|jogger|skater|cyclist|biker|rider",
     "sitting|seated|shoulder|hand|arm|leg|foot|feet|head|baby|toddler|infant|snow|statue|lego|cardboard|bust|riding|on the bike|on a bike|on the horse|crouch|kneel|lying|sleeping|bending|squat", (S, "height"), 0.5, 2.3,
     "people standing: children from about 0.5 m; tallest living adults about 2.3 m (kinematic.class_bound, unchanged)"),
    ("baby", r"baby|infant|toddler", "carriage|stroller|bottle|seat|bed|crib|chair|monitor|doll", (S, "height length"), 0.45, 1.0, "WHO child growth standards: length 0.45-0.55 m at birth, about 0.87 m at 2 years"),
    ("horse", r"horse|pony", "rocking|sea ?horse|carousel|toy|shoe|drawn|cart", (S, "height"), 0.7, 2.6, "Falabella miniature horse withers 0.7-0.86 m; Shire withers up to about 1.9 m, head up to about 2.5 m"),
    ("horse", r"horse|pony", "rocking|sea ?horse|carousel|toy|shoe|drawn|cart", (S, "length"), 1.0, 3.2, "horse body length about 1.0 m (miniature) to about 3 m (draught horse nose to tail)"),
    ("elephant", r"elephant", "toy|seal", (S, "height"), 0.8, 4.0, "elephant calves about 0.85-1.0 m at birth; African bush elephant bulls 3.0-4.0 m at the shoulder"),
    ("dog", r"dog|puppy", "hot ?dog|toy|house|bowl|leash|bed|sled|prairie", (S, "height"), 0.15, 1.1, "Chihuahua withers 0.15-0.23 m; Irish Wolfhound / Great Dane withers up to about 0.9 m, head about 1.1 m"),
    ("dog", r"dog|puppy", "hot ?dog|toy|house|bowl|leash|bed|sled|prairie", (S, "length"), 0.25, 2.0, "dog length about 0.25 m (toy breeds) to about 2 m (giant breeds with tail)"),
    ("cat", r"(?<!big )cat|kitten", "big|wild|lion|tiger|leopard|toy|cat ?walk|catamaran|caterpillar|food|tree|bed|scan", (S, "height"), 0.12, 0.6, "domestic cat withers about 0.23-0.25 m, head up to about 0.45 m (Maine Coon); kittens from about 0.12 m"),
    ("cat", r"(?<!big )cat|kitten", "big|wild|lion|tiger|leopard|toy|cat ?walk|catamaran|caterpillar|food|tree|bed|scan", (S, "length"), 0.25, 1.25, "domestic cat head-body about 0.46 m, with tail up to about 1.2 m (Maine Coon record 1.23 m); kittens from about 0.25 m"),
    ("rabbit", r"rabbit|bunny|hare", "toy|hole|ear", (S, "length"), 0.18, 0.80, "domestic rabbits: Netherland Dwarf about 0.2-0.25 m to Flemish Giant about 0.8 m; European hare 0.5-0.75 m"),
    ("rabbit", r"rabbit|bunny|hare", "toy|hole|ear", (S, "height"), 0.10, 0.5, "rabbits sitting about 0.1-0.3 m; hares and large breeds with raised ears about 0.4-0.5 m"),
    ("deer", r"deer|moose|elk|reindeer|caribou", "toy", (S, "height"), 0.35, 3.0, "pudu (smallest deer) withers 0.35-0.44 m; moose withers up to 2.1 m, top of head / antlers up to about 2.7-3.0 m"),
    ("deer", r"deer|moose|elk|reindeer|caribou", "toy", (S, "length"), 0.7, 3.2, "pudu length about 0.8 m; moose up to about 3.1 m"),
    ("antelope", r"antelope|gazelle|impala", "", (S, "height"), 0.25, 3.0, "royal antelope 0.25 m at the shoulder; common eland up to 1.8 m at the shoulder, head and horns to about 2.5 m; greater kudu horns up to about 1.8 m along the spiral"),
    ("antelope", r"antelope|gazelle|impala", "", (S, "length"), 0.4, 3.5, "royal antelope about 0.4-0.5 m; common eland up to about 3.45 m"),
    ("raccoon", r"raccoon|racoon", "", (S, "length"), 0.4, 1.05, "raccoon head-body 0.4-0.7 m, tail 0.2-0.4 m"),
    ("eagle", r"eagle", "toy|statue|logo", (S, "wingspan"), 0.85, 2.9, "eagles: smallest (e.g. little eagle 1.13-1.32 m, small serpent eagles about 0.9 m) to wedge-tailed eagle 2.84 m"),
    ("eagle", r"eagle", "toy|statue|logo", (S, "length"), 0.4, 1.1, "eagles: body length about 0.45 m (little eagle) to 1.0 m (Philippine / harpy eagle)"),
    ("bird", r"bird", "ostrich|emu|cassowary|rhea|penguin|bird ?house|bird ?cage|feeder|nest|bath", (S, "length"), 0.05, 1.8, "flying birds: body length from about 5-6 cm (bee hummingbird) to about 1.6-1.8 m (Dalmatian pelican); ratites excluded"),
    ("bird", r"bird", "ostrich|emu|cassowary|rhea|penguin|bird ?house|bird ?cage|feeder|nest|bath", (S, "wingspan"), 0.06, 3.7, "birds: wingspan from about 6 cm (bee hummingbird) to about 3.5-3.7 m (wandering albatross)"),
    ("butterfly", r"butterfly|moth", "", (S, "wingspan width"), 0.01, 0.31, "butterflies / moths: western pygmy blue about 1.2 cm to Queen Alexandra's birdwing / Atlas moth about 0.28-0.3 m"),
    ("whale", r"whale", "shark|toy", (S, "length"), 2.6, 33.0, "whales: dwarf sperm whale about 2.7 m to blue whale about 30-33 m"),
    ("duck", r"duck", "toy|rubber", (S, "length"), 0.28, 0.8, "ducks: about 0.3 m (teal) to 0.75 m (muscovy)"),
    ("table", r"(?<!tennis )(?<!pool )(?<!billiard )table", "lamp|leg|top|surface|tennis|pool|billiard|snooker|cloth|periodic|turn|time|water|saw|foosball|thick", (S, "height"), 0.25, 1.4, "tables: coffee tables from about 0.3 m, dining 0.72-0.76 m, bar tables about 1.05-1.1 m (kinematic.class_bound)"),
    ("desk", r"desk", "lamp|leg|top|surface|thick|organizer|mat", (S, "height"), 0.5, 1.3, "desks: children's desks about 0.55 m, standard 0.72-0.76 m, standing desks up to about 1.3 m"),
    ("chair", r"chair", "wheel|leg|seat|back|arm|lift|high ?chair|rocking|beach|deck|lawn", (S, "height"), 0.35, 1.4, "chairs: overall height about 0.5 m (child) to about 1.3 m (high-back); kinematic.SIZE3"),
    ("sofa", r"sofa|couch", "cushion|pillow|leg", (S, "height"), 0.4, 1.3, "sofas: overall height about 0.6-1.0 m (kinematic.SIZE3 0.4-1.3)"),
    ("sofa", r"sofa|couch", "cushion|pillow|leg", (S, "length width"), 0.6, 4.0, "sofas: seat depth about 0.8-1.0 m; length about 1.2 m (loveseat) to about 3.5 m"),
    ("bed", r"bed", "flower|river|sea|truck|bedside|bedroom|bunk|dog|cat|pet|sheet|cover", (S, "length"), 1.4, 2.6, "beds: about 1.9-2.1 m long (toddler 1.4 m, extra long 2.2 m; kinematic.SIZE3)"),
    ("bench", r"bench", "seat|leg|press|top|work|lab|test|tool", (S, "height"), 0.35, 1.1, "benches: seat height 0.4-0.5 m, backrest top about 0.8-1.0 m"),
    ("bench", r"bench", "seat|leg|press|top|work|lab|test|tool", (S, "width"), 0.25, 3.0, "benches: seat depth about 0.3-0.6 m, length about 1-3 m (an asked width can be either)"),
    ("door", r"door(?:way)?", "car|cabinet|fridge|refrigerator|oven|cage|truck|bus|van|microwave|closet|wardrobe|garage|barn|gate|trap|dog|cat|knob|handle|frame|mat|bell|step|front of|glass of|train|plane|airplane|elevator", (S, "height"), 1.6, 3.5, "doors: standard 2.03-2.1 m (80 in / EN 2040 mm); tall doors up to about 3 m (kinematic.class_bound)"),
    ("door", r"door(?:way)?", "car|cabinet|fridge|refrigerator|oven|cage|truck|bus|van|microwave|closet|wardrobe|garage|barn|gate|trap|dog|cat|knob|handle|frame|mat|bell|step|front of|glass of|train|plane|airplane|elevator", (S, "width"), 0.6, 2.0, "doors: single 0.7-0.9 m, double 1.5-1.8 m"),
    ("refrigerator", r"refrigerator|fridge", "magnet|door|handle", (S, "height"), 0.45, 2.1, "refrigerators: mini fridge about 0.5 m to full size about 1.9 m"),
    ("table lamp", r"(?:table|desk|bedside) lamp", "shade|base|bulb", (S, "height"), 0.2, 1.0, "table / desk lamps about 0.3-0.8 m"),
    ("floor lamp", r"floor lamp|standing lamp", "shade|base|bulb", (S, "height"), 1.0, 2.1, "floor lamps about 1.4-1.9 m"),
    ("tv", r"tv|television", "stand|cabinet|table|remote|antenna|tower|show", (S, "width"), 0.35, 2.3, "televisions 19-98 in: width about 0.43-2.2 m"),
    ("tv", r"tv|television", "stand|cabinet|table|remote|antenna|tower|show", (S, "height"), 0.22, 1.4, "televisions 19-98 in: height about 0.25-1.25 m (without stand), up to about 1.4 m with stand"),
    ("trash bin", r"trash ?bin|trash ?can|garbage ?(?:bin|can)|waste ?(?:bin|basket)|dustbin|wastebasket|rubbish bin|recycling bin", "bag|lid", (S, "height"), 0.2, 1.3, "bins: office bins about 0.25-0.4 m; wheelie bins about 1.0-1.1 m; 1100 l bins about 1.3 m (EN 840)"),
    ("toaster", r"toaster", "oven", (S, "length width"), 0.1, 0.5, "pop-up toasters: about 0.27-0.45 m long, 0.15-0.25 m wide"),
    ("toaster", r"toaster", "oven", (S, "height"), 0.12, 0.35, "pop-up toasters: about 0.18-0.22 m high"),
    ("kettle", r"kettle", "", (S, "height"), 0.12, 0.4, "kettles about 0.2-0.3 m high"),
    ("typewriter", r"typewriter", "key|ribbon|paper|carriage|platen|case", (S, "width length"), 0.25, 0.65, "typewriters: Olympia SM3 31 x 32 cm, Underwood No. 5 about 30 x 33 cm, IBM Selectric 53 x 38 cm"),
    ("typewriter", r"typewriter", "key|ribbon|paper|carriage|platen|case", (S, "height"), 0.08, 0.35, "typewriters: about 0.11 m (portable) to 0.28 m (Underwood No. 5) high"),
    ("keyboard", r"(?:computer )?keyboard", "piano|musical|key|space ?bar|spacebar", (S, "length"), 0.25, 0.65, "computer keyboards: compact about 0.29 m to full size with keypad about 0.45-0.48 m (kinematic.SIZE3 0.2-0.65)"),
    ("laptop", r"laptop|macbook|notebook computer", "", (S, "width length"), 0.2, 0.5, "laptops 11-18 in: about 0.27-0.41 m wide (kinematic.SIZE3)"),
    ("phone", r"(?:iphone|smartphone|phone|cellphone|cell phone|mobile phone|mobile)", "booth|box|stand|case|holder|screen|camera|tele|head|micro|xylo|saxo", (S, "width"), 0.045, 0.13, "phones: about 0.05-0.08 m wide; foldables up to about 0.13 m (kinematic.class_bound)"),
    ("phone", r"(?:iphone|smartphone|phone|cellphone|cell phone|mobile phone|mobile)", "booth|box|stand|case|holder|screen|camera|tele|head|micro|xylo|saxo", (S, "length height"), 0.09, 0.22, "phones: about 0.11-0.17 m long (kinematic.class_bound)"),
    ("book", r"(?:note)?book|textbook|novel", "shelf|case|stand|end|mark|let|page|cover|store|facebook|macbook|chromebook|notebook computer", (S, "length"), 0.08, 0.5, "books: pocket books about 0.1 m to large atlases about 0.5 m (kinematic.SIZE3)"),
    ("bottle", r"(?:water |wine |plastic |glass |beer |soda |milk |medicine |spray )?bottle", "cap|lid|label|text|letter|logo|neck|opener|brush|holder|rack|word|lettering", (S, "height length"), 0.04, 0.6, "bottles: about 0.05 m (miniatures) to about 0.5 m (wine magnum, 5 l water); kinematic.SIZE3"),
    ("can", r"(?:soda |coke |beer |tin |drink |aluminum |aluminium )?can", "trash|garbage|watering|paint|oil|gas|jerry|scan|toucan|pecan|pelican|american|mexican|cancel|candle|candy|canopy|canvas|cannon|canal|canoe", (S, "height"), 0.05, 0.25, "beverage cans: 8.4 oz about 0.13 m, 12 oz 0.122 m, 0.5 l 0.168 m; food tins about 0.06-0.18 m (kinematic.SIZE3)"),
    ("cup", r"(?:coffee |tea |paper |plastic )?(?:cup|mug)", "cupboard|cupcake|holder|world|trophy|measuring|suction|cupola|cupid", (S, "height"), 0.03, 0.35, "cups and mugs about 0.05-0.15 m; large travel mugs about 0.25 m (kinematic.SIZE3)"),
    ("cup", r"(?:coffee |tea |paper |plastic )?(?:cup|mug)", "cupboard|cupcake|holder|world|trophy|measuring|suction|cupola|cupid", (S, "diameter width"), 0.03, 0.15, "cups and mugs: rim diameter about 0.05-0.12 m"),
    ("glass", r"(?:drinking |wine |water |beer )glass", "", (S, "height"), 0.05, 0.3, "drinking glasses about 0.08-0.25 m high"),
    ("plate", r"(?:dinner |paper |dessert |side )?plate", "license|licence|number|tectonic|hot|steel|metal|base|glass|name|home|plate of|plateau", (S, "diameter width"), 0.1, 0.4, "plates: saucers about 0.12-0.15 m, dinner plates 0.25-0.30 m, chargers about 0.33 m"),
    ("pizza", r"pizza", "box|slice|cutter|oven|peel", (S, "diameter width"), 0.15, 0.6, "pizzas: personal about 0.2 m, large about 0.36-0.41 m, extra large about 0.45-0.5 m"),
    ("chess piece", r"(?:chinese |western |international )?chess(?: piece| man| token)?|xiangqi(?: piece)?", "board|table|clock", (S, "diameter width size"), 0.02, 0.06, "chess / xiangqi pieces: base diameters about 2.5-5 cm (FIDE Handbook C.02: king base diameter 40-50 % of its 9.5 cm height)"),
    ("chessboard", r"chess ?board", "", (S, "side length width"), 0.2, 0.7, "chess boards: travel sets about 0.2-0.3 m; FIDE squares 5-6.5 cm give about 0.4-0.52 m plus border"),
    ("chess piece", r"(?:chess )?(?:king|queen)(?: piece)?|chess piece", "size|bed|kingdom", (S, "height"), 0.03, 0.15, "FIDE Handbook C.02: king height 9.5 cm +-10 %; travel sets about 3-5 cm"),
    ("shoe", r"shoe|sneaker|boot", "box|lace|horse|shoe ?horn|rack|car boot", (S, "length"), 0.1, 0.36, "shoes: children's about 0.1-0.2 m, adults about 0.22-0.33 m"),
    ("pen", r"(?:ball ?point |fountain )?pen", "pencil|pendulum|penguin|pen holder|cap|open|pent|spent|pend", (S, "length"), 0.08, 0.2, "pens about 0.13-0.15 m"),
    ("pencil", r"pencil", "case|sharpener|box|holder|cup", (S, "length"), 0.03, 0.2, "pencils: new 0.175-0.19 m, worn down to a few cm"),
    ("toothbrush", r"tooth ?brush", "holder|cup|head", (S, "length"), 0.1, 0.25, "toothbrushes about 0.17-0.2 m; children's about 0.13 m"),
    ("credit card", r"(?:credit|debit|bank|id|business|playing) card", "", (S, "length width"), 0.045, 0.095, "ISO/IEC 7810 ID-1 85.60 x 53.98 mm; playing cards 88.9 x 63.5 mm; business cards 85-89 x 51-55 mm"),
    ("banknote", r"(?:bank ?note|bill|banknote|dollar bill|dollar)", "board|billboard|bill of|billiard", (S, "length"), 0.1, 0.18, "banknotes: USD 156 x 66 mm, EUR 120-160 mm, KRW 136-154 mm long"),
    ("wallet", r"wallet", "", (S, "length width"), 0.07, 0.25, "wallets: bifold about 0.11 x 0.09 m, long wallets about 0.19-0.2 m"),
    ("apple", r"apple", "pie|tree|logo|watch|pencil|macbook|pineapple|crab", (S, "diameter width size height"), 0.045, 0.12, "apples: USDA grades about 57-95 mm; large apples up to about 0.11 m"),
    ("avocado", r"avocado", "", (S, "length"), 0.06, 0.2, "avocados about 0.07-0.15 m long"),
    ("acorn", r"acorn", "", (S, "length diameter size height width"), 0.008, 0.06, "acorns about 1-6 cm (Quercus species range)"),
    ("baguette", r"baguette", "", (S, "length"), 0.3, 1.0, "baguettes: standard about 0.55-0.65 m long"),
    ("cookie", r"cookie|biscuit", "jar|sheet|cutter|box", (S, "diameter width size"), 0.03, 0.15, "cookies about 0.05-0.1 m in diameter"),
    ("coin", r"coin", "slot|purse", (S, "diameter width size"), 0.014, 0.041, "circulating coins: about 15 mm (USD dime 17.9 mm) to about 40 mm"),
    ("backboard", r"backboard", "", (S, "width"), 0.9, 1.85, "FIBA backboard 1.80 m wide, NBA 1.83 m; youth / portable boards from about 0.9 m"),
    ("backboard", r"backboard", "", (S, "height"), 0.6, 1.1, "FIBA backboard 1.05 m high, NBA 1.07 m; small boards about 0.6-0.8 m"),
    ("hoop", r"basketball (?:hoop|rim)|hoop", "hula|ring|cup|glass|mug|bowl|jar|bottle|pot|wheel|tire|tyre|vase|can|bucket|basket of|earring", (S, "diameter"), 0.4, 0.5, "FIBA Official Basketball Rules equipment: rim inner diameter 450-459 mm"),
    ("football pitch", r"(?:football|soccer) (?:pitch|field)", "", (S, "width"), 45.0, 90.0, "IFAB Law 1: pitch width 45-90 m"),
    ("football pitch", r"(?:football|soccer) (?:pitch|field)", "", (S, "length"), 90.0, 120.0, "IFAB Law 1: pitch length 90-120 m"),
    ("basketball court", r"basketball court", "", (S, "length"), 26.0, 29.5, "FIBA court 28 m, NBA 28.65 m long"),
    ("basketball court", r"basketball court", "", (S, "width"), 14.0, 15.5, "FIBA court 15 m, NBA 15.24 m wide"),
    ("tennis court", r"tennis court", "", (S, "length"), 23.0, 24.5, "ITF Rules of Tennis rule 1: court length 23.77 m"),
    ("tennis court", r"tennis court", "", (S, "width"), 8.0, 11.2, "ITF Rules of Tennis rule 1: singles 8.23 m, doubles 10.97 m"),
    ("goal", r"(?:soccer |football )goal(?: ?post)?", "keeper|line|area|kick", (S, "width"), 7.2, 7.4, "IFAB Law 1: goal width 7.32 m"),
    ("goal", r"(?:soccer |football )goal(?: ?post)?", "keeper|line|area|kick", (S, "height"), 2.4, 2.5, "IFAB Law 1: goal height 2.44 m"),
    ("table tennis table", r"(?:ping[ -]?pong|table tennis) table", "", (S, "length"), 2.6, 2.8, "ITTF Handbook 2.1: table 2.74 m long"),
    ("table tennis table", r"(?:ping[ -]?pong|table tennis) table", "", (S, "width"), 1.45, 1.6, "ITTF Handbook 2.1: table 1.525 m wide"),
    ("table tennis table", r"(?:ping[ -]?pong|table tennis) table", "", (S, "height"), 0.74, 0.78, "ITTF Handbook 2.1: playing surface 76 cm above the floor"),
    ("fire hydrant", r"(?:fire )?hydrant", "", (S, "height"), 0.4, 1.3, "above-ground fire hydrants about 0.6-1.0 m (AWWA C502 dry-barrel hydrants; EN 14384)"),
    ("streetlight", r"street ?light|street ?lamp|lamp ?post|light ?pole|lamppost|street lamp post", "bulb|head|shade|fixture|base", (S, "height"), 2.5, 20.0, "street lighting mounting heights about 3-12 m (EN 13201 / IES RP-8 practice); high roadway poles up to about 15-20 m"),
    ("house", r"house|home", "light|warehouse|greenhouse|dog|bird|doll|tree|boat|plant|door|roof|window|wall|step|stairs|green|horse|mouse|play|power|club|opera|court", (S, "height"), 2.5, 25.0, "houses: one storey about 3-5 m to the ridge, three to four storeys about 10-15 m"),
    ("step", r"(?:stair|first |second |third |bottom |top |front )?(?:step|stair)", "foot|steps of|ladder|step ?stool|stepping stone", (S, "height"), 0.08, 0.35, "stair risers: building codes cap about 0.18-0.20 m (IRC R311.7.5.1: 7.75 in); outdoor steps up to about 0.3 m"),
    ("floor tile", r"(?:floor |ceramic |wall )?tile", "roof|tiles of", (S, "side length width"), 0.07, 1.2, "tiles: mosaic about 0.1 m (smaller mosaics are not called tiles here) to large format 1.2 m"),
]
PERSON_RX = r"person|pedestrian|man|woman|walker|player|boy|girl|kid|child|lady|guy|runner|human|people|jogger|swimmer"
KIN = [
    ("swimmer", r"swimmer|(?:swimming )(?:person|man|woman|boy|girl)", "", ("speed avgspeed hvel",), 0.05, 2.5, "50 m freestyle world record 20.91 s: 2.39 m/s; kinematic.class_bound"),
    ("walking person", r"(?:walking |strolling )(?:" + PERSON_RX + r")|pedestrian|walker", "running|jogging|sprint|cycl|bike|ride|riding|skat|scoot|swim", ("speed avgspeed hvel",), 0.05, 3.0, "walking: preferred speed 1.2-1.5 m/s, walk-run transition about 2.0-2.5 m/s; kinematic.class_bound"),
    ("person", r"(?:" + PERSON_RX + r")", "cycl|bike|bicycle|ride|riding|rider|skat|scoot|ski|car|board|swim|horse|motor|wheelchair|ball", ("speed avgspeed hvel",), 0.02, 12.5, "fastest human sprint about 12.4 m/s (U. Bolt, Berlin 2009)"),
    ("car", r"(?<!cable )(?<!train )car", "race|racing|formula|f1|kart|toy|cable|train|rail|roller|rc|remote|model", ("speed avgspeed hvel",), 0.05, 70.0, "road cars: 250 km/h (69 m/s) electronic limit of most fast cars; kinematic.class_bound"),
    ("bicycle", r"(?:bicycle|bike|cyclist|biker)(?<!motorbike)", "motor|e-?bike|electric", ("speed avgspeed hvel",), 0.05, 20.0, "racing cyclists sprint about 20 m/s (72 km/h); kinematic.class_bound"),
    ("horse", r"horse|pony", "rocking|sea ?horse|carousel|toy", ("speed avgspeed hvel",), 0.05, 24.6, "Quarter Horse sprint about 24.6 m/s (88 km/h)"),
    ("dog", r"dog|puppy", "hot ?dog|toy|robot", ("speed avgspeed hvel",), 0.05, 20.0, "greyhound about 19.4 m/s (70 km/h)"),
    ("cat", r"(?<!big )cat|kitten", "big|wild|lion|tiger|leopard|cheetah|toy|robot", ("speed avgspeed hvel",), 0.02, 13.5, "domestic cat about 13.4 m/s (48 km/h)"),
    ("rabbit", r"rabbit|bunny|hare", "toy|robot", ("speed avgspeed hvel",), 0.02, 20.0, "European hare up to about 20 m/s (72 km/h)"),
    ("elephant", r"elephant", "toy|seal", ("speed avgspeed hvel",), 0.02, 7.0, "Hutchinson et al. 2003 (Nature 422): elephants up to about 6.8 m/s"),
    ("ball", r"ball", "toy|robot", ("speed avgspeed hvel",), 0.01, 75.0, "fastest recorded tennis serve 73.2 m/s (263 km/h); kinematic.SPEED3"),
    ("boat", r"boat|ship|speedboat|sailboat|yacht|canoe|kayak|ferry", "toy|model|paper|rc|remote", ("speed avgspeed hvel",), 0.02, 45.0, "high-performance powerboats about 45 m/s (160 km/h); kinematic.SPEED3"),
    ("droplet", r"(?:water |rain ?)?drop(?:let)?|raindrop", "", ("speed avgspeed hvel",), 0.001, 9.2, "Gunn & Kinzer 1949: terminal speed of water drops in air at most 9.2 m/s"),
    ("vehicle accel", r"(?<!cable )(?<!train )car|truck|bus|van|vehicle|forklift|bike|bicycle|motorcycle|scooter", "race|racing|formula|f1|kart|toy|rc|remote|model|rocket|cable|train", ("accel",), 0.01, 15.0, "production cars: 0-100 km/h in about 1.9 s is 14.6 m/s^2; braking about 1-1.2 g; kinematic.class_bound"),
    ("person accel", r"(?:" + PERSON_RX + r")", "cycl|bike|ride|riding|skat|ski|car|board|swim|horse|motor|astronaut|robot|ball", ("accel",), 0.01, 12.0, "sprint starts about 9-10 m/s^2; kinematic.class_bound"),
    ("passive object accel", r"ball|box|cup|mug|book|bottle|can|apple|coin|pen|pencil|card|acorn|block|cube|stone|rock|brick|toy|chess piece|pawn|knight|bishop|rook|eraser|wallet|phone|glass|jar",
     "rocket|robot|car|rc|remote|motor|drone|vehicle|launcher|propell", ("accel",), 0.0, 9.81,
     "an unpowered object pushed by no hand accelerates at most at free-fall g = 9.81 m/s^2 (drag and friction only reduce it)"),
]
NOT_ALL = r"\b(?:toy|model|miniature|mini|lego|figurine|statue|sculpture|cartoon|drawing|picture|poster|painting|photo|image|logo|sticker|print|replica|inflatable|plush|stuffed|doll|origami|shadow|reflection|icon|emoji|robot|robotic|rc|remote|animated|animation|pixel|screen|display|text|letter|lettering|word|reflection|projection|hologram|cardboard|cutout|cut-out|ornament|decoration|keychain|magnet|sign|symbol|mascot|costume|balloon|cake|candy|chocolate|cookie cutter|shaped|clay|wooden toy)\b"
CUT = re.compile(r"\s+(?:of|on|in|at|that|which|placed|with|near|next|to|from|behind|under|above|below|inside|beside|between|across|along|attached|held|carried|hanging|sitting|standing|lying|located|by|for|where|while|who|whose|being|is|are|was|were)\b|,")
USI = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "m/s": 1.0, "cm/s": 0.01, "mm/s": 0.001, "km/h": 1 / 3.6,
       "m/s^2": 1.0, "cm/s^2": 0.01, "mm/s^2": 0.001}
DIMSYN = {"tall": "height", "long": "length", "wide": "width", "size": "size"}


def core(phrase):
    p = re.sub(r"\(.*?\)", " ", str(phrase or "").lower())
    p = p.replace("“", '"').replace("”", '"')
    p = re.sub(r"\s+", " ", p).strip(" .?\"'")
    p = re.sub(r"^(?:the|a|an)\s+", "", p)
    m = CUT.search(p)
    if m:
        p = p[:m.start()]
    return p.strip()


def lookup(question, x):
    q = x.get("q") or {}
    kind, dim = q.get("kind"), (q.get("dim") or "")
    cp = core(q.get("target") or "")
    if not cp:
        return None, cp
    ql = str(question).lower()
    if re.search(NOT_ALL, cp) or re.search(r"\b(?:toy|model|miniature|lego|figurine|statue|sculpture|cartoon|drawing|picture|poster|painting|replica|robot|rc car|remote[- ]controlled)\b", ql):
        return None, cp
    if kind == "size":
        d = DIMSYN.get(dim, dim)
        for e in TABLE:
            cls, rx, nx, (k, dims), lo, hi, src = e
            if d not in dims.split():
                continue
            if not re.search(r"(?:^|\s)(?:" + rx + r")s?$", cp):
                continue
            if nx and (re.search(nx, cp) or re.search(nx, ql)):
                continue
            return e, cp
        return None, cp
    if kind in ("speed", "avgspeed", "hvel", "accel"):
        for e in KIN:
            cls, rx, nx, (ks,), lo, hi, src = e
            if kind not in ks.split():
                continue
            if not re.search(r"(?:^|\s)(?:" + rx + r")s?$", cp):
                continue
            if nx and (re.search(nx, cp) or re.search(nx, ql)):
                continue
            if cls.endswith("accel") and re.search(r"gravit|=\s*9\.8", str(x.get("prior_text") or "").lower()) and cls != "passive object accel":
                continue
            return e, cp
    return None, cp


def prior_scaled(x):
    for p in x.get("priors") or []:
        if p.get("kind") != "size" or not p.get("value_si"):
            continue
        obj = p.get("obj") or ((x.get("prior_obj") or {}).get("phrase")) or ""
        e, _ = lookup(str(p.get("raw") or ""), {"q": {"kind": "size", "dim": p.get("dim") or "", "target": obj}})
        if e is not None and not (e[4] <= float(p["value_si"]) <= e[5]):
            return True
    return False


def si_factor(x):
    u = ((x.get("q") or {}).get("unit") or "m").strip().lower().replace("²", "^2")
    return USI.get(u, np.nan)


def flag(v_si, lo, hi, m=MARGIN):
    return bool(np.isfinite(v_si) and (v_si <= 0 or v_si < lo / m or v_si > hi * m))


def choose(cands, lo, hi, m=MARGIN):
    for nm, c in cands:
        if np.isfinite(c) and c > 0 and lo <= c <= hi:
            return nm, c, "in"
    for nm, c in cands:
        if np.isfinite(c) and c > 0 and lo / m <= c <= hi * m:
            return nm, c, "near"
    return None




def main():
    BASE, CUR, GT = sys.argv[1:4]
    det = pd.read_csv(f"{R}/submissions/{BASE}_detail.csv", index_col=0)
    cur = pd.read_csv(f"{R}/submissions/{CUR}_id1.csv").parsed_value.values
    cf = pd.read_csv(f"{R}/preds/test_caw9b_orig.csv").parsed_value.values
    cc = pd.read_csv(f"{R}/preds/test_caw9b_orig_crop.csv").parsed_value.values
    gu = pd.read_csv(f"{R}/preds/test_geo_v4nt{GT}usrngsr_d_vw_raw.csv").geo_value.values
    P = [json.loads(l) for l in open(f"{R}/geo/out/parse_test_v2.jsonl")]
    assert len(P) == len(det) == len(cf) == len(cc) == len(gu) == len(cur)
    assert [int(x["id"]) for x in P] == [int(k) for k in det.index]
    out = {}
    for i in range(len(det)):
        x = P[i]
        e, cp = lookup(x["question"], x)
        if e is None:
            continue
        f = si_factor(x)
        a = float(cur[i]) * f
        cls, rx, nx, kd, lo, hi, src = e
        if not flag(a, lo, hi) or prior_scaled(x):
            continue
        fus, g, cs = float(det.answer.iloc[i]), float(det.g.iloc[i]), float(det.cs.iloc[i])
        other = cf[i] if np.isfinite(cs) and np.isclose(cs, cc[i]) and not np.isclose(cc[i], cf[i]) else cc[i]
        cands = [("fusion", fus * f), ("g", g * f), ("cs", cs * f), ("caw_other", float(other) * f), ("g_u", float(gu[i]) * f)]
        ch = choose(cands, lo, hi)
        if ch is None:
            continue
        out[str(int(det.index[i]))] = dict(value=ch[1] / f)
    json.dump(out, open(f"{R}/cache/tmp/ag_{TAG}_patch_test.json", "w"), indent=0)
    print("rul336 written", len(out))


if __name__ == "__main__":
    main()
