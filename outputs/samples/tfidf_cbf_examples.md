# TF-IDF CBF sample recommendations

Item text: `content_text_basic` from `data/items.csv`. 3 users picked at random (seed=42) from evaluable users with a profile.
Train interactions and mean train rating are computed over all users in `interactions_train.csv`.

## User `AGO3MQHEHK3XE2VJWGXYARGOVDPQ`

Validation hits: 0 of 3 relevant items.

### Liked in train (rating >= 4, 7 items)

| item_id | title | user rating | train interactions | mean train rating |
|---|---|---|---|---|
| B0016HDP58 | Gibraltar SC-CMBAC Medium Cymbal Boom Attachment Clamp | 5 | 56 | 4.93 |
| B014P7CDDU | Musiclily Pickguard Strat Back Plate Tremolo Cavity Cover Vintage Style Backplate for Fender US/Mexico Made Standard Stratocaster Modern Style Electric Guitar,4Ply White Pearl | 5 | 36 | 4.31 |
| B07DVX841V | Musiclily Pro 6 inline Guitar Locking Tuners String Tuning Pegs Keys Machines Heads Set for Stratocaster Telecaster Style Electric Guitar, Gold | 5 | 96 | 4.41 |
| B0B92LPGD2 | Fender American Series String Guides | 5 | 210 | 4.38 |
| B083KJPCVV | Ubblove Drumsticks 5A 2 pairs with ANTI-SLIP Handles for Drum Light Durable Wooden Drum 2 Pair Drum Sticks for Kids Adults Musical Instrument Percussion Accessories | 5 | 5 | 4.60 |
| B0B8Z6192W | Fender 5-way Selector Switch | 5 | 123 | 4.78 |
| B015IJIO5U | VOX AP2BS amPlug 2 Guitar/Bass Headphone Amplifier Black | 5 | 538 | 4.39 |

### Top-10 recommendations

| rank | item_id | title | score | train interactions | mean train rating | relevant in validation |
|---|---|---|---|---|---|---|
| 1 | B014ZZW8QE | Musiclily 6 Hole Strat Back Plate Tremolo Cavity Cover Backplate for Fender US/Mexico Made Standard ST Stratocaster Modern Style Electric Guitar, 4ply Black Pearl | 0.1918 | 76 | 4.55 |  |
| 2 | B07MF2HPQV | Musiclily Guitar Flat Wavy Strat String Guides String Trees Retainer for Fender American Vintage Stratocaster Parts, Gold (Pack of 4) | 0.1888 | 10 | 4.50 |  |
| 3 | B00GGMWAJU | Musiclily 6-in-line Guitar Locking Tuners Machine Head Tuning Pegs Keys Set for Right Hand Fender Strat Stratocaster Tele Telecaster Electric Guitar Parts, Black | 0.1861 | 37 | 3.70 |  |
| 4 | B01MD10XAU | Musiclily 13 Hole P Bass Pickguard Precision Bass Scratch Plate Pick Guards for 4 String Fender USA/Meixican American Standard Precision Bass Guitar, 4Ply Black Pearl | 0.1809 | 91 | 4.44 |  |
| 5 | B078NTTC3M | Musiclily Pro 10 Hole Jazz Bass Pickguard J Bass Scratch Plate for 4 String Fender American Standard Jazz Bass Modern Style, 4Ply Red Tortoise | 0.1803 | 14 | 4.50 |  |
| 6 | B07DCM96L3 | Musiclily Pro 6 in Line Semi Closed Guitar Tuners Tuning Pegs Machine Heads Set for Strat or Tele Style Replacement, Black | 0.1770 | 11 | 4.55 |  |
| 7 | B015CNC9TY | Musiclily HSS 11 Holes Strat Electric Guitar Pickguard and BackPlate Set for Fender US/Mexico Made Standard Stratocaster Modern Style Guitar Parts,4Ply Pearl Purple | 0.1760 | 26 | 4.54 |  |
| 8 | B00GGMWA5O | Musiclily 6 Inline Guitar Locking Tuners 6R String Tuning Pegs Keys Machine Heads Set for Fender Stratocaster Telecaster Parts, Chrome | 0.1754 | 56 | 3.70 |  |
| 9 | B07BHMGDZT | Musiclily 6 inline Semi Closed Guitar Tuners Tuning Pegs Keys Machine Heads Set for Squier Strat Electric Guitar, Black | 0.1736 | 2 | 4.00 |  |
| 10 | B0B8TL53TG | Fender Neck Plate Mounting Screws - American Standard/Deluxe Guitars | 0.1734 | 3 | 5.00 |  |

## User `AEFDMLY3LCEB4U7Y5IFOUFNQ4ZTQ`

Validation hits: 0 of 2 relevant items.

### Liked in train (rating >= 4, 5 items)

| item_id | title | user rating | train interactions | mean train rating |
|---|---|---|---|---|
| B001GH4H3W | [OLD MODEL] Vox AmPlug Bass Guitar Headphone Amp | 5 | 29 | 3.90 |
| B007IE5TQG | Hohner Special 20 Harmonica 3 piece Pro Pack keys of G,C,A | 5 | 15 | 4.53 |
| B09JXTHJKK | Mendini By Cecilio Eb Alto Saxophone - Case, Tuner, Mouthpiece, 10 Reeds, Pocketbook- Nickel E Flat Musical Instruments | 5 | 64 | 4.28 |
| B07WWZKZY9 | Amazon Basics Sax Stand with Flute/Clarinet Peg | 5 | 13 | 4.00 |
| B086W93GHN | Eno Solid Big Clip-on Professional Wind Instruments Alto Saxophone Tuner, Colorful LCD Display, Saxophone Accessories | 4 | 20 | 4.30 |

### Top-10 recommendations

| rank | item_id | title | score | train interactions | mean train rating | relevant in validation |
|---|---|---|---|---|---|---|
| 1 | B002PHVAQY | [DISCONTINUED] VOX APAG AmPlug Acoustic Guitar Headphone Amplifier | 0.1619 | 5 | 4.20 |  |
| 2 | B099ZS165M | Cecilio Alto Saxophone Reeds, 10-pack with Individual Plastic Case, Strength 3.0 | 0.1594 | 20 | 4.35 |  |
| 3 | B0002F7IQ2 | On-Stage Flute Peg (SXS7101B), Black | 0.1540 | 8 | 4.62 |  |
| 4 | B001EEO8IQ | Hamilton Alto/Tenor sax Stand, Black, includes Flute/Clarinet peg | 0.1485 | 63 | 4.56 |  |
| 5 | B01542U79M | Mendini by Cecilio Tenor Saxophone, L+92D B Flat, Case, Tuner, Mouthpiece, Gold | 0.1457 | 25 | 4.00 |  |
| 6 | B07KDYCQNC | HOHNER Harmonica, Stainless steel (3P1501BX) | 0.1375 | 12 | 4.00 |  |
| 7 | B0755Q1FFM | 【The Best Deal】OriGlam 10pcs Alto Sax Saxophone Reeds 2.5 Reed, Alto bE Saxophone Reeds Lade Bamboo 2-1/2 Reed Strength 2.5 for Clarinet, Soprano or Alto Sax | 0.1364 | 4 | 3.75 |  |
| 8 | B000V6767I | HollywoodWinds Curved Soprano Sax Peg | 0.1359 | 11 | 4.00 |  |
| 9 | B002XNN8B0 | Hohner Accordions 1501/7 Bluesband Harmonica 7-Piece Set with Carrying Case, Chrome | 0.1342 | 8 | 4.25 |  |
| 10 | B003K16N0C | Hohner Accordions 1501/7 Bluesband Harmonica 7-Piece Set with Carrying Case, Chrome | 0.1342 | 34 | 3.82 |  |

## User `AH4PQ6UJM6FVZXFOC7AFZY2G4B4Q`

Validation hits: 0 of 4 relevant items.

### Liked in train (rating >= 4, 21 items)

| item_id | title | user rating | train interactions | mean train rating |
|---|---|---|---|---|
| B07CZMZRWN | Soulo SM8525 Trumpet Harmon Style Wah Wah Mute | 5 | 30 | 4.83 |
| B01FRN3N1C | Gammon Percussion Full Size Complete Adult 5 Piece Drum Set with Cymbals Stands Stool and Sticks, Black | 5 | 51 | 4.16 |
| B01M18LT95 | Gearlux Attachable Music Stand Arm - Small | 5 | 29 | 4.28 |
| B077J6ZGD5 | Hola! Music HM-MTH Microphone Music Stand iPad Tablet Smartphone Holder Mount - Fits Devices from 6 to 15 Inch | 5 | 300 | 4.72 |
| B00SSVYYXW | Clarinet Oboe Soprano Saxophone Electronic Torch Sax Gig Bag Hot | 5 | 9 | 4.44 |
| B0BMV8CDXJ | Otto Link AOL OLMT6 Accordion Accessory (OLM-404-6) | 5 | 7 | 4.43 |
| B09VP8Y3MV | AKAI Professional EWI Solo - Electronic Wind Instrument With Built-in Speaker, Rechargeable Battery, 200 Sounds and USB MIDI Connectivity,Black | 5 | 13 | 3.46 |
| B00KDE3U5K | AxeRig LIFETIME WARRANTY Clip-On Chromatic Guitar Tuner for Acoustic, Bass, 6 & 12 string Guitars, Banjo, Mandolin, Ukulele, Violin, Cello, Trumpet, Brass, Sax, Flute, Woodwinds - SPARE BATTERY | 5 | 78 | 4.45 |
| B004LLV04E | PreSonus AudioBox Studio with Headphones, Microphone, Mic Cable, USB Cable, and StudioOne Artist Software (Download) | 5 | 7 | 4.14 |
| B0C34Z8KSM | Hohner 1896BX Marine Band Harmonic, Key of C Sharp | 5 | 11 | 4.91 |
| ... | (+11 more) | | | |

### Top-10 recommendations

| rank | item_id | title | score | train interactions | mean train rating | relevant in validation |
|---|---|---|---|---|---|---|
| 1 | B07NBTFB4B | East top Harmonica, Blues Harmonica Key of E, 10 Holes 008K Diatonic Blues harp Mouth Organ Harmonica with Silver Cover, Standard Harmonicas For Adults, Professionals and Students | 0.1168 | 8 | 4.62 |  |
| 2 | B07SB4413B | East top Diatonic Harmonica Key of D 10 Holes 20 Tones 008K Blues Harp Mouth Organ Harmonica with Black Cover, Top Grade Harmonica for Adults, Professionals and Students | 0.1164 | 19 | 4.68 |  |
| 3 | B01FJSEX3M | East top Diatonic Harmonica Key of C 10 Holes 20 Tones 008K Blues Harp Mouth Organ Harmonica with Black Cover, Top Grade Harmonica for Adults, Professionals and Students | 0.1164 | 3 | 4.67 |  |
| 4 | B08GS9C74W | East top Harmonica Set of 7, 10 Holes Blues Harp Mouth Organ Diatonic Blues Harmonica Set For Adults, Professionals and Students, as a Gift | 0.1152 | 3 | 4.00 |  |
| 5 | B07N8V6K46 | East top Diatonic Harmonica Key of Eb, 10 Holes 20 Tones 008K Diatonic Blues Harp Mouth Organ Harmonica with Blue Case, Standard Harmonica For Adults, Professionals and Students | 0.1150 | 26 | 4.50 |  |
| 6 | B07NQ8X3FV | East top Diatonic Blues Harmonica key of Paddy C, 10 Holes 20 Tones 008K Blues Harp Diatonic Mouth Organ Harmonica, Standard Harmonica for Adults, Professionals and Students | 0.1124 | 18 | 4.44 |  |
| 7 | B07MX8JD5M | East top 16 Holes 64 Tones Chromatic Harmonica Key of C, Chromatic Mouth Organ Harmonica For Adults, Professionals and Students (BK) | 0.1076 | 7 | 3.00 |  |
| 8 | B07NBMCG9T | East top Blues Harmonica, 10 Holes 20 Tones Diatonic Blues Mouth Organ Harmonica Key of G with White Cover, T008L Harmonica for Adults, Professionals and Students | 0.1037 | 10 | 4.80 |  |
| 9 | B07PXQXX4V | East top Harmonica, Diatonic Blues Deluxe Harmonica Key of C, 10 Holes 20 Tones Blues Harp Mouth Organ Harmonica For Adults, Beginners, Professionals and Students（Silver grey） | 0.0983 | 8 | 4.50 |  |
| 10 | B07NQBJPNP | East top Tremolo Harmonica Key of F#, 24 Holes Professional Tremolo Mouth Organ T2406S Harmonica For Adults, Professionals and Students | 0.0976 | 4 | 4.50 |  |
