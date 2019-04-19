OTB50=['Basketball', 'Biker', 'Bird1', 'BlurBody', 'BlurCar2',
       'BlurFace', 'BlurOwl', 'Bolt', 'Box', 'Car1',
       'Car4', 'CarDark', 'CarScale', 'ClifBar', 'Couple',
       'Crowds', 'David', 'Deer', 'Diving', 'DragonBaby',
       'Dudek', 'Football', 'Freeman4', 'Girl', 'Human3',
       'Human4', 'Human6', 'Human9', 'Ironman', 'Jump',
       'Jumping', 'Liquor', 'Matrix', 'MotorRolling', 'Panda',
       'RedTeam', 'Shaking', 'Singer2', 'Skating1', 'Skating2',
       'Skating2-2', 'Skiing', 'Soccer', 'Surfer', 'Sylvester',
       'Tiger2', 'Trellis', 'Walking', 'Walking2', 'Woman']

OTB100_rest=['Bird2', 'BlurCar1', 'BlurCar3', 'BlurCar4', 'Board',
             'Bolt2', 'Boy', 'Car2', 'Car24', 'Coke',
             'Coupon', 'Crossing', 'Dancer', 'Dancer2', 'David2',
             'David3', 'Dog', 'Dog1', 'Doll', 'FaceOcc1',
             'FaceOcc2', 'Fish', 'FleetFace', 'Football1', 'Freeman1',
             'Freeman3', 'Girl2', 'Gym', 'Human2', 'Human5',
             'Human7', 'Human8', 'Jogging-1', 'Jogging-2', 'KiteSurf',
             'Lemming', 'Man', 'Mhyang', 'MountainBike', 'Rubik',
             'Singer1', 'Skater', 'Skater2', 'Subway', 'Suv',
             'Tiger1', 'Toy', 'Trans', 'Twinnings', 'Vase']

OTB100=OTB50+OTB100_rest

IV=['Basketball', 'Box', 'Car1', 'Car2', 'Car24',
    'Car4', 'CarDark', 'Coke', 'Crowds', 'David',
    'Doll', 'FaceOcc2', 'Fish', 'Human2', 'Human4', 'Human7',
    'Human8', 'Human9', 'Ironman', 'KiteSurf', 'Lemming',
    'Liquor', 'Man', 'Matrix', 'Mhyang', 'MotorRolling',
    'Shaking','Singer1', 'Singer2', 'Skating1', 'Skiing',
    'Soccer', 'Sylvester', 'Tiger1', 'Tiger2', 'Trans', 'Trellis', 'Woman']

SV=['Biker', 'BlurBody', 'BlurCar2', 'BlurOwl', 'Board',
    'Box', 'Boy', 'Car1', 'Car24', 'Car4', 'CarScale',
    'ClifBar', 'Couple', 'Crossing', 'Dancer', 'David',
    'Diving', 'Dog', 'Dog1', 'Doll', 'DragonBaby',
    'Dudek', 'FleetFace', 'Freeman1', 'Freeman3', 'Freeman4',
    'Girl', 'Girl2', 'Gym', 'Human2', 'Human3', 'Human4',
    'Human5', 'Human6', 'Human7', 'Human8', 'Human9', 'Ironman',
    'Jump', 'Lemming', 'Liquor', 'Matrix', 'MotorRolling',
    'Panda', 'RedTeam', 'Rubik', 'Shaking', 'Singer1', 'Skater',
    'Skater2', 'Skating1', 'Skating2', 'Skating2-2',
    'Skiing', 'Soccer', 'Surfer', 'Toy', 'Trans', 'Trellis', 'Twinnings', 'Vase',
    'Walking', 'Walking2', 'Woman']

OCC=['Basketball', 'Biker', 'Bird2', 'Bolt', 'Box',
     'CarScale', 'ClifBar', 'Coke', 'Coupon', 'David', 'David3', 'Doll',
     'DragonBaby', 'Dudek', 'FaceOcc1', 'FaceOcc2', 'Football', 'Freeman4', 'Girl',
     'Girl2', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7', 'Ironman',
     'Jogging-1', 'Jogging-2', 'Jump', 'KiteSurf', 'Lemming', 'Liquor', 'Matrix', 'Panda',
     'RedTeam', 'Rubik', 'Singer1', 'Skating1', 'Skating2', 'Skating2-2', 'Soccer',
     'Subway', 'Suv', 'Tiger1', 'Tiger2', 'Trans', 'Walking', 'Walking2', 'Woman']

DEF=['Basketball', 'Bird1', 'Bird2', 'BlurBody', 'Bolt', 'Bolt2',
     'Couple', 'Crossing', 'Crowds', 'Dancer', 'Dancer2',
     'David', 'David3', 'Diving', 'Dog', 'Dudek', 'FleetFace', 'Girl2',
     'Gym', 'Human3', 'Human4', 'Human5', 'Human6', 'Human7', 'Human8', 'Human9',
     'Jogging-1', 'Jogging-2', 'Jump', 'Mhyang', 'Panda', 'Singer2', 'Skater', 'Skater2', 'Skating1', 'Skating2',
     'Skating2-2', 'Skiing', 'Subway', 'Tiger1', 'Tiger2', 'Trans', 'Walking', 'Woman']

GRAY=['Car1','Car4','ClifBar','Dudek','Football','Jumping','Freeman4','Car2','Coupon','Dancer','Dancer2',
      'David2','Dog1','FaceOcc2','FleetFace','Freeman1','Freeman3','Man','Mhyang','Skater',
      'SKater2','Suv','Toy','Twinnings','Vase']

COLOR=[item for item in OTB100 if item not in GRAY]

OTB2013=['Ironman','Matrix','MotorRolling','Soccer','Skiing','Freeman4','Freeman1','Skating1','Tiger2','Liquor',
         'Coke','Football','FleetFace','Couple','Tiger1','Woman','Bolt','Freeman3','Basketball','Lemming',
         'Singer2','Subway','CarScale','David3','Shaking','Sylvester','Girl','Jumping','Trellis','David',
         'Boy','Deer','FaceOcc2','Dudek','Football1','Suv','Jogging-1','Jogging-2','MountainBike','Crossing','Singer1',
         'Dog1','Walking','Walking2','Doll','Car4','David2','CarDark','Mhyang','FaceOcc1','Fish']

