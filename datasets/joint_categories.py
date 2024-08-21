import numpy as np

categories = [
    "Hips", #0
    "Spine", #1
    "SpineEnd", #2
    "Neck", #3
    "Head", #4
    "LeftUpLeg", #5
    "LeftLeg", #6
    "LeftFoot", #7
    "LeftToeBase", #8
    "RightUpLeg", #9
    "RightLeg", #10
    "RightFoot", #11
    "RightToeBase", #12
    "LeftShoulder", #13
    "LeftArm", #14
    "LeftForeArm", #15
    "LeftHand", #16
    "RightShoulder", #17
    "RightArm", #18
    "RightForeArm", #19
    "RightHand" #20
]

secondary_categories = [
    'Hips',
    'Head',
    'Neck',
    'SpineEnd',
    'Arm'
]

arm_categories = [
    'LeftShoulder',
    'LeftArm',
    'LeftForeArm',

    'RightShoulder',
    'RightArm',
    'RightForeArm'
]

kpts_parent_id = [
    -1,
    0,
    1,
    2,
    3,
    0,
    5,
    6,
    7,
    0,
    9,
    10,
    11,
    2,
    13,
    14,
    15,
    2,
    17,
    18,
    19
]


cat_idx = {}
for i in range(len(categories)):
    cat_idx[categories[i]] = i

cent_idx = []
left_idx = []
right_idx = []
for i, cat in enumerate(categories):
    if 'Left' in cat:
        left_idx.append(i)
    elif 'Right' in cat:
        right_idx.append(i)
    else:
        cent_idx.append(i)

cats_np = np.array(categories)
# print(cent_idx, left_idx, right_idx)

end_cats = ['Head', 'LeftToeBase', 'RightToeBase', 'LeftHand', 'RightHand']
end_idx = []
for i, cat in enumerate(categories):
    if cat in end_cats:
        end_idx.append(i)



seg_cls = ['Hair'
           'Hat',
           'Ear',
           'Tie',
           'LeftSleeve',
           'RightSleeve',
           'Dress',
           'Skirt',
           'Tail']


