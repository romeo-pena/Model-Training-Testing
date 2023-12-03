def reduce_id_labels(img):
    # Decrease the class count of cityscapes (make similar to fant city)
    #
    # CS_CLASSES = ['road' 0, 'sidewalk','building' 3, 'wall' 3, 'fence', 'pole', 'traffic_light', 'traffic_sign',
    #               'vegetation' 8, 'terrain', 'sky' 5, 'person', 'rider', 'car' 14, 'truck' 15, 'bus' 16, 'train',
    #               'motorcycle', 'bicycle']

    newImg = img
    newImg[newImg > 19] = 19

    # print(img)
    for i in range(0, 20):
        if i != 0 and i != 2 and i != 8 and i != 13 and i != 14 and i != 15 and i != 3 and i != 6 and i != 7 \
                and i != 5 and i != 4:
            # Other
            newImg[newImg == i] = 5
        elif i == 2 or i == 3:
            # building
            newImg[newImg == i] = 1
        elif i == 6 or i == 7 or i == 4 or i == 5:
            # Traffic Signs/Lights
            newImg[newImg == i] = 4
        elif i == 8:
            # tree?
            newImg[newImg == i] = 2
        elif i == 13:
            # car
            newImg[newImg == i] = 3
        elif i == 14:
            # car
            newImg[newImg == i] = 3
        elif i == 15:
            # car
            newImg[newImg == i] = 3

            # 1 road
            # 2 building
            # 3 tree
            # 4 car
            # 5 traffic
            # 6 other
    # print(newImg)
    return newImg



def reduce_id_labels_sky(img):
    newImg = img
    newImg[newImg > 19] = 19

    # print(img)
    for i in range(0, 20):
        if i != 10:
            # Other
            newImg[newImg == i] = 5
        else:
            # building
            newImg[newImg == i] = 0

    return newImg