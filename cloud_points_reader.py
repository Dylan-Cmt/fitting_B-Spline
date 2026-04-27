def cloud_points_reader(input_file):
    # opening the file in read mode
    my_file = open(input_file)
    # reading the file
    data = my_file.readlines()
    # store the points
    points = []
    for line in data:
        line_tmp = line.strip().split(' ')
        point = (float(line_tmp[0]), float(line_tmp[1]))
        points.append(point)
    # closing file
    my_file.close()
    # return the points
    return points

points = cloud_points_reader("apollo.txt")
# check if we have the right number of points
assert(len(points) == 140)

