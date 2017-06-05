import os
import os.path
from shutil import copy

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def make_paths(path):
    image_path=path+"/images/"
    skl_path=path+"/skeleton/"
    robot_path=path+"/robot/"
    depth_path=path+"/depth/"
    if not os.path.exists(image_path): os.makedirs(image_path)
    if not os.path.exists(skl_path): os.makedirs(skl_path)
    if not os.path.exists(depth_path): os.makedirs(depth_path)
    if not os.path.exists(robot_path): os.makedirs(robot_path)
    return image_path, skl_path, robot_path, depth_path

Dir='./no_consent/'
Dir2='output/'
if not os.path.exists(Dir2):
    os.makedirs(Dir2)

#if not os.path.exists(Dir2+"making_tea"):
#    os.makedirs(Dir2+"making_tea")

i=1
vids_in=1
incr=0
vis = 1

date_dirs=get_immediate_subdirectories(Dir)
for date_dir in date_dirs:

    #if date_dir not in ['2017-02-23', '2017-02-24']: continue
    dirs=get_immediate_subdirectories(Dir+date_dir)

    for dir in dirs:
        vids_in+=1
        if vis: print dir
        fname=Dir+date_dir+"/"+dir+"/labels.txt"
        if not os.path.isfile(fname): continue
        f=open(fname,'r')
        lines=f.readlines()
        if len(lines) > 4:
            incr+=1
            # print "\n",Dir, date_dir, dir
            # import pdb; pdb.set_trace()

        for line in lines:
            if vis: print line
            elements=line.split(":")

            label=elements[1]
            frames=elements[2].replace("\n","").split(",")
            sf=frames[0]
            ef=frames[1]
            # print [label,sf,ef]
            #if label=="making_tea":
            #    path=Dir2+"making_tea/vid"+str(i)
            #else:
            path=Dir2+"vid"+str(i)
            i+=1

            if not os.path.exists(path):
                os.makedirs(path)

            label_f=open(path+'/label.txt','w')
            label_f.write(date_dir+"/"+dir+"\r\n"+label)

            image_path, skl_path, robot_path, depth_path = make_paths(path)
            # for j in range(int(sf),int(ef)+1):
            #     copy(Dir+date_dir+"/"+dir+"/rgb/rgb_"+"%05d" % j+".jpg", image_path+"%05d"%(j-int(sf)+1)+".jpg")
            #     copy(Dir+date_dir+"/"+dir+"/depth/depth_"+"%05d" % j+".jpg", depth_path+"%05d"%(j-int(sf)+1)+".jpg")
            #     copy(Dir+date_dir+"/"+dir+"/robot/robot_"+"%05d" % j+".txt", robot_path+"%05d"%(j-int(sf)+1)+".txt")
            #     try:
            #         copy(Dir+date_dir+"/"+dir+"/cpm_skeleton/cpm_skl_"+"%05d" % j+".txt", skl_path+"%05d"%(j-int(sf)+1)+".txt")
            #     except:
            #         pass

print incr
if vis:
    print "vids in: %s" %vids_in
    print "clips out: %s" %i
